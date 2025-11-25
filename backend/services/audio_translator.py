import os
import tempfile
import subprocess
import logging
import requests
import re
import uuid
from app.core.secrets_manager import secrets_manager
from app.services.s3_uploader import s3_uploader
from app.core.config import settings
from pydub import AudioSegment
import torch
import torchaudio as ta
import assemblyai as aai
from num2words import num2words

logger = logging.getLogger(__name__)

class AudioTranslatorServiceUpdate:
    def __init__(self):
        self.deepl_api_key = secrets_manager.get("DEEPL_API_KEY")
        self.assemblyai_api_key = secrets_manager.get("ASSEMBLYAI_API_KEY")
        
        # Configure AssemblyAI
        aai.settings.api_key = self.assemblyai_api_key
        
        # Language mapping from frontend codes to DeepL codes  
        self.target_language_map = {
            "english": "EN",
            "spanish": "ES",
            "french": "FR", 
            "german": "DE",
            "italian": "IT",
            "portuguese": "PT",
            "chinese": "ZH",
            "japanese": "JA",
            "korean": "KO",
            "arabic": "AR",
            "russian": "RU"
        }
        
        # Language mapping from frontend codes to Chatterbox language IDs
        self.chatterbox_language_map = {
            "english": "en",
            "spanish": "es",
            "french": "fr", 
            "german": "de",
            "italian": "it",
            "portuguese": "pt",
            "chinese": "zh",
            "japanese": "ja",
            "korean": "ko",
            "arabic": "ar",
            "russian": "ru"
        }
        
        # Initialize TTS models
        self._init_tts_models()
    
    def _init_tts_models(self):
        """Initialize Chatterbox TTS models - prioritize multilingual model"""
        try:
            from chatterbox.tts import ChatterboxTTS
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS
            
            # Force CPU device for server deployment
            device = "cpu"
            
            logger.info(f"Loading Chatterbox models on {device}")
            
            # Prioritize multilingual model for better quality and language support
            self.multilingual_model = ChatterboxMultilingualTTS.from_pretrained(device=device)
            
            # Load normal model as fallback
            self.normal_model = ChatterboxTTS.from_pretrained(device=device)
            
            logger.info("Chatterbox multilingual and normal models loaded successfully")
            
        except ImportError:
            logger.warning("ChatterboxTTS not available")
            self.multilingual_model = None
            self.normal_model = None
        except Exception as e:
            logger.error(f"Failed to load Chatterbox models: {str(e)}")
            self.multilingual_model = None
            self.normal_model = None
    
    def extract_audio_from_video(self, input_video_path, output_audio_path):
        """Extract audio from video using ffmpeg"""
        # Validate paths to prevent path traversal
        if not os.path.abspath(input_video_path).startswith(tempfile.gettempdir()):
            raise ValueError("Invalid input video path")
        if not os.path.abspath(output_audio_path).startswith(tempfile.gettempdir()):
            raise ValueError("Invalid output audio path")
            
        os.makedirs(os.path.dirname(output_audio_path), exist_ok=True)
        command = [
            "/usr/bin/ffmpeg", "-y", "-threads", "2", "-i", input_video_path,
            "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2",
            output_audio_path
        ]
        subprocess.run(command, check=True, timeout=120)
    
    def separate_audio_stems(self, audio_path, temp_dir):
        """Separate audio into vocals and ambient using Demucs"""
        try:
            from demucs.pretrained import get_model
            from demucs.apply import apply_model
            
            # Load Demucs model
            model = get_model("htdemucs")  # best quality
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)
            
            # Load audio (keep stereo if present)
            wav, sr = ta.load(audio_path)
            if wav.shape[0] > 2:   # ensure max 2 channels
                wav = wav[:2, :]
            
            # Apply separation
            with torch.no_grad():
                sources = apply_model(model, wav[None].to(device), device=device)[0].cpu()
            
            # Stems: ["drums", "bass", "other", "vocals"]
            stem_names = model.sources
            stems = {name: source for name, source in zip(stem_names, sources)}
            
            # Save vocals (speech/singing)
            vocals_path = os.path.join(temp_dir, "vocals.wav")
            ta.save(vocals_path, stems["vocals"], sr)
            
            # Combine other stems into ambient
            ambient = stems["drums"] + stems["bass"] + stems["other"]
            ambient_path = os.path.join(temp_dir, "ambient.wav")
            ta.save(ambient_path, ambient, sr)
            
            logger.info(f"Audio separation completed: {vocals_path}, {ambient_path}")
            return vocals_path, ambient_path
            
        except ImportError:
            logger.warning("Demucs not available, using original audio")
            vocals_path = os.path.join(temp_dir, "vocals.wav")
            ambient_path = os.path.join(temp_dir, "ambient.wav")
            
            # Copy original audio as vocals
            import shutil
            shutil.copy(audio_path, vocals_path)
            
            # Create silent ambient
            original_audio = AudioSegment.from_wav(audio_path)
            silent_ambient = AudioSegment.silent(duration=len(original_audio))
            silent_ambient.export(ambient_path, format="wav")
            
            return vocals_path, ambient_path
    
    def create_segment_transcript_with_samples_assemblyai(self, audio_file, sample_dir):
        """
        Use AssemblyAI for transcription with speaker-based segmentation
        Creates voice samples for each speaker turn (segment = speaker turn)
        Each segment gets audio that contains ALL concatenated speech from that speaker
        """
        
        # Ensure sample directory exists
        os.makedirs(sample_dir, exist_ok=True)
        
        # Configure language detection options for better accuracy
        language_options = aai.LanguageDetectionOptions(
            expected_languages=["en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh", "nl", "pl", "sv", "da", "no", "fi", "cs", "hu", "ro", "bg", "el", "sk", "sl", "et", "lv", "lt"],
            fallback_language="auto"
        )
        
        # Configure AssemblyAI transcription WITH speaker diarization for speaker-turn segments
        config = aai.TranscriptionConfig(
            speaker_labels=True,  # ENABLE speaker diarization for speaker-turn based approach
            speakers_expected=None,  # Let AssemblyAI automatically detect number of speakers
            language_detection=True,  # Enable automatic language detection
            language_detection_options=language_options,  # Set language detection options
            language_confidence_threshold=0.5,  # Set confidence threshold
            punctuate=True,
            format_text=True
        )
        
        # Create transcriber
        transcriber = aai.Transcriber(config=config)
        
        logger.info("Starting speaker-turn based transcription with AssemblyAI...")
        logger.info("   - Speaker diarization: enabled")
        logger.info("   - Automatic language detection: enabled")
        logger.info("   - Language confidence threshold: 0.5")
        
        # Transcribe the audio file
        transcript = transcriber.transcribe(audio_file)
        
        if transcript.status == aai.TranscriptStatus.error:
            raise Exception(f"Transcription failed: {transcript.error}")
        
        # Get detected language from json_response
        detected_language = transcript.json_response.get("language_code", "unknown")
        language_confidence = transcript.json_response.get("language_confidence", 0.0)
        
        logger.info(f"Detected language: {detected_language}")
        logger.info(f"   Language confidence: {language_confidence:.2f}")
        
        # Load audio file for creating segment samples
        audio = AudioSegment.from_wav(audio_file)
        
        # Process utterances to create segments based on speaker turns
        segments = []
        segment_samples = {}
        segment_index = 0
        
        # Group consecutive utterances from the same speaker into single segments
        current_segment = None
        
        for utterance in transcript.utterances:
            speaker = utterance.speaker
            start_ms = utterance.start
            end_ms = utterance.end
            text = utterance.text
            
            # If this is a new speaker or first utterance, start a new segment
            if current_segment is None or current_segment['speaker'] != speaker:
                # Save the previous segment if it exists
                if current_segment is not None:
                    # Create the segment entry
                    segments.append({
                        'start': current_segment['start_ms'] / 1000.0,  # Convert to seconds
                        'end': current_segment['end_ms'] / 1000.0,
                        'text': current_segment['text'],
                        'speaker': current_segment['speaker'],
                        'index': segment_index
                    })
                    
                    segment_index += 1
                
                # Start new segment for this speaker
                current_segment = {
                    'speaker': speaker,
                    'start_ms': start_ms,
                    'end_ms': end_ms,
                    'text': text
                }
            else:
                # Same speaker continues - extend the current segment
                current_segment['end_ms'] = end_ms
                current_segment['text'] += " " + text
        
        # Don't forget the last segment
        if current_segment is not None:
            segments.append({
                'start': current_segment['start_ms'] / 1000.0,
                'end': current_segment['end_ms'] / 1000.0,
                'text': current_segment['text'],
                'speaker': current_segment['speaker'],
                'index': segment_index
            })
        
        # NEW APPROACH: Create concatenated audio for each speaker
        logger.info(f"Creating concatenated voice samples for each speaker...")
        
        # Group segments by speaker
        speaker_segments = {}
        for segment in segments:
            speaker = segment['speaker']
            if speaker not in speaker_segments:
                speaker_segments[speaker] = []
            speaker_segments[speaker].append(segment)
        
        # Create concatenated audio for each speaker
        speaker_concatenated_audio = {}
        for speaker, speaker_segment_list in speaker_segments.items():
            logger.info(f"Creating concatenated audio for {speaker} ({len(speaker_segment_list)} segments)...")
            
            # Sort segments by their original order (by index)
            speaker_segment_list.sort(key=lambda x: x['index'])
            
            # Concatenate all audio segments from this speaker
            concatenated_audio = AudioSegment.empty()
            total_duration = 0
            
            for segment in speaker_segment_list:
                start_ms = int(segment['start'] * 1000)
                end_ms = int(segment['end'] * 1000)
                segment_audio = audio[start_ms:end_ms]
                concatenated_audio += segment_audio
                duration_s = (end_ms - start_ms) / 1000
                total_duration += duration_s
            
            speaker_concatenated_audio[speaker] = concatenated_audio
            logger.info(f"   Total concatenated duration for {speaker}: {total_duration:.1f}s")
        
        # Now create voice samples for ALL segments using their speaker's concatenated audio
        logger.info(f"Saving voice samples...")
        for segment in segments:
            speaker = segment['speaker']
            segment_index = segment['index']
            
            # Use the concatenated audio for this speaker
            if speaker in speaker_concatenated_audio:
                sample_path = os.path.join(sample_dir, f"segment_{segment_index}.wav")
                speaker_concatenated_audio[speaker].export(sample_path, format="wav")
                segment_samples[f"segment_{segment_index}"] = sample_path
                
                concatenated_duration = len(speaker_concatenated_audio[speaker]) / 1000
                logger.info(f"Saved segment_{segment_index}.wav using {speaker}'s concatenated audio ({concatenated_duration:.1f}s)")
            else:
                logger.warning(f"No concatenated audio found for {speaker} in segment_{segment_index}")
        
        # Create formatted transcript with timestamps and speaker info
        transcript_lines = []
        for segment in segments:
            start_time = segment['start']
            end_time = segment['end']
            text = segment['text']
            speaker = segment['speaker']
            index = segment['index']
            
            if text.strip():
                transcript_lines.append(f"segment_{index}: {start_time:.2f} {end_time:.2f} \"{text}\" ({speaker})")
        
        return transcript_lines, segment_samples, segments, detected_language, language_confidence
    
    def translate_deepl(self, text, source_lang, target_lang="EN"):
        """Translate text using DeepL API"""
        url = "https://api-free.deepl.com/v2/translate"
        params = {
            "auth_key": self.deepl_api_key,
            "text": text,
            "source_lang": source_lang.upper(),
            "target_lang": target_lang.upper(),
        }
        response = requests.post(url, data=params)
        response.raise_for_status()
        return response.json()["translations"][0]["text"]
    
    def convert_symbols_to_words(self, text, target_language="english"):
        """Convert numbers with symbols, ordinals, and times into words for target language"""
        # Map frontend language codes to num2words language codes
        num2words_lang_map = {
            "english": "en",
            "spanish": "es", 
            "french": "fr",
            "german": "de",
            "italian": "it",
            "portuguese": "pt",
            "russian": "ru"
            # Note: num2words doesn't support Chinese, Japanese, Korean, Arabic
        }
        
        lang_code = num2words_lang_map.get(target_language, "en")
        
        # Pattern for symbols
        pattern_symbols = re.compile(r'([€$%#]|°[CF])?\s*([\d,]+)\s*([€$%#]|°[CF])?')
        # Pattern for ordinals (only for English)
        pattern_ordinals = re.compile(r'\b(\d+)(st|nd|rd|th)\b')
        # Pattern for times (HH:MM)
        pattern_times = re.compile(r'\b(\d{1,2}):(\d{2})\b')
        
        def replacer_symbols(match):
            pre, number_str, post = match.groups()
            number_str = number_str.replace(',', '') if number_str else ''
            try:
                number = int(number_str)
            except ValueError:
                return match.group(0)
            
            try:
                words = num2words(number, lang=lang_code)
            except Exception:
                # Fallback to English if language not supported
                words = num2words(number, lang="en")
            
            # Add currency/unit words (keep in English for TTS compatibility)
            if pre == '€' or post == '€':
                words += ' euros'
            elif pre == '$' or post == '$':
                words += ' dollars'
            elif pre == '%' or post == '%':
                words += ' percent'
            elif pre == '#' or post == '#':
                words = 'hashtag ' + words
            elif pre == '°C' or post == '°C':
                words += ' degrees Celsius'
            elif pre == '°F' or post == '°F':
                words += ' degrees Fahrenheit'
            
            return ' ' + words + ' '
        
        def replacer_ordinals(match):
            number = int(match.group(1))
            try:
                # Ordinals only work well in English
                if lang_code == "en":
                    return ' ' + num2words(number, to='ordinal', lang=lang_code) + ' '
                else:
                    # For other languages, just convert to cardinal number
                    return ' ' + num2words(number, lang=lang_code) + ' '
            except Exception:
                return match.group(0)
        
        def replacer_times(match):
            hours = int(match.group(1))
            minutes = int(match.group(2))
            try:
                if minutes == 0:
                    hours_word = num2words(hours, lang=lang_code)
                    return f' {hours_word} o\'clock '
                else:
                    hours_word = num2words(hours, lang=lang_code)
                    minutes_word = num2words(minutes, lang=lang_code)
                    return f' {hours_word} {minutes_word} '
            except Exception:
                return match.group(0)
        
        text = pattern_symbols.sub(replacer_symbols, text)
        text = pattern_ordinals.sub(replacer_ordinals, text)
        text = pattern_times.sub(replacer_times, text)
        
        # Normalize multiple spaces
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def get_deepl_language_code(self, assemblyai_code):
        """Convert AssemblyAI language codes to DeepL language codes"""
        mapping = {
            'en': 'EN',
            'es': 'ES',
            'fr': 'FR',
            'de': 'DE',
            'it': 'IT',
            'pt': 'PT',
            'ru': 'RU',
            'ja': 'JA',
            'ko': 'KO',
            'zh': 'ZH',
            'nl': 'NL',
            'pl': 'PL',
            'sv': 'SV',
            'da': 'DA',
            'no': 'NB',
            'fi': 'FI',
            'cs': 'CS',
            'et': 'ET',
            'hu': 'HU',
            'lv': 'LV',
            'lt': 'LT',
            'sk': 'SK',
            'sl': 'SL',
            'bg': 'BG',
            'el': 'EL',
            'ro': 'RO',
        }
        
        # Handle language codes with country variants (e.g., 'en_us' -> 'en')
        base_code = assemblyai_code.split('_')[0].lower()
        return mapping.get(base_code, 'EN')  # Default to English if not found
    
    def adjust_audio_duration(self, input_path, output_path, target_duration):
        """
        Adjust audio duration using Sox with high-quality time-stretching
        """
        # Validate target duration
        if target_duration <= 0:
            raise ValueError("Target duration must be greater than zero")
            
        # Get original duration with soxi
        result = subprocess.run(
            ["/usr/local/bin/soxi", "-D", input_path],
            capture_output=True,
            text=True,
            check=True,
            timeout=30
        )
        original_duration = float(result.stdout.strip())
        
        # Compute stretch factor
        factor = original_duration / target_duration
        
        # Apply high-quality time-stretch (phase vocoder, -s flag)
        subprocess.run(
            ["/usr/local/bin/sox", input_path, output_path, "tempo", "-s", str(factor)],
            check=True,
            timeout=60
        )
        
        return original_duration, target_duration
    
    def synthesize_chatterbox_segment(self, text, ref_audio_path, output_path, target_language="english"):
        """Synthesize voice segment with ChatterboxTTS - prioritize multilingual model"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Get language ID for Chatterbox
        language_id = self.chatterbox_language_map.get(target_language, "en")
        
        try:
            # Prioritize multilingual model for better quality
            if self.multilingual_model is not None:
                logger.info(f"Using multilingual model for TTS synthesis with language_id={language_id}")
                wav_segment = self.multilingual_model.generate(
                    text,
                    audio_prompt_path=ref_audio_path,
                    language_id=language_id,
                    exaggeration=0.5,
                    cfg_weight=0.5
                )
                ta.save(output_path, wav_segment, self.multilingual_model.sr)
            elif self.normal_model is not None:
                logger.info(f"Using normal model for TTS synthesis with language_id={language_id}")
                wav_segment = self.normal_model.generate(
                    text,
                    audio_prompt_path=ref_audio_path,
                    language_id=language_id,
                    exaggeration=0.5,
                    cfg_weight=0.5
                )
                ta.save(output_path, wav_segment, self.normal_model.sr)
            else:
                raise ImportError("No ChatterboxTTS models available")
            
            return output_path
            
        except (ImportError, Exception) as e:
            logger.warning(f"ChatterboxTTS synthesis failed: {str(e)}, using reference audio as placeholder")
            import shutil
            shutil.copy(ref_audio_path, output_path)
            return output_path
    
    def compose_final_audio_with_sox(self, vocals_path, segments, tmp_segments_dir, translated_segments, temp_dir, target_language="english"):
        """Compose final audio with Sox-based duration matching and ambient background"""
        
        # Create directories for Sox processing
        tmp_pre_dir = os.path.join(temp_dir, "tmp_pre")
        tmp_post_dir = os.path.join(temp_dir, "tmp_post")
        os.makedirs(tmp_pre_dir, exist_ok=True)
        os.makedirs(tmp_post_dir, exist_ok=True)
        
        # Map segments to voice samples and generate TTS for each segment
        segment_voice_mapping = {}
        available_voices = [f for f in os.listdir(tmp_segments_dir) if f.endswith(".wav")]
        
        for segment_name in [f"segment_{s['index']}" for s in segments]:
            voice_file = f"{segment_name}.wav"
            if voice_file in available_voices:
                segment_voice_mapping[segment_name] = os.path.join(tmp_segments_dir, voice_file)
        
        # Generate TTS for each segment (saved to tmp_pre_dir)
        segments_dict = {}
        for segment in translated_segments:
            segment_name = segment['segment_name']
            segments_dict[segment_name] = segment['text']
        
        for segment_name, segment_text in segments_dict.items():
            if segment_name not in segment_voice_mapping:
                logger.warning(f"Skipping {segment_name} - no voice sample available")
                continue
            
            voice_sample_path = segment_voice_mapping[segment_name]
            pre_output_path = os.path.join(tmp_pre_dir, f"{segment_name}.wav")
            
            logger.info(f"Generating audio for {segment_name}...")
            self.synthesize_chatterbox_segment(segment_text, voice_sample_path, pre_output_path, target_language)
        
        # Load original vocals audio
        original_audio = AudioSegment.from_wav(vocals_path)
        final_audio = original_audio - 20  # Heavy ducking for background only
        
        logger.info(f"Original vocals duration: {len(original_audio)/1000:.2f} seconds")
        
        # Process each segment with Sox-based duration matching
        for segment in translated_segments:
            segment_name = segment['segment_name']
            pre_audio_path = os.path.join(tmp_pre_dir, f"{segment_name}.wav")
            post_audio_path = os.path.join(tmp_post_dir, f"{segment_name}.wav")
            
            if not os.path.exists(pre_audio_path):
                logger.warning(f"Skipping {segment_name} - no pre-stretch audio found")
                continue
            
            start_ms = int(segment["start"] * 1000)
            end_ms = int(segment["end"] * 1000)
            target_duration_seconds = (end_ms - start_ms) / 1000.0
            
            logger.info(f"Processing {segment_name}: {start_ms/1000:.2f}s - {end_ms/1000:.2f}s")
            logger.info(f"   Target duration: {target_duration_seconds:.2f}s")
            
            try:
                orig_duration, target_duration = self.adjust_audio_duration(
                    pre_audio_path,
                    post_audio_path,
                    target_duration_seconds
                )
                
                logger.info(f"   Sox adjustment: {orig_duration:.2f}s -> {target_duration:.2f}s")
                
                matched_tts = AudioSegment.from_wav(post_audio_path)
                logger.info(f"   Final matched TTS duration: {len(matched_tts)/1000:.2f}s")
                
                final_audio = final_audio[:start_ms] + matched_tts + final_audio[end_ms:]
                
            except Exception as e:
                logger.error(f"Failed Sox processing for {segment_name}: {str(e)}")
                continue
        
        # Export final dubbed vocals
        final_dubbed_vocals_path = os.path.join(temp_dir, "final_dubbed_vocals.wav")
        final_audio.export(final_dubbed_vocals_path, format="wav")
        
        return final_dubbed_vocals_path
    
    def mix_with_ambient(self, dubbed_vocals_path, ambient_path, temp_dir):
        """Mix dubbed vocals with ambient background"""
        
        def load_resample(path, target_sr):
            """Load audio and resample to target_sr if needed."""
            wav, sr = ta.load(path)
            if sr != target_sr:
                wav = ta.functional.resample(wav, sr, target_sr)
            return wav
        
        def rms_energy(waveform):
            """Compute RMS energy of waveform."""
            return torch.sqrt(torch.mean(waveform**2))
        
        # Get reference sample rate from ambient
        _, ref_sr = ta.load(ambient_path)
        
        # Load and align all signals
        dubbed = load_resample(dubbed_vocals_path, ref_sr)
        ambient = load_resample(ambient_path, ref_sr)
        
        # Ensure channel match (keep stereo if ambient is stereo)
        if dubbed.shape[0] != ambient.shape[0]:
            dubbed = dubbed.mean(dim=0, keepdim=True).repeat(ambient.shape[0], 1)
        
        # Pad/trim dubbed to ambient length
        min_len = min(dubbed.shape[1], ambient.shape[1])
        max_len = max(dubbed.shape[1], ambient.shape[1])
        dubbed = torch.nn.functional.pad(dubbed[:, :min_len], (0, max_len - min_len))
        ambient = torch.nn.functional.pad(ambient[:, :min_len], (0, max_len - min_len))
        
        # Match volume ratio (using ambient as reference)
        rms_ambient = rms_energy(ambient)
        rms_dubbed = rms_energy(dubbed)
        
        scale = rms_ambient / rms_dubbed if rms_dubbed > 0 else 1.0
        dubbed_scaled = dubbed * scale * 2.0  # Boost dubbed vocals
        
        # Mix new dubbed voice with ambient
        final_mix = ambient + dubbed_scaled
        
        # Normalize to avoid clipping
        max_val = final_mix.abs().max()
        if max_val > 1.0:
            final_mix = final_mix / max_val
        
        # Save final result
        final_output_path = os.path.join(temp_dir, "final_mixed_audio.wav")
        ta.save(final_output_path, final_mix, ref_sr)
        
        logger.info(f"Final mixed audio saved: {final_output_path}")
        return final_output_path
    
    def _validate_video_url(self, url: str) -> bool:
        """Validate video URL to prevent SSRF attacks"""
        from urllib.parse import urlparse
        parsed = urlparse(url)
        
        # Only allow HTTPS URLs
        if parsed.scheme != 'https':
            return False
            
        # Block internal/private networks
        hostname = parsed.hostname
        if not hostname:
            return False
            
        # Block localhost and private IPs
        blocked_hosts = ['localhost', '127.0.0.1', '0.0.0.0']
        if hostname in blocked_hosts or hostname.startswith('192.168.') or hostname.startswith('10.') or hostname.startswith('172.'):
            return False
            
        return True

    async def process_video(self, video_url, target_language, user_id):
        """Process video using advanced AssemblyAI + ChatterboxTTS + Sox pipeline with target language"""
        # Validate URL to prevent SSRF
        if not self._validate_video_url(video_url):
            raise ValueError("Invalid or unsafe video URL")
            
        temp_dir = tempfile.mkdtemp()
        try:
            # Download video
            video_path = os.path.join(temp_dir, "input_video.mp4")
            original_audio_path = os.path.join(temp_dir, "extracted_audio.wav")
            
            # Download video using requests with proper headers
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(video_url, headers=headers, stream=True, timeout=30)
            response.raise_for_status()
            
            with open(video_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Extract audio from video
            self.extract_audio_from_video(video_path, original_audio_path)
            
            # Separate audio into vocals and ambient using Demucs
            vocals_path, ambient_path = self.separate_audio_stems(original_audio_path, temp_dir)
            
            # Create voice samples directory
            voices_dir = os.path.join(temp_dir, "voices")
            
            # Transcribe vocals with AssemblyAI (speaker diarization + language detection)
            transcript, samples, segments, detected_language, language_confidence = \
                self.create_segment_transcript_with_samples_assemblyai(vocals_path, voices_dir)
            
            logger.info(f"Original language detected: {detected_language} (confidence: {language_confidence:.2f})")
            
            # Validate language detection
            if detected_language == "unknown" or language_confidence < 0.5:
                logger.warning("WARNING: Language detection may be unreliable")
            
            # Get DeepL language codes
            source_lang_deepl = self.get_deepl_language_code(detected_language)
            target_lang_deepl = self.target_language_map.get(target_language, "EN")  # Use selected target language
            
            logger.info(f"Translation: {source_lang_deepl} -> {target_lang_deepl}")
            logger.info(f"   Auto-detected source: {detected_language}")
            logger.info(f"   Selected target: {target_language}")
            
            # Translate and process segments
            translated_segments = []
            for line in transcript:
                if line.strip():
                    # Parse format: segment_X: start_time end_time "text" (Speaker)
                    match = re.match(r'^(segment_\d+):\s+(\d+\.\d+)\s+(\d+\.\d+)\s+"(.*?)"\s*(\([^)]+\))?$', line)
                    if match:
                        segment_name = match.group(1)
                        start_time = float(match.group(2))
                        end_time = float(match.group(3))
                        original_text = match.group(4)
                        
                        # Only translate if the source language is not already English
                        if source_lang_deepl != target_lang_deepl:
                            translated_text = self.translate_deepl(original_text, source_lang_deepl, target_lang_deepl)
                        else:
                            translated_text = original_text
                        
                        # Convert symbols to words for better TTS
                        processed_text = self.convert_symbols_to_words(translated_text, target_language)
                        
                        # Store segment information for audio processing
                        translated_segments.append({
                            'segment_name': segment_name,
                            'start': start_time,
                            'end': end_time,
                            'text': processed_text,
                            'index': int(segment_name.split('_')[1])
                        })
            
            # Compose final audio with Sox-based duration matching
            final_dubbed_vocals_path = self.compose_final_audio_with_sox(
                vocals_path, segments, voices_dir, translated_segments, temp_dir, target_language
            )
            
            # Mix dubbed vocals with ambient background
            final_mixed_audio_path = self.mix_with_ambient(
                final_dubbed_vocals_path, ambient_path, temp_dir
            )
            
            # Upload translated audio to S3 with metadata
            s3_folder = f"translated-audios/{user_id}"
            original_filename = video_url.split('/')[-1].split('.')[0] if '/' in video_url else 'video'
            s3_filename = f"{original_filename}-multispeaker-{uuid.uuid4()}.wav"
            
            # Create metadata for the audio file
            metadata = {
                'originalname': original_filename,
                'sourcelanguage': detected_language,
                'targetlanguage': target_language,
                'confidence': str(language_confidence),
                'userid': user_id,
                'pipeline': 'assemblyai-chatterbox-sox'
            }
            
            translated_audio_url = s3_uploader.upload(
                final_mixed_audio_path, 
                folder=s3_folder, 
                filename=s3_filename,
                metadata=metadata
            )
            
            return translated_audio_url
            
        finally:
            # Clean up temporary files
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

# Initialize the service
audio_translator_update = AudioTranslatorServiceUpdate()
