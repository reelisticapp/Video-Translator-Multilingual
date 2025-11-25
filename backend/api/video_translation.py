from fastapi import APIRouter, HTTPException, Request, UploadFile, File, Form
from typing import Optional
import os
import json
import subprocess
import tempfile
import uuid
import logging
import requests
from app.services.s3_uploader import s3_uploader
from app.services.audio_translator import audio_translator_update

router = APIRouter()
logger = logging.getLogger(__name__)

def get_media_duration(file_path: str) -> float:
    """Get duration of audio/video file in seconds using ffprobe"""
    try:
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_format', file_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        duration = float(data['format']['duration'])
        return duration
    except Exception as e:
        logger.error(f"Error getting media duration: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Could not determine media duration: {str(e)}")

def calculate_translation_tokens(duration_seconds: float) -> int:
    """Calculate tokens needed for advanced translation based on duration"""
    from app.api.tokens import TOKEN_COSTS
    # Advanced pipeline uses more tokens due to speaker diarization, better TTS, etc.
    base_tokens_per_second = TOKEN_COSTS.get("translation", 2)
    advanced_multiplier = 2.5  # Advanced pipeline uses 2.5x more tokens
    return int(duration_seconds * base_tokens_per_second * advanced_multiplier)

@router.post("/calculate-cost-advanced")
async def calculate_advanced_translation_cost(
    file: UploadFile = File(...)
):
    """Calculate token cost for advanced multispeaker translation of uploaded media file"""
    try:
        # Validate file type
        allowed_extensions = {'.mp4', '.wav', '.mp3', '.m4a', '.avi', '.mov', '.mkv', '.webm'}
        file_extension = os.path.splitext(file.filename)[1].lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_extension}")
        
        # Save file temporarily to get duration
        with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Get duration and calculate cost
            duration = get_media_duration(temp_file_path)
            tokens_required = calculate_translation_tokens(duration)
            
            return {
                "duration_seconds": duration,
                "tokens_required": tokens_required,
                "filename": file.filename,
                "pipeline": "advanced-multispeaker",
                "features": [
                    "AssemblyAI speaker diarization",
                    "Automatic language detection", 
                    "Chatterbox TTS with voice cloning",
                    "Sox audio processing",
                    "Demucs audio separation",
                    "Ambient preservation"
                ]
            }
        finally:
            # Clean up temp file
            os.unlink(temp_file_path)
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Advanced cost calculation error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload-video-advanced")
async def upload_video_advanced(
    file: UploadFile = File(...),
    userId: str = Form(...),
    folder: str = Form("videos-to-translate-advanced")
):
    """Upload a video file to S3 bucket for advanced multispeaker processing"""
    try:
        logger.info(f"Uploading video for advanced processing - user: {userId[:50]} to folder: {folder[:100]}")
        
        # Read file content
        file_content = await file.read()
        
        # Generate unique filename with original extension
        original_extension = os.path.splitext(file.filename)[1]
        if not original_extension:
            original_extension = ".mp4"  # Default to mp4 if no extension
        
        filename = f"{uuid.uuid4()}{original_extension}"
        
        # Save temporarily to get duration
        duration = None
        with tempfile.NamedTemporaryFile(suffix=original_extension, delete=False) as temp_file:
            temp_file.write(file_content)
            temp_file_path = temp_file.name
        
        try:
            duration = get_media_duration(temp_file_path)
        except Exception as e:
            logger.warning(f"Could not get duration: {str(e)}")
        finally:
            os.unlink(temp_file_path)
        
        # Upload to S3 in the specified folder
        from io import BytesIO
        video_data = BytesIO(file_content)
        metadata = {
            "userId": userId,
            "pipeline": "advanced-multispeaker",
            "features": "speaker-diarization,language-detection,voice-cloning,audio-separation"
        }
        if duration:
            metadata["duration"] = str(duration)
            metadata["tokens_required"] = str(calculate_translation_tokens(duration))
        
        video_url = s3_uploader.upload(
            file_source=video_data,
            folder=folder,
            filename=filename,
            metadata=metadata
        )
        
        # Convert S3 URL to CloudFront URL for faster delivery if available
        cloudfront_domain = os.getenv('CLOUDFRONT_DOMAIN')
        if cloudfront_domain:
            # Extract the S3 key from the URL
            s3_key = video_url.split('.amazonaws.com/')[-1]
            video_url = f"https://{cloudfront_domain}/{s3_key}"
        
        response = {"url": video_url, "pipeline": "advanced-multispeaker"}
        if duration:
            response["duration"] = duration
            response["tokens_required"] = calculate_translation_tokens(duration)
        
        logger.info(f"Video uploaded successfully for advanced processing")
        return response
    except Exception as e:
        logger.error(f"Advanced video upload error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/translate-video-advanced")
async def translate_video_advanced(request: Request, data: dict):
    """
    Advanced video translation using:
    - AssemblyAI for speaker diarization and language detection
    - Chatterbox TTS for high-quality voice synthesis
    - Demucs for audio source separation
    - Sox for professional audio processing
    """
    try:
        logger.info(f"Advanced video translation request: {data}")
        
        # Extract parameters from request
        video_url = data.get("videoUrl")
        target_language = data.get("targetLanguage", "english")  # Target language selection
        user_id = data.get("userId", "default")
        duration = data.get("duration")  # Duration can be provided from frontend
        
        if not video_url:
            raise HTTPException(status_code=400, detail="Video URL is required")
        
        logger.info(f"Processing video with target language: {target_language[:20]}")
        
        # If duration not provided, try to get it from S3 metadata
        if not duration:
            try:
                # Try to get duration from S3 metadata first
                from urllib.parse import urlparse
                parsed_url = urlparse(video_url)
                s3_key = parsed_url.path.lstrip('/')
                
                head_response = s3_uploader.s3.head_object(
                    Bucket=s3_uploader.default_bucket, 
                    Key=s3_key
                )
                metadata = head_response.get('Metadata', {})
                duration = float(metadata.get('duration', 0))
            except Exception as e:
                logger.warning(f"Could not get duration from metadata: {str(e)}")
        
        # Consume tokens before processing if duration is known
        if duration and user_id != "default":
            try:
                from app.services.simplified_token_service import SimplifiedTokenService
                token_service = SimplifiedTokenService()
                tokens_needed = calculate_translation_tokens(duration)
                
                await token_service.consume_tokens(
                    user_id=user_id,
                    amount=tokens_needed,
                    feature="advanced_translation",
                    description=f"Advanced multispeaker video translation - {duration:.1f}s"
                )
                logger.info(f"Consumed {tokens_needed} tokens for advanced {duration:.1f}s translation")
            except Exception as e:
                logger.error(f"Token consumption failed: {str(e)}")
                raise HTTPException(status_code=400, detail=f"Insufficient tokens or token error: {str(e)}")
        
        # Process the video using the advanced audio translator service
        try:
            translated_audio_url = await audio_translator_update.process_video(
                video_url, target_language, user_id
            )
            
            logger.info(f"Advanced video translation completed:")
            logger.info(f"  - Target language: {target_language}")
            logger.info(f"  - Audio output: {translated_audio_url}")
            
            # Automatically create dubbed video
            dubbed_video_url = None
            try:
                logger.info("Creating dubbed video automatically...")
                
                # Create the dubbed video by combining original video with translated audio
                temp_dir = tempfile.mkdtemp()
                try:
                    # Download original video
                    original_video_path = os.path.join(temp_dir, "original_video.mp4")
                    headers = {
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                    }
                    video_response = requests.get(video_url, headers=headers, stream=True)
                    video_response.raise_for_status()
                    
                    with open(original_video_path, "wb") as f:
                        for chunk in video_response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    
                    # Download translated audio
                    translated_audio_path = os.path.join(temp_dir, "translated_audio.wav")
                    audio_response = requests.get(translated_audio_url, headers=headers, stream=True)
                    audio_response.raise_for_status()
                    
                    with open(translated_audio_path, "wb") as f:
                        for chunk in audio_response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    
                    # Combine video and audio using FFmpeg
                    final_video_path = os.path.join(temp_dir, "dubbed_video.mp4")
                    
                    ffmpeg_cmd = [
                        "ffmpeg", "-y", "-threads", "2",
                        "-i", original_video_path,
                        "-i", translated_audio_path,
                        "-c:v", "copy",  # Keep original video codec
                        "-map", "0:v:0",  # Map video from first input
                        "-map", "1:a:0",  # Map audio from second input
                        "-shortest",  # End when shortest stream ends
                        final_video_path
                    ]
                    
                    subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
                    
                    # Upload final dubbed video to S3
                    s3_folder = f"dubbed-videos/{user_id}"
                    original_filename = video_url.split('/')[-1].split('.')[0] if '/' in video_url else 'video'
                    s3_filename = f"{original_filename}-dubbed-advanced-{uuid.uuid4()}.mp4"
                    
                    metadata = {
                        'originalname': original_filename,
                        'userid': user_id,
                        'pipeline': 'advanced-multispeaker-dubbed',
                        'original_video_url': video_url,
                        'translated_audio_url': translated_audio_url,
                        'target_language': target_language
                    }
                    
                    dubbed_video_url = s3_uploader.upload(
                        final_video_path,
                        folder=s3_folder,
                        filename=s3_filename,
                        metadata=metadata
                    )
                    
                    # Convert S3 URL to CloudFront URL for faster delivery if available
                    cloudfront_domain = os.getenv('CLOUDFRONT_DOMAIN')
                    if cloudfront_domain:
                        s3_key = dubbed_video_url.split('.amazonaws.com/')[-1]
                        dubbed_video_url = f"https://{cloudfront_domain}/{s3_key}"
                    
                    logger.info(f"Dubbed video created successfully: {dubbed_video_url}")
                    
                finally:
                    # Clean up temp directory
                    import shutil
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    
            except Exception as video_error:
                logger.warning(f"Failed to create dubbed video, but audio translation succeeded: {str(video_error)}")
                # Continue without video if audio translation was successful
            
            logger.info(f"  - Features: Auto language detection, speaker diarization, voice cloning, ambient preservation")
            
            response_data = {
                "translatedAudioUrl": translated_audio_url,
                "pipeline": "advanced-multispeaker",
                "features": {
                    "speaker_diarization": True,
                    "automatic_language_detection": True,
                    "voice_cloning": True,
                    "audio_separation": True,
                    "ambient_preservation": True,
                    "high_quality_tts": True
                }
            }
            
            # Include dubbed video URL if creation was successful
            if dubbed_video_url:
                response_data["dubbedVideoUrl"] = dubbed_video_url
                logger.info(f"  - Dubbed video: {dubbed_video_url}")
            
            return response_data
        except Exception as processing_error:
            logger.error(f"Advanced video processing error: {str(processing_error)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Advanced processing failed: {str(processing_error)}")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Advanced video translation error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/translated-audios-advanced")
async def get_translated_audios_advanced(user_id: str):
    """Get a list of advanced translated audio files for a user"""
    try:
        logger.info(f"Getting advanced translated audios for user: {user_id}")
        
        # Get S3 client
        bucket = s3_uploader.default_bucket
        prefix = f"translated-audios/{user_id}/"
        
        # List audio files from S3
        audios = []
        try:
            response = s3_uploader.s3.list_objects_v2(
                Bucket=bucket,
                Prefix=prefix
            )
            
            if 'Contents' in response:
                for obj in response['Contents']:
                    key = obj['Key']
                    # Skip folder markers and filter for advanced pipeline files
                    if key.endswith('/') or not key.endswith('.wav'):
                        continue
                    
                    # Try to get metadata
                    try:
                        head_response = s3_uploader.s3.head_object(Bucket=bucket, Key=key)
                        metadata = head_response.get('Metadata', {})
                        
                        # Filter for advanced pipeline files
                        if metadata.get('pipeline') != 'assemblyai-chatterbox-sox':
                            continue
                        
                        # Convert S3 URL to CloudFront URL for faster delivery
                        file_url = f"https://{bucket}.s3.amazonaws.com/{key}"
                        cloudfront_domain = os.getenv('CLOUDFRONT_DOMAIN')
                        if cloudfront_domain:
                            file_url = f"https://{cloudfront_domain}/{key}"
                        
                        audios.append({
                            "url": file_url,
                            "filename": key.split('/')[-1],
                            "size": obj['Size'],
                            "timestamp": obj['LastModified'].isoformat(),
                            "originalname": metadata.get('originalname', 'unknown'),
                            "sourcelanguage": metadata.get('sourcelanguage', 'unknown'),
                            "confidence": metadata.get('confidence', 'unknown'),
                            "pipeline": "advanced-multispeaker",
                            "features": [
                                "Speaker diarization",
                                "Automatic language detection", 
                                "Voice cloning",
                                "Audio separation",
                                "Ambient preservation"
                            ]
                        })
                    except Exception as meta_error:
                        logger.warning(f"Could not get metadata for {key}: {str(meta_error)}")
                        continue
            
            # Sort by timestamp (newest first) and limit to recent items
            audios.sort(key=lambda x: x['timestamp'], reverse=True)
            audios = audios[:50]  # Limit to 50 most recent
            
        except Exception as e:
            logger.warning(f"Could not list S3 objects: {str(e)}")
            # Return empty list if S3 listing fails
        
        return {
            "audios": audios,
            "pipeline": "advanced-multispeaker",
            "total": len(audios)
        }
    except Exception as e:
        logger.error(f"Error getting advanced translated audios: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/create-dubbed-video")
async def create_dubbed_video(request: Request, data: dict):
    """Create a final dubbed video by combining original video with translated audio"""
    try:
        logger.info(f"Creating dubbed video: {data}")
        
        # Extract parameters
        original_video_url = data.get("originalVideoUrl")
        translated_audio_url = data.get("translatedAudioUrl")
        user_id = data.get("userId", "default")
        
        if not original_video_url or not translated_audio_url:
            raise HTTPException(status_code=400, detail="Both original video URL and translated audio URL are required")
        
        temp_dir = tempfile.mkdtemp()
        try:
            # Download original video
            original_video_path = os.path.join(temp_dir, "original_video.mp4")
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            
            import requests
            video_response = requests.get(original_video_url, headers=headers, stream=True)
            video_response.raise_for_status()
            
            with open(original_video_path, "wb") as f:
                for chunk in video_response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Download translated audio
            translated_audio_path = os.path.join(temp_dir, "translated_audio.wav")
            audio_response = requests.get(translated_audio_url, headers=headers, stream=True)
            audio_response.raise_for_status()
            
            with open(translated_audio_path, "wb") as f:
                for chunk in audio_response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Combine video and audio using FFmpeg
            final_video_path = os.path.join(temp_dir, "dubbed_video.mp4")
            
            ffmpeg_cmd = [
                "ffmpeg", "-y", "-threads", "2",
                "-i", original_video_path,
                "-i", translated_audio_path,
                "-c:v", "copy",  # Copy video stream without re-encoding
                "-map", "0:v:0",  # Use video from first input
                "-map", "1:a:0",  # Use audio from second input
                "-shortest",  # Finish when shortest input ends
                final_video_path
            ]
            
            subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
            
            # Upload final dubbed video to S3
            s3_folder = f"dubbed-videos/{user_id}"
            original_filename = original_video_url.split('/')[-1].split('.')[0] if '/' in original_video_url else 'video'
            s3_filename = f"{original_filename}-dubbed-advanced-{uuid.uuid4()}.mp4"
            
            metadata = {
                'originalname': original_filename,
                'userid': user_id,
                'pipeline': 'advanced-multispeaker-dubbed',
                'original_video_url': original_video_url,
                'translated_audio_url': translated_audio_url
            }
            
            dubbed_video_url = s3_uploader.upload(
                final_video_path,
                folder=s3_folder,
                filename=s3_filename,
                metadata=metadata
            )
            
            logger.info(f"Dubbed video created successfully: {dubbed_video_url}")
            
            return {
                "dubbedVideoUrl": dubbed_video_url,
                "pipeline": "advanced-multispeaker-dubbed",
                "originalVideoUrl": original_video_url,
                "translatedAudioUrl": translated_audio_url
            }
            
        finally:
            # Clean up temporary files
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Dubbed video creation error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
