import React, { useState, useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import { CurrencyDollarIcon, ChevronLeftIcon, ChevronRightIcon } from '@heroicons/react/24/solid';
import upload from '../../assets/icons/upload.png';
import edit from '../../assets/icons/edit.png';
import { useAuth } from '../register/AuthContext';
import { useTokens, TOKEN_COSTS } from '../tokens/TokenContext';
import { useTranslation } from '../../utils/i18n-simple';
import { getApiUrl } from '../../utils/apiConfig.js';

export default function VideoTranslator() {
  const { t } = useTranslation();
  const { user } = useAuth();
  const { tokens: userTokens, consumeTokens, calculateTranslationCost } = useTokens();
  const [videoFile, setVideoFile] = useState(null);
  const [targetLanguage, setTargetLanguage] = useState('english');
  const [detectedLanguage, setDetectedLanguage] = useState('');
  const [languageConfidence, setLanguageConfidence] = useState(0);
  const [tokenCost, setTokenCost] = useState(0);
  const [videoDuration, setVideoDuration] = useState(0); // in seconds
  const [isCalculatingCost, setIsCalculatingCost] = useState(false);
  const [showModal, setShowModal] = useState(false);
  const [isTranslating, setIsTranslating] = useState(false);
  const [translatedAudioUrl, setTranslatedAudioUrl] = useState('');
  const [dubbedVideoUrl, setDubbedVideoUrl] = useState('');
  const [originalVideoUrl, setOriginalVideoUrl] = useState('');
  const [error, setError] = useState('');
  const [translatedAudios, setTranslatedAudios] = useState([]);
  const [selectedAudio, setSelectedAudio] = useState(null);

  const [progress, setProgress] = useState(0);
  const [isCreatingVideo, setIsCreatingVideo] = useState(false);

  // Modal navigation state
  const [currentModalIndex, setCurrentModalIndex] = useState(0);
  const [modalTouchStart, setModalTouchStart] = useState(null);
  const [modalTouchEnd, setModalTouchEnd] = useState(null);

  // Available target language options
  const targetLanguageOptions = [
    { value: 'english', label: t('videoTranslator.languages.english') },
    { value: 'spanish', label: t('videoTranslator.languages.spanish') },
    { value: 'french', label: t('videoTranslator.languages.french') },
    { value: 'german', label: t('videoTranslator.languages.german') },
    { value: 'italian', label: t('videoTranslator.languages.italian') },
    { value: 'portuguese', label: t('videoTranslator.languages.portuguese') },
    { value: 'chinese', label: t('videoTranslator.languages.chinese') },
    { value: 'japanese', label: t('videoTranslator.languages.japanese') },
    { value: 'korean', label: t('videoTranslator.languages.korean') },
    { value: 'arabic', label: t('videoTranslator.languages.arabic') },
    { value: 'russian', label: t('videoTranslator.languages.russian') }
  ];

  const fileInputRef = useRef(null);
  const navigate = useNavigate();

  // Calculate tokens based on video duration (optimized for backend pricing)
  const calcTokens = (durationInSeconds) => {
    if (!durationInSeconds) return 0;
    // 2 tokens per second (matches backend TOKEN_COSTS.translation)
    return Math.ceil(durationInSeconds * 2);
  };

  const onFileChange = async (e) => {
    if (e.target.files?.[0]) {
      const file = e.target.files[0];
      setVideoFile(file);
      setIsCalculatingCost(true);
      setTokenCost(0);
      setVideoDuration(0);

      try {
        // Use backend to calculate exact cost for advanced pipeline
        const formData = new FormData();
        formData.append('file', file);

        const response = await axios.post(`${getApiUrl()}/api/calculate-cost-advanced`, formData);
        const { duration_seconds, tokens_required } = response.data;

        setVideoDuration(duration_seconds);
        setTokenCost(tokens_required);
      } catch (error) {
        console.error('Error calculating cost:', error);
        // Fallback to client-side duration detection
        const video = document.createElement('video');
        video.preload = 'metadata';

        video.onloadedmetadata = function () {
          window.URL.revokeObjectURL(video.src);
          const duration = Math.ceil(video.duration);
          setVideoDuration(duration);
          setTokenCost(calcTokens(duration));
        }

        video.src = URL.createObjectURL(file);
      } finally {
        setIsCalculatingCost(false);
      }
    }
  };

  const onUploadClick = () => fileInputRef.current?.click();

  const onTranslate = async () => {
    if (!videoFile) return alert(t('videoTranslator.uploadVideoFirst'));

    // Check if user has enough tokens
    if (userTokens < tokenCost) {
      setError(t('videoTranslator.insufficientTokens').replace('{cost}', tokenCost).replace('{available}', userTokens));
      setShowModal(true);
      return;
    }

    setIsTranslating(true);
    setError('');
    setShowModal(true);
    setProgress(0);

    try {
      // Get current user ID from auth context or fallback to default
      const userId = user?.username || 'guest';

      // Upload the video file for advanced processing
      const videoFormData = new FormData();
      videoFormData.append('file', videoFile);
      videoFormData.append('userId', userId);
      videoFormData.append('folder', 'videos-to-translate-advanced');

      // Simulate progress updates
      const progressInterval = setInterval(() => {
        setProgress(prev => {
          const newProgress = prev + Math.random() * 5;
          return newProgress > 95 ? 95 : newProgress;
        });
      }, 1000);

      try {
        // Upload video for advanced processing
        const uploadResponse = await axios.post(`${getApiUrl()}/api/upload-video-advanced`, videoFormData);
        const videoUrl = uploadResponse.data.url;

        // Call the advanced translation API with target language and duration
        const translationResponse = await axios.post(`${getApiUrl()}/api/translate-video-advanced`, {
          videoUrl,
          targetLanguage,
          userId,
          duration: videoDuration
        });

        clearInterval(progressInterval);
        setProgress(100);

        const newTranslatedAudioUrl = translationResponse.data.translatedAudioUrl;
        const newDubbedVideoUrl = translationResponse.data.dubbedVideoUrl;
        const detectedLang = translationResponse.data.features?.detected_language || 'unknown';
        const confidence = translationResponse.data.features?.language_confidence || 0;

        setTranslatedAudioUrl(newTranslatedAudioUrl);
        if (newDubbedVideoUrl) {
          setDubbedVideoUrl(newDubbedVideoUrl);
        }
        setDetectedLanguage(detectedLang);
        setLanguageConfidence(confidence);
        setOriginalVideoUrl(videoUrl);

        // Create dubbed video if not already created by backend
        if (!newDubbedVideoUrl) {
          setIsCreatingVideo(true);
          try {
            const dubbedVideoResponse = await axios.post(`${getApiUrl()}/api/create-dubbed-video`, {
              originalVideoUrl: videoUrl,
              translatedAudioUrl: newTranslatedAudioUrl,
              userId
            });

            setDubbedVideoUrl(dubbedVideoResponse.data.dubbedVideoUrl);
          } catch (videoErr) {
            console.error('Failed to create dubbed video:', videoErr);
            // Continue without video if audio translation succeeded
          } finally {
            setIsCreatingVideo(false);
          }
        }

        // Create a new translated audio object
        const newAudio = {
          originalName: videoFile.name,
          url: newTranslatedAudioUrl,
          videoUrl: newDubbedVideoUrl || dubbedVideoUrl,
          detectedLanguage: detectedLang,
          targetLanguage,
          languageConfidence: confidence,
          timestamp: new Date().toISOString(),
          pipeline: 'advanced-multispeaker'
        };

        // Add to translated audios list
        setTranslatedAudios(prev => [newAudio, ...prev].slice(0, 5));

        // Set as selected audio
        setSelectedAudio(newTranslatedAudioUrl);

        // Tokens are now consumed by the backend before processing
        // No need to consume tokens here as it's handled in the API

      } catch (err) {
        clearInterval(progressInterval);
        console.error('Video translation failed:', err);
        if (err.response && err.response.data && err.response.data.detail) {
          throw new Error(err.response.data.detail);
        } else {
          throw new Error('Error processing video');
        }
      }
    } catch (err) {
      console.error('Error in translation process:', err);
      setError(err.message || 'Unknown error occurred');
    } finally {
      setIsTranslating(false);
    }
  };

  const closeModal = () => {
    setShowModal(false);
    if (!translatedAudioUrl) {
      setSelectedAudio(null);

    }
    setCurrentModalIndex(0);
  };

  // Get all available audios for navigation
  const getAllAudios = () => {
    return [...translatedAudios];
  };

  // Modal navigation functions
  const goToPrevModal = () => {
    const allAudios = getAllAudios();
    const newIndex = currentModalIndex > 0 ? currentModalIndex - 1 : allAudios.length - 1;
    setCurrentModalIndex(newIndex);
    setSelectedAudio(allAudios[newIndex].url);
  };

  const goToNextModal = () => {
    const allAudios = getAllAudios();
    const newIndex = currentModalIndex < allAudios.length - 1 ? currentModalIndex + 1 : 0;
    setCurrentModalIndex(newIndex);
    setSelectedAudio(allAudios[newIndex].url);
  };

  // Modal touch handlers for swipe
  const handleModalTouchStart = (e) => {
    setModalTouchEnd(null);
    setModalTouchStart(e.targetTouches[0].clientX);
  };

  const handleModalTouchMove = (e) => {
    setModalTouchEnd(e.targetTouches[0].clientX);
  };

  const handleModalTouchEnd = () => {
    if (!modalTouchStart || !modalTouchEnd) return;

    const distance = modalTouchStart - modalTouchEnd;
    const isLeftSwipe = distance > 50;
    const isRightSwipe = distance < -50;

    if (isLeftSwipe) {
      goToNextModal();
    }
    if (isRightSwipe) {
      goToPrevModal();
    }
  };

  const goToCanvas = () => navigate('/canvas');

  // Transform API audio data to component format
  const transformAudioData = (apiAudio) => ({
    originalName: apiAudio.originalname || 'Unknown',
    url: apiAudio.url,
    detectedLanguage: apiAudio.sourcelanguage || 'Unknown',
    targetLanguage: apiAudio.targetlanguage || 'English',
    languageConfidence: apiAudio.confidence || 0,
    timestamp: apiAudio.timestamp,
    pipeline: apiAudio.pipeline || 'advanced-multispeaker'
  });

  // Fetch user's translated audios when component mounts or user changes
  useEffect(() => {
    const fetchTranslatedAudios = async () => {
      if (user) {
        try {
          const response = await fetch(`${getApiUrl()}/api/translated-audios-advanced?user_id=${user.username}`);
          if (response.ok) {
            const data = await response.json();
            if (data.audios && data.audios.length > 0) {
              const audios = data.audios.slice(0, 5).map(transformAudioData);
              setTranslatedAudios(audios);
            }
          }
        } catch (error) {
          console.error('Failed to fetch translated audios:', error);
          // Don't show error to user, just continue with empty audios
        }
      }
    };

    // Fetch translated audios only if logged in
    if (user) {
      fetchTranslatedAudios();
    }
  }, [user]);

  return (
    <TranslateTutorialProvider>
      <div className="w-full px-2 py-4 flex flex-col">
        {/* Header */}
        <h1 style={{ fontFamily: '"Space Grotesk",sans-serif' }}
          className="text-4xl font-extrabold mb-6 text-center drop-shadow-lg mt-8">
          <span className="bg-gradient-to-r from-purple-600 via-purple-300 to-purple-100 bg-clip-text text-transparent">{t('videoTranslator.title').split(' ')[0]}</span>
          <span className="ml-2 text-white">{t('videoTranslator.title').split(' ').slice(1).join(' ')}</span>
        </h1>
        <p className="text-gray-400 text-center mb-6">{t('videoTranslator.subtitle')}</p>

        {/* Upload area */}
        <div className="flex flex-col items-center w-full bg-gray-800 rounded-lg p-6 mb-6">
          <div
            onClick={onUploadClick}
            className="w-full h-48 border-2 border-dashed border-gray-500 rounded-lg flex flex-col items-center justify-center cursor-pointer hover:border-purple-500 transition-colors"
          >
            {videoFile ? (
              <div className="flex flex-col items-center">
                <video
                  src={URL.createObjectURL(videoFile)}
                  className="h-32 max-w-full object-contain rounded"
                  controls
                />
                <p className="text-gray-300 mt-2 text-sm">{videoFile.name}</p>
              </div>
            ) : (
              <>
                <div
                  className="w-12 h-12 mb-2"
                  style={{
                    maskImage: `url(${upload})`,
                    WebkitMaskImage: `url(${upload})`,
                    maskSize: 'contain',
                    WebkitMaskSize: 'contain',
                    maskRepeat: 'no-repeat',
                    WebkitMaskRepeat: 'no-repeat',
                    backgroundColor: '#9333ea'
                  }}
                />
                <p className="text-gray-300">{t('videoTranslator.uploadPrompt')}</p>
                <p className="text-gray-500 text-sm mt-1">{t('videoTranslator.fileFormats')}</p>
              </>
            )}
          </div>

          <input
            ref={fileInputRef}
            type="file"
            accept="video/*"
            onChange={onFileChange}
            className="hidden"
          />

          {/* Target language selection */}
          <div className="mt-6 w-full max-w-md">
            <div className="flex items-center">
              <span className="text-gray-300 text-sm mr-3">Target language:</span>
              <select
                value={targetLanguage}
                onChange={(e) => setTargetLanguage(e.target.value)}
                className="bg-gray-700 text-white rounded px-3 py-2 outline-none focus:ring-2 focus:ring-purple-500 flex-1"
              >
                <option value="english">English</option>
                <option value="spanish">Spanish</option>
                <option value="french">French</option>
                <option value="german">German</option>
                <option value="italian">Italian</option>
                <option value="portuguese">Portuguese</option>
                <option value="chinese">Chinese</option>
                <option value="japanese">Japanese</option>
                <option value="korean">Korean</option>
                <option value="arabic">Arabic</option>
                <option value="russian">Russian</option>
              </select>
            </div>
          </div>

          {/* Translate button */}
          <button
            onClick={onTranslate}
            disabled={!videoFile || userTokens < tokenCost || isCalculatingCost || tokenCost === 0}
            className={`mt-6 flex items-center ${!videoFile || userTokens < tokenCost || isCalculatingCost || tokenCost === 0 ? 'bg-gray-600 cursor-not-allowed' : 'bg-purple-600 hover:bg-purple-500'} text-white font-semibold rounded-lg px-6 py-3 transition`}
          >
            <span className="mr-2">
              {isCalculatingCost ? 'Calculating...' : t('videoTranslator.translateButton')}
            </span>
            <span className="flex items-center bg-gray-700 px-2 py-1 rounded text-sm">
              {isCalculatingCost ? '...' : tokenCost}
              <CurrencyDollarIcon className="h-4 w-4 ml-1" />
            </span>
          </button>
        </div>

        {/* User's translated audios */}
        {translatedAudios.length > 0 && (
          <div className="mt-6 w-full">
            <h2 className="text-2xl font-bold mb-4 bg-gradient-to-r from-purple-300 to-purple-600 bg-clip-text text-transparent">
              {t('videoTranslator.lastTranslations')}
            </h2>
            <div className="flex flex-wrap gap-4">
              {translatedAudios.map((audio, index) => (
                <div
                  key={index}
                  className="relative group cursor-pointer w-[calc(50%-8px)] sm:w-[calc(33.333%-11px)] md:w-[calc(25%-12px)] lg:w-[calc(20%-13px)]"
                  onClick={() => {
                    const allAudios = getAllAudios();
                    const audioIndex = allAudios.findIndex(a => a.url === audio.url);
                    setCurrentModalIndex(audioIndex >= 0 ? audioIndex : index);
                    setSelectedAudio(audio.url);
                    setShowModal(true);
                  }}
                >
                  <div className="relative w-full h-48 rounded-lg overflow-hidden bg-gray-800 flex flex-col items-center justify-center p-4">
                    <div className="w-16 h-16 mb-2 rounded-full bg-purple-600 flex items-center justify-center">
                      <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3" />
                      </svg>
                    </div>
                    <audio
                      src={audio.url}
                      className="w-full mt-2"
                      controls
                    />
                    <div className="mt-2 text-center">
                      <p className="text-xs text-white truncate w-full">{audio.originalName}</p>
                      <p className="text-xs text-blue-300 truncate w-full">
                        {audio.detectedLanguage && audio.detectedLanguage !== 'Unknown'
                          ? `From: ${audio.detectedLanguage}`
                          : 'Language: Auto-detected'}
                      </p>
                      <p className="text-xs text-purple-300 truncate w-full">
                        To: {targetLanguageOptions.find(l => l.value === audio.targetLanguage)?.label || audio.targetLanguage || 'English'}
                      </p>
                      {audio.pipeline && (
                        <p className="text-xs text-green-300 truncate w-full">
                          {audio.pipeline === 'advanced-multispeaker' ? '🎭 Multispeaker' : 'Standard'}
                        </p>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Video Modal */}
        {showModal && (
          <div
            onClick={closeModal}
            className="fixed inset-0 bg-black bg-opacity-50 filter backdrop-blur-md flex items-start justify-center z-50 pt-10"
          >
            <div
              onClick={(e) => e.stopPropagation()}
              onTouchStart={handleModalTouchStart}
              onTouchMove={handleModalTouchMove}
              onTouchEnd={handleModalTouchEnd}
              className="bg-transparent flex flex-col items-center justify-start max-w-xl w-full h-[90vh] min-h-[400px] relative"
              style={{ minWidth: 400 }}
            >
              {/* Modal content (video box) */}
              <div className="bg-gray-900 rounded-2xl shadow-lg w-full flex flex-col relative">
                {/* Close and Navigation */}
                <div className="absolute top-4 right-4 z-[70] flex items-center gap-2">
                  {/* Item counter */}
                  {!isTranslating && getAllAudios().length > 1 && (
                    <span className="text-gray-400 text-sm">
                      {currentModalIndex + 1} / {getAllAudios().length}
                    </span>
                  )}
                  <button
                    onClick={closeModal}
                    aria-label="Close"
                    className="text-white text-2xl leading-none"
                    style={{
                      background: 'rgba(0,0,0,0.2)',
                      borderRadius: '50%',
                      width: 36,
                      height: 36,
                    }}
                  >
                    &times;
                  </button>
                </div>

                {/* Navigation arrows for desktop */}
                {!isTranslating && getAllAudios().length > 1 && (
                  <>
                    <button
                      onClick={goToPrevModal}
                      className="hidden md:flex absolute left-4 top-1/2 transform -translate-y-1/2 bg-black/50 hover:bg-black/70 text-white p-2 rounded-full transition-colors z-10"
                    >
                      <ChevronLeftIcon className="w-6 h-6" />
                    </button>
                    <button
                      onClick={goToNextModal}
                      className="hidden md:flex absolute right-4 top-1/2 transform -translate-y-1/2 bg-black/50 hover:bg-black/70 text-white p-2 rounded-full transition-colors z-10"
                    >
                      <ChevronRightIcon className="w-6 h-6" />
                    </button>
                  </>
                )}

                {/* Swipe indicators for mobile */}
                {!isTranslating && getAllAudios().length > 1 && (
                  <div className="md:hidden absolute top-16 left-1/2 transform -translate-x-1/2 flex gap-2 z-10">
                    {getAllAudios().map((_, index) => (
                      <div
                        key={index}
                        className={`w-2 h-2 rounded-full transition-colors ${index === currentModalIndex ? 'bg-purple-500' : 'bg-gray-500'
                          }`}
                      />
                    ))}
                  </div>
                )}
                {/* Audio or Loading State */}
                {isTranslating ? (
                  <div className="w-full flex flex-col items-center justify-center bg-black rounded-2xl p-8" style={{ minHeight: 300, maxHeight: '65vh' }}>
                    <div className="animate-spin rounded-full h-16 w-16 border-t-2 border-b-2 border-purple-500 mb-4"></div>
                    <p className="text-white text-lg mb-2">{t('videoTranslator.extractingAudio')}</p>
                    <div className="w-full max-w-md bg-gray-800 rounded-full h-2.5">
                      <div
                        className="bg-purple-600 h-2.5 rounded-full transition-all duration-300"
                        style={{ width: `${progress}%` }}
                      ></div>
                    </div>
                    <p className="text-gray-400 text-sm mt-2">{Math.round(progress)}% complete</p>
                  </div>
                ) : error ? (
                  <div className="w-full flex flex-col items-center justify-center bg-black rounded-2xl p-8" style={{ minHeight: 300, maxHeight: '65vh' }}>
                    <svg className="w-12 h-12 text-red-500 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    <p className="text-red-500 text-lg text-center">{error}</p>
                    <p className="text-gray-400 text-sm mt-4 text-center">Please try again or contact support if the issue persists.</p>
                  </div>
                ) : selectedAudio ? (
                  <div className="w-full flex flex-col items-center justify-center bg-black rounded-2xl p-8" style={{ minHeight: 300 }}>
                    {dubbedVideoUrl ? (
                      <>
                        <div className="w-24 h-24 mb-6 rounded-full bg-purple-600 flex items-center justify-center">
                          <svg className="w-12 h-12 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                          </svg>
                        </div>
                        <h3 className="text-xl text-white mb-4">Dubbed Video ({targetLanguageOptions.find(l => l.value === targetLanguage)?.label || 'English'})</h3>
                        <div className="w-full max-w-md bg-gray-800 rounded-lg p-4">
                          <video
                            src={dubbedVideoUrl}
                            className="w-full rounded"
                            controls
                            autoPlay
                            muted
                          />
                          <div className="text-center mt-2">
                            {detectedLanguage && (
                              <p className="text-blue-300 text-sm">
                                Detected: {detectedLanguage}
                                {languageConfidence > 0 && ` (${Math.round(languageConfidence * 100)}% confidence)`}
                              </p>
                            )}
                            <p className="text-purple-300 text-sm">
                              Target: {targetLanguageOptions.find(l => l.value === targetLanguage)?.label || 'English'}
                            </p>
                            <p className="text-green-300 text-sm">🎭 Advanced Multispeaker</p>
                          </div>
                        </div>
                      </>
                    ) : (
                      <>
                        <div className="w-24 h-24 mb-6 rounded-full bg-purple-600 flex items-center justify-center">
                          <svg className="w-12 h-12 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3" />
                          </svg>
                        </div>
                        <h3 className="text-xl text-white mb-4">Translated Audio ({targetLanguageOptions.find(l => l.value === targetLanguage)?.label || 'English'})</h3>
                        <div className="w-full max-w-md bg-gray-800 rounded-lg p-4">
                          <audio
                            src={selectedAudio}
                            className="w-full"
                            controls
                            autoPlay
                          />
                          <div className="text-center mt-2">
                            {detectedLanguage && (
                              <p className="text-blue-300 text-sm">
                                Detected: {detectedLanguage}
                                {languageConfidence > 0 && ` (${Math.round(languageConfidence * 100)}% confidence)`}
                              </p>
                            )}
                            <p className="text-purple-300 text-sm">
                              Target: {targetLanguageOptions.find(l => l.value === targetLanguage)?.label || 'English'}
                            </p>
                            <p className="text-green-300 text-sm">🎭 Advanced Multispeaker</p>
                            {isCreatingVideo && (
                              <p className="text-yellow-300 text-sm">🎬 Creating dubbed video...</p>
                            )}
                          </div>
                        </div>
                      </>
                    )}
                  </div>
                ) : translatedAudioUrl ? (
                  <div className="w-full flex flex-col items-center justify-center bg-black rounded-2xl p-8" style={{ minHeight: 300 }}>
                    {dubbedVideoUrl ? (
                      <>
                        <div className="w-24 h-24 mb-6 rounded-full bg-purple-600 flex items-center justify-center">
                          <svg className="w-12 h-12 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                          </svg>
                        </div>
                        <h3 className="text-xl text-white mb-4">Dubbed Video ({targetLanguageOptions.find(l => l.value === targetLanguage)?.label || 'English'})</h3>
                        <div className="w-full max-w-md bg-gray-800 rounded-lg p-4">
                          <video
                            src={dubbedVideoUrl}
                            className="w-full rounded"
                            controls
                            autoPlay
                            muted
                          />
                          <div className="text-center mt-2">
                            {detectedLanguage && (
                              <p className="text-blue-300 text-sm">
                                Detected: {detectedLanguage}
                                {languageConfidence > 0 && ` (${Math.round(languageConfidence * 100)}% confidence)`}
                              </p>
                            )}
                            <p className="text-purple-300 text-sm">
                              Target: {targetLanguageOptions.find(l => l.value === targetLanguage)?.label || 'English'}
                            </p>
                            <p className="text-green-300 text-sm">🎭 Advanced Multispeaker</p>
                          </div>
                        </div>
                      </>
                    ) : (
                      <>
                        <div className="w-24 h-24 mb-6 rounded-full bg-purple-600 flex items-center justify-center">
                          <svg className="w-12 h-12 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3" />
                          </svg>
                        </div>
                        <h3 className="text-xl text-white mb-4">Translated Audio ({targetLanguageOptions.find(l => l.value === targetLanguage)?.label || 'English'})</h3>
                        <div className="w-full max-w-md bg-gray-800 rounded-lg p-4">
                          <audio
                            src={translatedAudioUrl}
                            className="w-full"
                            controls
                            autoPlay
                          />
                          <div className="text-center mt-2">
                            {detectedLanguage && (
                              <p className="text-blue-300 text-sm">
                                Detected: {detectedLanguage}
                                {languageConfidence > 0 && ` (${Math.round(languageConfidence * 100)}% confidence)`}
                              </p>
                            )}
                            <p className="text-purple-300 text-sm">
                              Target: {targetLanguageOptions.find(l => l.value === targetLanguage)?.label || 'English'}
                            </p>
                            <p className="text-green-300 text-sm">🎭 Advanced Multispeaker</p>
                            {isCreatingVideo && (
                              <p className="text-yellow-300 text-sm">🎬 Creating dubbed video...</p>
                            )}
                          </div>
                        </div>
                      </>
                    )}
                  </div>
                ) : (
                  <div className="w-full flex flex-col items-center justify-center bg-black rounded-2xl p-8" style={{ minHeight: 300, maxHeight: '65vh' }}>
                    <p className="text-white text-lg">Translated audio will appear here</p>
                  </div>
                )}
              </div>
              {/* Actions bar */}
              <div
                className="flex flex-col px-6 py-4 w-full mt-4"
                style={{
                  borderRadius: '12px',
                  border: '1.5px solid #b983ff',
                  background: '#0b0f17',
                  boxShadow: '0 2px 8px 0 rgba(0,0,0,0.10)',
                  minHeight: 64,
                  maxWidth: '100%',
                }}
              >
                {/* Action buttons */}
                <div className="flex justify-between items-center">
                  {/* Share button */}
                  <div
                    onClick={!isTranslating && (selectedAudio || translatedAudioUrl) ? () => navigator.clipboard.writeText(selectedAudio || translatedAudioUrl).then(() => alert('Audio URL copied to clipboard!')) : undefined}
                    className={`flex items-center rounded-lg px-4 py-2 transition duration-200 ${!isTranslating && (selectedAudio || translatedAudioUrl) ? 'cursor-pointer' : 'cursor-not-allowed opacity-50'}`}
                    style={{
                      background: '#374151',
                      border: '1px solid #3a4252',
                      minWidth: 120,
                      gap: 8,
                      fontFamily: '"Space Grotesk", sans-serif',
                      boxShadow: '0 0 0 transparent',
                    }}
                    onMouseEnter={!isTranslating && (selectedAudio || translatedAudioUrl) ? e => e.currentTarget.style.boxShadow = '0 0 12px 2px #3a4252' : undefined}
                    onMouseLeave={!isTranslating && (selectedAudio || translatedAudioUrl) ? e => e.currentTarget.style.boxShadow = '0 0 0 transparent' : undefined}
                  >
                    <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8.684 13.342C8.886 12.938 9 12.482 9 12c0-.482-.114-.938-.316-1.342m0 2.684a3 3 0 110-2.684m0 2.684l6.632 3.316m-6.632-6l6.632-3.316m0 0a3 3 0 105.367-2.684 3 3 0 00-5.367 2.684zm0 9.316a3 3 0 105.368 2.684 3 3 0 00-5.368-2.684z" />
                    </svg>
                    <span className="text-white font-semibold">{t('videoTranslator.share')}</span>
                  </div>

                  <div className="flex gap-2">
                    {/* Download Audio button */}
                    <div
                      onClick={!isTranslating && (selectedAudio || translatedAudioUrl) ? () => {
                        const a = document.createElement('a');
                        a.href = selectedAudio || translatedAudioUrl;
                        a.download = 'translated-audio.wav';
                        document.body.appendChild(a);
                        a.click();
                        document.body.removeChild(a);
                      } : undefined}
                      className={`flex items-center rounded-lg px-3 py-2 transition duration-200 ${!isTranslating && (selectedAudio || translatedAudioUrl) ? 'cursor-pointer' : 'cursor-not-allowed opacity-50'}`}
                      style={{
                        background: '#4c1d95',
                        border: '1px solid #6d28d9',
                        minWidth: 100,
                        gap: 6,
                        fontFamily: '"Space Grotesk", sans-serif',
                        boxShadow: '0 0 0 transparent',
                      }}
                      onMouseEnter={!isTranslating && (selectedAudio || translatedAudioUrl) ? e => e.currentTarget.style.boxShadow = '0 0 12px 2px #6d28d9' : undefined}
                      onMouseLeave={!isTranslating && (selectedAudio || translatedAudioUrl) ? e => e.currentTarget.style.boxShadow = '0 0 0 transparent' : undefined}
                    >
                      <svg className="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3" />
                      </svg>
                      <span className="text-white font-semibold text-sm">Audio</span>
                    </div>

                    {/* Download Video button */}
                    <div
                      onClick={!isTranslating && dubbedVideoUrl ? () => {
                        const a = document.createElement('a');
                        a.href = dubbedVideoUrl;
                        a.download = 'dubbed-video.mp4';
                        document.body.appendChild(a);
                        a.click();
                        document.body.removeChild(a);
                      } : undefined}
                      className={`flex items-center rounded-lg px-3 py-2 transition duration-200 ${!isTranslating && dubbedVideoUrl ? 'cursor-pointer' : 'cursor-not-allowed opacity-50'}`}
                      style={{
                        background: '#059669',
                        border: '1px solid #10b981',
                        minWidth: 100,
                        gap: 6,
                        fontFamily: '"Space Grotesk", sans-serif',
                        boxShadow: '0 0 0 transparent',
                      }}
                      onMouseEnter={!isTranslating && dubbedVideoUrl ? e => e.currentTarget.style.boxShadow = '0 0 12px 2px #10b981' : undefined}
                      onMouseLeave={!isTranslating && dubbedVideoUrl ? e => e.currentTarget.style.boxShadow = '0 0 0 transparent' : undefined}
                    >
                      <svg className="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                      </svg>
                      <span className="text-white font-semibold text-sm">Video</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        <TranslateTutorial />
        <TranslateTutorialButton />
      </div>
    </TranslateTutorialProvider>
  );
}
