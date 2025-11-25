import React, { useState, useEffect } from 'react';
import Section from './Section';
import Heading from './Heading';
import Button from './Button';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../register/AuthContext.jsx';
import { useTranslation } from '../../utils/i18n-simple.js';
import aiImageDemo from '../assets/reelistic/AI Image.png';
import aiTranslationDemo from '../assets/reelistic/AI Translation.png';

const VoiceCloning = () => {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const { user } = useAuth();
  const [selectedLanguage, setSelectedLanguage] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);

  const languages = [
    { code: 'en', name: 'English', flag: '🇺🇸', accent: 'American' },
    { code: 'es', name: 'Spanish', flag: '🇪🇸', accent: 'European' },
    { code: 'fr', name: 'French', flag: '🇫🇷', accent: 'Parisian' },
    { code: 'de', name: 'German', flag: '🇩🇪', accent: 'Standard' },
    { code: 'it', name: 'Italian', flag: '🇮🇹', accent: 'Roman' },
    { code: 'pt', name: 'Portuguese', flag: '🇧🇷', accent: 'Brazilian' },
    { code: 'ja', name: 'Japanese', flag: '🇯🇵', accent: 'Tokyo' },
    { code: 'ko', name: 'Korean', flag: '🇰🇷', accent: 'Seoul' },
    { code: 'zh', name: 'Chinese', flag: '🇨🇳', accent: 'Mandarin' },
    { code: 'hi', name: 'Hindi', flag: '🇮🇳', accent: 'Standard' },
    { code: 'ar', name: 'Arabic', flag: '🇸🇦', accent: 'Modern Standard' },
    { code: 'ru', name: 'Russian', flag: '🇷🇺', accent: 'Moscow' }
  ];

  const features = [
    {
      icon: '🎤',
      title: t('voiceCloning.features.voiceCloning.title'),
      description: t('voiceCloning.features.voiceCloning.description'),
      details: t('voiceCloning.features.voiceCloning.details')
    },
    {
      icon: '🌍',
      title: t('voiceCloning.features.languages.title'),
      description: t('voiceCloning.features.languages.description'),
      details: t('voiceCloning.features.languages.details')
    },
    {
      icon: '⚡',
      title: t('voiceCloning.features.realTime.title'),
      description: t('voiceCloning.features.realTime.description'),
      details: t('voiceCloning.features.realTime.details')
    },
    {
      icon: '🎯',
      title: t('voiceCloning.features.lipSync.title'),
      description: t('voiceCloning.features.lipSync.description'),
      details: t('voiceCloning.features.lipSync.details')
    }
  ];

  useEffect(() => {
    const interval = setInterval(() => {
      setSelectedLanguage((prev) => (prev + 1) % languages.length);
    }, 2000);
    return () => clearInterval(interval);
  }, []);

  const handleTryNow = () => {
    if (user) {
      navigate('/tools/translate');
    } else {
      navigate('/signup');
    }
  };

  const togglePlay = () => {
    setIsPlaying(!isPlaying);
    // Simulate audio playback
    if (!isPlaying) {
      setTimeout(() => setIsPlaying(false), 3000);
    }
  };

  return (
    <Section className="py-20 bg-gradient-to-b from-n-7 to-n-8" id="voice-cloning">
      <div className="container relative">
        <header className="text-center mb-12 md:mb-20">
          <h2 className="text-4xl md:text-5xl font-bold text-n-1 mb-6">
            {t('voiceCloning.title')}
          </h2>
          <p className="text-xl text-n-3 max-w-3xl mx-auto">
            {t('voiceCloning.subtitle')}
          </p>
        </header>

        <div className="max-w-7xl mx-auto">
          <div className="grid lg:grid-cols-2 gap-16 items-center">

            {/* Left Side - Interactive Demo */}
            <div className="relative">
              <div className="bg-gradient-to-r from-blue-500/20 to-purple-500/20 rounded-3xl p-8 backdrop-blur-sm border border-blue-500/20">

                {/* AI Translation Demo */}
                <div className="bg-n-8 rounded-2xl overflow-hidden shadow-2xl mb-6">
                  <img
                    src={aiTranslationDemo}
                    alt="AI Voice Translation Demo"
                    className="w-full h-48 object-cover"
                  />
                </div>

                {/* Voice Cloning Interface */}
                <div className="bg-n-8 rounded-2xl p-6 shadow-2xl">
                  <div className="flex items-center justify-between mb-6">
                    <h3 className="text-xl font-bold text-n-1">{t('voiceCloning.demo.studioTitle')}</h3>
                    <div className="flex items-center gap-2">
                      <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
                      <span className="text-n-3 text-sm">{t('voiceCloning.demo.aiReady')}</span>
                    </div>
                  </div>

                  {/* Original Voice Section */}
                  <div className="mb-6">
                    <div className="flex items-center gap-3 mb-3">
                      <div className="w-10 h-10 bg-gradient-to-r from-green-400 to-emerald-500 rounded-full flex items-center justify-center">
                        🎤
                      </div>
                      <div>
                        <h4 className="text-n-1 font-semibold">{t('voiceCloning.demo.originalVoice')}</h4>
                        <p className="text-n-4 text-sm">{t('voiceCloning.demo.yourVoice')}</p>
                      </div>
                    </div>
                    <div className="bg-n-7 rounded-lg p-4">
                      <p className="text-n-2 text-sm mb-3">"{t('voiceCloning.demo.welcomeMessage')}"</p>
                      <div className="flex items-center gap-3">
                        <button
                          onClick={togglePlay}
                          className="w-8 h-8 bg-green-500 rounded-full flex items-center justify-center hover:bg-green-600 transition-colors"
                        >
                          {isPlaying ? '⏸️' : '▶️'}
                        </button>
                        <div className="flex-1 bg-n-6 rounded-full h-2 overflow-hidden">
                          <div className={`h-full bg-green-400 transition-all duration-3000 ${isPlaying ? 'w-full' : 'w-0'}`} />
                        </div>
                        <span className="text-n-4 text-xs">0:03</span>
                      </div>
                    </div>
                  </div>

                  {/* Language Selection */}
                  <div className="mb-6">
                    <h4 className="text-n-1 font-semibold mb-3">{t('voiceCloning.demo.selectLanguage')}</h4>
                    <div className="grid grid-cols-4 gap-2 max-h-32 overflow-y-auto">
                      {languages.map((lang, index) => (
                        <button
                          key={lang.code}
                          onClick={() => setSelectedLanguage(index)}
                          className={`p-2 rounded-lg text-xs transition-all duration-300 ${selectedLanguage === index
                              ? 'bg-blue-500 text-white shadow-lg scale-105'
                              : 'bg-n-7 text-n-3 hover:bg-n-6'
                            }`}
                        >
                          <div className="text-lg mb-1">{lang.flag}</div>
                          <div className="font-medium">{lang.name}</div>
                        </button>
                      ))}
                    </div>
                  </div>

                  {/* Translated Voice Section */}
                  <div>
                    <div className="flex items-center gap-3 mb-3">
                      <div className="w-10 h-10 bg-gradient-to-r from-blue-400 to-purple-500 rounded-full flex items-center justify-center">
                        {languages[selectedLanguage].flag}
                      </div>
                      <div>
                        <h4 className="text-n-1 font-semibold">{t('voiceCloning.demo.clonedVoice')}</h4>
                        <p className="text-n-4 text-sm">{languages[selectedLanguage].name} • {languages[selectedLanguage].accent}</p>
                      </div>
                      <div className="ml-auto">
                        <span className="bg-blue-500 text-white text-xs px-2 py-1 rounded-full">{t('voiceCloning.demo.generated')}</span>
                      </div>
                    </div>
                    <div className="bg-n-7 rounded-lg p-4">
                      <p className="text-n-2 text-sm mb-3">
                        {selectedLanguage === 1 && "¡Bienvenidos a mi canal! Hoy vamos a aprender sobre la creación de videos con IA..."}
                        {selectedLanguage === 2 && "Bienvenue sur ma chaîne ! Aujourd'hui, nous allons apprendre la création vidéo IA..."}
                        {selectedLanguage === 3 && "Willkommen auf meinem Kanal! Heute lernen wir über KI-Videoerstellung..."}
                        {selectedLanguage === 4 && "Benvenuti nel mio canale! Oggi impareremo la creazione di video AI..."}
                        {selectedLanguage === 6 && "私のチャンネルへようこそ！今日はAI動画作成について学びます..."}
                        {![1, 2, 3, 4, 6].includes(selectedLanguage) && `"Welcome to my channel! Today we're learning about AI video creation..." (${languages[selectedLanguage].name})`}
                      </p>
                      <div className="flex items-center gap-3">
                        <button className="w-8 h-8 bg-blue-500 rounded-full flex items-center justify-center hover:bg-blue-600 transition-colors">
                          ▶️
                        </button>
                        <div className="flex-1 bg-n-6 rounded-full h-2">
                          <div className="h-full bg-blue-400 rounded-full w-0" />
                        </div>
                        <span className="text-n-4 text-xs">0:03</span>
                      </div>
                    </div>
                  </div>

                  {/* Processing Status */}
                  <div className="mt-4 p-3 bg-green-500/10 border border-green-500/20 rounded-lg">
                    <div className="flex items-center gap-2">
                      <div className="w-4 h-4 bg-green-400 rounded-full animate-pulse" />
                      <span className="text-green-300 text-sm font-medium">{t('voiceCloning.demo.voiceCloningComplete')}</span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Floating Stats */}
              <div className="absolute -top-6 -right-6 bg-gradient-to-r from-purple-500 to-pink-500 rounded-2xl p-4 shadow-xl">
                <div className="text-center">
                  <div className="text-2xl font-bold text-white">50+</div>
                  <div className="text-purple-100 text-xs">{t('voiceCloning.stats.languages')}</div>
                </div>
              </div>

              <div className="absolute -bottom-6 -left-6 bg-gradient-to-r from-blue-500 to-cyan-500 rounded-2xl p-4 shadow-xl">
                <div className="text-center">
                  <div className="text-2xl font-bold text-white">98.7%</div>
                  <div className="text-blue-100 text-xs">{t('voiceCloning.stats.accuracy')}</div>
                </div>
              </div>
            </div>

            {/* Right Side - Features */}
            <div className="space-y-8">
              {features.map((feature, index) => (
                <div
                  key={index}
                  className="group p-6 bg-n-7/50 rounded-2xl border border-n-6 hover:border-blue-500/30 hover:bg-blue-500/5 transition-all duration-300"
                >
                  <div className="flex items-start gap-4">
                    <div className="w-12 h-12 bg-gradient-to-r from-blue-500 to-purple-500 rounded-xl flex items-center justify-center text-2xl shadow-lg group-hover:scale-110 transition-transform duration-300">
                      {feature.icon}
                    </div>
                    <div className="flex-1">
                      <h3 className="text-xl font-bold text-n-1 mb-2 group-hover:text-blue-300 transition-colors">
                        {feature.title}
                      </h3>
                      <p className="text-n-3 mb-3">{feature.description}</p>
                      <p className="text-sm text-n-4 leading-relaxed">{feature.details}</p>
                    </div>
                  </div>
                </div>
              ))}

              {/* CTA */}
              <div className="bg-gradient-to-r from-blue-500/10 to-purple-500/10 rounded-2xl p-6 border border-blue-500/20">
                <h3 className="text-xl font-bold text-n-1 mb-3">{t('voiceCloning.cta.title')}</h3>
                <p className="text-n-3 mb-4">{t('voiceCloning.cta.description')}</p>
                <Button
                  onClick={handleTryNow}
                  className="bg-gradient-to-r from-blue-500 to-purple-500 hover:from-blue-600 hover:to-purple-600 text-white px-6 py-3 rounded-xl shadow-lg hover:shadow-xl transform hover:scale-105 transition-all duration-300"
                >
                  🎤 {user ? t('voiceCloning.cta.button') : t('voiceCloning.cta.buttonLoggedOut')}
                </Button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </Section>
  );
};

export default VoiceCloning;