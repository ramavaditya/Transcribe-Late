#!/usr/bin/env python3
"""
Enhanced Terminal Live Transcription & Translation
Combines the best features from all projects for optimal terminal-based operation
"""

import os
import time
import threading
import queue
import numpy as np
import sounddevice as sd
from pathlib import Path
import tempfile
import json
import argparse
from typing import Dict, List, Optional
import sys

# Fix OpenMP library conflict on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Try to import faster-whisper first, fallback to openai-whisper
try:
    from faster_whisper import WhisperModel
    USE_FASTER_WHISPER = True
    print("✓ Using faster-whisper for enhanced performance")
except ImportError:
    try:
        import whisper
        USE_FASTER_WHISPER = False
        print("✓ Using openai-whisper (faster-whisper recommended)")
    except ImportError:
        print("❌ Error: No Whisper implementation found!")
        print("Install with: pip install faster-whisper OR pip install openai-whisper")
        sys.exit(1)

# Translation providers
try:
    import requests
    USE_HTTP_TRANSLATION = True
except ImportError:
    USE_HTTP_TRANSLATION = False

try:
    import subprocess
    USE_OLLAMA = True
except ImportError:
    USE_OLLAMA = False

class TerminalTranscriber:
    def __init__(self, config: Dict):
        self.config = config
        self.audio_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.model = None
        self.languages = config.get('languages', ['en', 'ja', 'zh'])
        self.sample_rate = config.get('sample_rate', 16000)
        self.chunk_duration = config.get('chunk_duration', 3.0)
        self.silence_threshold = config.get('silence_threshold', 0.5)
        self.amplitude_threshold = config.get('amplitude_threshold', 0.01)
        
        # Initialize hallucination detection
        self._init_hallucination_detection()
        
        # Initialize Whisper model
        self._load_model()
        
        # Initialize translation
        self._init_translation()
        
    def _load_model(self):
        """Load the appropriate Whisper model"""
        try:
            if USE_FASTER_WHISPER:
                device = "cuda" if self.config.get('use_cuda', False) else "cpu"
                compute_type = "float16" if device == "cuda" else "int8"
                self.model = WhisperModel(
                    self.config.get('model_size', 'base'),
                    device=device,
                    compute_type=compute_type,
                    download_root="./models"
                )
                print(f"✓ Loaded faster-whisper model: {self.config.get('model_size', 'base')} on {device}")
            else:
                self.model = whisper.load_model(self.config.get('model_size', 'base'))
                print(f"✓ Loaded openai-whisper model: {self.config.get('model_size', 'base')}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            sys.exit(1)
    
    def _init_translation(self):
        """Initialize translation capabilities"""
        self.translation_providers = []
        
        if USE_OLLAMA:
            self.translation_providers.append('ollama')
            print("✓ Ollama translation available")
        
        if USE_HTTP_TRANSLATION:
            self.translation_providers.append('http')
            print("✓ HTTP translation available")
        
        if not self.translation_providers:
            print("⚠️  No translation providers available - transcription only")
    
    def _init_hallucination_detection(self):
        """Initialize hallucination detection patterns"""
        self.hallucination_patterns = {
            # Japanese common hallucination phrases
            'ja': [
                "ご視聴ありがとうございました", "ご視聴ありがとうございました。",
                "ありがとうございました", "ありがとうございました。",
                "どうもありがとうございました", "どうもありがとうございました。",
                "どうも、ありがとうございました", "どうも、ありがとうございました。",
                "おやすみなさい", " おやすみなさい。", "おやすみなさい。",
                "終わり", "おわり", "お疲れ様でした", "お疲れ様でした。",
                "最初からやりのをしてください", "やり直してください", "最初が"
            ],
            
            # English common hallucination phrases
            'en': [
                "Thanks for watching!", "Thanking You", "Thank you.",
                "All right.", "Bye.", "Okay.", "Yeah.", ".",
                "Do you agree?", "It's a little bit different.",
                "First", "Please try again.", "Please do the first one."
            ],
            
            # Chinese common hallucination phrases
            'zh': [
                "那他说他说", "最初からやりのをしてください",
                "最初", "重新开始吧。", "祝贺", "Congratulations"
            ],
            
            # Mixed language hallucinations
            'mixed': [
                " E aí", " Obrigado.", " Gracias.", " Obrigada.",
                " Продолжение следует...", " तो अस दे दाहीं के दाहीं"
            ],
            
            # Common patterns that indicate hallucinations
            'patterns': [
                "Thank you for watching", "End of", "Goodbye", "See you",
                "That's all", "The end", "Fin", "終了", "完了",
                "お疲れ様", "お疲れ様でした", "お疲れ様です",
                "ありがとう", "どうも", "さようなら", "バイバイ"
            ]
        }
    
    def audio_capture_worker(self):
        """Capture audio from microphone in real-time"""
        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.float32,
                blocksize=int(self.sample_rate * 0.1),  # 100ms blocks
                callback=self._audio_callback
            ) as stream:
                print("🎤 Audio capture started")
                
                while not self.stop_event.is_set():
                    time.sleep(0.1)
        except Exception as e:
            print(f"❌ Audio capture error: {e}")
    
    def _audio_callback(self, indata, frames, time, status):
        """Audio callback for real-time processing"""
        if status:
            print(f"⚠️  Audio status: {status}")
        
        if not self.stop_event.is_set():
            self.audio_queue.put(indata.copy())
    
    def process_audio_stream(self):
        """Process audio stream in chunks"""
        buffer = []
        samples_per_chunk = int(self.sample_rate * self.chunk_duration)
        
        print(f"\n🎤 Speak now! Audio is being captured in {self.chunk_duration}s chunks...")
        print("=" * 60)
        
        while not self.stop_event.is_set():
            # Collect audio data
            try:
                data = self.audio_queue.get(timeout=0.1)
                buffer.append(data.flatten())
                
                # Process chunk when we have enough data
                if len(buffer) * len(data) >= samples_per_chunk:
                    # Concatenate all buffered audio
                    audio_chunk = np.concatenate(buffer)
                    buffer = []  # Clear buffer
                    
                    # Check if audio has meaningful content
                    max_amplitude = np.max(np.abs(audio_chunk))
                    if max_amplitude > self.amplitude_threshold:
                        self._process_audio_chunk(audio_chunk)
                        
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Audio processing error: {e}")
                buffer = []  # Reset buffer on error
    
    def _process_audio_chunk(self, audio_data: np.ndarray):
        """Process a single audio chunk"""
        try:
            # Normalize audio
            audio_normalized = audio_data.astype(np.float32)
            
            # Transcribe
            if USE_FASTER_WHISPER:
                segments, info = self.model.transcribe(
                    audio_normalized,
                    language=None,  # Auto-detect
                    task="transcribe"
                )
                transcription = " ".join([seg.text for seg in segments])
                detected_lang = info.language
            else:
                result = self.model.transcribe(
                    audio_normalized,
                    language=None,
                    task="transcribe"
                )
                transcription = result["text"].strip()
                detected_lang = result.get("language", "unknown")
            
            if transcription.strip():
                # Check for hallucinations first
                if self._is_hallucination(transcription, detected_lang):
                    print(f"🚫 Hallucination detected: '{transcription}' - Skipping")
                    return
                
                # Only process if detected language is one of our supported languages
                if detected_lang in self.languages:
                    self._display_results(detected_lang, transcription)
                else:
                    # Map similar languages to our supported ones
                    mapped_lang = self._map_language(detected_lang)
                    if mapped_lang in self.languages:
                        self._display_results(mapped_lang, transcription)
                    else:
                        # Skip languages we don't support
                        print(f"⚠️  Skipping unsupported language: {detected_lang} ({transcription})")
                
        except Exception as e:
            print(f"❌ Transcription error: {e}")
    
    def _display_results(self, detected_lang: str, transcription: str):
        """Display transcription and translation results"""
        timestamp = time.strftime("%H:%M:%S")
        
        # Clear line and display results
        print(f"\r{' ' * 80}", end='\r')  # Clear line
        
        # Display transcription
        lang_name = self._get_language_name(detected_lang)
        print(f"[{timestamp}] [{lang_name}] {transcription}")
        
        # Get translations
        if self.translation_providers:
            translations = self._get_translations(transcription, detected_lang)
            for target_lang, translation in translations.items():
                if translation and translation != transcription:
                    target_name = self._get_language_name(target_lang)
                    print(f"           [{target_name}] {translation}")
        
        print("-" * 60)
    
    def _get_translations(self, text: str, source_lang: str) -> Dict[str, str]:
        """Get translations for the given text"""
        translations = {}
        
        for target_lang in self.languages:
            if target_lang != source_lang:
                translation = self._translate_text(text, source_lang, target_lang)
                if translation:
                    translations[target_lang] = translation
        
        return translations
    
    def _translate_text(self, text: str, source_lang: str, target_lang: str) -> Optional[str]:
        """Translate text using available providers"""
        try:
            if 'ollama' in self.translation_providers:
                return self._translate_with_ollama(text, source_lang, target_lang)
            elif 'http' in self.translation_providers:
                return self._translate_with_http(text, source_lang, target_lang)
        except Exception as e:
            print(f"⚠️  Translation error ({source_lang}→{target_lang}): {e}")
        return None
    
    def _translate_with_ollama(self, text: str, source_lang: str, target_lang: str) -> Optional[str]:
        """Translate using Ollama"""
        try:
            # Only translate between our three supported languages
            if source_lang not in self.languages or target_lang not in self.languages:
                return None
                
            lang_names = {
                'en': 'English', 'ja': 'Japanese', 'zh': 'Chinese'
            }
            
            prompt = f"Translate this text from {lang_names.get(source_lang, source_lang)} to {lang_names.get(target_lang, target_lang)}. Output only the translation:\n\n{text}"
            
            result = subprocess.run([
                "ollama", "run", "gemma3:1b"
            ], input=prompt.encode('utf-8'), capture_output=True, timeout=10)
            
            if result.returncode == 0:
                return result.stdout.decode('utf-8').strip()
        except Exception as e:
            print(f"Ollama translation error: {e}")
        return None
    
    def _translate_with_http(self, text: str, source_lang: str, target_lang: str) -> Optional[str]:
        """Translate using HTTP API (if available)"""
        # This would be implemented for your specific translation API
        return None
    
    def _get_language_name(self, lang_code: str) -> str:
        """Get human-readable language name"""
        lang_names = {
            'en': '🇺🇸 English',
            'ja': '🇯🇵 Japanese', 
            'zh': '🇨🇳 Chinese'
        }
        return lang_names.get(lang_code, f"🌐 {lang_code}")
    
    def _map_language(self, detected_lang: str) -> str:
        """Map detected language to supported language if possible"""
        # Language mapping for similar languages
        language_mapping = {
            # Hindi and similar South Asian languages -> English
            'hi': 'en',  # Hindi
            'bn': 'en',  # Bengali
            'ur': 'en',  # Urdu
            'ta': 'en',  # Tamil
            'te': 'en',  # Telugu
            'mr': 'en',  # Marathi
            'gu': 'en',  # Gujarati
            'kn': 'en',  # Kannada
            'ml': 'en',  # Malayalam
            'pa': 'en',  # Punjabi
            'or': 'en',  # Odia
            'as': 'en',  # Assamese
            
            # Korean -> Japanese (closer linguistically)
            'ko': 'ja',
            
            # Other languages -> English
            'es': 'en',  # Spanish
            'fr': 'en',  # French
            'de': 'en',  # German
            'ru': 'en',  # Russian
            'ar': 'en',  # Arabic
            'pt': 'en',  # Portuguese
            'it': 'en',  # Italian
            'nl': 'en',  # Dutch
            'sv': 'en',  # Swedish
            'da': 'en',  # Danish
            'no': 'en',  # Norwegian
            'fi': 'en',  # Finnish
            'pl': 'en',  # Polish
            'tr': 'en',  # Turkish
            'th': 'en',  # Thai
            'vi': 'en',  # Vietnamese
            'id': 'en',  # Indonesian
            'ms': 'en',  # Malay
        }
        
        return language_mapping.get(detected_lang, detected_lang)
    
    def _is_hallucination(self, text: str, detected_lang: str) -> bool:
        """Check if the transcribed text is a hallucination"""
        text_lower = text.lower().strip()
        
        # Check language-specific hallucinations
        if detected_lang in self.hallucination_patterns:
            for pattern in self.hallucination_patterns[detected_lang]:
                if pattern.lower() in text_lower or text_lower in pattern.lower():
                    return True
        
        # Check mixed language hallucinations
        for pattern in self.hallucination_patterns['mixed']:
            if pattern.lower() in text_lower or text_lower in pattern.lower():
                return True
        
        # Check common patterns
        for pattern in self.hallucination_patterns['patterns']:
            if pattern.lower() in text_lower or text_lower in pattern.lower():
                return True
        
        # Check for suspicious patterns
        suspicious_patterns = [
            # Very short or repetitive text
            len(text.strip()) < 3,
            text.count(text[0]) == len(text) if text else False,
            
            # Common ending phrases
            text_lower.endswith(('thank you', 'goodbye', 'bye', 'end', 'fin')),
            text_lower.endswith(('ありがとう', 'さようなら', 'バイバイ', '終了', '完了')),
            text_lower.endswith(('谢谢', '再见', '结束', '完了')),
            
            # Suspicious language mixing
            any(char in text for char in ['а', 'б', 'в', 'г', 'д', 'е', 'ё', 'ж', 'з']),  # Cyrillic
            any(char in text for char in ['अ', 'आ', 'इ', 'ई', 'उ', 'ऊ', 'ए', 'ऐ', 'ओ', 'औ']),  # Devanagari
            
            # Very generic responses
            text_lower in ['okay', 'yeah', 'yes', 'no', 'maybe', 'sure'],
            text_lower in ['はい', 'いいえ', 'うん', 'そう', 'まあ'],
            text_lower in ['是的', '不是', '好的', '可以', '也许']
        ]
        
        return any(suspicious_patterns)
    
    def start(self):
        """Start the transcription process"""
        print("🚀 Starting Enhanced Terminal Transcription & Translation")
        print(f"📝 Languages: {', '.join([self._get_language_name(lang) for lang in self.languages])}")
        print(f"🎵 Sample Rate: {self.sample_rate}Hz")
        print(f"⏱️  Chunk Duration: {self.chunk_duration}s")
        print(f"🔇 Silence Threshold: {self.silence_threshold}s")
        print("\nPress Ctrl+C to stop\n")
        
        # Start audio capture thread
        capture_thread = threading.Thread(target=self.audio_capture_worker, daemon=True)
        capture_thread.start()
        
        # Start processing thread
        processing_thread = threading.Thread(target=self.process_audio_stream, daemon=True)
        processing_thread.start()
        
        try:
            # Keep main thread alive
            while not self.stop_event.is_set():
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n\n🛑 Stopping transcription...")
            self.stop_event.set()
            
            # Wait for threads to finish
            capture_thread.join(timeout=2)
            processing_thread.join(timeout=2)
            
            print("✅ Transcription stopped successfully")
    
    def stop(self):
        """Stop the transcription process"""
        self.stop_event.set()

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Enhanced Terminal Live Transcription & Translation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python enhanced-terminal-transcribe.py
  python enhanced-terminal-transcribe.py --model large --chunk-duration 5
  python enhanced-terminal-transcribe.py --languages en ja zh --use-cuda
        """
    )
    
    parser.add_argument(
        '--model', 
        default='base',
        choices=['tiny', 'base', 'small', 'medium', 'large'],
        help='Whisper model size (default: base)'
    )
    
    parser.add_argument(
        '--languages',
        nargs='+',
        default=['en', 'ja', 'zh'],
        help='Languages to support (default: en ja zh - English, Japanese, Chinese)'
    )
    
    parser.add_argument(
        '--chunk-duration',
        type=float,
        default=3.0,
        help='Audio chunk duration in seconds (default: 3.0)'
    )
    
    parser.add_argument(
        '--sample-rate',
        type=int,
        default=16000,
        help='Audio sample rate (default: 16000)'
    )
    
    parser.add_argument(
        '--silence-threshold',
        type=float,
        default=0.5,
        help='Silence threshold in seconds (default: 0.5)'
    )
    
    parser.add_argument(
        '--use-cuda',
        action='store_true',
        help='Use CUDA acceleration (if available)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration JSON file'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = {
        'model_size': args.model,
        'languages': args.languages,
        'chunk_duration': args.chunk_duration,
        'sample_rate': args.sample_rate,
        'silence_threshold': args.silence_threshold,
        'use_cuda': args.use_cuda,
        'amplitude_threshold': 0.01
    }
    
    # Load from config file if specified
    if args.config and os.path.exists(args.config):
        try:
            with open(args.config, 'r') as f:
                file_config = json.load(f)
                config.update(file_config)
                print(f"✓ Loaded configuration from {args.config}")
        except Exception as e:
            print(f"⚠️  Error loading config file: {e}")
    
    # Create and start transcriber
    transcriber = TerminalTranscriber(config)
    
    try:
        transcriber.start()
    except KeyboardInterrupt:
        transcriber.stop()

if __name__ == "__main__":
    main()
