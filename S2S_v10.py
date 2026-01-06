"""
Modular Voice Assistant with RAG Support
Speech-to-Speech interface powered by RAG, Whisper, and voice synthesis
"""

import os
import struct
import logging
from typing import Optional
import sounddevice as sd
import pyttsx3
import pvporcupine
import groq
import numpy as np
import speech_recognition as sr
import time
from datetime import datetime
from dotenv import load_dotenv

from rag_system import RAGSystem
from prompt_manager import SystemPrompt, RAGPromptBuilder, ConversationManager

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VoiceAssistant:
    """
    Modular voice assistant with RAG, wake word detection, and continuous conversation.
    """
    
    def __init__(self):
        """Initialize the voice assistant."""
        self._validate_environment()
        self._initialize_components()
        self._initialize_rag()
        self._initialize_conversation_state()
    
    def _validate_environment(self):
        """Validate required environment variables and API keys."""
        required_keys = ['PORCUPINE_API_KEY', 'GROQ_API_KEY', 'OPEN_API']
        missing_keys = [key for key in required_keys if not os.getenv(key)]
        
        if missing_keys:
            raise ValueError(f"Missing environment variables: {', '.join(missing_keys)}")
        
        logger.info("✅ All required environment variables present")
    
    def _initialize_components(self):
        """Initialize speech recognition, TTS, and Groq client."""
        try:
            # Initialize Groq client
            self.groq_client = groq.Client(api_key=os.getenv('GROQ_API_KEY'))
            
            # Initialize TTS engine
            self.tts_engine = pyttsx3.init()
            self._configure_tts()
            
            # Initialize speech recognition
            self.recognizer = sr.Recognizer()
            self._configure_speech_recognition()
            
            # Initialize Porcupine wake word detector
            self._initialize_porcupine()
            
            logger.info("✅ All components initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize components: {e}")
            raise
    
    def _initialize_rag(self):
        """Initialize the RAG system."""
        try:
            self.rag_system = RAGSystem(
                kb_path="info.txt",
                faiss_path="faiss_index.idx",
                embeddings_path="embeddings.npy",
                chunks_path="doc_chunks.pkl"
            )
            self.rag_system.initialize()
            logger.info("✅ RAG system initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize RAG system: {e}")
            self.rag_system = None
    
    def _initialize_conversation_state(self):
        """Initialize conversation management."""
        self.conversation_manager = ConversationManager(max_history=4)
        self.in_conversation = False
        self.conversation_timeout = 60  # seconds
        self.last_interaction_time = None
    
    def _initialize_porcupine(self):
        """Initialize Porcupine wake word detector."""
        try:
            # Try to load custom wake word model
            wake_word_model = os.path.join(
                os.path.dirname(__file__),
                "Hey-Bruce_en_windows_v3_0_0.ppn"
            )
            
            # Check if custom model exists
            if os.path.exists(wake_word_model):
                try:
                    self.porcupine = pvporcupine.create(
                        access_key=os.getenv('PORCUPINE_API_KEY'),
                        keyword_paths=[wake_word_model]
                    )
                    logger.info("✅ Porcupine initialized with custom 'Hey Bruce' model")
                except Exception as custom_error:
                    logger.warning(f"⚠️ Custom model failed ({custom_error}), trying built-in keywords...")
                    # Fallback to built-in "hey google" keyword
                    self._initialize_porcupine_builtin()
            else:
                # Custom model doesn't exist, use built-in
                logger.info("ℹ️ Custom wake word model not found, using built-in 'hey google'")
                self._initialize_porcupine_builtin()
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Porcupine: {e}")
            raise
    
    def _initialize_porcupine_builtin(self):
        """Initialize Porcupine with built-in keyword."""
        try:
            self.porcupine = pvporcupine.create(
                access_key=os.getenv('PORCUPINE_API_KEY'),
                keywords=["google"]  # Built-in keyword
            )
            logger.info("✅ Porcupine initialized with built-in 'hey google' keyword")
            logger.info("   📝 To use 'Hey Bruce', download the macOS .ppn file from:")
            logger.info("      https://console.picovoice.ai/")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Porcupine with built-in keywords: {e}")
            raise
    
    def _configure_tts(self):
        """Configure text-to-speech settings."""
        try:
            self.tts_engine.setProperty('rate', 180)  # Speaking rate
            self.tts_engine.setProperty('volume', 0.9)  # Volume (0.0-1.0)
            logger.info("✅ TTS configured")
        except Exception as e:
            logger.warning(f"⚠️ Could not configure TTS: {e}")
    
    def _configure_speech_recognition(self):
        """Configure speech recognition settings."""
        self.recognizer.energy_threshold = 400
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 1.0
        self.recognizer.phrase_threshold = 0.2
        self.recognizer.non_speaking_duration = 0.8
    
    def speak(self, text: str):
        """
        Convert text to speech.
        
        Args:
            text: Text to speak
        """
        try:
            logger.info(f"🔊 Speaking: {text}")
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        except Exception as e:
            logger.error(f"❌ TTS error: {e}")
    
    def listen_for_command(self, timeout: int = 12, phrase_limit: int = 20) -> Optional[str]:
        """
        Listen for user command.
        
        Args:
            timeout: Maximum time to wait for speech
            phrase_limit: Maximum length of phrase in seconds
            
        Returns:
            Transcribed text or None if failed
        """
        try:
            with sr.Microphone() as source:
                logger.info("🎤 Listening for command...")
                
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                
                # Listen for audio
                audio = self.recognizer.listen(
                    source,
                    timeout=timeout,
                    phrase_time_limit=phrase_limit
                )
                
                # Try to recognize speech
                try:
                    text = self.recognizer.recognize_google(audio)
                    logger.info(f"✅ Recognized: {text}")
                    return text.strip()
                except sr.UnknownValueError:
                    logger.info("⚠️ Could not understand audio")
                    if self.in_conversation:
                        self.speak("Sorry, I didn't catch that. Could you repeat?")
                    return None
                
        except sr.WaitTimeoutError:
            logger.warning("⏱️ Listening timeout - no speech detected")
            if self.in_conversation:
                self.speak("I'm still here. What would you like to know?")
            return None
            
        except sr.RequestError as e:
            logger.error(f"❌ Speech recognition error: {e}")
            self.speak("Sorry, there was an error with speech recognition.")
            return None
    
    def _handle_special_commands(self, user_input: str) -> Optional[str]:
        """
        Handle special commands like time, date, etc.
        
        Args:
            user_input: User's input
            
        Returns:
            Response if special command handled, None otherwise
        """
        user_lower = user_input.lower().strip()
        
        # Check for time queries
        time_keywords = ['time', 'what time', 'current time', "what's the time"]
        if any(kw in user_lower for kw in time_keywords):
            current_time = datetime.now().strftime("%I:%M %p")
            date_str = datetime.now().strftime("%A, %B %d, %Y")
            return f"The current time is {current_time} on {date_str}."
        
        # Check for date queries
        date_keywords = ['date', 'what date', 'current date', "what's the date"]
        if any(kw in user_lower for kw in date_keywords):
            date_str = datetime.now().strftime("%A, %B %d, %Y")
            return f"Today is {date_str}."
        
        # Check for day queries
        day_keywords = ['what day', 'day of the week', 'day is today']
        if any(kw in user_lower for kw in day_keywords):
            day = datetime.now().strftime("%A")
            return f"Today is {day}."
        
        return None
    
    def _handle_conversation_control(self, user_input: str) -> Optional[str]:
        """
        Handle conversation control commands.
        
        Args:
            user_input: User's input
            
        Returns:
            Response if control command handled, None otherwise
        """
        user_lower = user_input.lower().strip()
        
        exit_keywords = ['goodbye', 'bye', 'stop', 'exit', 'quit', 'see you later']
        if any(kw in user_lower for kw in exit_keywords):
            self.conversation_manager.clear()
            self.in_conversation = False
            return "Goodbye! Say 'Hey Bruce' if you need me again."
        
        return None
    
    def get_ai_response(self, user_input: str) -> str:
        """
        Get AI response using RAG system.
        
        Args:
            user_input: User's input
            
        Returns:
            AI response text
        """
        try:
            user_lower = user_input.lower().strip()
            
            # Check for conversation control
            control_response = self._handle_conversation_control(user_input)
            if control_response:
                return control_response
            
            # Check for special commands
            special_response = self._handle_special_commands(user_input)
            if special_response:
                return special_response
            
            # Get RAG context
            retrieved_chunks = []
            confidence = "LOW"
            
            if self.rag_system:
                retrieved_chunks = self.rag_system.retrieve_context(user_lower, k=3)
                if retrieved_chunks:
                    confidence = self.rag_system.get_confidence_level(retrieved_chunks[0][1])
            
            # Build prompt
            prompt, confidence = RAGPromptBuilder.build_rag_prompt(
                question=user_lower,
                retrieved_chunks=retrieved_chunks,
                conversation_history=self.conversation_manager.get_history()
            )
            
            # Log retrieval info
            if retrieved_chunks:
                score = retrieved_chunks[0][1]
                logger.info(f"🔍 Retrieved context ({confidence} confidence, score={score:.3f})")
            else:
                logger.info(f"🧠 No context retrieved, using general knowledge")
            
            # Prepare messages for API
            messages = [
                {
                    "role": "system",
                    "content": SystemPrompt.get_content()
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            # Add conversation history
            messages.extend(self.conversation_manager.format_for_api())
            
            # Get response from Groq
            response = self.groq_client.chat.completions.create(
                messages=messages,
                model="gemma2-9b-it",
                max_tokens=150,
                temperature=0.7
            )
            
            reply = response.choices[0].message.content.strip()
            
            # Update conversation history
            self.conversation_manager.add_message("user", user_input)
            self.conversation_manager.add_message("assistant", reply)
            
            logger.info(f"✅ AI response: {reply}")
            return reply
            
        except Exception as e:
            logger.error(f"❌ Error getting AI response: {e}")
            return "Sorry, I'm having trouble processing your request right now."
    
    def start_conversation(self):
        """Start and maintain continuous conversation."""
        self.in_conversation = True
        self.last_interaction_time = time.time()
        self.conversation_manager.clear()
        
        logger.info("🎙️ Starting conversation")
        self.speak("Hi, I'm Bruce. How can I help you today?")
        
        while self.in_conversation:
            # Check for conversation timeout
            if time.time() - self.last_interaction_time > self.conversation_timeout:
                self.speak("I haven't heard from you in a while. Say 'Hey Bruce' if you need me again.")
                self.in_conversation = False
                break
            
            # Listen for user input
            user_command = self.listen_for_command(timeout=8)
            
            if user_command:
                self.last_interaction_time = time.time()
                
                # Get AI response
                ai_response = self.get_ai_response(user_command)
                
                # Speak response
                self.speak(ai_response)
                
                # Check if conversation should end
                if not self.in_conversation:
                    break
        
        logger.info("🛑 Conversation ended")
    
    def listen_for_wake_word(self):
        """Listen for wake word and start conversation when detected."""
        wake_word = "Hey Bruce" if hasattr(self.porcupine, 'keywords') and 'bruce' in str(self.porcupine.keywords).lower() else "Hey Google"
        logger.info(f"👂 Voice Assistant started. Listening for '{wake_word}'...")
        
        try:
            with sd.RawInputStream(
                samplerate=self.porcupine.sample_rate,
                blocksize=self.porcupine.frame_length,
                dtype='int16',
                channels=1
            ) as stream:
                
                while True:
                    if not self.in_conversation:
                        # Read audio data
                        pcm_data = stream.read(self.porcupine.frame_length)[0]
                        pcm = struct.unpack_from("h" * self.porcupine.frame_length, pcm_data)
                        
                        # Process for wake word
                        result = self.porcupine.process(pcm)
                        
                        if result >= 0:
                            logger.info("🎯 Wake word detected!")
                            self.start_conversation()
                    else:
                        time.sleep(0.1)
                        
        except KeyboardInterrupt:
            logger.info("⛔ Voice assistant stopped by user")
        except Exception as e:
            logger.error(f"❌ Wake word detection error: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        try:
            if hasattr(self, 'porcupine') and self.porcupine:
                self.porcupine.delete()
            logger.info("✅ Cleanup completed")
        except Exception as e:
            logger.error(f"❌ Cleanup error: {e}")
    
    def run(self):
        """Start the voice assistant."""
        try:
            self.listen_for_wake_word()
        except Exception as e:
            logger.error(f"❌ Fatal error: {e}")
        finally:
            self.cleanup()


def main():
    """Main entry point."""
    assistant = VoiceAssistant()
    assistant.run()


if __name__ == "__main__":
    main()
