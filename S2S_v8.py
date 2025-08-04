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
import faiss
import openai
import re
import tempfile
import scipy.io.wavfile
import pickle
from datetime import datetime
import pytz
#import numpy as np
import sounddevice as sd
import tempfile
import scipy.io.wavfile
import os
import openai
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()


PORCUPINE_API_KEY = os.getenv('PORCUPINE_API_KEY')  
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
OPEN_API = os.getenv('OPEN_API')
openai.api_key = OPEN_API


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VoiceAssistant:
    """
    A clean, modular voice assistant with wake word detection and continuous conversation.
    """
    
    def __init__(self):
        # API Keys (should be moved to environment variables in production)        
        self.PORCUPINE_API_KEY = os.getenv('PORCUPINE_API_KEY')  
        self.GROQ_API_KEY = os.getenv('GROQ_API_KEY')
        self.OPEN_API = os.getenv('OPEN_API')
        self.FAISS_PATH = "faiss_index.idx"
        self.CHUNKS_PATH = "doc_chunks.pkl"
        self.chat_history = []

        if not self.PORCUPINE_API_KEY or not self.GROQ_API_KEY or not self.OPEN_API:
            raise ValueError('Missing one or more required keys in the .env file')
        else:
            print('We are good')

        # Conversation state
        self.in_conversation = False
        self.conversation_timeout = 60  # seconds of inactivity before ending conversation
        self.last_interaction_time = None
        
        # Initialize clients and engines
        self._initialize_components()

    def handle_special_commands(self, user_input: str) -> Optional[str]:
        """
        Handle special commands like time, date, etc. before processing through RAG.
        
        Args:
            user_input: User's transcribed speech
            
        Returns:
            Response string if special command handled, None otherwise
        """
        user_input_lower = user_input.lower().strip()
        
        # Time-related queries
        time_keywords = ['time', 'what time is it', 'current time', 'tell me the time', 'what\'s the time']
        if any(keyword in user_input_lower for keyword in time_keywords):
            try:
                # Get current time - you can modify the timezone as needed
                current_time = datetime.now()
                # Format time in a natural way
                time_str = current_time.strftime("%I:%M %p")
                date_str = current_time.strftime("%A, %B %d, %Y")
                
                return f"The current time is {time_str} on {date_str}"
            except Exception as e:
                logger.error(f"Error getting current time: {e}")
                return "Sorry, I couldn't get the current time right now."
        
        # Date-related queries
        date_keywords = ['date', 'what date is it', 'current date', 'tell me the date', 'what\'s the date', 'today\'s date']
        if any(keyword in user_input_lower for keyword in date_keywords):
            try:
                current_date = datetime.now()
                date_str = current_date.strftime("%A, %B %d, %Y")
                return f"Today is {date_str}"
            except Exception as e:
                logger.error(f"Error getting current date: {e}")
                return "Sorry, I couldn't get the current date right now."
        
        # Day of week queries
        day_keywords = ['what day is it', 'what day', 'day of the week', 'day is today']
        if any(keyword in user_input_lower for keyword in day_keywords):
            try:
                current_day = datetime.now().strftime("%A")
                return f"Today is {current_day}"
            except Exception as e:
                logger.error(f"Error getting current day: {e}")
                return "Sorry, I couldn't get the current day right now."
        
        # Return None if no special command was handled
        return None

    def persist(self):
        if os.path.exists(self.FAISS_PATH) and os.path.exists(self.CHUNKS_PATH):
            print("🔄 Loading from disk...")
            self.index = faiss.read_index(self.FAISS_PATH)

            with open(self.CHUNKS_PATH, "rb") as f:
                self.chunks = pickle.load(f)

            self.embeddings_np = np.load("embeddings.npy")
            
        else:
            print("⚙️ Running first-time setup: chunk + embed + index")

            doc = r'C:\Users\Mubaraq\Downloads\S2S RAG\info.txt'
            with open(doc, 'r') as f:
                long_text = f.read()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            self.chunks = text_splitter.split_text(long_text)

            response = openai.Embedding.create(
                input=self.chunks,
                model="text-embedding-ada-002"
            )

            embeddings = [item["embedding"] for item in response["data"]]
            self.embeddings_np = np.array(embeddings).astype("float32")
            np.save("embeddings.npy", self.embeddings_np) # save embedding

            self.index = faiss.IndexFlatL2(self.embeddings_np.shape[1])
            self.index.add(self.embeddings_np)

            # Save both to disk
            faiss.write_index(self.index, self.FAISS_PATH)

            with open(self.CHUNKS_PATH, "wb") as f:
                pickle.dump(self.chunks, f)

            print("✅ Indexed and saved.")

    def _initialize_components(self):
        """Initialize all required components."""
        try:
            # Initialize Groq client
            self.groq_client = groq.Client(api_key=self.GROQ_API_KEY)
            
            # Initialize TTS engine
            self.tts_engine = pyttsx3.init()
            self._configure_tts()
            
            # Initialize speech recognition with better settings
            self.recognizer = sr.Recognizer()
            # More conservative recognition settings to avoid cutting off
            self.recognizer.energy_threshold = 400  # Higher threshold to reduce false triggers
            self.recognizer.dynamic_energy_threshold = True
            self.recognizer.pause_threshold = 1.0  # Longer pause detection
            self.recognizer.phrase_threshold = 0.2  # Shorter minimum audio length
            self.recognizer.non_speaking_duration = 0.8  # How long to wait after speech stops
            
            # Initialize Porcupine wake word detection
            self.porcupine = pvporcupine.create(
                access_key=self.PORCUPINE_API_KEY,
                keyword_paths=[r"C:\Users\Mubaraq\Downloads\S2S RAG\Hey-Bruce_en_windows_v3_0_0.ppn"]
            )
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    def _configure_tts(self):
        """Configure text-to-speech settings."""
        try:
            # Set speech rate (adjust as needed)
            self.tts_engine.setProperty('rate', 180)
            # Set volume (0.0 to 1.0)
            self.tts_engine.setProperty('volume', 0.9)
        except Exception as e:
            logger.warning(f"Could not configure TTS settings: {e}")
    
    def speak(self, text: str):
        """Convert text to speech."""
        try:
            logger.info(f"Speaking: {text}")
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        except Exception as e:
            logger.error(f"TTS error: {e}")

    def listen_for_command(self, timeout: int = 12, phrase_limit: int = 20) -> Optional[str]:
        """
        Listen for user command with improved sensitivity and longer patience.
        
        Args:
            timeout: Maximum time to wait for speech input
            phrase_limit: Maximum length of phrase in seconds
            
        Returns:
            Transcribed text or None if failed
        """
        try:
            with sr.Microphone() as source:
                logger.info("Listening for command...")
                
                # Adjust for ambient noise with longer duration for better calibration
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                
                # More patient listening settings
                self.recognizer.pause_threshold = 1.2  # Wait longer for pauses (increased from 0.8)
                self.recognizer.phrase_threshold = 0.2  # Shorter minimum audio length
                self.recognizer.non_speaking_duration = 0.8  # How long to wait after speech stops
                
                # Listen with longer timeout and phrase limit
                audio = self.recognizer.listen(
                    source, 
                    timeout=timeout, 
                    phrase_time_limit=phrase_limit
                )
                
                # Try multiple recognition services for better accuracy
                text = None
                
                # Try Google first (usually most accurate)
                try:
                    text = self.recognizer.recognize_google(audio)
                    logger.info(f"User said (Google): {text}")
                except sr.UnknownValueError:
                    logger.info("Google recognition failed, trying alternative...")
                    # You can add other recognition services here if needed
                    pass
                
                if text:
                    return text.strip()
                else:
                    raise sr.UnknownValueError("Could not understand audio")
                
        except sr.WaitTimeoutError:
            logger.warning("Listening timeout - no speech detected")
            if self.in_conversation:
                self.speak("I'm still here. What would you like to know?")
            return None
            
        except sr.UnknownValueError:
            logger.warning("Could not understand audio")
            if self.in_conversation:
                self.speak("Sorry, I didn't catch that. Could you repeat?")
            else:
                self.speak("Sorry, I couldn't understand what you said.")
            return None
            
        except sr.RequestError as e:
            logger.error(f"Speech recognition error: {e}")
            self.speak("Sorry, there was an error with speech recognition.")
            return None
    
    def get_ai_response(self, user_input: str) -> str:
        """
        Get AI response from Groq API with improved RAG accuracy.
        
        Args:
            user_input: User's transcribed speech
            
        Returns:
            AI response text
        """
        try:
            # Check for conversation control commands
            user_input_lower = user_input.lower().strip()
            
            if any(phrase in user_input_lower for phrase in ['goodbye', 'bye', 'stop', 'exit', 'quit', 'see you later', 'talk to you later']):
                self.chat_history = [] # reinitializing
                self.in_conversation = False
                return "Goodbye! Say 'Hey Bruce' if you need me again."
            
            # Check for special commands first (time, date, etc.)
            special_response = self.handle_special_commands(user_input)
            if special_response:
                return special_response
            
            # If no special command, proceed with improved RAG
            query_embedding = openai.Embedding.create(
                input=[user_input_lower],
                model="text-embedding-ada-002"
            )["data"][0]["embedding"]

            query_np = np.array([query_embedding], dtype="float32")
            
            # Get more candidates for better selection
            distances, indices = self.index.search(query_np, k=5)
            
            # Calculate multiple similarity scores
            similarities = []
            for i in range(len(indices[0])):
                if indices[0][i] != -1:  # Valid index
                    chunk_embedding = self.embeddings_np[indices[0][i]].reshape(1, -1)
                    similarity = cosine_similarity(query_np, chunk_embedding)[0][0]
                    similarities.append((similarity, indices[0][i], self.chunks[indices[0][i]]))
            
            # Sort by similarity score (descending)
            similarities.sort(key=lambda x: x[0], reverse=True)
            
            # Get the best similarity score
            best_similarity = similarities[0][0] if similarities else 0.0
            
            # More nuanced threshold logic
            if best_similarity >= 0.85:
                print("🔍 Using RAG context (HIGH confidence, score =", round(best_similarity, 3), ")")
                # Use only the top 2 most relevant chunks
                relevant_chunks = [item[2] for item in similarities[:2]]
                context = "\n\n".join(relevant_chunks)
                
                prompt = f"""You are a helpful AI assistant. Use the context below to answer the question accurately and concisely.

                Context:
                {context}

                Question: {user_input_lower}
                
                Instructions: Answer based strictly on the provided context. If the context doesn't contain enough information to answer the question completely, say so.
                
                Answer:"""
                
            elif best_similarity >= 0.75:
                print("🔍 Using RAG context (MEDIUM confidence, score =", round(best_similarity, 3), ")")
                # Use top chunk but mention uncertainty
                context = similarities[0][2]
                
                prompt = f"""You are a helpful AI assistant. Use the context below to help answer the question, but note that the match may not be perfect.

                Context:
                {context}

                Question: {user_input_lower}
                
                Instructions: Answer based on the context provided, but acknowledge if you're not completely certain about the answer.
                
                Answer:"""
                
            else:
                print("🧠 Using reasoning model without RAG (LOW confidence, score =", round(best_similarity, 3), ")")
                prompt = f"""You are a helpful AI assistant named Bruce. Answer the following question using your general knowledge. Be helpful and conversational.

                Question: {user_input_lower}
                Answer:"""

            messages=[
                    {
                        "role": "system",
                        "content": (
                            "Your name is Bruce. You are a helpful voice assistant that answers "
                            "questions clearly and concisely. Always respond in simple English "
                            "that a 12-year-old can understand. Keep responses brief and friendly. "
                            "You are currently in a conversation, so respond naturally as if continuing a chat."
                            "Don't include emoji in the output."
                            "Don't include symbols in the output."
                            "At the same time be informative with the response."
                        )
                    },
                    {"role": "user", "content": prompt},
                ]
            
                # Append the last 2 user-assistant pairs
            if len(self.chat_history) >= 6:
                messages.extend(self.chat_history[-4:])  # last 2 pairs = 4 messages
            else:
                messages.extend(self.chat_history)

            # Add the latest user message
            messages.append({"role": "user", "content": prompt})

            response = self.groq_client.chat.completions.create(
                messages=messages,
                model="gemma2-9b-it",
                max_tokens=150,  # Limit response length for voice
                temperature=0.7)

            
            reply = response.choices[0].message.content.strip()
            logger.info(f"AI response: {reply}")
            self.chat_history.append({"role": "assistant", "content": reply})
            self.chat_history.append({"role": "user", "content": prompt})
            return reply
            
        except Exception as e:
            logger.error(f"AI response error: {e}")
            return "Sorry, I'm having trouble processing your request right now."
    
    def start_conversation(self):
        """Start and maintain continuous conversation."""
        self.in_conversation = True
        self.last_interaction_time = time.time()
        
        # Initializing RAG operations
        self.persist()

        # Initial greeting
        self.speak("Hi, I'm Bruce. How can I help you today?")
        
        while self.in_conversation:
            # Check for conversation timeout
            if time.time() - self.last_interaction_time > self.conversation_timeout:
                self.speak("I haven't heard from you in a while. Say 'Hey Bruce' if you need me again.")
                self.in_conversation = False
                break
            
            # Listen for user command
            user_command = self.listen_for_command(timeout=8)
            
            if user_command:
                self.last_interaction_time = time.time()
                
                # Get AI response
                ai_response = self.get_ai_response(user_command)
                
                # Speak the response
                self.speak(ai_response)
                
                # If the AI response indicates conversation end, break
                if not self.in_conversation:
                    break
                    
            else:
                # If no command received, continue listening but update timeout check
                continue
        
        logger.info("Conversation ended")
    
    def listen_for_wake_word(self):
        """Listen for wake word detection once, then start conversation."""
        logger.info("Voice Assistant started. Listening for 'Hey Bruce'...")
        
        try:
            with sd.RawInputStream(
                samplerate=self.porcupine.sample_rate,
                blocksize=self.porcupine.frame_length,
                dtype='int16',
                channels=1
            ) as stream:
                
                while True:
                    # If we're not in conversation, listen for wake word
                    if not self.in_conversation:
                        # Read audio data
                        pcm_data = stream.read(self.porcupine.frame_length)[0]
                        pcm = struct.unpack_from("h" * self.porcupine.frame_length, pcm_data)
                        
                        # Process for wake word
                        result = self.porcupine.process(pcm)
                        
                        if result >= 0:
                            logger.info("Wake word 'Hey Bruce' detected!")
                            # Start continuous conversation
                            self.start_conversation()
                    else:
                        # Small delay to prevent excessive CPU usage during conversation
                        time.sleep(0.1)
                        
        except KeyboardInterrupt:
            logger.info("Voice assistant stopped by user")
        except Exception as e:
            logger.error(f"Wake word detection error: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        try:
            if hasattr(self, 'porcupine') and self.porcupine:
                self.porcupine.delete()
            logger.info("Cleanup completed")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
    
    def run(self):
        """Start the voice assistant."""
        try:
            self.listen_for_wake_word()
        except Exception as e:
            logger.error(f"Fatal error: {e}")
        finally:
            self.cleanup()


def main():
    """Main entry point."""
    assistant = VoiceAssistant()
    assistant.run()


if __name__ == "__main__":
    main()