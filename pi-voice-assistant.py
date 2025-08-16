import sys
import time
import torch
import numpy as np
import pyaudio
import logging
import threading
import queue
import ollama
import soundfile as sf
import tempfile
import os
import re
from kittentts import KittenTTS
import whisper

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Audio configuration
INPUT_FORMAT = pyaudio.paInt16
INPUT_CHANNELS = 1
INPUT_RATE = 16000
INPUT_CHUNK = 1024

class Assistant:
    def __init__(self):
        logging.info("Initializing OnDevice Pi Voice Assistant...")
        
        # Configuration - hardcoded for simplicity
        self.ollama_host = "http://localhost:11434"
        self.ollama_model = "gemma3:270m-it-fp16"  # Lightweight model perfect for Pi 4
        self.whisper_model = "tiny"   # Using tiny model for speed
        
        # TTS Configuration
        self.tts_voice = "expr-voice-2-f"  # Available: expr-voice-2-m, expr-voice-2-f, etc.
        self.tts_speed = 1.0
        
        # Initialize audio
        self.audio = pyaudio.PyAudio()
        self._test_audio()
        
        # Load Whisper model
        print(f"Loading Whisper {self.whisper_model} model...")
        try:
            self.model = whisper.load_model(self.whisper_model)
            print("✓ Whisper model loaded successfully!")
        except Exception as e:
            print(f"Error loading Whisper model: {e}")
            sys.exit(1)
        
        # Initialize KittenTTS
        print("Loading KittenTTS model (first run downloads ~25MB)...")
        self.tts = KittenTTS("KittenML/kitten-tts-nano-0.1")
        print("KittenTTS loaded successfully!")
        
        # Initialize Ollama client
        self.ollama_client = ollama.Client(host=self.ollama_host)
        
        # Conversation history
        self.conversation_history = []
        self.max_history_length = 10
        
        self.is_speaking = False
        
        print("OnDevice Pi Voice Assistant initialized successfully!")
        print("Commands:")
        print("  Press ENTER to start recording (speak for at least 1 second)")
        print("  Type 'quit' or 'exit' to quit")
        print("  Type 'clear' to clear conversation history")
        print("  Type 'voice' to change TTS voice")

    def _test_audio(self):
        """Test if audio input is available"""
        try:
            stream = self.audio.open(format=INPUT_FORMAT,
                                   channels=INPUT_CHANNELS,
                                   rate=INPUT_RATE,
                                   input=True,
                                   frames_per_buffer=INPUT_CHUNK)
            stream.close()
        except Exception as e:
            logging.error(f"Audio initialization failed: {str(e)}")
            print("Error: Could not initialize audio input. Please check your microphone.")
            sys.exit(1)

    def record_audio(self):
        """Record audio until Enter is pressed again"""
        print("\n🎤 Recording... Press ENTER to stop (speak for at least 1 second)")
        
        stream = self.audio.open(format=INPUT_FORMAT,
                               channels=INPUT_CHANNELS,
                               rate=INPUT_RATE,
                               input=True,
                               frames_per_buffer=INPUT_CHUNK)
        
        frames = []
        recording = True
        start_time = time.time()
        
        def stop_recording():
            nonlocal recording
            input()  # Wait for Enter key
            recording = False
        
        # Start thread to listen for Enter key
        stop_thread = threading.Thread(target=stop_recording)
        stop_thread.daemon = True
        stop_thread.start()
        
        # Record audio
        while recording:
            try:
                data = stream.read(INPUT_CHUNK, exception_on_overflow=False)
                frames.append(data)
                
                # Show recording progress every second
                current_time = time.time()
                elapsed = current_time - start_time
                if int(elapsed) > int(elapsed - 0.1):  # Print roughly every second
                    print(f"Recording: {elapsed:.1f}s", end='\r')
                    
            except Exception as e:
                logging.error(f"Recording error: {e}")
                break
                
        stream.stop_stream()
        stream.close()
        
        elapsed_time = time.time() - start_time
        print(f"\n🛑 Recording stopped ({elapsed_time:.1f}s recorded)")
        
        if not frames:
            return None
        
        # Convert to numpy array for Whisper
        audio_data = np.frombuffer(b''.join(frames), np.int16).astype(np.float32) / 32768.0
        
        # Warn if recording is very short
        if elapsed_time < 0.5:
            print("⚠️  Very short recording. Try speaking for longer next time.")
        
        return audio_data

    def speech_to_text(self, audio_data):
        """Convert audio to text using Whisper"""
        if audio_data is None or len(audio_data) == 0:
            return ""
            
        print("🗣️ Converting speech to text with Whisper...")
        
        try:
            # Use Whisper to transcribe
            result = self.model.transcribe(
                audio_data,
                language="en",  # Set to English for better performance
                fp16=torch.cuda.is_available()  # Use fp16 if CUDA available
            )
            
            text = result["text"].strip()
            
            if text:
                print(f"You said: {text}")
                return text
            else:
                print("No speech detected.")
                return ""
                
        except Exception as e:
            logging.error(f"Speech recognition error: {str(e)}")
            print(f"Error: Could not process speech - {str(e)}")
            return ""

    def ask_ollama(self, prompt):
        """Send prompt to Ollama and get response"""
        if not prompt:
            return ""
            
        print("🤖 Thinking...")
        
        # Add system prompt for voice assistant behavior
        system_prompt = {
            "role": "system", 
            "content": "You are Pi Assistant, a helpful voice assistant running on a Raspberry Pi. Keep responses brief, conversational, and under 100 words. When users introduce themselves, remember their name and use it in conversation. You are Pi Assistant, not the user. Speak naturally as if having a casual conversation."
        }
        
        # Add to conversation history
        self.conversation_history.append({"role": "user", "content": prompt})
        
        # Keep only recent history and include system prompt
        messages = [system_prompt] + self.conversation_history[-self.max_history_length:]
        
        try:
            response = self.ollama_client.chat(
                model=self.ollama_model,
                messages=messages
            )
            
            assistant_response = response['message']['content']
            
            # Add response to history
            self.conversation_history.append({"role": "assistant", "content": assistant_response})
            
            # Trim history if too long
            if len(self.conversation_history) > self.max_history_length * 2:
                self.conversation_history = self.conversation_history[-self.max_history_length * 2:]
            
            print(f"\nPi Assistant: {assistant_response}\n")
            return assistant_response
            
        except Exception as e:
            logging.error(f"Ollama error: {str(e)}")
            error_msg = "Sorry, I encountered an error. Please try again."
            print(f"\nPi Assistant: {error_msg}\n")
            return error_msg

    def _clean_text_for_tts(self, text):
        """Clean text for TTS by removing emojis, asterisks, and special characters"""
        if not text:
            return ""
        
        # Remove text between asterisks (like *smiles* or *laughs*)
        text = re.sub(r'\*[^*]*\*', '', text)
        
        # Remove standalone asterisks
        text = re.sub(r'\*', '', text)
        
        # Simple emoji removal - remove common emoji ranges
        text = re.sub(r'[\U0001F600-\U0001F64F]', '', text)  # emoticons
        text = re.sub(r'[\U0001F300-\U0001F5FF]', '', text)  # symbols & pictographs
        text = re.sub(r'[\U0001F680-\U0001F6FF]', '', text)  # transport & map
        text = re.sub(r'[\U0001F1E0-\U0001F1FF]', '', text)  # flags
        
        # Clean up multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def text_to_speech(self, text):
        """Convert text to speech using KittenTTS with streaming playback"""
        if not text or self.is_speaking:
            return
            
        def speak():
            try:
                self.is_speaking = True
                print("🔊 Generating speech...")
                
                # Clean text for TTS (remove emojis, asterisks, etc.)
                cleaned_text = self._clean_text_for_tts(text)
                
                if not cleaned_text.strip():
                    print("⚠️  No speakable text after cleaning")
                    return
                
                # Truncate text if too long for KittenTTS
                if len(cleaned_text) > 200:
                    text_to_speak = cleaned_text[:200].rsplit('.', 1)[0]
                    if text_to_speak:
                        text_to_speak += "."
                    else:
                        text_to_speak = cleaned_text[:200]
                    print(f"📝 Text truncated: {len(cleaned_text)} → {len(text_to_speak)} chars")
                else:
                    text_to_speak = cleaned_text
                
                # Generate audio with KittenTTS (in memory)
                audio_data = self.tts.generate(text_to_speak, voice=self.tts_voice)
                
                print("🔊 Streaming speech...")
                
                # Stream audio directly to speakers (NO FILE SAVING)
                self._stream_audio_to_speakers(audio_data, 24000)
                
            except Exception as e:
                logging.error(f"Text-to-speech error: {str(e)}")
                print(f"Warning: Could not generate or play speech: {e}")
                print("💡 Trying to continue without TTS...")
            finally:
                self.is_speaking = False
        
        # Run speech in background
        speech_thread = threading.Thread(target=speak)
        speech_thread.daemon = True
        speech_thread.start()

    def _stream_audio_to_speakers(self, audio_data, sample_rate):
        """Stream audio data directly to speakers without saving to file"""
        try:
            # Convert to 16-bit integers for playback
            if audio_data.dtype != np.int16:
                audio_data = (audio_data * 32767).astype(np.int16)
            
            # Create pyaudio stream for output
            stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,  # KittenTTS is mono
                rate=int(sample_rate),
                output=True,
                frames_per_buffer=1024
            )
            
            # Stream audio data in chunks (NO FILE I/O)
            chunk_size = 1024
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i + chunk_size]
                stream.write(chunk.tobytes())
            
            # Clean up stream
            stream.stop_stream()
            stream.close()
            
        except Exception as e:
            logging.error(f"Audio streaming error: {str(e)}")
            print("Warning: Could not stream audio")

    def _play_audio_file(self, file_path):
        """Legacy method - kept for compatibility but not used in streaming mode"""
        print("⚠️  Using legacy file playback - consider using streaming instead")
        try:
            # Read audio file
            data, sample_rate = sf.read(file_path)
            
            # Convert to 16-bit integers
            if data.dtype != np.int16:
                data = (data * 32767).astype(np.int16)
            
            # Create pyaudio stream
            stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1 if len(data.shape) == 1 else data.shape[1],
                rate=int(sample_rate),
                output=True
            )
            
            # Play audio
            stream.write(data.tobytes())
            
            # Clean up
            stream.stop_stream()
            stream.close()
            
        except Exception as e:
            logging.error(f"Audio playback error: {str(e)}")
            print("Warning: Could not play audio")

    def change_voice(self):
        """Change TTS voice"""
        voices = [
            'expr-voice-2-m', 'expr-voice-2-f', 'expr-voice-3-m', 'expr-voice-3-f',
            'expr-voice-4-m', 'expr-voice-4-f', 'expr-voice-5-m', 'expr-voice-5-f'
        ]
        
        print("\nAvailable voices:")
        for i, voice in enumerate(voices, 1):
            mark = " (current)" if voice == self.tts_voice else ""
            print(f"  {i}. {voice}{mark}")
        
        try:
            choice = input("Enter voice number (1-8): ").strip()
            voice_index = int(choice) - 1
            
            if 0 <= voice_index < len(voices):
                self.tts_voice = voices[voice_index]
                print(f"Voice changed to: {self.tts_voice}")
                
                # Test the voice
                self.text_to_speech("Hello! This is my new voice.")
            else:
                print("Invalid choice. Voice unchanged.")
                
        except ValueError:
            print("Invalid input. Voice unchanged.")

    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        print("Conversation history cleared.")

    def shutdown(self):
        """Clean shutdown"""
        print("\nShutting down OnDevice Pi Voice Assistant...")
        self.audio.terminate()
        sys.exit(0)

    def run(self):
        """Main application loop"""
        print("\n" + "="*50)
        print("Welcome to OnDevice Pi Voice Assistant!")
        print("="*50)
        
        try:
            while True:
                command = input("\nPress ENTER to record, or type a command: ").strip().lower()
                
                if command in ['quit', 'exit', 'q']:
                    self.shutdown()
                elif command == 'clear':
                    self.clear_history()
                elif command == 'voice':
                    self.change_voice()
                elif command == '' or command == 'record':
                    # Record and process audio
                    audio_data = self.record_audio()
                    if audio_data is not None:
                        text = self.speech_to_text(audio_data)
                        if text:
                            response = self.ask_ollama(text)
                            if response:
                                self.text_to_speech(response)
                else:
                    print("Unknown command. Available: record (ENTER), clear, voice, quit")
                    
        except KeyboardInterrupt:
            self.shutdown()
        except Exception as e:
            logging.error(f"Unexpected error: {str(e)}")
            self.shutdown()

def main():
    assistant = Assistant()
    assistant.run()

if __name__ == "__main__":
    main()