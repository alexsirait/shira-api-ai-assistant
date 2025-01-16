import tkinter as tk
from tkinter import messagebox
from threading import Thread
import speech_recognition as sr
import pyttsx3
import requests
import pygame
import time

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty("rate", 150)  # Set speaking rate
engine.setProperty("voice", "com.apple.speech.synthesis.voice.samantha")  # Choose voice (change as needed)

# Initialize Pygame for playing sound
pygame.mixer.init()

class VoiceAssistantApp:
    def __init__(self, root):
        self.root = root
        self.root.title("TechFusion AI Assistant")
        self.root.geometry("600x400")
        
        # Initialize state variables
        self.transcript = ""
        self.response = ""
        self.is_listening = False
        self.is_processing = False
        self.is_speaking = False
        self.audio = None

        # GUI components
        self.input_text = tk.StringVar()
        self.response_text = tk.StringVar()
        
        # Create GUI layout
        self.create_widgets()

        # Speech recognizer
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Load beep sound
        self.beep_sound = "beep-22.mp3"

    def create_widgets(self):
        # Title
        tk.Label(self.root, text="TechFusion AI Assistant", font=("Arial", 16, "bold")).pack(pady=10)
        
        # Transcript area
        self.transcript_label = tk.Label(self.root, text="Speech Transcript:", font=("Arial", 12))
        self.transcript_label.pack()
        
        self.transcript_display = tk.Label(self.root, text="", wraplength=500, font=("Arial", 10), bg="white", relief="sunken")
        self.transcript_display.pack(pady=5, padx=20, fill=tk.BOTH, expand=True)

        # Response area
        self.response_label = tk.Label(self.root, text="AI Response:", font=("Arial", 12))
        self.response_label.pack()
        
        self.response_display = tk.Label(self.root, text="", wraplength=500, font=("Arial", 10), bg="white", relief="sunken")
        self.response_display.pack(pady=5, padx=20, fill=tk.BOTH, expand=True)

        # Input area
        self.input_entry = tk.Entry(self.root, textvariable=self.input_text, font=("Arial", 12), width=50)
        self.input_entry.pack(pady=10)
        self.input_entry.bind("<Return>", self.handle_keypress)

        # Control buttons
        self.listen_button = tk.Button(self.root, text="Start Listening", command=self.toggle_listening, font=("Arial", 12))
        self.listen_button.pack(side=tk.LEFT, padx=10)
        
        self.ask_button = tk.Button(self.root, text="Ask AI", command=self.handle_voice_interaction, font=("Arial", 12))
        self.ask_button.pack(side=tk.LEFT, padx=10)

    def play_beep_sound(self):
        pygame.mixer.music.load(self.beep_sound)
        pygame.mixer.music.play()

    def toggle_listening(self):
        if self.is_listening:
            self.is_listening = False
            self.listen_button.config(text="Start Listening")
        else:
            self.is_listening = True
            self.listen_button.config(text="Stop Listening")
            Thread(target=self.start_listening).start()

    def start_listening(self):
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
            try:
                self.play_beep_sound()
                audio = self.recognizer.listen(source, timeout=5)
                self.transcript = self.recognizer.recognize_google(audio, language="id-ID")
                self.transcript_display.config(text=self.transcript)
            except sr.WaitTimeoutError:
                messagebox.showerror("Timeout", "No speech detected.")
            except sr.UnknownValueError:
                messagebox.showerror("Error", "Could not understand the audio.")

    def handle_keypress(self, event):
        self.handle_voice_interaction()

    def handle_voice_interaction(self):
        if not self.transcript:
            messagebox.showerror("Error", "No transcript available. Please speak or type a question.")
            return
        
        self.is_processing = True
        self.ask_button.config(state=tk.DISABLED)
        Thread(target=self.ask_api, args=(self.transcript,)).start()

    def ask_api(self, prompt):
        try:
            self.play_beep_sound()
            response = requests.post("http://127.0.0.1:8000/api/assistant/prompt_view", json={"prompt": prompt, "user_id": "alex"})
            response.raise_for_status()
            data = response.json()
            self.response = data["response"]
            self.response_text.set(self.response)
            self.response_display.config(text=self.response)
            self.speak_response(self.response)
        except Exception as e:
            messagebox.showerror("Error", f"API error: {e}")
        finally:
            self.is_processing = False
            self.ask_button.config(state=tk.NORMAL)

    def speak_response(self, text):
        self.is_speaking = True
        engine.say(text)
        engine.runAndWait()
        self.is_speaking = False

if __name__ == "__main__":
    root = tk.Tk()
    app = VoiceAssistantApp(root)
    root.mainloop()
