
# import os
# from yt_dlp import YoutubeDL
# import whisper
# import torch
# import subprocess

# # Ensure that the Whisper model runs on GPU if available
# device = "cuda" if torch.cuda.is_available() else "cpu"

# model = whisper.load_model("base").to(device)  # Use the base model; you can change this to "small", "medium", or "large"

# def download_audio_from_youtube(url, output_path='.'):
#     if not os.path.exists(output_path):
#         os.makedirs(output_path)
    
#     # Simplified filename to avoid special character issues
#     ydl_opts = {
#         'format': 'bestaudio/best',
#         'outtmpl': f'{output_path}/sample.%(ext)s',  # Set a simple name for the downloaded audio
#         'postprocessors': [{
#             'key': 'FFmpegExtractAudio',
#             'preferredcodec': 'wav',  
#             'preferredquality': '192',
#         }],
#         'ffmpeg_location': '/opt/homebrew/bin/ffmpeg'  # Adjust if necessary
#     }

#     with YoutubeDL(ydl_opts) as ydl:
#         ydl.download([url])
    
#     return os.path.join(output_path, "sample.wav")

# def convert_audio_for_transcription(input_filename):
#     output_filename = os.path.join(os.path.dirname(input_filename), "temp_converted.wav")
#     try:
#         # Convert audio to 16 kHz, mono format required for Whisper
#         subprocess.run([
#             'ffmpeg', '-i', input_filename, '-ar', '16000', '-ac', '1', output_filename
#         ], check=True)
#         return output_filename
#     except subprocess.CalledProcessError as e:
#         print(f"Error converting audio: {e}")
#         return None

# def transcribe_audio_file(audio_filename):
#     temp_filename = convert_audio_for_transcription(audio_filename)
#     if temp_filename:
#         try:
#             result = model.transcribe(temp_filename, fp16=torch.cuda.is_available())
#             os.remove(temp_filename)
#             return result['text']
#         except Exception as e:
#             print(f"Error transcribing audio file {audio_filename}: {e}")
#             os.remove(temp_filename)
#             return "[Error processing the audio file]"
#     else:
#         return "[Conversion failed, no transcription performed]"

# if __name__ == "__main__":
#     youtube_url = input("Enter the YouTube video URL: ")
#     audio_file = download_audio_from_youtube(youtube_url)
#     print(f"Audio downloaded and saved to: {audio_file}")
    
#     transcript = transcribe_audio_file(audio_file)
#     print("Final Transcript:")
#     print(transcript)


import os
import subprocess
import torch
from yt_dlp import YoutubeDL
import whisper
import requests
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from googletrans import Translator
from ttsmms import TTS
from scipy.io import wavfile
import numpy as np
from gtts import gTTS

# Ensure that the Whisper model runs on GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model("base").to(device)  # Use the base model; you can change this to "small", "medium", or "large"

# Function to download audio from YouTube
def download_audio_from_youtube(url, output_path='.'):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': f'{output_path}/sample.%(ext)s',  # Set a simple name for the downloaded audio
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',  
            'preferredquality': '192',
        }],
        'ffmpeg_location': '/opt/homebrew/bin/ffmpeg'  # Adjust if necessary
    }

    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    
    return os.path.join(output_path, "sample.wav")

# Function to convert the downloaded audio for transcription
def convert_audio_for_transcription(input_filename):
    output_filename = os.path.join(os.path.dirname(input_filename), "temp_converted.wav")
    try:
        subprocess.run([
            'ffmpeg', '-i', input_filename, '-ar', '16000', '-ac', '1', output_filename
        ], check=True)
        return output_filename
    except subprocess.CalledProcessError as e:
        print(f"Error converting audio: {e}")
        return None

# Function to transcribe the audio file
def transcribe_audio_file(audio_filename):
    temp_filename = convert_audio_for_transcription(audio_filename)
    if temp_filename:
        try:
            result = model.transcribe(temp_filename, fp16=torch.cuda.is_available())
            os.remove(temp_filename)
            return result['text']
        except Exception as e:
            print(f"Error transcribing audio file {audio_filename}: {e}")
            os.remove(temp_filename)
            return "[Error processing the audio file]"
    else:
        return "[Conversion failed, no transcription performed]"

# Translator class to handle translation
class TranslatorApp:
    def __init__(self, model_path="facebook/nllb-200-distilled-600M", tokenizer_path="facebook/nllb-200-distilled-600M"):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.translator_offline = self.initialize_offline_translator()
        self.src_lang = "en"
        self.tgt_lang = "ta"

    def initialize_offline_translator(self):
        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path)
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        return pipeline('translation', model=model, tokenizer=tokenizer, src_lang="eng_Latn", tgt_lang="tam_Taml", max_length=400)

    @staticmethod
    def is_connected():
        try:
            requests.get('https://www.google.com', timeout=5)
            return True
        except requests.ConnectionError:
            return False

    @staticmethod
    def translate_online(text, src_lang, dest_lang):
        translator = Translator()
        result = translator.translate(text, src=src_lang, dest=dest_lang)
        return result.text

    def translate_offline(self, text):
        return self.translator_offline(text)[0]['translation_text']

    def run(self, content):
        if self.is_connected():
            translation = self.translate_online(content, self.src_lang, self.tgt_lang)
        else:
            translation = self.translate_offline(content)
        return translation

# Text-to-Speech class to handle speech synthesis
class TextToSpeech:
    def _init_(self, model_path="data/tam"):
        self.model_path = model_path
        self.tts_offline = self.initialize_offline_tts()

    def initialize_offline_tts(self):
        return TTS(self.model_path)

    @staticmethod
    def is_connected():
        try:
            requests.get('https://www.google.com', timeout=5)
            return True
        except requests.ConnectionError:
            return False

    def synthesize_speech_offline(self, text):
        wav = self.tts_offline.synthesis(text)
        return wav

    def save_audio_offline(self, wav, output_path):
        wavfile.write(output_path, wav["sampling_rate"], np.array(wav["x"]))
        print(f"Audio saved to {output_path}")

    def synthesize_speech_online(self, text, lang='ta'):
        tts = gTTS(text=text, lang=lang, slow=False)
        output_path = "on_speech.mp3"
        tts.save(output_path)
        print(f"Audio saved to {output_path}")
        return output_path

    def run(self, content):
        if self.is_connected():
            self.synthesize_speech_online(content)
        else:
            wav = self.synthesize_speech_offline(content)
            self.save_audio_offline(wav, "off_speech.wav")

# Main function to orchestrate the workflow
if __name__ == "__main__":
    youtube_url = input("Enter the YouTube video URL: ")
    
    # Step 1: Download the audio from YouTube
    audio_file = download_audio_from_youtube(youtube_url)
    print(f"Audio downloaded and saved to: {audio_file}")
    
    # Step 2: Transcribe the audio
    transcript = transcribe_audio_file(audio_file)
    print("Transcript:")
    print(transcript)
    
    # Step 3: Translate the transcript
    translation_app = TranslatorApp()
    translated_text = translation_app.run(transcript)
    print("Translated Text:")
    print(translated_text)
    
    # Step 4: Convert the translated text to speech
    tts_app = TextToSpeech()
    tts_app.run(translated_text)