import speech_recognition as sr
import playsound
import gTTS

class OnlineSST:
    def __init__(self):
        self.recognizer = sr.Recognizer()

    def speak(self, text):
        # tts = gTTS(text=text, lang='ta')
        # filename = 'voice.mp3'
        # tts.save(filename)
        # playsound.playsound(filename)
        print(text)

    def get_audio(self):
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source)
            audio = self.recognizer.listen(source)
            said = ""

            try:
                said = self.recognizer.recognize_google(audio,language='ta-IN')
                print(said)
            except sr.UnknownValueError:
                print("Could not understand audio")
            except sr.RequestError as e:
                print("Could not request results; {0}".format(e))

        return said.lower()

if __name__ == "__main__":
    voice_assistant = OnlineSST()
    voice_assistant.get_audio()


# class Speech:
#     def getText(self):
#             text = self.said
#             self.said = ''
#             return text

#     def whisper_get_audio(self):
#         parser = argparse.ArgumentParser()
#         parser.add_argument("--model", default="tiny", help="Model to use",
#                             choices=["tiny", "base", "small", "medium", "large"])
#         parser.add_argument("--non_english", action='store_true',
#                             help="Don't use the english model.")
#         parser.add_argument("--energy_threshold", default=10000,
#                             help="Energy level for mic to detect.", type=int)
#         parser.add_argument("--record_timeout", default=0,
#                             help="How real time the recording is in seconds.", type=float)
#         parser.add_argument("--phrase_timeout", default=0,
#                             help="How much empty space between recordings before we "
#                                 "consider it a new line in the transcription.", type=float)
        
#         args = parser.parse_args()


#         model = args.model
#         if args.model != "small" and not args.non_english:
#             model = model + ".en"
#             print(model)
#         audio_model = whisper.load_model(model)

#         record_timeout = args.record_timeout
#         phrase_timeout = args.phrase_timeout

#         transcription = ['']

#         with self.source:
#             self.recorder.adjust_for_ambient_noise(self.source)

#         def record_callback(_, audio:sr.AudioData) -> None:
#             data = audio.get_raw_data()
#             self.data_queue.put(data)

        
#         with self.source as source: 
#             audio_data = self.recorder.listen(source).get_raw_data()
#             audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
#             result = audio_model.transcribe(audio_np, fp16=torch.cuda.is_available())
#             text = result['text'].strip()
#             return text
#         return ''


#     def speak(self, text):
#         """Speak the given text using the TTS engine"""
#         self.tts_engine.say(text)
#         self.tts_engine.runAndWait()

# import argparse
# import numpy as np
# import torch
# import whisper
# import speech_recognition as sr
# from queue import Queue

# class Speech:
#     def __init__(self):
#         self.said = ''
#         self.recorder = sr.Recognizer()
#         self.source = sr.Microphone()
#         self.data_queue = Queue()

#     def getText(self):
#         text = self.said
#         self.said = ''
#         return text

#     def whisper_get_audio(self):
#         parser = argparse.ArgumentParser()
#         parser.add_argument("--model", default="base", help="Model to use",
#                             choices=["tiny", "small", "base", "medium", "large"])
#         parser.add_argument("--non_english", action='store_true',
#                             help="Don't use the English model.")
#         parser.add_argument("--energy_threshold", default=10000,
#                             help="Energy level for mic to detect.", type=int)
#         parser.add_argument("--record_timeout", default=0,
#                             help="How real-time the recording is in seconds.", type=float)
#         parser.add_argument("--phrase_timeout", default=0,
#                             help="How much empty space between recordings before we "
#                                  "consider it a new line in the transcription.", type=float)

#         args = parser.parse_args()

#         model_name = args.model
#         if model_name != "small" and not args.non_english:
#             model_name += ".en"
#         audio_model = whisper.load_model(model_name)

#         with self.source:
#             self.recorder.adjust_for_ambient_noise(self.source)

#         print("Listening...")  
#         with self.source as source:
#             audio_data = self.recorder.listen(source).get_raw_data()
#             audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
#             result = audio_model.transcribe(audio_np, fp16=torch.cuda.is_available())
#             text = result['text'].strip()
    
#             return text

#         return ''  

# if __name__ == "__main__":
#     speech = Speech()
#     spoken_text = speech.whisper_get_audio()
#     print("You said:", spoken_text)
    

    
