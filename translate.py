import requests
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from googletrans import Translator

class TranslatorApp:
    def __init__(self, model_path="facebook/nllb-200-distilled-600M", tokenizer_path="facebook/nllb-200-distilled-600M"):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.translator_offline = self.initialize_offline_translator()
        self.src_lang = None
        self.tgt_lang = None

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

    def run(self,content):
        self.src_lang = "en"
        self.tgt_lang = "ta"

    
        # text = input("Enter text to translate (or type 'exit' to quit): ").strip()
        if content.lower() == 'exit':
            exit()
        if self.is_connected():
            translation = self.translate_online(content, self.src_lang, self.tgt_lang)
        else:
            translation = self.translate_offline(content)
        print(translation)
        return translation

if __name__ == "__main__":
    app = TranslatorApp()
    content =  input(": ")
    app.run(content)
