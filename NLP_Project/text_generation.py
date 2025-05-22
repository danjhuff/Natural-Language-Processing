import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from googletrans import Translator
import pytesseract
from langdetect import detect
from PIL import Image
from nltk.stem import PorterStemmer
from pdf2image import convert_from_path
import re

class text_generation:
    def __init__(self, *args):
        self.translator = Translator()
        print("Translator initialized...\nStarting data preprocessing...")
        results = {}
        for image in args:
            raw_text = self.ocr(image)
            if raw_text:  
                if detect(raw_text) != "en": 
                    raw_text = self.translateSwahili(raw_text)
                raw_text = self.remove_stop_words(raw_text)
                words = raw_text.lower().split()
                results[image] = self.stem(words)
        print("Finished data preprocessing for all given documents...")
        print(results)

    def remove_stop_words(self, text):
        stop_words = set(stopwords.words('english'))
        words = word_tokenize(text)
        filtered_words = [word for word in words if word.lower() not in stop_words]
        return ' '.join(filtered_words)

    def stem(self, text):
        stemmer = PorterStemmer()
        stemmed_words = [stemmer.stem(word) for word in text]
        return stemmed_words

    def translateSwahili(self, text):
        try:
            translation = self.translator.translate(text, dest='en')
            return translation.text  
        except Exception as e:
            print(f"Error translating text: {e}")
            return text 

    def ocr(self, i):
        try:
            return pytesseract.image_to_string(Image.open(i))
        except Exception as e:
            print(f"Error occurred: {e}")
            try:
                images = convert_from_path(i)
                images[0].save('temp.png', 'PNG')
                raw_text = pytesseract.image_to_string(Image.open('temp.png'))
                return raw_text
            except Exception as e:
                print(f"Error converting PDF to image: {e}")
                return ""  
    
    def regex(self, text):
        pattern = r"READINGS:\s*(.*?)\s*END\."
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip() 
        return None

