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
from nltk.stem import WordNetLemmatizer

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

                # 1. Extract speaker utterances directly from raw text
                speaker_pairs = self.extract_speaker_utterances(raw_text)

                with open("structured_speaker_output.txt", "a", encoding="utf-8") as f:
                    for speaker, utterance in speaker_pairs:
                        f.write(f"SPEAKER: {speaker}\n")
                        f.write(f"UTTERANCE: {utterance.strip()}\n\n")

                # 2. Only for model training: process and lemmatize
                words = raw_text.lower().split()
                results[image] = self.lemmatize(words)


        print("Finished data preprocessing for all given documents...")

        with open("preprocessed_output.txt", "w", encoding="utf-8") as f:
            for filename, words in results.items():
                f.write(f"File: {filename}\n")
                f.write(" ".join(words))
                f.write("\n\n")


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
                full_text = ""
                for page_num, image in enumerate(images):
                    text = pytesseract.image_to_string(image)
                    full_text += f"\n\n--- Page {page_num + 1} ---\n\n{text}"
                return full_text
            except Exception as e:
                print(f"Error converting PDF to image: {e}")
                return ""
    
    def regex(self, text):
        pattern = r"READINGS:\s*(.*?)\s*END\."
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip() 
        return None

    def lemmatize(self, words):
        lemmatizer = WordNetLemmatizer()
        return [lemmatizer.lemmatize(word) for word in words]
    
    def extract_speaker_utterances(self, text):
        # Regex pattern to capture speaker lines (e.g., 'sen. sifuna:', 'the speaker (hon. kingi):', etc.)
        pattern = re.compile(r"\n(?:the\s+)?(?:temporary\s+)?(?:chairperson|speaker)?\s*\(?sen(?:ator)?\.?\s*([a-zA-Z\s.'()-]+)\)?:", re.IGNORECASE)

        matches = list(pattern.finditer(text))
        speaker_pairs = []

        for i, match in enumerate(matches):
            speaker = match.group(1).strip()
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            utterance = text[start:end].strip()
            if len(utterance) > 0:
                speaker_pairs.append((speaker, utterance))

        return speaker_pairs




processor = text_generation("/Users/danhuff/Desktop/KenyaSample.pdf")
