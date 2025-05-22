import os
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
import fitz
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import re
from hdbscan import HDBSCAN
from umap import UMAP
hdbscan_model = HDBSCAN(min_cluster_size=5, prediction_data=True)
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
nltk.download('stopwords')
nltk.download('punkt')
from openai import OpenAI
nltk.download('words')
from nltk.corpus import words as nltk_words
english_vocab = set(w.lower() for w in nltk_words.words())
from collections import Counter, defaultdict
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))
custom_stopwords = {
    "senate", "committee", "house", "representative", "congress", "bill", "january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "year", "years", "day", "days", 
    "madam", 'auditor', 'billion', 'chairman', 'city', 'clause', 'cotton', 'division', 'drought', 'election', 'ended', 'family', 'food', 'maize', 'mental','party', 'peace', 'petition', 'police', 'quorum', 'river', 'road', 'sang', 'turkana', 'veronica', 'welcome', 'word', 'youth'
    'aker', 'Kingi', 'Cheruiyot', 'Orwoba', 'Kinyua', 'Faki', 'Tobiko', 'Kingi ', 'Miraj', 'Wafula', 'Osotsi', 'Madzayo', 'Chute', 'Kisang‚Äô', 'Wakili Sigei', 'Okenyuri', 'Veronica Maina', 'Dr Khalwale', 'Kibwana', 'Kavindu Muthama', 'Ogola', 'Methu', 'Mwaruma', 'Dr Murango', 'M Kajwang‚Äô', ' Veronica Maina', 'Wamatinga', 'Joe Nyutu', 'Murgor', 'Wakili Sige i', 'Veron ica Maina', 'Tabitha Mutinda', 'Wambua', '', 'M Kajwang ‚Äô', 'Veronica Maina ', 'Dr Oburu', 'Maanzo', 'Dr Khalwa le', 'Olekina', 'Sifuna', 'Mungatana MGH', 'Chimera', 'Oyomo', 'Abass', 'Cherarkey', 'A bdul Haji', 'Ch eruiyot', 'Cherarke y', 'Crystal Asi ge', 'Crystal Asige', 'Oke tch Gicheru', 'Ali Roba', 'Veronica Mai na', 'Munyi Mundigi', 'Muyeka', 'Mumma', 'Mun gatana MGH', 'Gataya', 'Oketch Gicheru', 'Tabitha Mu tinda', 'Prof Tom Ojienda SC', 'Senator', 'Thang‚Äôwa', 'Kavindu Mut hama', 'Kathuri ', 'Kathuri', 'Kat huri', 'Lomenen', 'Chirchir', 'Kisang', 'Ch irchir', 'Dullo', 'Mbugua', 'Mandago', 'Gataya Mo Fire', 'Ka thuri', 'Korir', 'Abdul Haji', 'Omogeni', 'K athuri ', 'Murkomen', 'Mungatana  MGH', 'Onyonka', ' Kinyua', ' Kathuri', 'Tabitha Keroche', 'Cheptumo', 'Senators', 'Kathur i', 'Mungatan a MGH', 'Githuku', 'Dr Lelegwe  Ltumbesi', 'Chesang', 'Wakili  Sigei', 'Ki ngi', 'Seki', 'Ogolla', 'King i', 'Dr Lelegwe Ltumbesi', 'Mundigi', 'Thangw‚Äôa', 'Tom Ojienda SC', 'Abbas', 'Lemaltian', 'Ali Rob a', 'Mumm a', 'W akili Sigei', 'Mutinda', 'Beth Syengo', 'Wam bua', 'Dr Khalw ale', 'Dr Kh alwale', 'Ver onica Maina', 'Cheruiyo t', 'Okiya Omtatah', 'Cherargei', 'Prof Kamar', 'Veronicah  Maina', 'Kib wana', 'Kibw ana', ' Kingi', 'Okiyah Omtatah', 'Veronicah Main a', 'Veronicah Maina', 'Mu ngatana MGH', 'K ingi', 'Dr  Khalwale', 'Shakila Abdalla', 'Sifun a', 'M Kajwang', 'Speaker Hon Kingi', 'Mar iam Omar', 'Mariam Omar', 'Cherar key', 'Sifu na', 'Kin gi', 'Montet Betty', 'Oketch Giche ru', 'Mariam  Omar', 'Dr Kha lwale', 'Oketch Gicher u', 'Kathu ri', 'Boy', 'Cher uiyot', 'Cry stal Asige', 'M ungatana MGH', 'Malonza ', 'Malonza', 'Njeru', 'Veronic a Maina', 'Cher arkey', ' Abdul Haji', 'Gataya  Mo Fire', 'Abdul Ha ji', 'O kenyuri', 'Wakili Sigei ', 'Dr Murang o', 'Wamatangi', 'Nyamu', ' Olekina', 'Tabith Mutinda', ' Dr Khalwale', ' Cheruiyot', 'Wakoli', 'Cryst al Asige', 'Mungatana M GH', 'Abas s', 'Cheruiy ot', 'Tab itha Keroche', 'Okenyu ri', 'KIngi', 'Dr K halwale', 'M umma', 'W amatinga', 'Chera rkey', 'Veronica Main a', 'Veronica Ma ina', 'veronica Maina', 'C herargei', 'Ve ronica Maina', 'Madza yo', 'Olek ina', 'Miano', 'King', 'M iano', 'Cheru iyot', 'Mia no', 'Macho gu', 'Machogu', 'Mvurya', 'Shakil a', 'Ch erarkey', 'Mv urya', 'Ab dul Haji', 'Wambu a', 'Thang‚Äô wa', 'Mun yi Mundigi', 'Wak ili Sigei', 'Dr Khalwale ', 'S ifuna', 'Oketch Gacheru', 'Hezena Lemal etian', 'Lemal etian', 'Dr Lelegwe Ltumb esi', ' Wakili Sigei', ' Mungatana MGH', 'Lemaletian', 'Mungatan MGH', 'Kinyu a', 'Tabitha Keroc he', 'K athuri', 'Ali Ro ba', 'Gataya Mo  Fire', 'Oketch  Gicheru', 'Dr  Murango', 'Veroni ca Maina', 'Veronica Mania', 'Okiya Omt atah', 'Tabitha Mut inda', 'Cherakey', 'Cheptum o', 'Lema letian', 'Veronicah Maina ', 'Dr Muran go', 'Wa kili Sigei', 'Wakili Sig ei', 'Wakili Si gei', 'Che ruiyot', 'O gola', 'Okiya O mtatah', 'Dr Mu rango', 'Nyutu', 'M  Kajwang‚Äô', 'Mungatana', 'aki', 'Murango', 'Murkomen ', 'Dullo ', 'Olekin a', 'S enators', 'Cheruiyot ', 'Omogeni ', 'Faki ', 'Cheptumo ', 'Dr Boni Khalwale', 'Dr Bon i Khalwale', 'Stewart Madzayo', 'Kipchumba Murkomen', 'Soipan Tuya', 'Mohamed Faki', 'John Kinyua', 'Gloria Orwoba', 'Samson Cherar key', 'Andrew Okoiti Omtatah', 'William Cheptumo', 'Mohamed Said Chute', 'Miraj Abdillahi', 'Julius Gataya', 'Ekomwa James Lomenen', 'Jackson Mandago', 'Maureen Tabitha Mutinda', 'Methu John Muhia', 'Danson Mungatana', 'Aaron Cheruiyot', ' James Kamau Murango', 'Julius Murgor', 'Joseph Nyutu Ngugi', 'Wa matinga Wahome', 'Lenku Seki', 'Allan Chesang', 'Kathuri Murungi', 'Shakila Abdallah', 'Dr Murang‚Äôo', 'Kath uri', 'Beth  Syengo', 'D r Khalwale', 'Shakil a Abdalla', 'Meth u', 'V eronica Maina', 'Cheptu mo', 'Dr Lelegwe Ltum besi', 'Munyi Mund igi', 'Vero nica Maina', 'Ososti', 'Ma dzayo', 'Wafula Wakoli', 'Wakoli Wafula', 'Joe Nyu tu', 'William Kipkemoi Kisang‚Äô', 'William Kipkemoi Kisang', 'Dr Khalwal e', ' Kavindu Muthama', 'Man dago', 'Ma ndago', 'Sif una', 'Dr Lelegwe', 'Or woba', 'Mung atana MGH', 'Munga tana MGH', 'Shakilla Abdala', 'Wetangula', 'Dr Ruto', 'Mwaru ma', 'Verinica Maina', 'Mungatana EGH', ' Dr  Lelegwe', 'Ole kina', 'Wakil i Sigei', 'Dr Khal wale', 'Dr Khlawale', 'M Kajwa ng‚Äô', 'Kis ang', 'Mungatana MG H', 'Oketch Gic heru', 'Dr Lele gwe Ltumbesi', ' Njeru', 'Manda go', 'Mvurya ', 'Prof  Kamar', 'Waki li Sigei', ' Beth Syengo', 'Dullo  and say', 'Wakili S igei', 'Madzayo ', 'Pr of Tom Ojienda SC', 'M adzayo', 'Ol ekina', 'Senat or', 'Kibwana The point is', 'Waliki Sigei', 'Mpuri Aburi', 'Beth S yengo', 'Munyi Mu ndigi', 'Beth Sye ngo', 'Crystal As ige', 'Cherark ey', 'Ki nyua', 'Kuria ', 'Linturi', 'Dr  Lelegwe Ltumbesi', 'Os otsi', 'Soipan', 'Onyonk a', 'Dr Leleg we Ltumbesi', 'C heptumo', 'Members', 'Maanz o', 'Orw oba', 'Orwo ba', 'Abdul  Haji', 'Dr Kkalwale', ' Wambua', 'Dr Leleg we', 'Lome nen', 'r Khalwale', 'Murgo r', 'Chute Said Mohamed', 'Chute Sa id Mohamed', 'Ali Roba My question is', 'Catherine Muyeka Mumma', 'Beth Kalunda Syengo', 'Si funa', 'Oket ch Gicheru', 'Shakila  Abdalla', 'athuri', 'Ma riam Omar', 'Okiy a Omtatah', 'Dr Lelengwe Ltumbesi', 'Crystal Asig e', 'M Kajw ang‚Äô', 'Githu ku', 'Gith uku', 'Chesang‚Äô', 'Mum ma', 'Ki sang'
}
stop_words.update(custom_stopwords)
hdbscan_model = HDBSCAN(min_cluster_size=2, min_samples=1, prediction_data=True)
client = OpenAI(api_key=os.getenv(""))

def identify_common_stopwords(texts, threshold=0.80):

    # Add words to stop_words if they appear in more than `threshold` percentage of documents.
    doc_count = len(texts)
    word_document_occurrences = defaultdict(int)

    for text in texts:
        words_in_doc = set(re.findall(r'\b[a-z]{3,}\b', text.lower()))
        for word in words_in_doc:
            word_document_occurrences[word] += 1

    for word, count in word_document_occurrences.items():
        if count / doc_count >= threshold:
            stop_words.add(word)

def is_english_word(word):
    return word.lower() in english_vocab

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = word_tokenize(text)
    filtered = [
        word for word in tokens
        if word not in stop_words and len(word) > 2 and is_english_word(word)
    ]

    return ' '.join(filtered)

    # stemmer.stem(word)

def extract_text_from_pdf(path, poppler_path=None):
    texts, names = [], []
    if os.path.isdir(path):
        pdfs = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(".pdf")]
    elif path.lower().endswith(".pdf") and os.path.isfile(path):
        pdfs = [path]
    else:
        print(f"No PDF(s) found at {path!r}")
        return texts, names

    for pdf in pdfs:
        print(f"\n>> Processing {pdf!r}")
        text = ""
        # 1) native text extraction
        try:
            doc = fitz.open(pdf)
            for page in doc:
                text += page.get_text()
            doc.close()
            if text.strip():
                print("   ✓ Text via PyMuPDF")
        except Exception as e:
            print(f"   ⚠️ PyMuPDF failed: {e}")

        # 2) fallback to OCR
        if not text.strip():
            print("Falling back to OCR…")
            try:
                imgs = convert_from_path(pdf, poppler_path=poppler_path)
                for img in imgs:
                    text += pytesseract.image_to_string(img)
                if text.strip():
                    print("   ✓ OCR succeeded")
                else:
                    print("   ✗ OCR gave no text")
            except Exception as e:
                print(f"   ✗ OCR pipeline failed: {e}")

        if text.strip():
            texts.append(text)
            names.append(os.path.basename(pdf))
        else:
            print(f"   ✗ No text extracted from {pdf!r}")


    valid_texts = []
    for t in texts:
        cleaned = str(t).strip()
        if cleaned and len(cleaned) > 50:  # Minimum 50 characters
            valid_texts.append(cleaned)
    
    return valid_texts, names

def generate_topic_descriptions(topic_model, num_words=10, model="gpt-4", max_tokens=150):
    topic_info = topic_model.get_topic_info()
    descriptions = {}

    for _, row in topic_info.iterrows():
        topic_id = row['Topic']
        if topic_id == -1:
            continue

        words = topic_model.get_topic(topic_id)
        if not words:
            continue

        top_words = ', '.join([w for w, _ in words[:num_words]])
        prompt = f"Given these top words:\n{top_words}\nDescribe the topic in simple terms."

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.5
            )
            description = response.choices[0].message.content.strip()
        except Exception as e:
            description = f"Error: {e}"

        descriptions[topic_id] = {
            "keywords": top_words,
            "description": description
        }

    return descriptions



def generate_topic_descriptions_csv(topic_model, output_csv="topic_descriptions.csv", num_words=10, model="gpt-4", max_tokens=150):
    topic_info = topic_model.get_topic_info()
    rows = []

    for _, row in topic_info.iterrows():
        topic_id = row['Topic']
        if topic_id == -1:
            continue  # Skip outlier topic

        topic_words = topic_model.get_topic(topic_id)
        if not topic_words:
            continue

        keywords = ', '.join([word for word, _ in topic_words[:num_words]])
        prompt = (
            f"Given the following keywords from a topic model:\n\n"
            f"{keywords}\n\n"
            f"Describe what this topic is about in simple terms. What does it likely refer to?"
        )

        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.5
            )
            description = response['choices'][0]['message']['content'].strip()
        except Exception as e:
            description = f"Error: {e}"

        rows.append({
            "Topic": topic_id,
            "Top Keywords": keywords,
            "Description": description
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"\n✓ Topic descriptions saved to {output_csv}")


def run_topic_modeling(texts, embedding_model_name="paraphrase-multilingual-mpnet-base-v2"):

    umap_model = UMAP(n_components=5, metric='cosine', random_state=42)
    hdbscan_model = HDBSCAN(min_cluster_size=2, min_samples=1, prediction_data=True)

    try:
        embedder = SentenceTransformer(embedding_model_name)
        topic_model = BERTopic(
            embedding_model=embedder,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            verbose=True
        )
    except Exception as e:
        topic_model = BERTopic(
            embedding_model=None,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            verbose=True
        )

    topics, _ = topic_model.fit_transform(texts)
    return topic_model, topics


def main(pdf_path, poppler_path=None):
    texts, file_names = extract_text_from_pdf(pdf_path, poppler_path)
    if not texts:
        print("No text found in any PDF. Exiting.")
        return

    identify_common_stopwords(texts, threshold=0.8)

    texts = [preprocess_text(text) for text in texts]
    texts = [text for text in texts if len(text.split()) > 5 and len(text) > 50]

    topic_model, topics = run_topic_modeling(texts)

    if topics is None:
        print("Failed to generate topics")
        return

    print("\n--- Topic Modeling Results ---")
    for fname, topic in zip(file_names, topics):
        print(f"{fname}: Topic {topic}")

    info = topic_model.get_topic_info()
    print(info)

    info2 = topic_model.get_topics()
    print(info2)

    info3 = topic_model.get_document_info(texts)
    print("\n--- Document Info ---")
    print(info3)

    print("\n--- Generating Topic Descriptions with OpenAI ---")
    descriptions = generate_topic_descriptions(topic_model)
    for tid, desc in descriptions.items():
        print(f"\nTopic {tid}:")
        print(f"Keywords: {desc['keywords']}")
        print(f"Description: {desc['description']}")

    print("\n--- Generating and Saving Topic Descriptions ---")
    generate_topic_descriptions_csv(topic_model)
    
if __name__ == "__main__":
    pdf_path = "/Users/sebastianchacon/Desktop/STARS/files"
    poppler_path = None

    main(pdf_path, poppler_path)
