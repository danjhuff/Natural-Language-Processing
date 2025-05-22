
import os
import re
import torch
import fitz  # PyMuPDF
import pytesseract
from pdf2image import convert_from_path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from collections import defaultdict, Counter
from transformers import AutoTokenizer, pipeline


# Load Kornosk's pretrained stance detection model 
# Bert Model tuned with 2020 election tweets on stance surrounding Biden
tokenizer = AutoTokenizer.from_pretrained("kornosk/bert-election2020-twitter-stance-biden")
model = AutoModelForSequenceClassification.from_pretrained("kornosk/bert-election2020-twitter-stance-biden")




model.eval()
id2label = {0: "AGAINST", 1: "FAVOR", 2: "NONE"}

def extract_vote_blocks(raw_text):
    vote_blocks = {"AYES": [], "NOES": [], "ABSTENTIONS": []}
    pattern = r"(AYES|NOES|ABSTENTIONS):\s*(.*?)\n\n"
    matches = re.findall(pattern, raw_text, re.DOTALL | re.IGNORECASE)

    for label, block in matches:
        names = re.findall(r"Sen\.?\s+[A-Za-z .'-]+", block)
        cleaned = [re.sub(r"\s+", " ", name.strip()) for name in names]
        vote_blocks[label.upper()].extend(cleaned)

    return vote_blocks

def normalize_name(name):
    name = name.lower()
    name = re.sub(r"sen\.?\s*", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name

def clean_speaker_name(name):
    name = name.strip()

    # Remove common procedural titles (with or without "The")
    procedural_patterns = [
        r"\b(T?he\s)?(Temporary\s)?(Chairperson|Speaker|Deputy Speaker|Clerk|Teller|Senate\s+(Majority|Minority)\s+Leader|Leader\s+of\s+Majority)\b",
        r"\b(Some|An)?\s*hon\.?\s*Senator(s)?\b",
        r"\bABSENTIONS?\b", r"\bABSTENTIONS?\b", r"\bAYES\b", r"\bNOES\b", r"\bDisclaimer\b"
    ]

    for pattern in procedural_patterns:
        if re.search(pattern, name, flags=re.IGNORECASE):
            return None

    # Keep only names that explicitly contain "Sen." or "Senator"
    if not re.search(r"\bSen(\.|ator)?\b", name, re.IGNORECASE):
        return None

    # Normalize "Senator" to "Sen."
    name = re.sub(r"\bSenator\b", "Sen.", name, flags=re.IGNORECASE)

    # Remove anything in parentheses
    name = re.sub(r"\(.*?\)", "", name)

    # Normalize spacing
    name = name.replace("Sen ", "Sen. ")
    name = re.sub(r"\s+", " ", name).strip()

    return name if len(name.split()) >= 2 else None






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
        try:
            doc = fitz.open(pdf)
            for page in doc:
                text += page.get_text()
            doc.close()
            if text.strip():
                print("   ✓ Text via PyMuPDF")
        except Exception as e:
            print(f"   ⚠️ PyMuPDF failed: {e}")

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
        if cleaned and len(cleaned) > 50:
            valid_texts.append(cleaned)
    with open("newMethodSpeakers.txt", "w", encoding="utf-8") as f:
        for name, text in zip(names, valid_texts):
            f.write(f"===== {name} =====\n\n")
            f.write(text.strip())
            f.write("\n\n")

    return valid_texts, names

def parse_speaker_utterances_from_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    pattern = re.compile(r"(?:^|\n)([A-Z][A-Za-z .'\-()]+):\s*", re.MULTILINE)
    matches = list(pattern.finditer(text))
    utterances = []

    for i, match in enumerate(matches):
        speaker = match.group(1).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        if len(content) > 10:
            utterances.append({"speaker": speaker, "text": content})

    return utterances

def predict_stance(utterance, context=None, force_binary=True):
    if context:
        input_text = f"{context} [SEP] {utterance}"
    else:
        input_text = utterance

    inputs = tokenizer(
        input_text,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze()

        pred_id = torch.argmax(probs).item()
        label = id2label[pred_id]

        if force_binary:
            if label == "NONE":
                forced_id = torch.argmax(probs[:2]).item()  # pick FOR or AGAINST
                label = id2label[forced_id]

        return "FOR" if label == "FAVOR" else "AGAINST" if label == "AGAINST" else "NONE"


# Run Processing 
texts, names = extract_text_from_pdf("/Users/danhuff/Desktop/Bills/KenyaSample.pdf")
utterances = parse_speaker_utterances_from_file("/Users/danhuff/newMethodSpeakers.txt")

total_for = 0
total_against = 0
total_none = 0

for i, utt in enumerate(utterances):
    context = utterances[i - 1]["text"] if i > 0 else None
    utt["stance"] = predict_stance(utt["text"], context=context)

    # Tally stance counts
    if utt["stance"] == "FOR":
        total_for += 1
    elif utt["stance"] == "AGAINST":
        total_against += 1
    elif utt["stance"] == "NONE":
        total_none += 1

    # Print each result (can comment out for just the report)
    #print(f"\nSPEAKER: {utt['speaker']}") 
    #print(f"STANCE: {utt['stance']}")
    #print(f"UTTERANCE: {utt['text'][:300]}...")  

print("\n=== STANCE SUMMARY ===")
print(f"Total FOR: {total_for}")
print(f"Total AGAINST: {total_against}")
print(f"Total NONE: {total_none}")

speaker_stances = defaultdict(list)

speaker_stances = defaultdict(list)
for utt in utterances:
    cleaned = clean_speaker_name(utt["speaker"])
    if cleaned:
        speaker_stances[cleaned].append(utt["stance"])

# Decide majority stance for each speaker
speaker_majority = {}
for speaker, stances in speaker_stances.items():
    counts = Counter(stances)
    if counts["FOR"] > counts["AGAINST"]:
        speaker_majority[speaker] = "FOR"
    elif counts["AGAINST"] > counts["FOR"]:
        speaker_majority[speaker] = "AGAINST"
    else:
        speaker_majority[speaker] = "NONE"  # Mixed or unclear

# Print results
print("\n=== SPEAKER STANCES ===")
for speaker, stance in speaker_majority.items():
    print(f"{speaker}: {stance}")

# Get raw text from your file
with open("/Users/danhuff/newMethodSpeakers.txt", "r", encoding="utf-8") as f:
    full_text = f.read()

votes = extract_vote_blocks(full_text)

# Flatten ground-truth stance mapping
true_stance = {}
for name in votes["AYES"]:
    true_stance[normalize_name(name)] = "FOR"
for name in votes["NOES"]:
    true_stance[normalize_name(name)] = "AGAINST"

# Compare predicted vs true
correct = 0
total = 0
for speaker, stance in speaker_majority.items():
    norm = normalize_name(speaker)
    if norm in true_stance:
        total += 1
        if stance == true_stance[norm]:
            correct += 1
        else:
            print(f"MISMATCH: {speaker} predicted {stance} but voted {true_stance[norm]}")

print(f"\nEvaluation Accuracy on voters: {correct}/{total} = {correct/total:.2%}")

print(f"Total speakers in vote list: {len(true_stance)}")
print(f"Matched speakers with predictions: {total}")

voted_senators = set(normalize_name(name) for name in votes["AYES"] + votes["NOES"])
speakers = set(normalize_name(speaker) for speaker in speaker_majority.keys())
silent_voters = voted_senators - speakers
print(f"Senators who voted but did not speak: {silent_voters}")


y_true = []
y_pred = []

for speaker, stance in speaker_majority.items():
    norm = normalize_name(speaker)
    if norm in true_stance:
        y_true.append(true_stance[norm])
        y_pred.append(stance)

#print(classification_report(y_true, y_pred, labels=["FOR", "AGAINST", "NONE"]))




