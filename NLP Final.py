#NLP Final Project: Stance Detection 
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from collections import defaultdict

#Bert Model
#Dialog accounts of previous statements to contextualize them 

class BERTDialogStance(nn.Module):
    def __init__(self, bert_type='bert-base-uncased', num_stances=3, hidden_size=768):
        super(BERTDialogStance, self).__init__()
        self.bert = AutoModel.from_pretrained(bert_type)
        self.tokenizer = AutoTokenizer.from_pretrained(bert_type)
        self.classifier = nn.Linear(hidden_size, num_stances)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_text):
        enc = self.tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
        enc = {k: v.to(next(self.parameters()).device) for k, v in enc.items()}

        with torch.no_grad():
            outputs = self.bert(**enc)
            pooled_output = outputs.last_hidden_state[:, 0, :] 
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

#group utterances from each speaker 
def group_utterances_by_speaker(utterances):
    speaker_utterances = defaultdict(list)

    for utt in utterances:
        if ':' not in utt:
            continue
        speaker, text = utt.split(':', 1)
        speaker = speaker.strip()
        text = text.strip()
        speaker_utterances[speaker].append(text)

    grouped = {speaker: " ".join(texts) for speaker, texts in speaker_utterances.items()}
    return grouped


#Predict speakers stance 
def predict_stance_per_speaker(model, grouped, stance_labels=None):
    model.eval()
    stance_labels = stance_labels or {0: "Against", 1: "Neutral", 2: "Support"}

    for speaker, text in grouped.items():
        logits = model(text)
        prediction = torch.argmax(logits, dim=1).item()
        print(f"{speaker}: {stance_labels[prediction]}")

#sample dialog
utterances = [
    "Senator A: Thank you all for attending today’s session. I want to begin by saying I support stronger regulations on carbon emissions — we cannot keep ignoring the science.",
    "Senator B: I agree that climate change is a real issue, but pushing aggressive regulations could cripple small businesses. We need a balanced approach.",
    "Senator C: I’m still reviewing the data, but I’m skeptical that immediate regulations will have the intended long-term impact.",
    "Senator A: With all due respect, this isn’t about short-term discomfort — it’s about long-term survival. The projections are clear.",
    "Senator B: But what about the job losses in the energy sector? We need a transition plan first, not just blanket restrictions.",
    "Senator C: Senator A, can you point to evidence that the last set of regulations directly improved environmental outcomes?",
    "Senator A: Yes — emissions dropped by 12% after the 2015 clean energy act. That’s not a coincidence.",
    "Senator B: Yet energy prices soared during that same period, disproportionately affecting rural communities.",
    "Senator C: I believe we need a phased approach — perhaps pilot programs before committing nationally.",
    "Senator A: A phased approach is fine, as long as it's backed by strong enforcement and deadlines. Otherwise, it's just a delay tactic.",
    "Senator B: Then let’s discuss incentive structures — can we agree to subsidize clean energy startups in parallel?",
    "Senator C: That’s something I could support, especially if it’s tied to measurable outcomes.",
    "Senator A: Now we’re getting somewhere. Let’s make sure this is more than just talk."
]


device = torch.device("cpu")
model = BERTDialogStance().to(device)
grouped = group_utterances_by_speaker(utterances)
predict_stance_per_speaker(model, grouped)




