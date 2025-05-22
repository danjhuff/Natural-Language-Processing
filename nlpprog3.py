import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

def parse_labeled_sentences(filename):
    pairs = []
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                parts = line.split(maxsplit=1)
                if len(parts) == 2 and parts[0] in {'1', '2'}:
                    pairs.append((int(parts[0]), parts[1]))
                else:
                    print(f"Skipping malformed line: {line}")
    return pairs

def train_and_save_model(txt_path, model_path):
    pairs = parse_labeled_sentences(txt_path)
    if not pairs:
        raise ValueError(f"No valid lines found in {txt_path}")
    
    labels, sentences = zip(*pairs)

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        sentences, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Build pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LogisticRegression(max_iter=1000))
    ])

    # Train model
    pipeline.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel trained on: {txt_path}")
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    # Save model
    joblib.dump(pipeline, model_path)
    print(f"Model saved to {model_path}")

# Train and evaluate each model
train_and_save_model('/Users/danhuff/desktop/camper.txt', 'camper_model.pkl')
train_and_save_model('/Users/danhuff/desktop/conviction.txt', 'conviction_model.pkl')
train_and_save_model('/Users/danhuff/desktop/deed.txt', 'deed_model.pkl')
