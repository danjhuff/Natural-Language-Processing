import nltk
from nltk.corpus import inaugural, PlaintextCorpusReader
from CorpusReader_TFIDF import CorpusReader_TFIDF
import math
from collections import defaultdict

# Download required NLTK data (run once)
nltk.download('inaugural')
nltk.download('stopwords')

def run_tests():
    print("Starting tests for CorpusReader_TFIDF...\n")

    # ... (previous tests) ...

    # Enhanced cosine_sim test
    print("\nTesting cosine_sim with detailed output...")
    fileid1, fileid2 = fileids[0], fileids[1]  # e.g., '1789-Washington.txt', '1793-Washington.txt'
    v1 = tfidf_corpus.tfidf(fileid1)
    v2 = tfidf_corpus.tfidf(fileid2)
    
    print(f"TF-IDF vector for {fileid1}: {list(v1.items())[:5]}...")  # First 5 terms
    print(f"TF-IDF vector for {fileid2}: {list(v2.items())[:5]}...")

    dotProduct = 0
    for term in v1:
        dotProduct += v1[term] * v2.get(term, 0)
    n1 = math.sqrt(sum(val ** 2 for val in v1.values()))
    n2 = math.sqrt(sum(val ** 2 for val in v2.values()))

    sim = dotProduct / (n1 * n2) if n1 and n2 else 0
    print(f"Dot product: {dotProduct}")
    print(f"Norm 1: {n1}, Norm 2: {n2}")
    print(f"cosine_sim({fileid1}, {fileid2}): {sim}")

    # Similarly for cosine_sim_new
    print("\nTesting cosine_sim_new with detailed output...")
    new_words = ["citizens", "freedom", "nation"]
    v1_new = tfidf_corpus.tfidfNew(new_words)
    v2_doc = tfidf_corpus.tfidf(fileids[0])

    print(f"TF-IDF vector for new words {new_words}: {v1_new}")
    print(f"TF-IDF vector for {fileids[0]}: {list(v2_doc.items())[:5]}...")

    dotProduct_new = 0
    for term in v1_new:
        dotProduct_new += v1_new[term] * v2_doc.get(term, 0)
    n1_new = math.sqrt(sum(val ** 2 for val in v1_new.values()))
    n2_doc = math.sqrt(sum(val ** 2 for val in v2_doc.values()))

    sim_new = dotProduct_new / (n1_new * n2_doc) if n1_new and n2_doc else 0
    print(f"Dot product (new): {dotProduct_new}")
    print(f"Norm 1 (new): {n1_new}, Norm 2 (doc): {n2_doc}")
    print(f"cosine_sim_new({new_words}, {fileids[0]}): {sim_new}")

    # ... (rest of tests) ...

    if __name__ == "__main__":
      run_tests()