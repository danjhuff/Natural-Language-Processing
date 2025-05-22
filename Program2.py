from transformers import BertModel, BertTokenizer
import torch
from sklearn.metrics.pairwise import cosine_similarity as cs
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import wordnet as wn
nltk.download('wordnet')
from transformers import AutoTokenizer, AutoModel


#8 Words with 10 sentences each
words = ["bottle"]
bottleS = [
    "I bought a bottle of water at the grocery store.",
    "The bar does not sell beer by the bottle.",
    "The mom fed her baby with the bottle.",
    "The bottle is sitting on the counter.",
    "The basketball knocked over the bottle on the sideline.",
    "The bottle fell off the boat and sank in the river.",
    "My teacher told me to put my bottle in the recycling bin.",
    "The magician made the bottle disappear.",
    "My son made a painting of a blue glass bottle.",
    "The bottles shattered as the angered man threw them at the ground."
]
sentences = [bottleS]



def genBERTVector(model_name, tokenizer, word, sentence):
    #Use BERT to generate the hidden states

    model = BertModel.from_pretrained(model_name)   

    inputs = tokenizer(sentence, padding=True, return_tensors="pt")
    # print(inputs)

    with torch.no_grad():
        outputs = model(**inputs)

    last_hidden_states = outputs.last_hidden_state
    last_hidden_states = last_hidden_states.numpy()
    truncated_vectors = last_hidden_states[:, :, :50]

    # print(last_hidden_states.shape) 
    # for x0 in last_hidden_states:
    #    print("-----")
    #    for x in x0:
    #        print(x)
    # print("-----")
    # print(truncated_vectors.shape)
    # for x0 in last_hidden_states:
    #     print("-----")
    #     for x in x0:
    #        print(x)
    # print(len(truncated_vectors))
    # print(truncated_vectors[0][0])

    return truncated_vectors

def calculate_similarities(vectors):
    # Reshape to 2D if needed (num_sentences, 50)
    if vectors.ndim == 3:
        vectors = vectors.reshape(-1, 50)
    
    similarities = []
    for i in range(len(vectors)):
        for j in range(i+1, len(vectors)):
            # Use vectors directly (already 2D)
            sim = cs(vectors[i].reshape(1, -1), 
            vectors[j].reshape(1, -1))[0][0]
            similarities.append(sim)
    return similarities

def plot_cs(similarities, word):
    # Plot the distribution of cosine similarities
    bins = np.arange(0, 1.1, 0.1)

    plt.hist(similarities, bins=bins, edgecolor='black', color='cyan')
    plt.title(f'Cosine Similarity Distribution for Word "{word}"')
    plt.xlabel('Cosine Similarity Range')
    plt.ylabel('Frequency')
    plt.xticks(bins)
    plt.grid(axis='y', alpha=0.75)
    plt.show()

def random360(model, tokenizer):
    all_bert_vectors = []
    
    for word, single_sentence in zip(words, sentences):
        vectors = genBERTVector('nlpaueb/legal-bert-base-uncased', tokenizer, word, single_sentence)
        
        if vectors.ndim == 3:
            vectors = vectors.reshape(-1, 50)  
        
        for vector in vectors:
            all_bert_vectors.append(vector)
    
    all_bert_vectors = np.array(all_bert_vectors)
    
    random_indices = np.random.choice(len(all_bert_vectors), 360, replace=False)
    random_vectors = all_bert_vectors[random_indices]
    random_similarities = calculate_similarities(random_vectors)
    
    plot_cs(random_similarities, "Random 360 Vectors")
    print(f"\nRandom 360 Average similarity: {np.mean(random_similarities):.4f}")
    print(f"Random 360 Standard deviation of Similarity: {np.std(random_similarities):.4f}")

def synonyms(word):
    #get synsets for the word
    syn = wn.synonyms(word)
    if not syn:
        print(f"No synsets found for the word '{word}'.")
        return []

    first_syn = syn[0]
    
    return first_syn

def __main__():
    model_name = 'bert-base-uncased'
    other_model =  'nlpaueb/legal-bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(other_model)
    model = BertModel.from_pretrained(other_model)

    x = genBERTVector(model_name, tokenizer, "bottle", bottleS)
    y = calculate_similarities(x)
    plot_cs(y, "bottle")

    # for word, single_sentence in zip(words, sentences):
    #     vectors = genBERTVector(model_name, tokenizer, word, single_sentence)
    #     similarities = calculate_similarities(vectors)
    #     allsentences.append(similarities)
    #     print(f"\nWord: {word}")
    #     print(f"\nSentences: {single_sentence}")
    #     print(f"Average similarity: {np.mean(similarities):.4f}")
    #     print(f"Standard deviation of Similarity: {np.std(similarities):.4f}")
    #     plot_cs(similarities, word)

    # Calculate and print the average similarity for all words
    #all_similarities = np.concatenate(allsentences)
    # print(f"\nOverall Average similarity: {np.mean(all_similarities):.4f}")
    # print(f"Overall Standard deviation of Similarity: {np.std(all_similarities):.4f}")
    # plot_cs(all_similarities, "All Words")
    
    #Randomly select 360 vectors and calculate similarities
    # random360(model, tokenizer)

    #Calculate similarity between every pair of synonyms
    # for word_pair, sentences in zip(syns, syn_sentences):
    #     vectors = genBERTVector(model, tokenizer, word_pair[0], sentences)
    #     similarities = calculate_similarities(vectors)
    #     allsentences.append(similarities)
    #     print(f"\nWord Pair: {word_pair}")
    #     print(f"Average similarity: {np.mean(similarities):.4f}")
    #     print(f"Standard deviation of Similarity: {np.std(similarities):.4f}")
    #     plot_cs(similarities, word_pair)

    #Calculate similarity between every pair of synonyms
    # for word_pair, sentences in zip(syns, syn_sentences):
    #     vectors = genBERTVector(model, tokenizer, word_pair[0], sentences)
    #     similarities = calculate_similarities(vectors)
    #     allsentences.append(similarities)
    #     # print(f"\nWord Pair: {word_pair}")
    #     # print(f"Average similarity: {np.mean(similarities):.4f}")
    #     # print(f"Standard deviation of Similarity: {np.std(similarities):.4f}")
    #     # plot_cs(similarities, word_pair)
    
    #Calculate and print average and std dev of similarity for all synonyms
    # all_similarities = np.concatenate(allsentences)
    # print(f"\nOverall Average similarity: {np.mean(all_similarities):.4f}")
    # print(f"Overall Standard deviation of Similarity: {np.std(all_similarities):.4f}")
    # plot_cs(all_similarities, "All Synonyms")


__main__()