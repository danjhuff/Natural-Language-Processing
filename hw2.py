
from transformers import BertModel, BertTokenizer
import torch
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')  # For extended WordNet data
from nltk.corpus import wordnet as wn

class Word:
    def __init__(self, target, sentences):
        """
        Args:
            target (str): The target word (e.g., 'class').
            sentences (List[str]): A list of sentences containing the word.
        """
        self.target = target
        self.sentences = sentences
        

    def __repr__(self):
        return f"Word(target='{self.target}', num_sentences={len(self.sentences)})"
    

    def get_sentence(self, index):
        return self.sentences[index]








################################################################################################################
#Word sentences 
################################################################################################################
classSentences = [
    "I have class tomorrow morning.",
    "Seb said his math class is very boring.",
    "Alden is walking to class right now.",
    "He bought an expensive testbook for his history class.",
    "The largest class at her school has 45 people.",
    "The kindergarteners have a pizza party during their class on Friday.",
    "A class with less than seven students is shown to have learning benefits.",
    "The football player was dismissed from his class early for the big game.",
    "The college student has to take a summer class to graduate early.",
    "The professor cancelled class in light of the inclement weather."
]

bottleSentences = [
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

phoneSentences = [
    "The student had to put her phone away during the test.",
    "The comedian asked an audience member to put his phone on silent mode after it rang.",
    "The average teenager uses their phone for over six hours every day.",
    "Apple’s new phone has a battery that can last for three days.",
    "Most modern homes do not have a home phone anymore.",
    "The students gathered around the phone to watch the basketball game.",
    "She forgot her phone on the kitchen counter before leaving for work.",
    "The phone rang loudly, startling everyone in the quiet library.",
    "He upgraded to a new phone with a better camera last week.",
    "The cracked screen on his phone made it hard to read messages."
]

sawSentences = [
    "I saw a shooting star while walking home last night.",
    "She saw her friend across the street and waved excitedly.",
    "We saw a great movie at the theater over the weekend.",
    "He saw the sign too late while driving and missed the turn.",
    "The friends saw the sunrise from the top of the mountain.",
    "I saw a deer run across the road early this morning.",
    "He saw an old photograph that brought back memories.",
    "We saw lightning strike a tree during the storm.",
    "As we turned the corner, we saw the cafe glowing with string lights.",
    "Despite the fog, they still saw the lighthouse blinking in the distance."
]

lightSentences = [
    "Through the curtains, a soft light spilled into the quiet room.",
    "What caught his attention first was the flickering light above the doorway.",
    "In the darkness, even the tiniest light felt like hope.",
    "A strange light hovered over the field, pulsing rhythmically.",
    "Only when the power returned did the light reveal the damage.",
    "Beneath the stained glass, colored light danced across the floor.",
    "That kind of light—gentle, golden, and warm—only came at sunset.",
    "She walked toward the light without hesitation, her fears fading.",
    "Light filled the hallway just as the clock struck midnight.",
    "Hidden behind clouds, the morning light struggled to shine through."
]

floorSentences = [
    "Scattered across the floor were notes she hadn’t touched in years.",
    "He dropped his keys, and they slid across the polished floor.",
    "The floor creaked beneath their footsteps, warning of every move.",
    "On the floor, a trail of muddy footprints led to the door.",
    "Not a single thing moved on that cold, silent floor.",
    "What surprised them most was the mosaic embedded in the floor.",
    "She collapsed to the floor, laughing so hard she couldn't breathe.",
    "Beneath the carpet, the original wooden floor had been perfectly preserved.",
    "Light from the window cast long lines across the dusty floor.",
    "The man waited to mop the floor as the people walked by."
]

droppedSentences = [
    "He dropped the glass before realizing his hands were still wet.",
    "Without a word, she dropped the envelope onto the desk and left.",
    "The chef dropped the marinated chicken into the pan.",
    "What he dropped wasn’t just his phone—it was his only lifeline.",
    "As the bell rang, she dropped everything and ran out the door.",
    "From the balcony, someone accidentally dropped a scarf into the wind.",
    "The child began to cry because he dropped his ice cream.",
    "The building dropped immediately during the planned demolition.",
    "The leaves dropped on the ground as summer became fall.",
    "Moments after takeoff, the plane dropped slightly before leveling out."
]

largeSentences = [
    "A large shadow loomed behind the curtain, unmoving.",
    "What stood in the center of the room was a large wooden table, scratched with age.",
    "She carried a large bag that seemed heavier with every step.",
    "A large portion of the company's assets were lost after losing the lawsuit.",
    "The crowd gathered around a large screen showing the final moments of the game.",
    "In one corner sat a large dog, quietly watching everyone pass.",
    "Large windows lined the hallway, flooding it with morning light.",
    "There was something comforting about that large, overstuffed chair in the corner.",
    "A large portion of the wall had crumbled, revealing the bricks beneath.",
    "Even in a large group, she somehow felt alone."
]
################################################################################################################

w1 = Word("class", classSentences)
w2 = Word("bottle", bottleSentences)
w3 = Word("phone", phoneSentences)
w4 = Word("saw", sawSentences)
w5 = Word("light", lightSentences)
w6 = Word("floor", floorSentences)
w7 = Word("dropped", droppedSentences)
w8 = Word("large", largeSentences)



def getBERTVector(model, tokenizer, word, list):
    vectors = {}
    target = word.target.lower()

    for i, j in enumerate(list):
        inputs = tokenizer(j, return_tensors="pt", padding = True)
        with torch.no_grad():
            outputs = model(**inputs)

        tokenIDs = inputs["input_ids"][0]
        tokens = tokenizer.convert_ids_to_tokens(tokenIDs)

        token_indices = [
            idx for idx, token in enumerate(tokens)
            if target in token.replace("##", "")
        ]

        if not token_indices:
            print("Word not found in sentence.")
        
        token_vectors = outputs.last_hidden_state[0][token_indices]
        avg_vector = torch.mean(token_vectors, dim=0).cpu().numpy()
        vectors[f"w{i+1}"] = avg_vector[:50]

    return vectors


def getCosine(v):
    similarities = []
    for(x, y) in combinations(v.keys(), 2):
        v1 = v[x].reshape(1,-1)
        v2 = v[y].reshape(1,-1)
        similarity = cosine_similarity(v1, v2)[0][0]
        similarities.append(similarity)
    return similarities

def cosineHist(similarities, w):
    mean = np.mean(similarities)
    std = np.std(similarities)
    
    plt.figure(figsize=(8, 5))
    bins = np.arange(0, 1.1, 0.1)
    plt.hist(similarities, bins=bins, edgecolor='black', range=(0, 1))

    # Vertical line at the mean
    plt.axvline(mean, color='red', linestyle='dashed', linewidth=1.5, label=f"Mean = {mean:.3f}")

    # Text box for mean and std
    textstr = f"Mean: {mean:.3f}\nSD: {std:.3f}"
    plt.text(0.65, max(plt.ylim()) * 0.9, textstr, fontsize=10, bbox=dict(facecolor='white', edgecolor='gray'))

    plt.title(f"Cosine Similarity Histogram for '{w}'")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def multiWordCosine(vecs):
    similarities = []

    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            wordi = vecs[i]
            wordj = vecs[j]

            for veci in wordi.values():
                for vecj in wordj.values():
                    sim = cosine_similarity(veci.reshape(1, -1), vecj.reshape(1, -1))[0][0]
                    similarities.append(sim)

    return similarities




# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

################################################################################################################
#Cosine Histograms
################################################################################################################
# vw1 = getBERTVector(model, tokenizer, w1, w1.sentences)
# sw1 = getCosine(vw1)
# # cosineHist(sw1, w1.target)

# vw2 = getBERTVector(model, tokenizer, w2, w2.sentences)
# sw2 = getCosine(vw2)
# # cosineHist(sw2, w2.target)

# vw3 = getBERTVector(model, tokenizer, w3, w3.sentences)
# sw3 = getCosine(vw3)
# # cosineHist(sw3, w3.target)

# vw4 = getBERTVector(model, tokenizer, w4, w4.sentences)
# sw4 = getCosine(vw4)
# # cosineHist(sw4, w4.target)

# vw5 = getBERTVector(model, tokenizer, w5, w5.sentences)
# sw5 = getCosine(vw5)
# # cosineHist(sw5, w5.target)

# vw6 = getBERTVector(model, tokenizer, w6, w6.sentences)
# sw6 = getCosine(vw6)
# # cosineHist(sw6, w6.target)

# vw7 = getBERTVector(model, tokenizer, w7, w7.sentences)
# sw7 = getCosine(vw7)
# # cosineHist(sw7, w7.target)

# vw8 = getBERTVector(model, tokenizer, w8, w8.sentences)
# sw8 = getCosine(vw8)
# # cosineHist(sw8, w8.target)

# totalGraph = sw1 + sw2 + sw3 + sw4 + sw5 + sw6 + sw7 + sw8
# cosineHist(totalGraph, "Total Graph")

################################################################################################################
################################################################################################################
#Part 2 (Function Above)

# allVecs = [vw1, vw2, vw3, vw4, vw5, vw6, vw7, vw8]

# mwc = multiWordCosine(allVecs)
# cosineHist(mwc, "Multi Word Cosine")

################################################################################################################
################################################################################################################
#Part 3 

def get_synonyms(word):
    synonyms = set()
    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            if lemma.name().lower() != word.lower():
                synonyms.add(lemma.name().replace("_", " "))
    return list(synonyms)

#To find pairs
# print(get_synonyms("hot"))

#test, examination
#car, automobile
#jump, leap
#restaurant, diner 
#tiny, petite 
#disk, record
#trash, junk
#hot, spicy

################################################################################################################
################################################################################################################
testSentences = [
    "The software failed the stress test under heavy user load.",
    "After hours of practice, she finally aced the piano proficiency test.",
    "The new filtration system must pass a purity test before approval.",
    "We conducted a blind taste test to determine the most popular coffee brand.",
    "The blood test came back positive for an iron deficiency."
]

examinationSentences = [
    "She studied all night to prepare for her history examination.",
    "The final examination will cover all chapters from the beginning of the semester.",
    "His high score on the entrance examination earned him a scholarship.",
    "The certification examination is known for its difficult essay questions.",
    "Due to illness, he had to reschedule his math examination for next week."
]

carSentences = [
    "She parked the car under a tree to keep it cool in the sun.",
    "The rental car broke down halfway through their road trip.",
    "His car is electric, so he charges it at home every night.",
    "A strange noise started coming from the back of the car.",
    "They loaded the car with camping gear and hit the road."
]

automobileSentences = [
    "The vintage automobile was the centerpiece of the museum exhibit.",
    "He spent years restoring the old automobile to mint condition.",
    "The invention of the automobile revolutionized personal transportation.",
    "Her father collects rare automobile models from the early 20th century.",
    "An automobile accident on the highway caused major traffic delays."
]

jumpSentences = [
    "He cleared the hurdle with a single, effortless jump.",
    "There was a sudden jump in temperature overnight.",
    "Her heart did a little jump when she saw the message.",
    "The athlete set a new record in the long jump event.",
    "The video showed a jump in the footage, indicating it had been edited."
]

leapSentences = [
    "The dancer landed the leap with perfect balance and grace.",
    "Moving to a new country was a huge leap of faith for her.",
    "The data shows a leap in profits compared to last quarter.",
    "With one powerful leap, the cat reached the top of the fridge.",
    "His invention marked a significant leap in battery technology."
]

restaurantSentences = [
    "The new Italian restaurant downtown has already become a local favorite.",
    "They celebrated their anniversary at a rooftop restaurant with a view of the city.",
    "The restaurant was fully booked, so we had to wait for a table.",
    "She’s always dreamed of opening her own vegan restaurant.",
    "We read the reviews before choosing a restaurant for dinner."
] 

dinerSentences = [
    "The small beachside diner served the freshest seafood I’ve ever had.",
    "We found a cozy little diner tucked away in a quiet alley.",
    "That late-night diner is popular with students after concerts.",
    "The food truck turned into a permanent diner after gaining a loyal following.",
    "Every diner in the neighborhood had its own unique charm and flavor."
]

tinySentences = [
    "The kitten was so tiny it fit comfortably in the palm of his hand.",
    "She placed a tiny charm inside the envelope for good luck.",
    "Only a tiny crack in the window let in the cold air.",
    "They lived in a tiny cabin deep in the forest.",
    "He made a tiny adjustment that completely fixed the code."
] 

petiteSentences = [
    "She wore a petite dress that was tailored perfectly to her frame.",
    "The boutique specializes in clothing for petite women.",
    "A petite figure can make oversized sweaters look even cozier.",
    "Her petite frame made it easy to move through the crowded space.",
    "The sculpture featured a petite ballerina mid-twirl."
] 

diskSentences = [
    "He inserted the disk into the drive to transfer the files.",
    "The old computer still used floppy disks for saving documents.",
    "A backup of the entire system was stored on an external disk.",
    "She found a labeled disk with their childhood photos on it.",
    "The technician replaced the damaged disk in the server rack."
]

recordSentences = [
    "He played his favorite jazz record on a vintage turntable.",
    "The store had a rare vinyl record she had been searching for.",
    "They listened to an old Beatles record on a rainy afternoon.",
    "Her grandfather kept a record collection in mint condition.",
    "The sound of a spinning record always brought back memories."
]

trashSentences = [
    "She took out the trash before the garbage truck arrived.",
    "The park was littered with trash after the weekend festival.",
    "He threw the broken headphones into the trash without a second thought.",
    "The smell of the overflowing trash bin filled the kitchen.",
    "Volunteers gathered bags of trash during the beach cleanup."
]

junkSentences = [
    "The garage was packed with old furniture and random junk.",
    "He spent the weekend sorting through junk he'd been hoarding for years.",
    "They hired a service to haul away all the construction junk.",
    "That drawer is full of receipts, batteries, and other junk.",
    "She donated the useful items and threw out the rest of the junk."
]

hotSentences = [
    "The wings were so hot they made his eyes water.",
    "She prefers hot chili over the mild version any day.",
    "That hot sauce is too intense for most people to handle.",
    "He accidentally bit into a hot pepper hidden in the dish.",
    "The restaurant is famous for its hot and flavorful curry."
]

spicySentences = [
    "He added extra peppers to make the curry more spicy.",
    "She warned them that the salsa was extremely spicy.",
    "The spicy aroma filled the kitchen as the stew simmered.",
    "He couldn’t handle food that was too spicy for his taste.",
    "They served a spicy noodle dish that made everyone sweat."
]

################################################################################################################
################################################################################################################

w9  = Word("test", testSentences)
w10 = Word("examination", examinationSentences)
w11 = Word("car", carSentences)
w12 = Word("automobile", automobileSentences)
w13 = Word("jump", jumpSentences)
w14 = Word("leap", leapSentences)
w15 = Word("restaurant", restaurantSentences)
w16 = Word("diner", dinerSentences)
w17 = Word("tiny", tinySentences)
w18 = Word("petite", petiteSentences)
w19 = Word("disk", diskSentences)
w20 = Word("record", recordSentences)
w21 = Word("trash", trashSentences)
w22 = Word("junk", junkSentences)
w23 = Word("hot", hotSentences)
w24 = Word("spicy", spicySentences)

vw9  = getBERTVector(model, tokenizer, w9, w9.sentences)
vw10 = getBERTVector(model, tokenizer, w10, w10.sentences)
vw11 = getBERTVector(model, tokenizer, w11, w11.sentences)
vw12 = getBERTVector(model, tokenizer, w12, w12.sentences)
vw13 = getBERTVector(model, tokenizer, w13, w13.sentences)
vw14 = getBERTVector(model, tokenizer, w14, w14.sentences)
vw15 = getBERTVector(model, tokenizer, w15, w15.sentences)
vw16 = getBERTVector(model, tokenizer, w16, w16.sentences)
vw17 = getBERTVector(model, tokenizer, w17, w17.sentences)
vw18 = getBERTVector(model, tokenizer, w18, w18.sentences)
vw19 = getBERTVector(model, tokenizer, w19, w19.sentences)
vw20 = getBERTVector(model, tokenizer, w20, w20.sentences)
vw21 = getBERTVector(model, tokenizer, w21, w21.sentences)
vw22 = getBERTVector(model, tokenizer, w22, w22.sentences)
vw23 = getBERTVector(model, tokenizer, w23, w23.sentences)
vw24 = getBERTVector(model, tokenizer, w24, w24.sentences)

def cosineSynonyms(v1, v2):
    similarities = []

    for vec1 in v1.values():
        for vec2 in v2.values():
            sim = cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0][0]
            similarities.append(sim)

    return similarities

cs1  = cosineSynonyms(vw9, vw10)   # test vs examination
cs2  = cosineSynonyms(vw11, vw12)  # car vs automobile
cs3  = cosineSynonyms(vw13, vw14)  # jump vs leap
cs4  = cosineSynonyms(vw15, vw16)  # restaurant vs diner
cs5  = cosineSynonyms(vw17, vw18)  # tiny vs petite
cs6  = cosineSynonyms(vw19, vw20)  # disk vs record
cs7  = cosineSynonyms(vw21, vw22)  # trash vs junk
cs8  = cosineSynonyms(vw23, vw24)  # hot vs spicy


# cosineHist(cs1, "test vs examination")
# cosineHist(cs2, "car vs automobile")
# cosineHist(cs3, "jump vs leap")
# cosineHist(cs4, "restaurant vs diner")
# cosineHist(cs5, "tiny vs petite")
# cosineHist(cs6, "disk vs record")
# cosineHist(cs7, "trash vs junk")
# cosineHist(cs8, "hot vs spicy")

totalGraphSyn = cs1 + cs2 + cs3 + cs4 + cs5 + cs6 + cs7 + cs8
cosineHist(totalGraphSyn, "Total Graph")