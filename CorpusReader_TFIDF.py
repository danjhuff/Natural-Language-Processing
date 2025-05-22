#Program 1 
#Natural Language Processing
#Daniel Huff
import nltk
import math
from nltk.corpus.reader import CorpusReader
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from collections import defaultdict

class CorpusReader_TFIDF:
#########################################################################################################
    #Constructor 
#########################################################################################################
    def __init__(self, corpus, tf = "raw", idf = "base", stopWord = "none", toStem = False, stemFirst = False, ignoreCase = True):

        self.corpus = corpus
        self.tfMethod = tf
        self.idfMethod = idf
        self.stopWord = stopWord
        self.toStem = toStem
        self.stemFirst = stemFirst
        self.ignoreCase = ignoreCase
    
        self.stemmer = SnowballStemmer("english") if toStem else None
        self.stopwords = set(stopwords.words('english')) if stopWord == "standard" else set()
        if stopWord not in ["none", "standard"]:
            with open(stopWord, 'r') as f:
                self.stopwords = set(word.strip().lower() for word in f.readlines())
        
        self.docWordCounts = {}
        self.idfValues = {}
        self.processCorpus()
#########################################################################################################
    #Functions
#########################################################################################################
    #TFIDF Functions
    #DONE
    def tfidf(self, fileid, returnZero = False):
        termCounts = self.docWordCounts[fileid]
        ret = {}
        for term, count in termCounts.items(): #What is items?????
            tfidfValue = self._compute_tf(term, count) * self.idfValues.get(term, 0)
            if returnZero or tfidfValue > 0:
                ret[term] = tfidfValue
        return ret
    #DONE
    def tfidfAll(self, returnZero = False):
        ret = {}
        for fileid in self.corpus.fileids():
            ret[fileid] = self.tfidf(fileid, returnZero)
        return ret
    #DONE
    def tfidfNew(self, words):
        words = self._preprocess(words)
        termCounts = defaultdict(int)
        for word in words:
            termCounts[word] = termCounts[word] + 1
        ret = {}
        for term, count in termCounts.items():
            ret[term] = self._compute_tf(term, count) * self.idfValues.get(term, 0)
        return ret

    #DONE
    def listIDF(self):
        return self.idfValues
    #DONE
    def cosine_sim(self, fileid1, fileid2):
        v1 = self.tfidf(fileid1)
        v2 = self.tfidf(fileid2)
        dotProduct = 0
        n1 = 0
        n2 = 0

        for term in v1:
            dotProduct += v1[term] * v2.get(term, 0)
        
        for val in v1.values():
            n1 += (val ** 2)
        n1 = math.sqrt(n1)
        for val in v2.values():
            n2 += (val ** 2)
        n2 = math.sqrt(n2)

        return dotProduct / (n1 * n2)
    #DONE
    def cosine_sim_new(self, words, fileid): 
        v1 = self.tfidfNew(words)
        v2 = self.tfidf(fileid)
        dotProduct = 0
        n1 = 0
        n2 = 0

        for term in v1:
            dotProduct += v1[term] * v2.get(term, 0)

        for val in v1.values():
            n1 += val ** 2
        n1 = math.sqrt(n1)
        for val in v2.values():
            n2 += val ** 2
        n2 = math.sqrt(n2)
        
        return dotProduct / (n1 * n2)
    #DONE
    def query(self, words):
        vec = self.tfidfNew(words)
        results = []

        for fileid in self.corpus.fileids():
            fileVec = self.tfidf(fileid)
            
            dotProduct = 0
            for term in vec:
                dotProduct += (vec[term] * fileVec.get(term, 0))
            
            normVec = 0
            for val in vec.values():
                normVec += val ** 2
            normVec = math.sqrt(normVec)
            
            normFile = 0
            for val in fileVec.values():
                normFile += val ** 2
            normFile = math.sqrt(normFile)
            
            if normVec and normFile:
                similarity = dotProduct / (normVec * normFile)
            else:
                similarity = 0

            results.append((fileid, similarity))

        sortedResults = []
        sortedResults = sorted(results, key=lambda x: x[1], reverse=True)
        
        return sortedResults
    
    #DONE
    def _compute_tf(self, term, termCount):
        if(self.tfMethod == 'log'):
            if (termCount) > 0:
                return 1 + math.log2(termCount)
            else:
                return 0
        return termCount
    #DONE
    def fileids(self):
        return self.corpus.fileids()
    #DONE
    def words(self, fileids=None):
        words = self.corpus.words(fileids)
        return self._preprocess(words)
    def raw(self, fileids=None):
        return self.corpus.raw(fileids)
    
#########################################################################################################
    #Processing DONE
#########################################################################################################
    #DONE
    def processCorpus(self):
        docCount = defaultdict(int)
        numDocs = len(self.corpus.fileids())

        for fileid in self.corpus.fileids():
            words = self._preprocess(self.corpus.words(fileid))
            termCounts = defaultdict(int)
            for word in words:
                termCounts[word] = termCounts[word] + 1
            self.docWordCounts[fileid] = termCounts
            for term in set(words):
                docCount[term] = docCount[term] + 1

        for term, df in docCount.items():
            if self.idfMethod == "smooth":
                self.idfValues[term] = math.log2((numDocs + 1)/(df+ 1)) + 1
            elif df > 0:
                self.idfValues[term] = math.log2(numDocs/df)
            else:
                self.idfValues[term] = 0

    #DONE
    def _preprocess(self, words):
        if self.ignoreCase:
            processed = []
            for x in words:
                processed.append(x.lower())
        else:
            processed = words
        
        if self.toStem and self.stemFirst:
            stemmedWords = []
            for x in processed:
                stemmedWords.append(self.stemmer.stem(x))
            processed = stemmedWords
        
        validWords = []
        for x in processed:
            if x not in self.stopwords:
                validWords.append(x)
        processed = validWords
        
        if self.toStem and not self.stemFirst:
            stemmedWords = []
            for x in processed:
                stemmedWords.append(self.stemmer.stem(x))
            processed = stemmedWords
        
        return processed
        
    