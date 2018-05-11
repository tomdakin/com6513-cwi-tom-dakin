import numpy as np
from collections import Counter

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer

from nltk.corpus import wordnet as wn

class English_Model(object):

    def __init__(self, language, classifier):
        self.name = "English Model"
        self.language = language

        # from 'Multilingual and Cross-Lingual Complex Word Identification' (Yimam et. al, 2017)
        if language == 'english':
            self.avg_word_length = 5.3
            self.avg_synset = 1.55 # c.f. sum([len(wn.synsets(wd)) for wd in wn.words()])/len([wd for wd in wn.words()])

        self.model = classifier


    def extract_all_features(self, trainset):

        self.avg_target_phrase_len = np.mean([len(x["target_word"].split()) for x in trainset])

        print ("extracting all features...")

        feature_index = {} # initialise the vector as a dict

        # get features for every example in train set
        for instance in trainset:
            for feature in self.extract_features(instance):
                feature_index[feature] = 0

        feature_vectorizer = DictVectorizer()
        feature_vectorizer.fit_transform(feature_index)

        print ("features extracted.")

        return feature_vectorizer


    def extract_features(self, instance):

        x_features = Counter()

        target_word = instance["target_word"]
        spacy_tokens = instance["spacy_tokens"]

        x_features["NUM_CHARS"] = len(target_word) / self.avg_word_length # character count feature
        x_features["NUM_TOKENS"] = len(target_word.split(' ')) /self.avg_target_phrase_len # token count feature
        x_features["FIRST_CHAR_UPPER"] += int(target_word[0].isupper()) # leading character upper case feature

        for token in spacy_tokens:

            x_features["LEMMA_" + token.lemma_] +=1 /len(spacy_tokens) # lemmatised token feature
            x_features["SHAPE_" + token.shape_] +=1 /len(spacy_tokens) # word shape feature
            x_features["NERTYPE_" + token.ent_type_] +=1 /len(spacy_tokens) # named entity type feature
            x_features["TAG_" + token.tag_] +=1 /len(spacy_tokens) # fine-grained POS tag feature
            x_features["AMBIGUITY"] += len(wn.synsets(token.text))/ (self.avg_synset * len(spacy_tokens)) # word ambiguity feature

            ## REJECTED FEATURES ###############################################
            # x_features["WORD_LOWER" + token.lower_] +=1 /len(spacy_tokens)
            # x_features["WORD_" + token.text] +=1 /len(spacy_tokens)
            ####################################################################

        return x_features
        
    def train(self, trainset):

        self.feature_vectorizer = self.extract_all_features(trainset)
        X = np.zeros((len(trainset), len(self.feature_vectorizer.vocabulary_)))

        print ("dim X: ", X.shape)

        y = []

        for i in range(len(trainset)):
            instance = trainset[i]
            x_features = self.extract_features(instance)

            x = np.hstack([self.feature_vectorizer.transform(x_features).A[0]])
            X[i,:] = x

            y.append(instance['gold_label'])

        print ("training model")
        self.model.fit(X, y)
        print ("model trained. making preds")


    def test(self, testset):
        X = np.zeros((len(testset), len(self.feature_vectorizer.vocabulary_)))

        for i in range(len(testset)):
            sent = testset[i]

            if i %200 == 0: print ("\rtested... %d" %i, end = '\r')
            x_features = self.extract_features(sent)

            x = np.hstack([self.feature_vectorizer.transform(x_features).A[0]])
            X[i,:] = x

        return self.model.predict(X)

################################################################################

class Spanish_Model(object):

    def __init__(self, language, classifier):
        self.name = "Spanish Model"
        self.language = language

        # from 'Multilingual and Cross-Lingual Complex Word Identification' (Yimam et. al, 2017)
        self.avg_word_length = 6.2
        self.model = classifier
        self.freq_index = {}

        self.esp_freqs = {}

        with open("spanish_subtitle_words_frequency_indexes.txt", "r") as f:
            for line in f.readlines():
                wd = line.split(",")[0]
                FI = int(line.split(",")[1])

                self.freq_index[wd] = FI

    def extract_all_features(self, trainset):
        print ("extracting all features...")

        self.word_index = Counter()
        self.char_tri_index = Counter()

        feature_index = {} # initialise the vector as a dict

        # get features for every example in train set
        for instance in trainset:
            for feature in self.extract_features(instance):
                feature_index[feature] = 0

            # building an index of the words for frequency analysis
            for token in instance["sentence"].split():
                self.word_index[token] += 1
                for i in range(len(token)-3):
                    self.char_tri_index[token[i:i+3]] += 1

        feature_vectorizer = DictVectorizer()
        feature_vectorizer.fit_transform(feature_index)

        print ("features extracted.")

        return feature_vectorizer


    def extract_features(self, instance):

        x_features = Counter()

        sentence = instance["sentence"]
        target_word = instance["target_word"]
        spacy_tokens = instance["spacy_tokens"]
        spacy_target_word = instance["spacy_target_word"]

        sentence_counts = Counter(sentence.split())

        # character count feature
        x_features["NUM_CHARS"] = len(target_word) / self.avg_word_length
        # token count feature
        x_features["NUM_TOKENS"] = len(target_word.split(' '))

        for token in spacy_tokens:

            x_features["LEMMA_" + token.lemma_] +=1/len(spacy_tokens)
            x_features["SHAPE_" + token.shape_] += 1/len(spacy_tokens)




            if token.lemma_ in self.freq_index:
                x_features["FI_" + str(self.freq_index[token.lemma_])] += 1/len(spacy_tokens)
            else:
                x_features["FI_0"] += 1/len(spacy_tokens)


            # x_features["WORD_" + token.text] +=1/len(spacy_tokens)

        return x_features



    def embedding(self, instance):

        target_word = instance["target_word"]
        spacy_tokens = instance["spacy_tokens"]
        spacy_target_word = instance["spacy_target_word"]

        embeddings = []

        for token in spacy_tokens:
            embeddings.append(token.vector)
        if len(embeddings)>1:
            out = np.mean(np.vstack(embeddings), axis=0)
        elif len(embeddings) ==1:
            out = embeddings[0]
        else:
            out = np.zeros(50,)
        return (out)

    def train(self, trainset):

        self.feature_vectorizer = self.extract_all_features(trainset)
        X = np.zeros((len(trainset), len(self.feature_vectorizer.vocabulary_)))


        # X = np.zeros((len(trainset), len(self.feature_vectorizer.vocabulary_) + 50)) ##would need to extend this by embedding (this is the 384)

        print ("dim X: ", X.shape)

        y = []

        for i in range(len(trainset)): # change sent >> instance
            sent = trainset[i]
            x_features = self.extract_features(sent) ## could add on an embedding here if wanted.

            # x = np.hstack([self.feature_vectorizer.transform(x_features).A[0], self.embedding(sent)])
            x = np.hstack([self.feature_vectorizer.transform(x_features).A[0]])

            X[i,:] = x

            y.append(sent['gold_label'])

        print ("dim X: ", X.shape)
        self.model.fit(X, y)

    def test(self, testset):

        X = np.zeros((len(testset), len(self.feature_vectorizer.vocabulary_)))
        for i in range(len(testset)):
            sent = testset[i]
            if i %200 == 0: print ("\rtested... %d" %i, end = '\r')

            x_features = self.extract_features(sent)

            x = self.feature_vectorizer.transform(x_features).A[0]

            X[i,:] = x

        return self.model.predict(X)
