import csv
import spacy

# Main Dataset class built upon code provided by Fernando Manchego / University of Sheffield
class Dataset(object):

    def __init__(self, language):
        self.language = language

        trainset_path = "datasets/{}/{}_Train.tsv".format(language, language.capitalize())
        devset_path = "datasets/{}/{}_Dev.tsv".format(language, language.capitalize())
        testset_path = "datasets/{}/{}_Test.tsv".format(language, language.capitalize())

        # Loading the spaCy model for the language
        if language == "english":
            self.nlp = spacy.load("en_core_web_lg")
        elif language == "spanish":
            self.nlp = spacy.load("es_core_news_md")

        # Reading the datasets
        print ("Generating spaCy objects for training instances...")
        self.trainset = self.read_dataset(trainset_path)
        print ("Generating spaCy objects for development instances...")
        self.devset = self.read_dataset(devset_path)
        print ("Generating spaCy objects for test instances...")
        self.testset = self.read_dataset(testset_path)


    def read_dataset(self, file_path):
        with open(file_path) as file:
            fieldnames = ['hit_id', 'sentence', 'start_offset', 'end_offset', 'target_word', 'native_annots',
                          'nonnative_annots', 'native_complex', 'nonnative_complex', 'gold_label', 'gold_prob']
            reader = csv.DictReader(file, fieldnames=fieldnames, delimiter='\t')

            dataset = [sent for sent in reader][:100]

        ## Create spacy doc for each sentence and target word
        i = 0 # counter for progress bar
        for instance in dataset:

            printProgressBar(i + 1, len(dataset), prefix = 'Progress:', suffix = 'Complete', length = 50)
            i+=1

            spacy_sentence = self.nlp(instance["sentence"])
            spacy_target_word = self.nlp(instance["target_word"])

            instance["spacy_sentence"] = spacy_sentence
            instance["spacy_target_word"] = spacy_target_word
            instance["spacy_tokens"] = self.get_spacy_tokens(spacy_sentence, spacy_target_word)

        return dataset

    # Function get_spacy_tokens finds the words from the target word and uses
    # their tokens from the HIT context.
    def get_spacy_tokens(self, spacy_sentence, spacy_target_word):

        spacy_token_list = []

        for target in spacy_target_word:
            for wd in spacy_sentence:
                if target.text == wd.text:
                    spacy_token_list.append(wd)
                    break

        return spacy_token_list


# PrintProgressBar function code taken from a Stack Overflow question
# https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console/27871113
# Question by bobber205:
# https://stackoverflow.com/users/186808/bobber205
# Answer by Greenstick:
# https://stackoverflow.com/users/2206251/greenstick

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total:
        print()
