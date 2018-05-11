import sklearn
# report_score code provided by Fernando Manchego
def report_score(gold_labels, predicted_labels, detailed=True):
    macro_F1 = sklearn.metrics.f1_score(gold_labels, predicted_labels, average='macro')
    print("macro-F1: {:.3f}".format(macro_F1))
    if detailed:
        scores = sklearn.metrics.precision_recall_fscore_support(gold_labels, predicted_labels)
        print("{:^10}{:^10}{:^10}{:^10}{:^10}".format("Label", "Precision", "Recall", "F1", "Support"))
        print('-' * 50)
        print("{:^10}{:^10.2f}{:^10.2f}{:^10.2f}{:^10}".format(0, scores[0][0], scores[1][0], scores[2][0], scores[3][0]))
        print("{:^10}{:^10.2f}{:^10.2f}{:^10.2f}{:^10}".format(1, scores[0][1], scores[1][1], scores[2][1], scores[3][1]))
    print()

import numpy as np
import random


## DEFINE A FUNCTION TO RANDOMLY SAMPLE FROM CONFUSION MATRIX FOR ERROR ANALYSIS
def sample_errors(test_set, predictions, sample_size=20):

    # positive class = complex, neg class = simple

    words = [instance['target_word'] for instance in test_set]
    labels = [instance['gold_label'] for instance in test_set]

    results = np.vstack([words, labels, predictions])

    false_pos = []
    false_neg = []
    true_pos = []
    true_neg = []

    for i in range(len(words)):

        if labels[i] != predictions[i]:
            if predictions[i] == "0":
                false_neg.append(words[i])
            elif predictions[i] == "1":
                false_pos.append(words[i])

        elif labels[i] == predictions[i]:
            if predictions[i] == "0":
                true_neg.append(words[i])
            elif predictions[i] == "1":
                true_pos.append(words[i])

    false_neg_sample = random.sample(false_neg, 20)
    false_pos_sample = random.sample(false_pos, 20)
    true_neg_sample = random.sample(true_neg, 20)
    true_pos_sample = random.sample(true_pos, 20)


    print ("\n--- Sample from {} False Negative Results ---".format(len(false_neg)))
    for instance in false_neg_sample:
        print (instance)

    print ("\n--- Sample from {} True Positive Results ---".format(len(true_pos)))
    for instance in true_pos_sample:
        print (instance)

    print ("\n--- Sample from {} True Negative Results ---".format(len(true_neg)))
    for instance in true_neg_sample:
        print (instance)

    print ("\n--- Sample from {} False Positive Results ---".format(len(false_pos)))
    for instance in false_pos_sample:
        print (instance)
