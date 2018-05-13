from utils.evaluation import report_score, sample_errors
from utils.dataset import Dataset
from utils.cross_validation import spanish_cross_validation, english_cross_validation
from model import English_Model, Spanish_Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import sys
import numpy as np

### DEFINE FUNCTION TO EXECUTE TRAIN AND TEST ON THE DATASET ###################
def execute_demo(language, model, classifier, error_sampling=False):
    cwi_model = model(language, classifier)

    if language == "english":
        dataset = Dataset("english")
    else: #spanish
        dataset = Dataset("spanish")

    print("\nmodel: {}".format(cwi_model.name))
    print("{}: {} training - {} dev - {} test".format(language,
                                                      len(dataset.trainset),
                                                      len(dataset.devset),
                                                      len(dataset.testset)))

    cwi_model.train(dataset.trainset)


    #dev set eval
    devset_gold_labels = [instance['gold_label'] for instance in dataset.devset]
    devset_preds = cwi_model.test(dataset.devset)
    print ("\nevaluation on dev set")
    report_score(devset_gold_labels, devset_preds)


    ## test set eval
    testset_preds = cwi_model.test(dataset.testset)
    testset_gold_labels = [instance['gold_label'] for instance in dataset.testset]
    print ("\nevaluation on test set:")
    report_score(testset_gold_labels, testset_preds)


    ## trainset eval
    trainset_preds = cwi_model.test(dataset.trainset)
    trainset_gold_labels = [instance['gold_label'] for instance in dataset.trainset]
    print ("\nevaluation on training set:")
    report_score(trainset_gold_labels, trainset_preds)

    ## ERROR SAMPLING ##
    if error_sampling == True:
        sample_errors(dataset.devset, predictions)
        sample_errors(dataset.testset, predictions)

    print ("**************************************************")
################################################################################

##### RUN PROGRAM #####
if __name__ == '__main__':
    args = sys.argv[1:]

    if "cv" in args:
        # Load the data, train and test the English model with hyperparameter ##
        # tuning using cross-validation, report the results ####################
        eng_cv = english_cross_validation(RandomForestClassifier(random_state=0))
        execute_demo("english", English_Model, eng_cv)
        print ("best parameters:")
        print (eng_cv.cv_results_['params'][eng_cv.best_index_])

        # Load the data, train and test the Spanish model with hyperparameter ##
        # tuning using cross-validation, report the results ####################
        esp_cv = spanish_cross_validation(LogisticRegression(solver="saga", n_jobs = -1, max_iter = 1000))
        execute_demo("spanish", Spanish_Model, esp_cv)
        print ("best parameters:")
        print (esp_cv.cv_results_['params'][esp_cv.best_index_])

    else:
        # Load the data, train and test the English model, report the results
        rf = RandomForestClassifier(random_state=0, n_estimators=75,
                                    min_samples_split=8, min_samples_leaf=1,
                                    max_features="sqrt", max_depth=None,
                                    bootstrap=False)
        execute_demo('english', English_Model, rf)

        # Load the data, train and test the Spanish model, report the results
        lr = LogisticRegression(penalty="l2", C=1)
        execute_demo('spanish', Spanish_Model, lr)
