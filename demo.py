from utils.evaluation import report_score, sample_errors
from utils.dataset import Dataset
from model import English_Model, Spanish_Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.linear_model import LogisticRegression
import numpy as np

def execute_demo(language, model, classifier):
    cwi_model = model(language, classifier)

    if language == "english":
        dataset = Dataset("english") #switch these out later
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
    #     sample_errors(dataset.devset, predictions)
    #     sample_errors(dataset.testset, predictions)


    print ("*************************************************")



### RUN THE MODEL
def english_cross_validation():
    # Number of trees in random forest
    n_estimators = [10, 25, 50, 75, 100, 125, 150, 175, 200, 250, 300, 500]
    max_features = ['auto', "sqrt"]
    max_depth = [None, 10, 15, 20, 25, 30]
    min_samples_split = [2, 4, 6, 8]
    min_samples_leaf = [1,2,3,4]
    bootstrap = [True, False]

    ## delete the line below
    # n_estimators = [10]
    # max_features = ['auto']
    # max_depth = [None]
    # min_samples_split = [2]
    # min_samples_leaf = [1]
    # bootstrap = [True]

    # Create the random grid
    param_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    rf = RandomForestClassifier(random_state=0)
    rf_random = RandomizedSearchCV(estimator = rf,
                                   param_distributions = param_grid,
                                   n_iter = 1,
                                   cv = 5,
                                   verbose=4,
                                   random_state=0,
                                   n_jobs = -1,
                                   scoring = "f1_macro")

    execute_demo("english", English_Model, rf_random)

    print ("best parameters:")
    print (rf_random.cv_results_['params'][rf_random.best_index_])


def spanish_cross_validation():
    # Number of trees in random forest
    penalty = ["l1", "l2"]
    C = np.linspace(0.01, 1, num=100).tolist()

    # Create the random grid
    param_grid = {'penalty': penalty,
                   "C": C}

    rf = LogisticRegression(solver="saga", n_jobs = -1, max_iter = 1000)
    rf_random = RandomizedSearchCV(estimator = rf,
                                   param_distributions = param_grid,
                                   n_iter = 1,
                                   cv = 3,
                                   verbose=4,
                                   random_state=0,
                                   n_jobs = -1,
                                   scoring = "f1_macro")

    execute_demo("spanish", Spanish_Model, rf_random)

    print ("best parameters:")
    print (rf_random.cv_results_['params'][rf_random.best_index_])



##### RUN PROGRAM #####

spanish_cross_validation()
english_cross_validation()
