from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

### DEFINE CROSS VALIDATION FUNCTIONS FOR USE IN HYPERPARAMETER TUNING #########

def english_cross_validation(model):
    # Change the options here for a different search
    n_estimators = [10, 25, 50, 75, 100, 125, 150, 175, 200]
    max_features = ['auto', "sqrt"]
    max_depth = [None, 10, 15, 20, 25, 30]
    min_samples_split = [2, 4, 6, 8]
    min_samples_leaf = [1,2,3,4]
    bootstrap = [True, False]

    # Create the param grid
    param_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    cv = RandomizedSearchCV(estimator = model,
                                   param_distributions = param_grid,
                                   n_iter = 100,
                                   cv = 3,
                                   verbose=4,
                                   random_state=0,
                                   n_jobs = -1,
                                   scoring = "f1_macro")

    return cv

def spanish_cross_validation(model):
    # Change the options here for a different search
    penalty = ["l1", "l2"]
    C = [0.5, 1, 2.5, 5, 10]

    # Create the param grid
    param_grid = {'penalty': penalty,
                   "C": C}

    # model =
    cv = RandomizedSearchCV(estimator = model,
                                   param_distributions = param_grid,
                                   n_iter = 10,
                                   cv = 3,
                                   verbose=4,
                                   random_state=0,
                                   n_jobs = -1,
                                   scoring = "f1_macro")

    return cv
