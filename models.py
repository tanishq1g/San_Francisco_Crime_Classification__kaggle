import xgboost as xgb
from sklearn.feature_extraction import text
from sklearn import preprocessing
from sklearn.metrics import log_loss, accuracy_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics.classification import accuracy_score
from sklearn.ensemble import VotingClassifier, ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier

class Model:
    
    def __init__(self):
        self.curr_model = None
        self.predictions = None

    def score_calc(self, predictions, test_data[labels], score_type = 'logloss'):
        if (score_type == 'logloss'):
            print(("Log loss: ") + str(log_loss(test_data[labels], predictions)))
        elif (score_type == 'accuracy'):
            print(("Accuracy score: ") + str(accuracy_score(test_data[labels], predictions)))
        else
            print("-")

    def data_split(self, train, split_ratio):
        train_data, test_data = train_test_split(train, train_size = split_ratio)
        return train_data, test_data
    
    #documentation link - https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression
    def lr_clf(self, train_data, test_data, features, labels, max_iter = 50, solver = 'sag', 
    print_score = True, score_type = 'logloss'):
        self.curr_model = LogisticRegression(
            random_state = 200, 
            solver = solver,  #optimisation algo
            multi_class = 'multinomial',
            penalty = 'l2',
            max_iter = max_iter,
            verbose = 1,
            n_jobs = -1
        )
        self.curr_model.fit(train_data[features], train_data[labels])
        predictions = np.array(self.curr_model.predict_proba(test_data[features]))
        self.predictions = predictions
        if (print_score == True):
            score_calc(predictions, test_data[labels], score_type)
        
    #documentation link - https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html#sklearn.naive_bayes.BernoulliNB
    def bernb_clf(self, train_data, test_data, features, labels, print_score = True, 
    score_type = 'logloss'):
        self.curr_model = BernoulliNB()
        self.curr_model.fit(train_data[features], train_data[labels])
        predictions = np.array(self.curr_model.predict_proba(test_data[features]))
        self.predictions = predictions
        if (print_score == True):
            score_calc(predictions, test_data[labels], score_type)
        
    #documentation link - https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html#sklearn.linear_model.SGDClassifier
    def sgd_clf(self, train_data, test_data, features, labels, 
    loss = 'log', penalty = 'l2', alpha = 0.001, l1_ratio = 0.15, fit_intercept = True, 
    max_iter = None, tol = None, shuffle = True, verbose = 1, epsilon = 0.1, 
    learning_rate = 'optimal', eta0 = 0.0, power_t = 0.5, early_stopping = False, 
    validation_fraction = 0.1, n_iter_no_change = 5, class_weight = None, warm_start = False, 
    average = False, max_iter = 50, print_score = True, score_type = 'logloss'):
        self.curr_model = SGDClassifier(
            random_state = 200,
            loss = loss,
            verbose = 1,
            learning_rate = learning_rate,
            penalty = penalty,
            tol = tol,
            shuffle = shuffle,
            average = average,
            alpha = alpha,
            max_iter = max_iter,
            n_jobs = -1,
            validation_fraction = validation_fraction,
            n_iter_no_change = n_iter_no_change,
            class_weight = class_weight,
            warm_start = warm_start,
            l1_ratio = l1_ratio,
            fit_intercept = fit_intercept,
            epsilon = epsilon,
            eta0 = eta0,
            power_t = power_t,
            early_stopping = early_stopping
        )
        self.curr_model.fit(train_data[features], train_data[labels])
        predictions = np.array(self.curr_model.predict_proba(test_data[features]))
        self.predictions = predictions
        if (print_score == True):
            score_calc(predictions, test_data[labels], score_type)

    #documentation link - https://xgboost.readthedocs.io/en/latest/python/python_api.html
    def xgb_clf(self, train_data, test_data, features, labels, max_depth = 3, learning_rate = 0.1, 
    n_estimators = 100, silent = True, objective = 'binary:logistic', booster = 'gbtree', nthread = None, 
    gamma = 0, min_child_weight = 1, max_delta_step = 0, subsample = 1, colsample_bytree = 1, 
    colsample_bylevel = 1, reg_alpha = 0, reg_lambda = 1, scale_pos_weight = 1, base_score = 0.5, 
    early_stopping_rounds = 5, eval_metric = 'mlogloss', print_score = True, score_type = 'logloss'):
        self.curr_model = xgb.XGBClassifier(
            random_state = 200,
            max_depth = max_depth,
            learning_rate = learning_rate,
            n_estimators = n_estimators,
            silent = silent,
            objective = objective,
            booster = booster,
            reg_alpha = reg_alpha,
            reg_lambda = reg_lambda,
            n_jobs = -1,
            nthread = nthread,
            gamma = gamma,
            min_child_weight = min_child_weight,
            max_delta_step = max_delta_step,
            colsample_bylevel = colsample_bylevel,
            colsample_bytree = colsample_bytree, 
            scale_pos_weight = scale_pos_weight,
            base_score = base_score,
            subsample = subsample
        )

        self.curr_model.fit(
            X = train_data[features],
            y = train_data[labels],
            verbose = True,
            early_stopping_rounds = early_stopping_rounds,
            eval_set = [(test_data[features], test_data[labels])],
            eval_metric = eval_metric
        )
        predictions = np.array(self.curr_model.predict_proba(test_data[features]))
        self.predictions = predictions
        if (print_score == True):
            score_calc(predictions, test_data[labels], score_type)
        
            
    # documentation link - https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier
    def randfor_clf(self, train_data, test_data, features, labels, n_estimators = 100, 
    criterion = 'gini', max_depth = None, min_samples_split = 2, min_samples_leaf = 1, 
    min_weight_fraction_leaf = 0.0, max_features = 'auto', max_leaf_nodes = None, 
    min_impurity_decrease = 0.0, bootstrap = True, oob_score = False, warm_start = False, 
    class_weight = None, print_score = True, score_type = 'logloss'):
        self.curr_model = RandomForestClassifier(
            random_state = 200, 
            criterion = criterion,
            max_depth = max_depth,
            max_features = max_features,
            max_leaf_nodes = max_leaf_nodes,
            min_impurity_decrease = min_impurity_decrease,
            min_samples_leaf = min_samples_leaf,
            min_samples_split = min_samples_split,
            min_weight_fraction_leaf = min_weight_fraction_leaf,
            bootstrap = bootstrap,
            oob_score = oob_score,
            warm_start = warm_start,
            class_weight = class_weight,
            n_estimators = n_estimators,
            verbose = 1,
            n_jobs = -1
        )
        self.curr_model.fit(train_data[features], train_data[labels])
        predictions = np.array(self.curr_model.predict_proba(test_data[features]))
        self.predictions = predictions
        if (print_score == True):
            score_calc(predictions, test_data[labels], score_type)
