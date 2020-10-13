import pandas as pd
from hyperopt import STATUS_OK
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, KFold
import numpy as np


df = pd.read_csv('finale_Dzeko_10oct.csv')

X = df.drop('xG', axis = 1).copy()
y = df['xG'].copy()

def objective(params, n_folds=10):

    xgbr = XGBRegressor(max_depth=int(list(params.values())[3]),
                        learning_rate=float(list(params.values())[2]),
                        subsample=float(list(params.values())[8]),
                        colsample_bytree=float(list(params.values())[0]),
                        gamma=float(list(params.values())[1]),
                        n_estimators=int(list(params.values())[5]),
                        reg_alpha=float(list(params.values())[6]),
                        reg_lambda=float(list(params.values())[7]),
                        min_child_weight=int(list(params.values())[4])
                        )

    # Perform n_fold cross validation with hyperparameters

    kfold = KFold(n_splits=n_folds, shuffle=True)
    kf_cv_scores = cross_val_score(xgbr, X, y, cv=kfold, scoring='neg_mean_squared_error')
    score = - kf_cv_scores.mean()

    # Dictionary with information for evaluation
    return {'loss': score, 'params': params, 'status': STATUS_OK}

from hyperopt import hp

space = {
    'max_depth': hp.quniform('max_depth', 6, 25, 2),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.001), np.log(0.2)),
    'subsample': hp.loguniform('subsample', np.log(0.1), np.log(1.0)),
    'colsample_bytree': hp.loguniform('colsample_bytree', np.log(0.8), np.log(1.0)),
    'gamma': hp.loguniform('gamma', np.log(0.01), np.log(3.0)),
    'n_estimators': hp.quniform('n_estimators', 100, 10000, 90),
    'reg_alpha': hp.loguniform('reg_alpha', np.log(0.01), np.log(1.0)),
    'reg_lambda': hp.loguniform('reg_lambda', np.log(1.0), np.log(4.5)),
    'min_child_weight': hp.quniform('min_child_weight', 1, 7, 1)

}


# Optimization algorithm

from hyperopt import tpe

# Algorithm
tpe_algorithm = tpe.suggest

# Now it is time to optimize

from hyperopt import fmin

MAX_EVALS = 1150

# Optimize
best = fmin(fn = objective, space = space, algo = tpe.suggest, max_evals = MAX_EVALS)

def objective2(params):
    print(params)

    return 1