from hyperopt import hp
import numpy as np

xgb_classifier_space = {
    #'booster': hp.choice('booster', ['gbtree', 'gblinear', 'dart']),
    'eta': hp.uniform('eta', 0.01, 0.5),
    'max_depth': hp.choice('max_depth', np.arange(2, 10, dtype=int)),
    'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
    'subsample': hp.uniform('subsample', 0.7, 1.0),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.7, 1.0),
    'gamma': hp.uniform('gamma', 0, 1.0),
    'lambda': hp.uniform('lambda', 1, 2),
    'alpha': hp.uniform('alpha', 0, 2),
    'scale_pos_weight': hp.uniform('scale_pos_weight', 0, 2),
    'n_estimators': hp.choice('n_estimators', np.arange(50, 1200, 50, dtype=int)),
    'early_stopping_rounds': hp.choice('early_stopping_rounds', np.arange(10, 40, 5, dtype=int))
}

xgb_regressor_space = {
    #'booster': hp.choice('booster', ['gbtree', 'gblinear', 'dart']),
    'eta': hp.uniform('eta', 0.01, 0.5),
    'max_depth': hp.choice('max_depth', np.arange(2, 10, dtype=int)),
    'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
    'subsample': hp.uniform('subsample', 0.7, 1.0),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.7, 1.0),
    'gamma': hp.uniform('gamma', 0, 1.0),
    'lambda': hp.uniform('lambda', 1, 2),
    'alpha': hp.uniform('alpha', 0, 2),
    'n_estimators': hp.choice('n_estimators', np.arange(50, 1200, 50, dtype=int)),
    'early_stopping_rounds': hp.choice('early_stopping_rounds', np.arange(10, 40, 5, dtype=int))
}