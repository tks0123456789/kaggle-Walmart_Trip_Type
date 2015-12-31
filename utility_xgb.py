import xgboost as xgb
import numpy as np

# Returns: all the cols of X used by xgboost
def feature_selection(X, y, xgb_params, num_round):
    dtrain = xgb.DMatrix(X, label = y)
    bst = xgb.train(xgb_params, dtrain, num_round)
    imp = bst.get_fscore() #  {'f1':123, 'f12':344, 'f15':131, ..}
    keys = imp.keys()
    cols = np.sort([int(k[1:]) for k in keys])
    return cols
