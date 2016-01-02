"""
Target parameters: max_depth(tc), num_round(nt)
XGB
useUpc:True
Time: 14h20m
The running time on Ubuntu 14.04 LTS[i7 4790k, 32G MEM, GTX660]
"""

import numpy as np
import scipy as sp
import pandas as pd
from datetime import datetime
import xgboost as xgb
from sklearn.metrics import log_loss

from utility_common import feature_extraction, data_path

# r087
# 2015/12/16 14h20m
# Ensemble
# XGB
# params: nt (=num_round)

# ncol: 138610
X1, target, v_train, v_test = feature_extraction(useUpc=True)
y = pd.get_dummies(target).values.argmax(1)

X1 = X1[v_train-1]

nModels = 10

sh = .2
cs = .4
bf = .8
xgb_params = {'eta':sh, 'silent':1, 'objective':'multi:softprob', 'num_class':38,
              'colsample_bytree':cs, 'subsample':bf,
              'eval_metric':'mlogloss', 'nthread':8}

nt_dict = {4:range(500, 951, 50), 5:range(300, 701, 50)}

pr_xgb_dict = {key:[np.zeros((v_train.size, 38)) for _ in range(len(nt_lst))] \
               for key, nt_lst in nt_dict.iteritems()}
scores = []
t0 = datetime.now()
for fold, idx in enumerate(kf):
    train_idx, valid_idx = idx
    dtrain = xgb.DMatrix(X1[train_idx], label = y[train_idx])
    dvalid = xgb.DMatrix(X1[valid_idx])    
    for tc in [4, 5]:
        nt_lst = nt_dict[tc]
        nt = np.max(nt_lst)
        xgb_params['max_depth'] = tc
        params = {'tc':tc, 'fold':fold}
        for i in range(1, nModels+1):
            params.update({'nModels':i})
            xgb_params['seed'] = 13913*i+32018
            bst = xgb.train(xgb_params, dtrain, nt)
            for j, ntree in enumerate(nt_lst):
                pr = bst.predict(dvalid, ntree_limit = ntree)
                pr_xgb_dict[tc][j][valid_idx] += pr
                sc = params.copy()
                sc.update({'ntree':ntree,
                           'each':log_loss(y[valid_idx], pr),
                           'avg':log_loss(y[valid_idx], pr_xgb_dict[tc][j][valid_idx]/i)})
                scores.append(sc)
            print scores[-1], datetime.now() - t0

pr_xgb_dict = {key:[pr/nModels for pr in pr_lst] for key, pr_lst in pr_xgb_dict.iteritems()}

output = open(data_path + 'pr_xgb087.pkl', 'wb')
pickle.dump(pr_xgb_dict, output)
output.close()

r087 = pd.DataFrame(scores)
r087.to_csv('logs/r087.csv')

params = ['tc', 'ntree']

grouped = r087.groupby(params)
pd.DataFrame({'avg':grouped.avg.mean().unstack().min(1),
              'ntree':grouped.avg.mean().unstack().idxmin(1)})
#         avg  ntree
# tc                
# 4   0.64834    750
# 5   0.64744*   550

pd.DataFrame({'each':grouped.each.mean().unstack().min(1),
              'ntree':grouped.each.mean().unstack().idxmin(1)})
#        each  ntree
# tc                
# 4   0.66263    700
# 5   0.66407    550
