"""
Target parameters: max_epochs
Feature selection by xgb + NN(2 hidden layers), useUpc:False
Time: 26h
The running time on Ubuntu 14.04 LTS[i7 4790k, 32G MEM, GTX660]
"""

import numpy as np
import scipy as sp
import pandas as pd
import xgboost as xgb
from datetime import datetime
from sklearn.metrics import log_loss

from lasagne.layers import InputLayer, DropoutLayer, DenseLayer

from utility_common import feature_extraction, data_path
from utility_nn import build_net_sparse_input
from utility_xgb import feature_selection

# r096
# 2015/12/23-24 1 day, 1:47:32.535909
# (h1,h2):(60, 100), (70, 90)
# CV pred
# Feature selection by XGB
# Shuffle data, No scaling, No nomalizing
# NN 2 hidden layers
# params: epochs (=max_epochs)

# ncol:13916
X4, target, v_train, v_test = feature_extraction(training, test, useUpc=False)
N = X4.shape[0]
X4 = X4[v_train-1]

# params for xgb
nt = 400
tc = 6
sh = .2
cs = .4
bf = .8
xgb_params = {'eta':sh, 'silent':1, 'objective':'multi:softprob', 'num_class':38,
              'max_depth':tc, 'colsample_bytree':cs, 'subsample':bf,
              'eval_metric':'mlogloss', 'nthread':8}

# params for nn
nModels = 20
epochs = 10
nIter = 42
# total epochs=epochs*nIter
lr = .02
mm = .2
p = .1
bs = 256


h1h2_lst = [{'h1':60, 'h2':100}, {'h1':70, 'h2':90}]
pr_nn_dict = {(params['h1'], params['h2']):[np.zeros((N/2, 38)) for _ in range(nIter)] \
              for params in h1h2_lst}
scores = []
t0 = datetime.now()
for fold, idx in enumerate(kf):
    train_idx, valid_idx = idx
    params_dict = {'fold':fold}
    for j in range(nModels):
        seed = 11281*j + 9108
        xgb_params['seed'] = seed
        cols = feature_selection(X4[train_idx], y[train_idx], xgb_params, nt)
        X = X4.tocsc()[:,cols].tocsr().astype(np.float32)
        for params in h1h2_lst:
            h1 = params['h1']
            h2 = params['h2']
            params_dict.update(params)
            pr_avg = pr_nn_dict[(h1, h2)]
            np.random.seed(seed)
            net1 = build_net_sparse_input(h1, h2, p, mm, max_epochs=epochs, num_in=X.shape[1],
                                    shuffle=True, eval_size=False)
            for k in range(nIter):
                net1.fit(X[train_idx], target[train_idx])
                pr = net1.predict_proba(X[valid_idx])
                pr_avg[k][valid_idx] += pr
                sc = params_dict.copy()
                sc.update({'epochs':epochs*(k+1), 'nModels':j+1,
                           'each':log_loss(target[valid_idx], pr),
                           'avg':log_loss(target[valid_idx], pr_avg[k][valid_idx]/(j+1))})
                scores.append(sc)
                print scores[-1], datetime.now() - t0

pr_nn_dict = {key:[pr/nModels for pr in val] for key, val in pr_nn_dict.iteritems()}

output = open(data_path + 'pr_nn096.pkl', 'wb')
pickle.dump(pr_nn_dict, output)
output.close()

r096 = pd.DataFrame(scores)
r096.to_csv('logs/r096.csv')

params = ['epochs', 'h1']
tmp = r096[r096.nModels==nModels].groupby(params).avg.mean().unstack()
r096_summary = tmp.join(r096.groupby(params).each.mean().unstack(), lsuffix='avg')
r096_summary.to_csv('logs/r096_summary.csv')

# h1        60avg    70avg       60       70
# epochs                                    
# 10      1.15125  1.13743  1.17607  1.16236
# 20      0.95668  0.94878  0.98184  0.97295
# 30      0.87463  0.86904  0.89940  0.89301
# 40      0.82361  0.81848  0.84922  0.84341
# 50      0.78531  0.78070  0.81242  0.80699
# 60      0.75493  0.75016  0.78406  0.77839
# 70      0.72993  0.72541  0.76065  0.75573
# 80      0.70898  0.70495  0.74258  0.73747
# 90      0.69139  0.68774  0.72781  0.72311
# 100     0.67697  0.67347  0.71653  0.71231
# 110     0.66444  0.66130  0.70752  0.70417
# 120     0.65427  0.65189  0.70070  0.69822
# 130     0.64581  0.64336  0.69645  0.69395
# 140     0.63863  0.63669  0.69438  0.69170
# 150     0.63261  0.63150  0.69227* 0.69151
# 160     0.62857  0.62767  0.69288  0.69123*
# 170     0.62478  0.62403  0.69390  0.69431
# 180     0.62229  0.62128  0.69708  0.69738
# 190     0.61952  0.61950  0.70056  0.70114
# 200     0.61795  0.61779  0.70376  0.70510
# 210     0.61717  0.61650* 0.71076  0.71096
# 220     0.61619  0.61674  0.71524  0.71566
# 230     0.61546* 0.61687  0.72208  0.72308
# 240     0.61608  0.61752  0.72810  0.72959
# 250     0.61598  0.61802  0.73326  0.73591
# 260     0.61647  0.61826  0.74046  0.74594
# 270     0.61753  0.61933  0.74769  0.75180
# 280     0.61786  0.62040  0.75558  0.75975
# 290     0.61836  0.62243  0.76415  0.76839
# 300     0.62003  0.62356  0.77196  0.77736
# 310     0.62038  0.62493  0.77810  0.78279
# 320     0.62278  0.62672  0.78843  0.79139
# 330     0.62321  0.62799  0.79354  0.80004
# 340     0.62479  0.62965  0.80367  0.81201
# 350     0.62568  0.63165  0.81175  0.81790
# 360     0.62721  0.63301  0.81956  0.82653
# 370     0.62977  0.63477  0.82804  0.83328
# 380     0.62961  0.63576  0.83590  0.84313
# 390     0.63131  0.63792  0.84264  0.85038
# 400     0.63335  0.63927  0.85234  0.85886
# 410     0.63425  0.64007  0.86152  0.86710
# 420     0.63500  0.64212  0.86660  0.87358
