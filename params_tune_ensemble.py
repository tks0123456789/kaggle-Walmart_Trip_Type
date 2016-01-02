"""
Target parameters: max_epochs
.3*(NN([6k+, 60, 100, 38])_avg + NN([6k+, 70, 90, 38]))_avg +
  .4*XGB(max_depth:5, num_round550)_avg
"""

import pandas as pd
import cPickle as pickle
from utility_common import data_path, file_train
from sklearn.metrics import log_loss

nModels = 20

training = pd.read_csv(file_train)
grouped_train = training.groupby('VisitNumber')
target = grouped_train.TripType.first().values

f = open(data_path + 'pr_nn096.pkl')
pr_nn_dict = pickle.load(f)
f.close()

f = open(data_path + 'pr_xgb087.pkl')
pr_xgb_all = pickle.load(f)
f.close()

# max_depth:5, num_round:550
pr_xgb_5 = pr_xgb_all[5][5]


# h1,h2:(60,100), (70,90)
scores = {'%dnn_xgb' % key[0]:[log_loss(target, .6*val[i]+.4*pr_xgb_5) \
                                for i in range(10, 42)] \
          for key, val in pr_nn_dict.iteritems()}

# r096_summary.csv: params_tune_nn.py
r096_summary = pd.read_csv('logs/r096_summary.csv', index_col=0, header=[0,1])

# epochs:110-420
r104 = r096_summary.iloc[10:,:]
r104.columns = ['60avg', '70avg', '60', '70']
r104 = r104.join(pd.DataFrame(scores, index=range(110, 421, 10)))
r104 = r104.reindex(columns=['60', '60avg', '60nn_xgb', '70', '70avg', '70nn_xgb'])
r104.columns = pd.MultiIndex(levels=[[60, 70], ['NN', 'NN_avg', 'NN_XGB']],
                             labels=[[0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2]],
                             names=[u'h1', u'model'])
r104.to_csv('logs/r104.csv')
for h1 in [60, 70]:
    ax = r104[h1].plot()
    ax.set_title('Experiment to choose max_epochs of NN(h1:%d)' % h1)
    ax.set_ylabel('Logloss')
    fig = ax.get_figure()
    fig.savefig('logs/exp_ens_h1_%d.png' % h1)

print pd.DataFrame({'loss':r104.min(0), 'epochs':r104.idxmin(0)})
#            epochs      loss
# h1 model                   
# 60 NN         150  0.692270
#    NN_avg     230  0.615458
#    NN_XGB     390  0.562667
# 70 NN         160  0.691234
#    NN_avg     210  0.616497
#    NN_XGB     410  0.563851
