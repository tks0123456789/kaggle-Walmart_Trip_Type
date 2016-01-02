"""
Target parameters: max_epochs
.3*(NN([6k+, 60, 100, 38])_avg + NN([6k+, 70, 90, 38]))_avg +
  .4*XGB(max_depth:5, num_round550)_avg
"""

import pandas as pd
import cPickle as pickle
from utility_common import data_path, file_train
from sklearn.metrics import log_loss

pd.set_option('display.precision', 4)

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
scores = {'%davg_xgb' % key[0]:[log_loss(target, .6*val[i]+.4*pr_xgb_5) \
                                for i in range(10, 42)] \
          for key, val in pr_nn_dict.iteritems()}

# r096_summary.csv: params_tune_nn.py
r096_summary = pd.read_csv('logs/r096_summary.csv', index_col=0)

# epochs:110-420
r104 = r096_summary.iloc[10:,:]

r104 = r104.join(pd.DataFrame(scores, index=range(110, 421, 10)))

r104 = r104.reindex(columns=['60', '60avg', '60avg_xgb', '70', '70avg', '70avg_xgb'])
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

print r104[60]
# model       NN  NN_avg  NN_XGB
# epochs                        
# 110     0.7075  0.6644  0.6039
# 120     0.7007  0.6543  0.5980
# 130     0.6965  0.6458  0.5932
# 140     0.6944  0.6386  0.5888
# 150     0.6923* 0.6326  0.5851
# 160     0.6929  0.6286  0.5820
# 170     0.6939  0.6248  0.5791
# 180     0.6971  0.6223  0.5770
# 190     0.7006  0.6195  0.5748
# 200     0.7038  0.6179  0.5729
# 210     0.7108  0.6172  0.5713
# 220     0.7152  0.6162  0.5699
# 230     0.7221  0.6155* 0.5686
# 240     0.7281  0.6161  0.5680
# 250     0.7333  0.6160  0.5669
# 260     0.7405  0.6165  0.5663
# 270     0.7477  0.6175  0.5657
# 280     0.7556  0.6179  0.5649
# 290     0.7642  0.6184  0.5647
# 300     0.7720  0.6200  0.5644
# 310     0.7781  0.6204  0.5637
# 320     0.7884  0.6228  0.5639
# 330     0.7935  0.6232  0.5636
# 340     0.8037  0.6248  0.5635
# 350     0.8117  0.6257  0.5634
# 360     0.8196  0.6272  0.5630
# 370     0.8280  0.6298  0.5633
# 380     0.8359  0.6296  0.5630
# 390     0.8426  0.6313  0.5627*
# 400     0.8523  0.6333  0.5630
# 410     0.8615  0.6342  0.5630
# 420     0.8666  0.6350  0.5628
