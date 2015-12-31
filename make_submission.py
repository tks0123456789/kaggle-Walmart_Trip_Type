import numpy as np
import pandas as pd

from utility_common import data_path, file_train, file_test

training = pd.read_csv(file_train)
test = pd.read_csv(file_test)
target = training.groupby('VisitNumber').TripType.first().values
v_train = training.VisitNumber.unique()
v_test = test.VisitNumber.unique()

pr_nn_h1_60 = np.load(data_path + 'pr_nn002_h1_60.npy')
pr_nn_h1_70 = np.load(data_path + 'pr_nn002_h1_70.npy')
pr_xgb = np.load(data_path + 'pr002_xgb_test.npy')
pr_nn = (pr_nn_h1_60 + pr_nn_h1_70) / 2

type_str_lst = ['TripType_'+str(c) for c in np.unique(target)]
pred = .6*pr_nn + .4*pr_xgb

# Public: 0.52832
# Private: 0.52625
pred002 = pd.DataFrame({'VisitNumber':v_test}).join(pd.DataFrame(pred, columns=type_str_lst))
pred002.to_csv(data_path+'pred002.csv', index = False)
