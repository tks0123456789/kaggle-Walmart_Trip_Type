import numpy as np
import scipy as sp
import pandas as pd

# Path
data_path = '../Data/'
file_train = data_path + 'train.csv'
file_test = data_path + 'test.csv'

def sign_log1p_abs(x):
    return np.sign(x) * np.log1p(np.abs(x))

# Parameters:
#   df:2 or 3 columns DataFrame
#   row: a column for row labels
#   col: a column for column labels
#   val: a column for values or None(values=1)
# Returns:
#   mat: csr matrix The columns are sorted by their frequency(decending).
#   label2row: a map from a row label to a row number of mat
#   label2column: a map from a column label to a column number of mat
def DataFrame_tocsr(df, row, col, val=None, label2row=None, label2col=None,
                    return_tbl=False, min_count=1):
    if label2row is None:
        row_labels = df[row].dropna().unique() # pd.Series.unique does not sort
        label2row = pd.Series(range(row_labels.size), index=row_labels)
    if val is None:
        df = df[[row, col]].dropna()
        vals = pd.Series(np.ones(df.shape[0]))
    else:
        df = df[[row, col, val]].dropna()
        vals = df[val].values
    if label2col is None:
        col_label_cnt = df[col].value_counts()
        if min_count > 1:
            col_label_cnt = col_label_cnt[col_label_cnt >= min_count]
        col_labels = col_label_cnt.index
        label2col = pd.Series(range(col_labels.size), index=col_labels)
    rows = df[row].map(label2row)
    cols = df[col].map(label2col)
    if cols.size == 0:
        return False
    mat = sp.sparse.coo_matrix((vals, (rows, cols)), shape=(label2row.size, label2col.size)).tocsr()
    if return_tbl:
        return mat, label2row, label2col
    else:
        return mat

def feature_extraction(training=None, test=None, useUpc=False):
    if training is None and test is None:
        training = pd.read_csv(file_train)
        test = pd.read_csv(file_test)

    v_train = training.VisitNumber.unique()
    num_train = v_train.size
    v_test = test.VisitNumber.unique()
    grouped_train = training.groupby('VisitNumber')
    target = grouped_train.TripType.first().values

    data_all = training.append(test)
    data_all = data_all.sort('VisitNumber')
    data_all.ScanCount=data_all.ScanCount.astype(float)

    w2int = pd.Series(range(7), index=['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'])

    grouped = data_all.groupby('VisitNumber')
    Weekday = grouped.Weekday.first()

    data_all['ScanCount_log1p'] = sign_log1p_abs(data_all.ScanCount)
    data_all.loc[data_all.DepartmentDescription=='MENSWEAR', 'DepartmentDescription'] = 'MENS WEAR'
    data_all.DepartmentDescription.fillna('-1', inplace=True)
    data_all.FinelineNumber.fillna(-1, inplace=True)
    data_all.Upc.fillna(-1, inplace=True)

    X_wday = sp.sparse.coo_matrix(pd.get_dummies(Weekday.map(w2int)).values)
    N = X_wday.shape[0]
    X_SC_sum_sign = data_all.groupby('VisitNumber').ScanCount.apply(lambda x:1 if x.sum()>0 else 0).reshape((N, 1))
    X_SC_sum = data_all.groupby('VisitNumber').ScanCount.apply(lambda x:x.sum()).reshape((N, 1))
    X_dept = DataFrame_tocsr(data_all,
                             row='VisitNumber',
                             col='DepartmentDescription',
                             val='ScanCount')

    X_fine = DataFrame_tocsr(data_all,
                             row='VisitNumber',
                             col='FinelineNumber',
                             val='ScanCount_log1p')


    fine_dept = data_all[['FinelineNumber', 'DepartmentDescription']].drop_duplicates()
    fine_dept_cnt = fine_dept.FinelineNumber.value_counts()
    tmp = data_all.DepartmentDescription + '_' + data_all.FinelineNumber.astype(str)
    tmp[data_all.FinelineNumber.isin(fine_dept_cnt[fine_dept_cnt<2].index)] = np.nan
    data_all['Dept_Fine'] = tmp
    X_dept_fine = DataFrame_tocsr(data_all,
                                  row='VisitNumber',
                                  col='Dept_Fine',
                                  val='ScanCount_log1p')

    W_int = w2int.ix[Weekday]
    W_diff = W_int.diff().fillna(0)
    W_diff[W_diff!=0] = 1
    day = (W_diff.cumsum() + 1).values
    X_day = pd.get_dummies(day)
    # quasiHour_float = []
    # for i in range(1, 32):
    #     tmp = day[day == i]
    #     n = tmp.size
    #     quasiHour_float = np.append(quasiHour_float, np.arange(n) / float(n))
    # hour = (quasiHour_float * 24).astype(int)

    X = sp.sparse.hstack((X_day, X_SC_sum_sign, sign_log1p_abs(X_SC_sum),
                          X_dept, X_fine, X_dept_fine)).tocsr()
    if useUpc:
        X_upc = DataFrame_tocsr(data_all,
                                row='VisitNumber',
                                col='Upc',
                                val='ScanCount_log1p')
        X = sp.sparse.hstack((X, X_upc)).tocsr()
    return X, target, v_train, v_test
