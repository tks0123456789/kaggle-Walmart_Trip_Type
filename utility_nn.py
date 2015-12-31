import numpy as np
from nolearn.lasagne import BatchIterator, NeuralNet
from nolearn.lasagne import TrainSplit
from lasagne.layers import InputLayer, DropoutLayer, DenseLayer
from lasagne.nonlinearities import softmax
from lasagne.updates import nesterov_momentum

## function
class BatchIterator_sparse(BatchIterator):
    def __init__(self, batch_size, shuffle=False):
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        bs = self.batch_size
        idx = range(self.n_samples)
        if self.shuffle:
            np.random.shuffle(idx)
        for i in range((self.n_samples + bs - 1) // bs):
            n_last = min((i + 1) * bs, self.n_samples)
            sl = slice(i * bs, n_last)
            Xb = self.X[idx[sl]]
            if self.y is not None:
                yb = self.y[idx[sl]]
            else:
                yb = None
            yield self.transform(Xb, yb)

    @property
    def n_samples(self):
        X = self.X
        return X.shape[0]

    def transform(self, Xb, yb):
        return Xb.todense(), yb

class NeuralNet_sparse(NeuralNet):
    def _check_good_input(self, X, y=None):
        x_len = X.shape[0]

        if y is not None:
            if len(y) != x_len:
                raise ValueError("X and y are not of equal length.")

        if self.regression and y is not None and y.ndim == 1:
            y = y.reshape(-1, 1)

        return X, y

class TrainSplit_sparse(TrainSplit):
    def __call__(self, X, y, net):
        if self.eval_size:
            if net.regression or not self.stratify:
                kf = KFold(y.shape[0], round(1. / self.eval_size))
            else:
                kf = StratifiedKFold(y, round(1. / self.eval_size))

            train_indices, valid_indices = next(iter(kf))
            X_train, y_train = X[train_indices], y[train_indices]
            X_valid, y_valid = X[valid_indices], y[valid_indices]
        else:
            X_train, y_train = X, y
            X_valid, y_valid = X[X.shape[0]:], y[len(y):]

        return X_train, X_valid, y_train, y_valid

# nolearn sparse input
def build_net_sparse_input(h1, h2, p, mm, bs=256, max_epochs=10, lr=.02, num_in=100, num_out=38,
                           shuffle=False, eval_size=.25, verbose=0):
    return NeuralNet_sparse(
        layers=[
            ('input', InputLayer),
            ('hidden1', DenseLayer),
            ('dropout1', DropoutLayer),
            ('hidden2', DenseLayer),
            ('dropout2', DropoutLayer),
            ('output', DenseLayer),
        ],
        batch_iterator_train=BatchIterator_sparse(batch_size=bs, shuffle=shuffle),
        batch_iterator_test=BatchIterator_sparse(batch_size=bs),
        input_shape=(None, num_in),
        hidden1_num_units=h1,
        hidden2_num_units=h2,
        dropout1_p = p,
        dropout2_p = p,
        output_nonlinearity=softmax,
        output_num_units=num_out,
        update=nesterov_momentum,
        update_learning_rate=lr,
        update_momentum=mm,
        regression=False,
        max_epochs=max_epochs,
        train_split=TrainSplit_sparse(eval_size=eval_size),
        use_label_encoder=True,
        verbose=verbose,
    )
