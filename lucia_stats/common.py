import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectorMixin
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.decomposition import PCA

sys.path.append(os.environ["GIT"])
from label_axes import label_axes

class Computation:
    def __init__(self, *args, **kwargs):
        self.computed = False
        
    def compute(self, *args, **kwargs):
        raise NotImplementedError("compute() method not implemented") 

class Figure:
    def __init__(self, shape=None, figsize=(15, 10)):
        self.shape = shape
        self.figsize = figsize
        self.fig = None
        self.ax = None

    @staticmethod
    def _cell_index(v):
        # Normalize an int / slice / (start, stop) tuple into a GridSpec index.
        return slice(*v) if isinstance(v, tuple) else v

    @staticmethod
    def _cell_extent(v):
        # Largest cell index touched, for inferring the grid shape.
        if isinstance(v, tuple):
            return v[1] - 1
        if isinstance(v, slice):
            return (v.stop or 0) - 1
        return v

    def plot(self, panels):
        # Lay out a figure from a list of (row, col, label, plot_func, *args) panels.
        # row/col may be an int, a slice, or a (start, stop) tuple to span cells.
        # plot_func is any callable; it is called as plot_func(ax, *args), so any
        # trailing tuple elements are forwarded to it. label may be None to skip.
        shape = self.shape
        if shape is None:
            nrows = max(self._cell_extent(row) for row, col, *_ in panels) + 1
            ncols = max(self._cell_extent(col) for row, col, *_ in panels) + 1
            shape = (nrows, ncols)

        self.fig = plt.figure(figsize=self.figsize)
        gs = GridSpec(*shape)

        self.ax = []
        labeled = []
        for row, col, label, plot_func, *fn_args in panels:
            ax = plt.subplot(gs[self._cell_index(row), self._cell_index(col)])
            plot_func(ax, *fn_args)
            self.ax.append(ax)
            if label is not None:
                labeled.append((ax, label))

        plt.tight_layout()
        if labeled:
            ax_list, labs = zip(*labeled)
            label_axes.label_axes(list(ax_list), labs=list(labs), fontweight='bold', fontsize=14)

        return self.fig, self.ax

def create_mock_confusion_matrix(which_type, n_rows, row_counts, seed):
    rng = np.random.default_rng(seed)
    n_cols = n_rows
    cm = np.zeros((n_rows, n_cols))
    if which_type == "perfect":
        return np.diag(row_counts).astype(int)
    elif which_type == "chance":
        for i, rc in enumerate(row_counts):
            ch = rng.choice(n_cols, rc)
            for j in ch:
                cm[i, j] += 1
    elif which_type == "uniform":
        for i, rc in enumerate(row_counts):
            cm[i, :] = rc//n_cols
            rem = int(rc - np.sum(cm_uniform[i, :]))
            ch = rng.choice(n_cols, rem)
            for j in ch:
                cm[i, j] += 1
    else:
        raise ValueError(f"Unknown which_type: {which_type}")

    return cm.astype(int)

def get_pc_score(X):
    pca = PCA(n_components=3)
    pca.fit(X)
    ev  = pca.explained_variance_ratio_
    pcs = pca.components_
    pc_score = np.sum(np.diag(ev) @ np.abs(pcs[:3]), axis=0) # Multiply the absolute loadings by the variance explained, then sum across the first 3 PCs
    return pc_score

class BaseSelector(SelectorMixin, BaseEstimator):
     def _get_support_mask(self):
        check_is_fitted(self, 'support_mask_')
        return self.support_mask_   

class VarianceSelector(BaseSelector):
    def __init__(self, k=10):
        self.k = k
    def fit(self, X, y=None):
        if hasattr(X, 'columns'):
            self.feature_names_in_ = X.columns
        X = check_array(X)
        self.n_features_in_ = X.shape[1]

        variances = X.var(axis=0)
        order = np.argsort(variances)[::-1]
        mask = np.zeros(self.n_features_in_, dtype=bool)
        mask[order[:self.k]] = True

        self.variances_ = variances
        self.support_mask_ = mask
        return self

class PCScoreSelector(BaseSelector):
    def __init__(self, k=10):
        self.k = k
    def fit(self, X, y=None):
        if hasattr(X, 'columns'):
            self.feature_names_in_ = X.columns
        X = check_array(X)
        self.n_features_in_ = X.shape[1]
        pc_score = get_pc_score(X)
        order = np.argsort(pc_score)[::-1]
        mask = np.zeros(self.n_features_in_, dtype=bool)
        mask[order[:self.k]] = True

        self.pc_score_ = pc_score
        self.support_mask_ = mask
        return self

class RandomSelector(BaseSelector):
    def __init__(self, k=10, random_state=None):
        self.k = k
        self.random_state = random_state
    def fit(self, X, y=None):
        if hasattr(X, 'columns'):
            self.feature_names_in_ = X.columns
        X = check_array(X)
        self.n_features_in_ = X.shape[1]
        
        rng = np.random.default_rng(self.random_state)
        idx = rng.choice(self.n_features_in_, size=self.k, replace=False)
        mask = np.zeros(self.n_features_in_, dtype=bool)
        mask[idx] = True

        self.random_order_ = idx    
        self.support_mask_ = mask
        return self

class RandomNonPCSelector(BaseSelector):
    def __init__(self, k=10, random_state=None):
        self.k = k
        self.random_state = random_state
    def fit(self, X, y=None):
        if hasattr(X, 'columns'):
            self.feature_names_in_ = X.columns
        X = check_array(X)
        self.n_features_in_ = X.shape[1]
        pc_score = get_pc_score(X)
        non_pc_idx = np.argsort(pc_score)[:self.n_features_in_-self.k] # Get the indices of the features with the lowest PC scores
        rng = np.random.default_rng(self.random_state)
        idx = rng.choice(non_pc_idx, size=self.k, replace=False) # Randomly select k features from the non-PC set
        mask = np.zeros(self.n_features_in_, dtype=bool)
        mask[idx] = True

        self.random_non_pc_order_ = idx    
        self.support_mask_ = mask
        return self
