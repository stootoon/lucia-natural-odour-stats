import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectorMixin
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.decomposition import PCA

class Computation:
    def __init__(self, *args, **kwargs):
        self.computed = False
        
    def compute(self, *args, **kwargs):
        raise NotImplementedError("compute() method not implemented") 

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
