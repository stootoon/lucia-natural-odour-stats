import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate, cross_val_score
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

from lucia_stats.common import PCScoreSelector, VarianceSelector, RandomSelector
from lucia_stats.data import get_noni_ripeness


class Analysis:
    def __init__(self, k=6):
        self.df, self.odour_cols = get_noni_ripeness(do_zscore=True)
        self.X = self.df[self.odour_cols]
        self.y = self.df['Ripeness'].astype(float)
        self.groups = self.df['Sample']
        self.cv = LeaveOneGroupOut()
        self.k = k

        alphas = np.logspace(-3, 3, 10)
        
        self.models = {
            "lasso":  make_pipeline(LassoCV(cv=5, max_iter=50000, alphas=50)),
            "pc_k":   make_pipeline(PCScoreSelector(k=self.k), LinearRegression()),
            "var_k":  make_pipeline(VarianceSelector(k=self.k), LinearRegression()),
            "corr_k": make_pipeline(SelectKBest(score_func=f_regression, k=self.k), RidgeCV(alphas=alphas)),
            "dummy":  make_pipeline(DummyRegressor(strategy='mean'))
        }

    def fit_models(self, results = None):
        if results is not None:
            self.results = results
        else:
            self.results = {name:cross_validate(model, self.X, self.y,
                                            groups=self.groups,
                                            cv=self.cv,
                                            scoring='neg_root_mean_squared_error',
                                            return_indices=True,
                                            return_estimator=True) for name, model in self.models.items()}
        return self.results

    def compute_lasso_sparsity(self):
        # Get the coefficients from the fitted Lasso model for each fold
        self.lasso_coefs = [estimator.named_steps['lassocv'].coef_ for estimator in self.results['lasso']['estimator']]
        # Count the number of non-zero coefficients for each fold
        self.lasso_sparsity = [np.sum(coef != 0) for coef in self.lasso_coefs]

    def compute_pc_var_overlap(self):
        # Determine the overlap betwen the features selected by the PCScoreSelector and VarianceSelector
        pc_selected = [estimator.named_steps['pcscoreselector'].get_support() for estimator in self.results['pc_k']['estimator']]
        var_selected = [estimator.named_steps['varianceselector'].get_support() for estimator in self.results['var_k']['estimator']]
        self.pc_var_overlap = [set(self.odour_cols[pc & var].tolist()) for pc, var in zip(pc_selected, var_selected)]
        
    def compute_confusion_matrix(self):
        # Build a confusion matrix for the lasso estimator by aggregating predictions over folds
        # Round the predictions to 1,2,3,4, and compare against the true values
        self.confusion_matrix = np.zeros((4, 4), dtype=int)
        for i, estimator in enumerate(self.results['lasso']['estimator']):
            test_idx = self.results["lasso"]["indices"]["test"][i]
            y_pred = estimator.predict(self.X.iloc[test_idx])
            y_true = self.y.iloc[test_idx]
            y_pred_rounded = np.clip(np.round(y_pred), 1, 4).astype(int) 
            y_true_rounded = np.clip(np.round(y_true), 1, 4).astype(int)
            # Use the true and pred as indices to increment the confusion matrix
            for true, pred in zip(y_true_rounded, y_pred_rounded):
                self.confusion_matrix[true-1, pred-1] += 1


    def compute_p_values(self, n_rand = 100):
        self.rand_results = np.array([
            cross_val_score(make_pipeline(RandomSelector(k=self.k, random_state=i), LinearRegression()),
                            self.X, self.y,
                            groups=self.groups,
                            cv=self.cv,
                            scoring='neg_root_mean_squared_error')
            for i in tqdm(range(n_rand))])

        # For each model and each fold, compute the p-value as the proportion of random scores that are better than the model's score
        self.p_values = {}
        for name, result in self.results.items():
            model_scores = result['test_score']
            p_vals = []
            for fold_idx, score in enumerate(model_scores):
                rand_scores = self.rand_results[:, fold_idx]
                p_val = np.mean(rand_scores > score)
                p_vals.append(p_val)
            self.p_values[name] = np.array(p_vals).astype(float)
        
        
        
        
