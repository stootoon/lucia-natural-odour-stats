import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.gridspec import GridSpec
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

sys.path.append(os.environ["GIT"])

from label_axes import label_axes

from lucia_stats.common import PCScoreSelector, VarianceSelector, RandomSelector, RandomNonPCSelector, Figure
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
            "corr_k": make_pipeline(SelectKBest(score_func=f_regression, k=self.k), RidgeCV()), 
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
        
    def compute_confusion_matrices(self):
        # Build a confusion matrix for the lasso estimator by aggregating predictions over folds
        # Round the predictions to 1,2,3,4, and compare against the true values
        self.confusion_matrices = {k: np.zeros((4, 4), dtype=int) for k in self.results.keys()}
        for name, result in self.results.items():
            for i, estimator in enumerate(result['estimator']):
                test_idx = result["indices"]["test"][i]
                y_pred = estimator.predict(self.X.iloc[test_idx])
                y_true = self.y.iloc[test_idx]
                y_pred_rounded = np.clip(np.round(y_pred), 1, 4).astype(int) 
                y_true_rounded = np.clip(np.round(y_true), 1, 4).astype(int)
                # Use the true and pred as indices to increment the confusion matrix
                for true, pred in zip(y_true_rounded, y_pred_rounded):
                    self.confusion_matrices[name][true-1, pred-1] += 1


    def compute_p_values(self, n_rand = 100, non_pc = True, agg_fun = np.median):
        self.rand_results = np.array([
            cross_val_score(make_pipeline((RandomNonPCSelector if non_pc else RandomSelector)(k=self.k, random_state=i), LinearRegression()),
                            self.X, self.y,
                            groups=self.groups,
                            cv=self.cv,
                            scoring='neg_root_mean_squared_error')
            for i in tqdm(range(n_rand))])
        
        self.rand_results_agg = agg_fun(self.rand_results, axis=-1)
        # For each model and each fold, compute the p-value as the proportion of random scores that are better than the model's score
        self.p_values_per_fold = {}
        self.p_values_agg = {}
        self.model_scores_agg = {}
        self.model_scores = {}
        for name, result in self.results.items():
            self.model_scores[name] = result['test_score']
            p_vals = []
            for fold_idx, score in enumerate(self.model_scores[name]):
                rand_scores = self.rand_results[:, fold_idx]
                p_val = (1 + np.sum(rand_scores > score))/(1 + n_rand)
                p_vals.append(p_val)
            self.p_values_per_fold[name] = np.array(p_vals).astype(float)

            self.model_scores_agg[name] = agg_fun(self.model_scores[name])
            self.p_values_agg[name] = ((1 + np.sum(self.rand_results_agg> self.model_scores_agg[name]))/(1 + n_rand)).astype(float)
                                       

# Turn top and right spines off
spines_off = lambda ax: [ax.spines[spine].set_visible(False) for spine in ['top', 'right']]
class PlotResults:
    def __init__(self, analysis, colorby="fold"):
        self.analysis = analysis
        if colorby == "fold":
            self.fold_cols = cm.rainbow(np.linspace(0,1,len(self.analysis.groups.unique()))) 
        # Color folds by ripeness of the leftout sample, using the ripeness colormap
        elif colorby == "ripeness":
            ripeness = self.analysis.df.groupby('Sample')['Ripeness'].first().sort_index()
            ripeness_norm = (ripeness - ripeness.min()) / (ripeness.max() - ripeness.min())
            self.fold_cols = cm.rainbow(ripeness_norm.values)
        else:
            raise ValueError(f"Unknown colorby value: {colorby}")
        self.model_cols = {"lasso": "C0", "pc_k": "C1", "var_k": "C2", "corr_k": "C3", "dummy": "gray"}
        self.model_names= {"lasso": "Lasso", "pc_k": "$PC_k$", "var_k": "$Var_k$", "corr_k": "$Corr_k$", "dummy": "Dummy"}
        plt.style.use("default")

    def plot_rmse(self, ax):
        # There are only 12 points per model, so instead of a box plot,
        # plot the RMSE for each fold as a scatter plot with jitter, and the mean as a horizontal line
        # Colour the points by fold, indicating the leftout sample
        # Each model will be at its own x value.
        for i, (name, result) in enumerate(self.analysis.results.items()):
            #x = np.random.normal(i, 0.04, size=len(result['test_score']))
            ts = result['test_score']
            x = np.linspace(i-0.2, i+0.2, len(ts))
            ax.scatter(x, -result['test_score'], alpha=0.5, color=self.fold_cols, s=20)
            if i == 0:
                # Add a legend for the folds, labeled by the leftout sample, but only for the first model
                for j, col in enumerate(self.fold_cols):
                    ax.scatter([], [], color=col, label=f'{self.analysis.groups.unique()[j]}')
            
            # Put whiskers at median and IQR
            median = np.median(-result['test_score'])
            q1 = np.percentile(-result['test_score'], 25)
            q3 = np.percentile(-result['test_score'], 75)
            ax.hlines(median, i-0.2, i+0.2, color='black', lw=1)
            ax.vlines(i, q1, q3, color='black', lw=1)
        ax.set_xticks(range(len(self.analysis.results)))
        ax.set_xticklabels([self.model_names[name] for name in self.analysis.results.keys()])
        ax.set_ylabel("RMSE")
        ax.set_title("RMSE for each model")
        # Add the legend for the folds, as 3 rows and 4 cols
        ax.legend(ncol=4, fontsize=8, title="Leftout sample", loc='upper left')
        spines_off(ax)

    def plot_sparsity(self, ax):
        # Plot a histogram of the sparsity level of the lasso model
        # Plot a vertical line at the mean sparsity level
        ax.hist(self.analysis.lasso_sparsity, bins=np.arange(0, self.analysis.X.shape[1]+1)-0.5, density=False, alpha=0.7)
        mean_sparsity = np.mean(self.analysis.lasso_sparsity)
        ax.axvline(mean_sparsity, color='red', linestyle='--', label=f'Mean: {mean_sparsity:.2f}')
        ax.set_xlabel("Number of non-zero coefficients")
        ax.set_ylabel("Count")
        ax.set_title("Classifier sparsity level")
        ax.set_xlim(3,9)
        spines_off(ax)

    def plot_overlap(self, ax):
        # Plot a histogram of the overlap of features selected by PCScoreSelector and VarianceSelector
        overlap_counts = [len(overlap) for overlap in self.analysis.pc_var_overlap]
        ax.hist(overlap_counts, bins=np.arange(0, self.analysis.k+2)-0.5, density=False, alpha=0.7)
        #mean_overlap = np.mean(overlap_counts)
        #ax.axvline(mean_overlap, color='red', linestyle='--', label=f'Mean: {mean_overlap:.2f}')
        ax.set_xlabel("Number of overlapping features")
        ax.set_ylabel("Count")
        ax.set_title("Overlap of $PC_k$ and $Var_k$ features")
        spines_off(ax)

    def plot_pc_score_vs_coef(self, ax):
        # Use the estimators to get the lasso coefs and pc scores for each fold,
        # and plot the pc score vs the lasso coefficient for each fold
        for i, (lasso, pck) in enumerate(zip(self.analysis.results['lasso']['estimator'], self.analysis.results['pc_k']['estimator'])): 
            lasso_coef = lasso.named_steps['lassocv'].coef_
            pc_score = pck.named_steps['pcscoreselector'].pc_score_
            ax.scatter(pc_score, lasso_coef, alpha=0.5, color=self.fold_cols[i], label=f'Fold {i+1}')
        ax.set_xlabel("PC Score")
        ax.set_ylabel("Lasso Coefficient")
        ax.set_title("PC Score vs Lasso Coefficient")
        spines_off(ax)

    def plot_confusion_matrix(self, ax, which_model):
        # Plot the confusion matrix for the specified model
        # The confusion matrices are in counts so normalize by the number of samples in each true class
        # Annotate the matrix with the counts
        cm = self.analysis.confusion_matrices[which_model]
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        im = ax.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(cm.shape[1]),
                yticks=np.arange(cm.shape[0]),
                xticklabels=np.arange(1, cm.shape[1]+1),
                yticklabels=np.arange(1, cm.shape[0]+1),
                ylabel='True ripeness',
                xlabel='Predicted ripeness',
                title=f'Confusion Matrix for {self.model_names[which_model]}')
        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                    rotation_mode="anchor")
        # At the center of each cell, place the count
        fmt = 'd'
        thresh = cm_normalized.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm_normalized[i, j] > thresh else "black")

    def plot_aggregate_p_values(self, ax):
        # Plot a histogram of the aggregate rand scores, and the aggregate model scores for each model, with a vertical line at the model score
        # Histogram should be light gray
        ax.hist(-self.analysis.rand_results_agg,
                bins=int(np.sqrt(len(self.analysis.rand_results_agg))),
                density=True, alpha=0.7, label='Random',
                color='lightgray'
                )
        for name, score in self.analysis.model_scores_agg.items():
            label = f'{self.model_names[name]} (p={self.analysis.p_values_agg[name]:.3f})'
            ax.axvline(-score, color=self.model_cols[name],
                       linestyle='--',
                       label=label)
        ax.set_xlabel("Aggregate RMSE")
        ax.set_ylabel("Density")
        ax.set_title("Aggregate RMSE and p-values")
        ax.set_xlim(0.2,1.2)
        ax.legend(fontsize=8)
        spines_off(ax)

    def plot_per_fold_p_values(self, ax, whiskers=True):
        # Just liken in the RMSE plot, plot the p-values for each fold as a scatter plot with jitter, and the mean as a horizontal line
        for i, (name, p_vals) in enumerate(self.analysis.p_values_per_fold.items()):
            x = np.linspace(i-0.2, i+0.2, len(p_vals))
            y = -np.log10(p_vals)
            ax.scatter(x, y, alpha=0.5, color=self.fold_cols, s=20)
            if whiskers:
                # Put whiskers at median and IQR
                median = np.median(y)
                q1 = np.percentile(y, 25)
                q3 = np.percentile(y, 75)
                ax.hlines(median, i-0.2, i+0.2, color='black', lw=1)
                ax.vlines(i, q1, q3, color='black', lw=1)
        ax.set_xticks(range(len(self.analysis.p_values_per_fold)))
        ax.set_xticklabels([self.model_names[name] for name in self.analysis.p_values_per_fold.keys()])
        ax.set_ylabel("-log10(p-value)")
        ax.set_title("p-values for each model per fold")
        spines_off(ax)
        

    def default_panels(self):
        # (row, col, label, plot_func) for the main figure. row/col may be an
        # int, a slice, or a (start, stop) tuple to span multiple cells.
        return [
            (0, 0, "A", self.plot_rmse),
            (0, 1, "B", self.plot_sparsity),
            (0, 2, "C", self.plot_overlap),
            (1, 0, "D", self.plot_pc_score_vs_coef),
            (1, 1, "E", self.plot_aggregate_p_values),
            (1, 2, "F", self.plot_per_fold_p_values),
            (2, 0, "G", lambda ax: self.plot_confusion_matrix(ax, "lasso")),
            (2, 1, "H", lambda ax: self.plot_confusion_matrix(ax, "pc_k")),
            (2, 2, "I", lambda ax: self.plot_confusion_matrix(ax, "dummy")),
        ]

    def plot_figure(self, panels=None, shape=(3, 3), figsize=(15, 10)):
        if panels is None:
            panels = self.default_panels()
        return Figure(shape=shape, figsize=figsize).plot(panels)

    
# def plot_results(obj):
# Panels to plot
# 1. Boxplot of RMSE for each model
# 2. Histogram of lasso sparsity level
# 3. Histogram of overlap of features selected by PCScoreSelector and VarianceSelector
# 4. Confusion matrix for lasso predictions
    
        
        
        
