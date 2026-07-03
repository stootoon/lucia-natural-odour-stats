import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV 
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

sys.path.append(os.environ["GIT"])

from label_axes import label_axes

from lucia_stats.common import PCScoreSelector, VarianceSelector, RandomSelector, RandomNonPCSelector
from lucia_stats.data import tidy_pandan 


class Analysis:
    def __init__(self, k=6):
        self.df, self.odour_cols = tidy_pandan(do_zscore=True)
        self.X = self.df[self.odour_cols]
        self.y = self.df['Sample']
        self.classes_ = self.y.unique()
        self.groups = self.df['Replicate']
        self.cv = LeaveOneGroupOut()
        self.k = k

        self.models = {
            "logreg":  make_pipeline(LogisticRegressionCV(cv=2, penalty='l1', max_iter=50000, Cs=50, solver='saga')),
            "pc_k":   make_pipeline(PCScoreSelector(k=self.k), LogisticRegression(max_iter=10000)),
            "var_k":  make_pipeline(VarianceSelector(k=self.k), LogisticRegression(max_iter=10000)),
            "corr_k": make_pipeline(SelectKBest(score_func=f_classif, k=self.k), LogisticRegression(max_iter=10000)), 
            "dummy":  make_pipeline(DummyClassifier(strategy='most_frequent'))
        }

    def fit_models(self, results = None):
        if results is not None:
            self.results = results
        else:
            self.results = {name:cross_validate(model, self.X, self.y,
                                            groups=self.groups,
                                            cv=self.cv,
                                            scoring='accuracy',
                                            return_indices=True,
                                            return_estimator=True) for name, model in self.models.items()}
        return self.results

    def compute_sparsity(self):
        # Get the coefficients from the fitted logistic regression model for each fold
        self.logreg_coefs = [estimator.named_steps['logisticregressioncv'].coef_ for estimator in self.results['logreg']['estimator']]
        # Count the number of non-zero coefficients for each fold
        # There is one coefficient per class, so we can sum across classes to get the total number of non-zero coefficients
        self.sparsity = [int(np.any(coef != 0, axis=0).sum()) for coef in self.logreg_coefs]

    def compute_pc_var_overlap(self):
        # Determine the overlap betwen the features selected by the PCScoreSelector and VarianceSelector
        pc_selected = [estimator.named_steps['pcscoreselector'].get_support() for estimator in self.results['pc_k']['estimator']]
        var_selected = [estimator.named_steps['varianceselector'].get_support() for estimator in self.results['var_k']['estimator']]
        self.pc_var_overlap = [set(self.odour_cols[pc & var].tolist()) for pc, var in zip(pc_selected, var_selected)]
        
    def compute_confusion_matrices(self):
        # Build a confusion matrix for the lasso estimator by aggregating predictions over folds
        n_classes = self.y.nunique() 
        self.confusion_matrices = {k: np.zeros((n_classes, n_classes), dtype=int) for k in self.results.keys()}
        for name, result in self.results.items():
            for i, estimator in enumerate(result['estimator']):
                test_idx = result["indices"]["test"][i]
                y_pred = estimator.predict(self.X.iloc[test_idx])
                y_true = self.y.iloc[test_idx]
                # Use the true and pred as indices to increment the confusion matrix
                self.confusion_matrices[name] += confusion_matrix(y_true, y_pred, labels=self.classes_)


    def compute_p_values(self, n_rand = 100, non_pc = True, agg_fun = np.median):
        self.rand_results = np.array([
            cross_val_score(make_pipeline((RandomNonPCSelector if non_pc else RandomSelector)(k=self.k, random_state=i), LogisticRegression(max_iter=10000)),
                            self.X, self.y,
                            groups=self.groups,
                            cv=self.cv,
                            scoring='accuracy')
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
    def __init__(self, analysis):
        self.analysis = analysis
        self.fold_cols = cm.rainbow(np.linspace(0,1,len(self.analysis.groups.unique()))) 
        self.n_classes = len(self.analysis.classes_)
        self.chance = 1.0 / self.n_classes
        self.model_cols = {"logreg": "C0", "pc_k": "C1", "var_k": "C2", "corr_k": "C3", "dummy": "gray"}
        self.model_names= {"logreg": "LogReg", "pc_k": "$PC_k$", "var_k": "$Var_k$", "corr_k": "$Corr_k$", "dummy": "Dummy"}
        plt.style.use("default")

    def plot_accuracy(self, ax):
        for i, (name, result) in enumerate(self.analysis.results.items()):
            #x = np.random.normal(i, 0.04, size=len(result['test_score']))
            ts = result['test_score']
            x = np.linspace(i-0.2, i+0.2, len(ts))
            ax.scatter(x, ts, alpha=0.5, color=self.fold_cols, s=20)
            if i == 0:
                for j, col in enumerate(self.fold_cols):
                    ax.scatter([], [], color=col, label=f'{self.analysis.groups.unique()[j]}')
            
            # Put whiskers at median and IQR
            q1, median, q3= np.percentile(ts, [25, 50, 75])
            ax.hlines(median, i-0.2, i+0.2, color='black', lw=1)
            ax.vlines(i, q1, q3, color='black', lw=1)
        ax.axhline(self.chance, color='gray', linestyle=':', lw=1, label='Chance')
        ax.set_xticks(range(len(self.analysis.results)))
        ax.set_xticklabels([self.model_names[name] for name in self.analysis.results.keys()])
        ax.set_ylabel("Accuracy")
        ax.set_ylim(-0.05,1.05)
        ax.set_title("Classification accuracy per model")
        # Add the legend for the folds, as 3 rows and 4 cols
        ax.legend(ncol=4, fontsize=8, title="Left-out replicate")
        spines_off(ax)

    def plot_sparsity(self, ax):
        sparsity = self.analysis.sparsity
        ax.hist(sparsity, bins=np.arange(0, self.analysis.X.shape[1]+1)-0.5, density=False, alpha=0.7, width=0.8, color='C0')
        mean_sparsity = np.mean(sparsity)
        ax.axvline(mean_sparsity, color='red', linestyle='--', label=f'Mean: {mean_sparsity:.2f}')
        ax.set_xlim(0, np.max(sparsity)+1)
        ax.set_xlabel("Number of odours used") 
        ax.set_ylabel("Count")
        ax.set_title("Classifier sparsity level")
        #ax.set_xlim(3,9)
        ax.legend(fontsize=8)
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
        # Use the estimators to get the logreg coefs and pc scores for each fold,
        # and plot the pc score vs the logreg coefficient for each fold
        for i, (lr, pck) in enumerate(zip(self.analysis.results['logreg']['estimator'],
                                          self.analysis.results['pc_k']['estimator'])): 
            coef = lr.named_steps['logisticregressioncv'].coef_
            pc_score = pck.named_steps['pcscoreselector'].pc_score_
            x = np.tile(pc_score, coef.shape[0])
            y = coef.ravel()
            ax.scatter(x, y, alpha=0.5, color=self.fold_cols[i], label=f'Fold {i+1}')
        ax.axhline(0, color='gray', lw=0.5)
        ax.set_xlabel("PC Score")
        ax.set_ylabel("Logistic Coefficient")
        ax.set_title("PC Score vs Logistic Coefficient")
        spines_off(ax)

    def plot_confusion_matrix(self, ax, which_model):
        # Plot the confusion matrix for the specified model
        # The confusion matrices are in counts so normalize by the number of samples in each true class
        # Annotate the matrix with the counts
        cm = self.analysis.confusion_matrices[which_model]
        row_sums = cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums!=0) 
        im = ax.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        labels = self.analysis.classes_
        ax.set(xticks=np.arange(cm.shape[1]),
                yticks=np.arange(cm.shape[0]),
                xticklabels=labels,
                yticklabels=labels, 
                ylabel='True fruit',
                xlabel='Predicted fruit',
                title=f'Confusion matrix for {self.model_names[which_model]}')
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
        ax.hist(self.analysis.rand_results_agg,
                bins=int(np.sqrt(len(self.analysis.rand_results_agg))),
                density=True, alpha=0.7, label='Random',
                color='lightgray'
                )
        for name, score in self.analysis.model_scores_agg.items():
            label = f'{self.model_names[name]} (p={self.analysis.p_values_agg[name]:.3f})'
            ax.axvline(score, color=self.model_cols[name],
                       linestyle='--',
                       label=label)
        ax.axvline(self.chance, color='k', linestyle=':', lw=1, label='Chance')
        ax.set_xlabel("Aggregate accuracy")
        ax.set_ylabel("Density")
        ax.set_title("Aggregate accuracy and p-values")
        ax.set_xlim(0,1)
        ax.legend(fontsize=8)
        spines_off(ax)

    def plot_per_fold_p_values(self, ax, whiskers=True):
        # Just liken in the accuracy plot, plot the p-values for each fold as a scatter plot with jitter, and the mean as a horizontal line
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
        

    def plot_figure(self):
        plt.figure(figsize=(15,10))
        gs = GridSpec(3,3)
        ax_acc        = plt.subplot(gs[0,0])
        ax_sparsity   = plt.subplot(gs[0,1])
        ax_overlap    = plt.subplot(gs[0,2])
        ax_pc_vs_coef = plt.subplot(gs[1,0])
        ax_cm_logreg  = plt.subplot(gs[2,0])
        ax_cm_pc      = plt.subplot(gs[2,1])
        #ax_cm_var     = plt.subplot(gs[2,0])
        #ax_cm_corr    = plt.subplot(gs[2,1])
        ax_cm_dummy   = plt.subplot(gs[2,2])
        ax_p_agg      = plt.subplot(gs[1,1])
        ax_p_fold     = plt.subplot(gs[1,2])
        panels =   [(self.plot_accuracy, ax_acc),
                    (self.plot_sparsity, ax_sparsity),
                    (self.plot_overlap, ax_overlap),
                    (self.plot_pc_score_vs_coef, ax_pc_vs_coef),
                    (self.plot_aggregate_p_values, ax_p_agg),
                    (self.plot_per_fold_p_values, ax_p_fold), 
                    (lambda ax: self.plot_confusion_matrix(ax, "logreg"), ax_cm_logreg),
                    (lambda ax: self.plot_confusion_matrix(ax, "pc_k"), ax_cm_pc),
#                    (lambda ax: self.plot_confusion_matrix(ax, "var_k"), ax_cm_var),
#                    (lambda ax: self.plot_confusion_matrix(ax, "corr_k"), ax_cm_corr),
                    (lambda ax: self.plot_confusion_matrix(ax, "dummy"), ax_cm_dummy),
                    
                  ]
        
        for plot_func, ax in panels:
            plot_func(ax)

        ax_list = [ax for _,ax  in panels]
        plt.tight_layout()
        label_axes.label_axes(ax_list, labs="ABCDEFGHI", fontweight='bold', fontsize=14) 
        
    
# def plot_results(obj):
# Panels to plot
# 1. Boxplot of RMSE for each model
# 2. Histogram of lasso sparsity level
# 3. Histogram of overlap of features selected by PCScoreSelector and VarianceSelector
# 4. Confusion matrix for lasso predictions
    
        
        
        
