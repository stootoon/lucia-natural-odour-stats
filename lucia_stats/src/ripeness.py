import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import zscore
from tqdm import tqdm
from sklearn.model_selection import LeaveOneGroupOut, cross_validate, cross_val_score
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import make_pipeline
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Resolve label_axes via its installed package before $GIT goes on sys.path:
# once $GIT is on the path, $GIT/label_axes shadows it as an empty namespace
# package and common.py's import of it breaks.
import label_axes  # noqa: F401

sys.path.append(os.environ["GIT"])
from lucia_stats.src.common import PCScoreSelector, VarianceSelector, RandomSelector, RandomNonPCSelector, Figure, create_mock_confusion_matrix
from lucia_stats.src.data import SHEET_URL

# Tab with noni fruit independently assessed for ripeness:
# 46 odours (rows) x 15 fruits (columns), 3 fruits per ripeness stage.
ripeness_gid = "165686008"
STAGES = ["G", "GY", "Y", "YB", "B"]  # green -> green/yellow -> yellow -> yellow/brown -> brown


def get_independent_ripeness(do_zscore=False):
    """Load the independently-assessed ripeness dataset.

    Returns a tidy dataframe with one row per fruit (columns: Sample, Stage,
    Fruit, Ripeness, plus one column per odour), and the odour column index.
    Ripeness is the ordinal stage index (G=1 ... B=5).
    """
    df = pd.read_csv(SHEET_URL(ripeness_gid))
    df = df.rename(columns={df.columns[0]: 'Odour'}).set_index('Odour')
    df.index = df.index.str.strip()

    tidy = df.T.reset_index().rename(columns={'index': 'Sample'})
    tidy.columns.name = None
    tidy['Stage'] = tidy['Sample'].str.extract(r'^([A-Z]+)')[0]
    tidy['Fruit'] = tidy['Sample'].str.extract(r'(\d+)$')[0].astype(int)
    tidy['Ripeness'] = tidy['Stage'].map({s: i + 1 for i, s in enumerate(STAGES)})
    unknown = set(tidy.loc[tidy['Ripeness'].isna(), 'Stage'])
    assert not unknown, f"Unknown ripeness stages: {unknown}"
    tidy['Ripeness'] = tidy['Ripeness'].astype(int)

    meta_cols = ['Sample', 'Stage', 'Fruit', 'Ripeness']
    odour_cols = tidy.columns.difference(meta_cols)

    if do_zscore:
        tidy[odour_cols] = zscore(tidy[odour_cols].values.astype(float), axis=1)

    return tidy, odour_cols


class Analysis:
    def __init__(self, k=6, seed=0, inner_cv=3):
        self.df, self.odour_cols = get_independent_ripeness(do_zscore=True)
        self.X = self.df[self.odour_cols]
        self.y = self.df['Ripeness'].astype(int)
        # Each column is a different fruit, so each is its own group and
        # leave-one-group-out is leave-one-fruit-out.
        self.groups = self.df['Sample']
        self.cv = LeaveOneGroupOut()
        self.k = k
        self.seed = seed
        self.n_classes = len(STAGES)

        self.models = {
            "logreg_l1": make_pipeline(
                LogisticRegressionCV(Cs=np.logspace(-2, 2, 10), cv=inner_cv,
                                     penalty='l1', solver='saga',
                                     scoring='accuracy', max_iter=50000,
                                     random_state=seed)),
            "pc_k":   make_pipeline(PCScoreSelector(k=self.k), LogisticRegression(max_iter=5000)),
            "var_k":  make_pipeline(VarianceSelector(k=self.k), LogisticRegression(max_iter=5000)),
            "corr_k": make_pipeline(SelectKBest(score_func=f_classif, k=self.k), LogisticRegression(max_iter=5000)),
            # NB 'most_frequent' degenerates under leave-one-fruit-out (the
            # held-out class is always the least frequent in training, so it is
            # never predicted and accuracy is 0); uniform guessing sits at the
            # 1/n_classes chance level instead.
            "dummy":  make_pipeline(DummyClassifier(strategy='uniform', random_state=seed)),
        }

    def fit_models(self, results=None):
        if results is not None:
            self.results = results
        else:
            self.results = {name: cross_validate(model, self.X, self.y,
                                                 groups=self.groups,
                                                 cv=self.cv,
                                                 scoring='accuracy',
                                                 return_indices=True,
                                                 return_estimator=True) for name, model in self.models.items()}
        return self.results

    def compute_sparsity(self):
        # Number of odours used by the L1 logistic model in each fold: a
        # feature counts as used if its coefficient is non-zero for any class.
        self.logreg_coefs = [est.named_steps['logisticregressioncv'].coef_
                             for est in self.results['logreg_l1']['estimator']]
        self.sparsity = [int(np.sum(np.any(coef != 0, axis=0))) for coef in self.logreg_coefs]

    def compute_pc_var_overlap(self):
        # Overlap between the features selected by the PCScoreSelector and VarianceSelector
        pc_selected = [est.named_steps['pcscoreselector'].get_support() for est in self.results['pc_k']['estimator']]
        var_selected = [est.named_steps['varianceselector'].get_support() for est in self.results['var_k']['estimator']]
        self.pc_var_overlap = [set(self.odour_cols[pc & var].tolist()) for pc, var in zip(pc_selected, var_selected)]

    def compute_logreg_pc_overlap(self):
        # Overlap between the odours used by the L1 logistic model (non-zero
        # coefficient for any class) and those selected by the PCScoreSelector
        logreg_selected = [np.any(est.named_steps['logisticregressioncv'].coef_ != 0, axis=0)
                           for est in self.results['logreg_l1']['estimator']]
        pc_selected = [est.named_steps['pcscoreselector'].get_support() for est in self.results['pc_k']['estimator']]
        self.logreg_pc_overlap = [set(self.odour_cols[lr & pc].tolist()) for lr, pc in zip(logreg_selected, pc_selected)]

    def compute_confusion_matrices(self):
        # Aggregate test predictions over folds. Predictions are class labels
        # (1..5), so no rounding is needed.
        n = self.n_classes
        self.confusion_matrices = {name: np.zeros((n, n), dtype=int) for name in self.results.keys()}
        for name, result in self.results.items():
            for i, estimator in enumerate(result['estimator']):
                test_idx = result["indices"]["test"][i]
                y_pred = estimator.predict(self.X.iloc[test_idx])
                y_true = self.y.iloc[test_idx]
                for true, pred in zip(y_true, y_pred):
                    self.confusion_matrices[name][true - 1, pred - 1] += 1

        row_counts = self.confusion_matrices[name].sum(axis=1)
        for fld in ["perfect", "chance", "uniform"]:
            self.confusion_matrices[fld] = create_mock_confusion_matrix(fld, n, row_counts, self.seed)

    def compute_p_values(self, n_rand=100, non_pc=True, agg_fun=np.mean):
        # Null distribution: classification accuracy using a random set of k
        # odours (optionally excluding the top-k PC-score odours).
        self.rand_results = np.array([
            cross_val_score(make_pipeline((RandomNonPCSelector if non_pc else RandomSelector)(k=self.k, random_state=i),
                                          LogisticRegression(max_iter=5000)),
                            self.X, self.y,
                            groups=self.groups,
                            cv=self.cv,
                            scoring='accuracy')
            for i in tqdm(range(n_rand))])

        self.rand_results_agg = agg_fun(self.rand_results, axis=-1)
        # p-value: proportion of random feature sets that do at least as well.
        # Accuracy is discrete so ties are counted against the model (>=).
        self.p_values_per_fold = {}
        self.p_values_agg = {}
        self.model_scores_agg = {}
        self.model_scores = {}
        for name, result in self.results.items():
            self.model_scores[name] = result['test_score']
            p_vals = []
            for fold_idx, score in enumerate(self.model_scores[name]):
                rand_scores = self.rand_results[:, fold_idx]
                p_val = (1 + np.sum(rand_scores >= score)) / (1 + n_rand)
                p_vals.append(p_val)
            self.p_values_per_fold[name] = np.array(p_vals).astype(float)

            self.model_scores_agg[name] = agg_fun(self.model_scores[name])
            self.p_values_agg[name] = ((1 + np.sum(self.rand_results_agg >= self.model_scores_agg[name])) / (1 + n_rand)).astype(float)


# Turn top and right spines off
spines_off = lambda ax: [ax.spines[spine].set_visible(False) for spine in ['top', 'right']]


class PlotResults:
    def __init__(self, analysis, colorby="ripeness"):
        self.analysis = analysis
        if colorby == "fold":
            self.fold_cols = cm.rainbow(np.linspace(0, 1, len(self.analysis.groups.unique())))
        # Color folds by the ripeness of the leftout fruit
        elif colorby == "ripeness":
            ripeness = self.analysis.df.groupby('Sample')['Ripeness'].first()
            ripeness = ripeness.loc[self.analysis.groups.unique()]
            ripeness_norm = (ripeness - 1) / (len(STAGES) - 1)
            self.fold_cols = cm.rainbow(ripeness_norm.values)
        else:
            raise ValueError(f"Unknown colorby value: {colorby}")
        self.model_cols = {"logreg_l1": "C0", "pc_k": "C1", "var_k": "C2", "corr_k": "C3", "dummy": "gray"}
        self.model_names = {"logreg_l1": "LogReg$_{L1}$", "pc_k": "$PC_k$", "var_k": "$Var_k$", "corr_k": "$Corr_k$",
                            "dummy": "Dummy", "perfect": "Perfect", "chance": "Chance", "uniform": "Uniform"}
        plt.style.use("default")

    def plot_accuracy(self, ax):
        # One point per fold (leave-one-fruit-out, so each is 0 or 1);
        # whiskers at median and IQR, as in the sechellia RMSE plot.
        for i, (name, result) in enumerate(self.analysis.results.items()):
            ts = result['test_score']
            x = np.linspace(i - 0.2, i + 0.2, len(ts))
            ax.scatter(x, ts, alpha=0.5, color=self.fold_cols, s=20)
            if i == 0:
                for j, col in enumerate(self.fold_cols):
                    ax.scatter([], [], color=col, label=f'{self.analysis.groups.unique()[j]}')
            mean = np.mean(ts)
            ax.hlines(mean, i - 0.2, i + 0.2, color='black', lw=1)
        ax.axhline(1 / self.analysis.n_classes, color='gray', linestyle=':', lw=1, label='Chance')
        ax.set_xticks(range(len(self.analysis.results)))
        ax.set_xticklabels([self.model_names[name] for name in self.analysis.results.keys()])
        ax.set_ylabel("Accuracy")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title("Leave-one-fruit-out accuracy per model")
        ax.legend(ncol=5, fontsize=7, title="Leftout fruit", loc='lower left')
        spines_off(ax)

    def plot_sparsity(self, ax):
        # Histogram of the number of odours used by the L1 logistic model
        ax.hist(self.analysis.sparsity, bins=np.arange(0, self.analysis.X.shape[1] + 1) - 0.5, density=False, alpha=0.7)
        mean_sparsity = np.mean(self.analysis.sparsity)
        ax.axvline(mean_sparsity, color='red', linestyle='--', label=f'Mean: {mean_sparsity:.2f}')
        ax.set_xlabel("Number of odours used")
        ax.set_ylabel("Count")
        ax.set_title("Classifier sparsity level")
        ax.set_xlim(-0.5, max(self.analysis.sparsity) + 3)
        ax.legend(fontsize=8)
        spines_off(ax)

    def plot_overlap(self, ax):
        # Histograms of the per-fold overlap of the PC_k features with the
        # Var_k features and (when computed) with the odours used by the L1
        # logistic model.
        bins = np.arange(0, self.analysis.k + 2) - 0.5
        counts = {'$PC_k \\cap Var_k$': [len(o) for o in self.analysis.pc_var_overlap]}
        if hasattr(self.analysis, 'logreg_pc_overlap'):
            counts['$PC_k \\cap$ LogReg$_{L1}$'] = [len(o) for o in self.analysis.logreg_pc_overlap]
        ax.hist(list(counts.values()), bins=bins, density=False, alpha=0.7, label=list(counts.keys()))
        ax.set_xlabel("Number of overlapping features")
        ax.set_ylabel("Count")
        ax.set_title("Feature overlap with $PC_k$")
        ax.legend(fontsize=8)
        spines_off(ax)

    def plot_pc_score_vs_coef(self, ax):
        # PC score vs the size of the L1 logistic coefficient for each odour.
        # The multinomial model has one coefficient per class per odour, so
        # summarize each odour by its largest absolute coefficient.
        for i, (logreg, pck) in enumerate(zip(self.analysis.results['logreg_l1']['estimator'],
                                              self.analysis.results['pc_k']['estimator'])):
            coef = np.max(np.abs(logreg.named_steps['logisticregressioncv'].coef_), axis=0)
            pc_score = pck.named_steps['pcscoreselector'].pc_score_
            ax.scatter(pc_score, coef, alpha=0.5, color=self.fold_cols[i], label=f'Fold {i+1}')
        ax.set_xlabel("PC Score")
        ax.set_ylabel("max |LogReg$_{L1}$ coefficient|")
        ax.set_title("PC Score vs LogReg$_{L1}$ Coefficient")
        spines_off(ax)

    def plot_confusion_matrix(self, ax, which_model):
        # Confusion matrices are in counts; normalize rows for the colours and
        # annotate with the counts. Classes are labeled by ripeness stage.
        cmat = self.analysis.confusion_matrices[which_model]
        cm_normalized = cmat.astype('float') / cmat.sum(axis=1)[:, np.newaxis]
        im = ax.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=1)
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(cmat.shape[1]),
               yticks=np.arange(cmat.shape[0]),
               xticklabels=STAGES,
               yticklabels=STAGES,
               ylabel='True stage',
               xlabel='Predicted stage',
               title=f'Confusion Matrix for {self.model_names[which_model]}')
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        thresh = cm_normalized.max() / 2.
        for i in range(cmat.shape[0]):
            for j in range(cmat.shape[1]):
                ax.text(j, i, format(cmat[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm_normalized[i, j] > thresh else "black")

    def plot_aggregate_p_values(self, ax, which_models=None):
        # Histogram of the aggregate accuracy of random odour sets, with a
        # vertical line at each model's aggregate accuracy.
        ax.hist(self.analysis.rand_results_agg,
                bins=np.arange(-0.5, len(self.analysis.groups.unique()) + 1.5) / len(self.analysis.groups.unique()),
                density=True, alpha=0.7, label='Random',
                color='lightgray')

        if which_models is None:
            which_models = list(self.model_names.keys())

        for name, score in self.analysis.model_scores_agg.items():
            if name not in which_models:
                continue
            label = f'{self.model_names[name]} (p={self.analysis.p_values_agg[name]:.3f})'
            ax.axvline(score, color=self.model_cols[name],
                       linestyle='--',
                       label=label)
        ax.set_xlabel("Aggregate accuracy")
        ax.set_ylabel("Density")
        ax.set_title("Aggregate accuracy and p-values")
        ax.set_xlim(0, 1)
        ax.legend(fontsize=8)
        spines_off(ax)

    def plot_per_fold_p_values(self, ax, whiskers=True):
        # Per-fold p-values. NB with leave-one-fruit-out each fold's accuracy
        # is 0 or 1, so these are coarse; the aggregate p-value is the
        # meaningful statistic.
        for i, (name, p_vals) in enumerate(self.analysis.p_values_per_fold.items()):
            x = np.linspace(i - 0.2, i + 0.2, len(p_vals))
            y = -np.log10(p_vals)
            ax.scatter(x, y, alpha=0.5, color=self.fold_cols, s=20)
            if whiskers:
                median = np.median(y)
                q1 = np.percentile(y, 25)
                q3 = np.percentile(y, 75)
                ax.hlines(median, i - 0.2, i + 0.2, color='black', lw=1)
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
            (0, 0, "A", self.plot_accuracy),
            (0, 1, "B", self.plot_sparsity),
            (0, 2, "C", self.plot_overlap),
            (1, 0, "D", self.plot_pc_score_vs_coef),
            (1, 1, "E", self.plot_aggregate_p_values),
            (1, 2, "F", self.plot_per_fold_p_values),
            (2, 0, "G", self.plot_confusion_matrix, "logreg_l1"),
            (2, 1, "H", self.plot_confusion_matrix, "pc_k"),
            (2, 2, "I", self.plot_confusion_matrix, "dummy"),
        ]

    def plot_figure(self, panels=None, shape=(3, 3), figsize=(15, 10)):
        if panels is None:
            panels = self.default_panels()
        return Figure(shape=shape, figsize=figsize).plot(panels)
