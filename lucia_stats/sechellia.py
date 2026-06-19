import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

from lucia_stats.common import PCScoreSelector
from lucia_stats.data import get_noni_ripeness


class Computations:
    def __init__(self, k=6):
        self.df, self.odour_cols = get_noni_ripeness(do_zscore=True)
        self.X = self.df[self.odour_cols]
        self.y = self.df['Ripeness'].astype(float)
        self.groups = self.df['Sample']
        self.cv = LeaveOneGroupOut()
        self.k = k

    def lasso_vs_pc_score(self):
        models = {
            "lasso": make_pipeline(LassoCV(cv=5, max_iter=50000, n_alphas=50)),
            "pc_k": make_pipeline(PCScoreSelector(k=self.k), LinearRegression()), 
        }

        self.estimators = {
            name:
            cross_validate(model,
                           X=self.X,
                           y=self.y,
                           cv=self.cv,
                           groups=self.groups,
                           return_estimator=True)["estimator"]
                                    
            for name, model in models.items()
        }

        self.coef_per_fold = np.array([e.named_steps["lassocv"].coef_ for e in self.estimators["lasso"]])
        self.pc_per_fold   = np.array([e.named_steps["pcscoreselector"].pc_score_ for e in self.estimators["pc_k"]])
        

        return self
        
        
        
        
        
        
        
    
