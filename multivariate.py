"""
Compute C-index for CoxPH model on clinical features

cox: https://scikit-survival.readthedocs.io/en/stable/user_guide/00-introduction.html
"""

import numpy as np
import pandas as pd
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from typing import List, Tuple

class MultivariateModel:
    def __init__(self, data_train: pd.DataFrame, data_val: pd.DataFrame, 
                 clinical_attrs: List[str], modality: str, random_state=42):
        """
        Initialize the MultivariateModel class with pre-split data.
        
        Args:
            data: DataFrame containing features and survival data
            clinical_attrs: List of clinical attribute column names
            modality: What kind of data we are using
        """
        self.clinical_attrs = clinical_attrs
        self.modality = modality
        self.random_state = random_state
        self.results = {}
        
        # Convert data to structured array format
        self.X_train = data_train[clinical_attrs].values
        self.y_train = np.array(list(zip(data_train['event'].values.astype(bool), data_train['duration'].values.astype('float64'))), dtype=[('event', '?'), ('duration', '<f8')])

        self.X_val = data_val[clinical_attrs].values
        self.y_val = np.array(list(zip(data_val['event'].values.astype(bool), data_val['duration'].values.astype('float64'))), dtype=[('event', '?'), ('duration', '<f8')])

    def run_analysis(self) -> Tuple[List[float], List[float]]:
        """Train CoxPH survival model and compute C-indices"""
        # Initialize containers for results
        all_train_c_idx = []
        all_val_c_idx = []
        
        # 1. CoxPH Model
        coxph = CoxPHSurvivalAnalysis()
        coxph.fit(self.X_train, self.y_train)
        
        # Training C-index
        coxph_train_c_idx = coxph.score(self.X_train, self.y_train)
        all_train_c_idx.append(coxph_train_c_idx)
        
        # Validation C-index
        coxph_val_pred = coxph.predict(self.X_val)
        coxph_val_c_idx = concordance_index_censored(
            self.y_val["event"], self.y_val["duration"], coxph_val_pred
        )[0]
        all_val_c_idx.append(coxph_val_c_idx)
        
        print(f'CoxPH Training c-index: {coxph_train_c_idx:.3f}')
        print(f'CoxPH Validation c-index: {coxph_val_c_idx:.3f}')
        
        return all_train_c_idx, all_val_c_idx
