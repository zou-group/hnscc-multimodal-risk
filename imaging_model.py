import os
import argparse
import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from lifelines.utils import concordance_index
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.plotting import add_at_risk_counts
import ast

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import optuna

from utils import set_seed
from preprocessing import preprocess_data_pfs, split_data

import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning, message=".*torch.load.*")

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

class MultimodalSurvivalDataset(Dataset):
    """Handles multiple data modalities with treatment OHE"""
    def __init__(self, feat_dir, clinical_data, clinical_attrs, use_clinical=False, use_molecular=False):
        self.feat_dir = os.path.join(feat_dir, 'pt_files')

        self.pt_files = [f for f in os.listdir(self.feat_dir) if f.endswith('.pt')]
        unique_files = {}
        
        self.clinical_data = clinical_data
        self.clinical_attrs = clinical_attrs
        self.treatment_map = {'io': 0, 'chemo': 1, 'combo': 2}

        self.use_clinical = use_clinical
        self.use_molecular = use_molecular

        # Initialize scalers
        self.scalers = {}
        if self.use_clinical:
            self._init_scalers(['age'])

        # Match pt files with clinical data
        self.matched_data = []
        for pt_file in self.pt_files:
            
            # TODO: use the first file if they have same names
            segments = pt_file.split('_')
            key = f"{segments[0]}_{segments[1]}"
            if key not in unique_files:
                unique_files[key] = pt_file
                
                patient_id = pt_file.split('_')[0]
                patient_record = self.clinical_data[self.clinical_data['patient_id'] == patient_id]

                if not patient_record.empty:
                    self.matched_data.append(self._process_record(pt_file, patient_record))

    def _init_scalers(self, cols):
        """Initialize scalers for continuous variables"""
        self.scalers = {col: StandardScaler() for col in cols}
        for col in cols:
            self.scalers[col].fit(self.clinical_data[[col]].values)

    def _process_record(self, pt_file, record):
        """Process individual patient record"""
        clinical_features = []
        for var in self.clinical_attrs:
            if var not in self.scalers.keys():
                val = record[var].values[0]
                clinical_features.append(val)

        processed = {
            'pt_file': pt_file,
            'patient_id': record['patient_id'].values[0],
            'event': record['event'].values[0],
            'duration': record['duration'].values[0],
            'treatment': self._one_hot_treatment(record['treatment'].values[0]),
            'slide_id': pt_file.split('.')[0],
        }

        # Add clinical features
        if self.use_clinical:
            clinical_features.append(self._get_scaled_features(record, ['age']))
            processed['clinical'] = clinical_features
            
        # Add molecular features
        if self.use_molecular:
            processed['molecular'] = record['risk_score'].values[0]
            
        return processed

    def _one_hot_treatment(self, treatment):
        """Create treatment one-hot encoding"""
        vec = torch.zeros(len(self.treatment_map))
        vec[self.treatment_map[treatment]] = 1.0
        return vec

    def _get_scaled_features(self, record, cols):
        """Get scaled features from record"""
        return torch.tensor(
            [self.scalers[col].transform([[record[col].values[0]]])[0][0] 
             for col in cols],
            dtype=torch.float32
        )

    def __len__(self):
        return len(self.matched_data)

    def __getitem__(self, idx):
        item = self.matched_data[idx]
        pt_path = os.path.join(self.feat_dir, item['pt_file'])
        image_features = torch.load(pt_path)
        
        return {
            'image_features': image_features,
            'treatment': item['treatment'],
            'event': torch.tensor(item['event'], dtype=torch.float32),
            'duration': torch.tensor(item['duration'], dtype=torch.float32),
            'patient_id': item['patient_id'],
            'slide_id': item['slide_id'],
            **({'clinical': torch.tensor(item['clinical'], dtype=torch.float32)} if self.use_clinical else {}),
            **({'molecular': torch.tensor(item['molecular'], dtype=torch.float32)} if self.use_molecular else {}),
        }

class AttentionMIL(nn.Module):
    """Attention MIL model with multimodal support"""
    def __init__(self, input_dim=512, hidden_dim=64, use_clinical=False, use_molecular=False):
        super().__init__()
        self.use_clinical = use_clinical
        self.use_molecular = use_molecular

        # Image feature processing
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # Attention mechanisn (from https://github.com/AMLab-Amsterdam/AttentionDeepMIL)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.LayerNorm(hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim//2, 1)
        )

        # Multimodal integration
        self.treatment_dim = 3
        combined_dim = hidden_dim + self.treatment_dim  # 3 treatments (io, chemo, combo)
        
        if use_clinical:
            self.clinical_dim = 5
            combined_dim += (self.clinical_dim // 2) # Clinical features ('hpv', 'gender', 'age', 'biopsy_hnscc', 'biopsy_lung')

            self.clinical_processor = nn.Sequential(
                nn.Linear(self.clinical_dim, self.clinical_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.3)
            )

        if use_molecular:
            self.molecular_dim = 1
            combined_dim += (self.molecular_dim)  # Molecular feature: risk score

        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim//3),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim//3, 1)
        )

    def forward(self, x, treatment, clinical_features=None, molecular_features=None):
        # x shape: [num_patches, input_dim]

        # Feature extraction
        H = self.feature_extractor(x) # [num_patches, hidden_dim]

        # Calculate attention scores
        A = self.attention(H)  # [num_patches, 1]
        A = torch.transpose(A, 1, 0)  # [1, num_patches]
        A = F.softmax(A, dim=1)  # Softmax over patches

        # Compute weighted average of patch features
        image_embed = torch.mm(A, H)  # [1, hidden_dim]

        # Combine modalities
        combined = torch.cat([image_embed, treatment.unsqueeze(0)], dim=1) # [1, hidden_dim + self.treatment_dim]

        if self.use_clinical and clinical_features is not None:
            C = self.clinical_processor(clinical_features.unsqueeze(0))  # [1, clinical_dim // 2]
            combined = torch.cat([combined, C], dim=1)
        
        if self.use_molecular and molecular_features is not None:
            M = molecular_features.view(1,1)
            combined = torch.cat([combined, M], dim=1)

        risk_score = self.classifier(combined).squeeze() #  predicted risk score: [1, 1]
        
        return risk_score, A.squeeze(), image_embed.squeeze()

# Custom loss function for Cox survival analysis
class NegativeLogLikelihood(nn.Module):
    def __init__(self):
        super(NegativeLogLikelihood, self).__init__()
    
    def forward(self, risk_scores, events, durations):
        # risk_scores: predicted risk for each patient
        # events: 1 if event occurred, 0 if censored
        # durations: time to event or censoring
        
        # Sort by duration in descending order
        idx = torch.argsort(durations, descending=True)
        risk_scores = risk_scores[idx]
        events = events[idx]
        durations = durations[idx]
        
        # Calculate negative log likelihood
        hazard_ratio = torch.exp(risk_scores)
        log_risk = torch.log(torch.cumsum(hazard_ratio, dim=0) + 1e-8) # add a small epsilon
        uncensored_likelihood = risk_scores - log_risk
        censored_likelihood = uncensored_likelihood * events
        
        neg_likelihood = -torch.sum(censored_likelihood)
        return neg_likelihood

# Function to collate batches for the dataloader
def custom_collate(batch):

    return {
        'image_features': [item['image_features'] for item in batch],
        'event': torch.stack([item['event'] for item in batch]),
        'duration': torch.stack([item['duration'] for item in batch]),
        'treatment': torch.stack([item['treatment'] for item in batch]),
        'patient_id': [item['patient_id'] for item in batch],
        'slide_id': [item['slide_id'] for item in batch],
        **({'clinical': torch.stack([item['clinical'] for item in batch])} if 'clinical' in batch[0] else {}),
        **({'molecular': torch.stack([item['molecular'] for item in batch])} if 'molecular' in batch[0] else {}),
    }


def calculate_risk_score(row, coefficients):
    return sum(coefficients[i] * row.iloc[i] for i in range(len(coefficients)))

def get_risk_score(data, treatment, use_clinical):
    if use_clinical:
        lasso_df = pd.read_csv('./outputs/molecular+clinical/lasso_results_alpha=0.1.csv')
    else:
        lasso_df = pd.read_csv('./outputs/molecular/lasso_results_alpha=0.1.csv')
    
    treatment_mapping = {'io': 'IO', 'combo': 'IO+chemo', 'combo+io': 'IO+/-chemo'}
    if treatment in treatment_mapping.keys():
        t = treatment_mapping[treatment]
    else:
        t = treatment

    genes = ast.literal_eval(lasso_df.loc[
        (lasso_df['treatment_type'] == t) & (lasso_df['PFS'] == True), 
        'sig_genes'
    ].values[-1])

    coef = ast.literal_eval(lasso_df.loc[
        (lasso_df['treatment_type'] == t) & (lasso_df['PFS'] == True), 
        'coefficients'
    ].values[-1])
    
    lasso_dict = {'genes': genes, 'coef': coef}

    median_risk_score = lasso_df.loc[
        (lasso_df['treatment_type'] == t) & (lasso_df['PFS'] == True), 
        'median'
    ].values[-1]

    # compute molecular-based risk score for patients
    data_lasso = data[lasso_dict['genes']]  # based on train genes & coefs
    lasso_coef = [x[0] for x in lasso_dict['coef']]

    data_lasso = data_lasso.copy()
    data_lasso['risk_score'] = data_lasso.apply(calculate_risk_score, coefficients=lasso_coef, axis=1)
    data_lasso['patient_id'] = data['patient_id']
    
    # Check for NaNs
    nan_count = data_lasso['risk_score'].isna().sum()
    if nan_count > 0:
        print(f"Warning: {nan_count} NaN risk scores found for treatment {treatment}")

    return data_lasso

def load_pfs_data(data_path, image_path, treatment, use_clinical, use_molecular):
    matching_file_names = os.listdir(image_path)
    matching_file_names = [name.replace('_HE.tiff', '').replace('_HE.svs', '').split('_')[0] for name in matching_file_names]

    data, clinical_attrs, label_cols, all_genes, unique_patients = preprocess_data_pfs(data_path,  mode='rna+attr', treatment_type = treatment)
    subset_data = data[data['patient_id'].isin(matching_file_names)]

    # Remove pdl1_cps if it exists
    if 'pdl1_cps' in subset_data.columns:
        subset_data = subset_data.drop('pdl1_cps', axis=1)
        if 'pdl1_cps' in clinical_attrs:
            clinical_attrs.remove('pdl1_cps')

    if use_molecular:
        data_lasso = get_risk_score(subset_data, treatment, use_clinical)
        subset_data = pd.merge(subset_data, data_lasso[['patient_id', 'risk_score']], on='patient_id')

    data_train, data_val, train_patient_ids, val_patient_ids = split_data(subset_data, include_test=False, split='random', test_size=0.3)

    data_train['patient_id'] = train_patient_ids
    data_val['patient_id'] = val_patient_ids
    
    return data_train, data_val, train_patient_ids, val_patient_ids, clinical_attrs

def train_model(model, train_loader, val_loader, criterion, optimizer, fold, feature, num_epochs):
    best_val_cindex = -1.0
    train_losses = []
    val_cindices = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Fold {fold+1}, Epoch {epoch+1}/{num_epochs}"):
            image_features = batch['image_features']
            treatments = batch['treatment'].to(device)
            events = batch['event'].to(device)
            durations = batch['duration'].to(device)

            clinical_features = batch['clinical'] if 'clinical' in batch.keys() else [None] * len(image_features)
            molecular_features = batch['molecular'] if 'molecular' in batch.keys() else [None] * len(image_features)
            
            batch_risk_scores = []
            
            # Process each image separately
            for image_feature, treatment, clinical_feature, molecular_feature in zip(image_features, treatments, clinical_features, molecular_features):
                image_feature = image_feature.to(device)
                
                if clinical_feature is not None:
                    clinical_feature = clinical_feature.to(device)
                if molecular_feature is not None:
                    molecular_feature = molecular_feature.to(device)
                
                # Get risk score for this patient
                risk_score, attention_weights, _ = model(image_feature, treatment, clinical_feature, molecular_feature)
                batch_risk_scores.append(risk_score)
            
            # Combine risk scores
            risk_scores = torch.stack(batch_risk_scores)
            
            # Calculate loss
            loss = criterion(risk_scores, events, durations)

            if torch.isnan(loss):
                print("NaN loss detected, skipping batch")
                continue
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            # add gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Calculate validation C-index
        val_cindex = evaluate_model(model, val_loader)
        
        # Save metrics
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        val_cindices.append(val_cindex)
        
        print(f"Fold {fold+1}, Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Val C-index: {val_cindex:.4f}")
        
        # Save best model
        if val_cindex > best_val_cindex:
            best_val_cindex = val_cindex

            if clinical_features[0] is not None and molecular_features[0] is not None:
                save_path = f'./outputs/checkpoints/imaging+clinical+molecular/best_fold{fold+1}_{feature}.pth'
            elif clinical_features[0] is not None:
                save_path = f'./outputs/checkpoints/imaging+clinical/best_fold{fold+1}_{feature}.pth'
            elif molecular_features[0] is not None:
                save_path = f'./outputs/checkpoints/imaging+molecular/best_fold{fold+1}_{feature}.pth'
            else:
                save_path = f'./outputs/checkpoints/imaging/best_fold{fold+1}_{feature}.pth'
            
            torch.save(model.state_dict(), save_path)
            print(f"Saved new best model with C-index: {val_cindex:.4f}")
    
    return train_losses, val_cindices

# Evaluation function
def evaluate_model(model, data_loader):
    model.eval()
    
    all_risk_scores = []
    all_events = []
    all_durations = []
    
    with torch.no_grad():
        for batch in data_loader:
            image_features = batch['image_features']
            treatments = batch['treatment'].to(device)
            events = batch['event'].numpy()
            durations = batch['duration'].numpy()
            patient_ids = batch['patient_id']

            clinical_features = batch['clinical'] if 'clinical' in batch.keys() else [None] * len(image_features)
            molecular_features = batch['molecular'] if 'molecular' in batch.keys() else [None] * len(image_features)

            batch_risk_scores = []
            batch_events = []
            batch_durations = []

            # Process each image separately
            for i, (image_feature, treatment, clinical_feature, molecular_feature) in enumerate(zip(image_features, treatments, clinical_features, molecular_features)):
                image_feature = image_feature.to(device)
                
                if clinical_feature is not None:
                    clinical_feature = clinical_feature.to(device)
                if molecular_feature is not None:
                    molecular_feature = molecular_feature.to(device)

                # Get risk score for this patient
                risk_score, attention_weights, _ = model(image_feature, treatment, clinical_feature, molecular_feature)

                if torch.isnan(risk_score):
                    print(f"NaN risk score detected for patient {patient_ids[i]}, skipping")
                    continue
                
                batch_risk_scores.append(risk_score.item())
                batch_events.append(events[i])
                batch_durations.append(durations[i])
            
            all_risk_scores.extend(batch_risk_scores)
            all_events.extend(batch_events)
            all_durations.extend(batch_durations)
    
    # Calculate C-index
    cindex = concordance_index(all_durations, -np.array(all_risk_scores), all_events)
    return cindex

def optimize_hyperparameters(data, feature, input_dim, feat_dir, clinical_attrs, use_clinical, use_molecular, n_trials=30):
    
    def objective(trial):
        # Hyperparameters to optimize
        hidden_dim = trial.suggest_int('hidden_dim', 32, 256, step=32)
        dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.7, step=0.1)
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
        
        # Initialize model with trial hyperparameters
        model = AttentionMIL(
            input_dim=input_dim, 
            hidden_dim=hidden_dim, 
            use_clinical=use_clinical, 
            use_molecular=use_molecular
        ).to(device)
        
        # Modify dropout rates in model
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.p = dropout_rate
        
        # Define loss function and optimizer with trial learning rate
        criterion = NegativeLogLikelihood()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)
        
        # Use 3-fold cross-validation for faster evaluation
        from sklearn.model_selection import StratifiedKFold
        kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        fold_cindices = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(data, data['event'])):
            # Split data for this fold
            data_train = data.iloc[train_idx].reset_index(drop=True)
            data_val = data.iloc[val_idx].reset_index(drop=True)
            
            # Create datasets and loaders
            train_dataset = MultimodalSurvivalDataset(feat_dir, data_train, clinical_attrs, use_clinical=use_clinical, use_molecular=use_molecular)
            train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=custom_collate)
            
            val_dataset = MultimodalSurvivalDataset(feat_dir, data_val, clinical_attrs, use_clinical=use_clinical, use_molecular=use_molecular)
            val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=custom_collate)
            
            # Train for fewer epochs during optimization
            _, val_cindices = train_model(model, train_loader, val_loader, criterion, optimizer, fold, feature, num_epochs=10)
            fold_cindices.append(max(val_cindices))  # Use best validation c-index from training
        
        # Return average validation c-index
        return np.mean(fold_cindices)
    
    # Create and run study
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    
    # Get best hyperparameters
    best_params = study.best_params
    print(f"Best hyperparameters: {best_params}")
    print(f"Best validation c-index: {study.best_value:.4f}")
    
    return best_params

def train(data, feature, input_dim, feat_dir, clinical_attrs, use_clinical, use_molecular, num_epochs):
    # # Run hyperparameter optimization first
    # print("Starting hyperparameter optimization...")
    # best_params = optimize_hyperparameters(
    #     data, feature, input_dim, feat_dir, clinical_attrs, 
    #     use_clinical, use_molecular, n_trials=30
    # )
    
    # # Use optimized hyperparameters
    # hidden_dim = best_params['hidden_dim']
    # dropout_rate = best_params['dropout_rate']
    # learning_rate = best_params['learning_rate']
    # print(hidden_dim, dropout_rate, learning_rate)

    if use_clinical and use_molecular:
        hidden_dim = 96
        dropout_rate = 0.4
        learning_rate = 3e-4
    elif use_clinical:
        hidden_dim = 256
        dropout_rate = 0.5
        learning_rate = 7e-4
    elif use_molecular:
        hidden_dim = 160
        dropout_rate = 0.3
        learning_rate = 5e-4
    else:
        hidden_dim = 224
        dropout_rate = 0.2
        learning_rate = 8e-4

    # Initialize model
    MIL_model = AttentionMIL(input_dim=input_dim, hidden_dim=hidden_dim, use_clinical=use_clinical, use_molecular=use_molecular).to(device)

    # Update dropout rates
    for module in MIL_model.modules():
        if isinstance(module, nn.Dropout):
            module.p = dropout_rate
    
    # Define loss function and optimizer
    criterion = NegativeLogLikelihood()
    optimizer = torch.optim.Adam(MIL_model.parameters(), lr=learning_rate, weight_decay=1e-3)

    kfold = StratifiedKFold(n_splits=5, shuffle=True)
    fold_cindices = []
    fold_risk_preds = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(data, data['event'])):
        print(f"\nFold {fold+1}/5")

        # Split data for this fold
        data_train = data.iloc[train_idx].reset_index(drop=True)
        data_val = data.iloc[val_idx].reset_index(drop=True)

        train_dataset = MultimodalSurvivalDataset(feat_dir, data_train, clinical_attrs, use_clinical=use_clinical, use_molecular=use_molecular)
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=custom_collate)

        val_dataset = MultimodalSurvivalDataset(feat_dir, data_val, clinical_attrs, use_clinical=use_clinical, use_molecular=use_molecular)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=custom_collate)

        # print(f"Training dataset size: {len(train_dataset)}")
        # print(f"Validation dataset size: {len(val_dataset)}")

        train_losses, val_cindices = train_model(MIL_model, train_loader, val_loader, criterion, optimizer, fold, feature, num_epochs)
        fold_cindices.append(val_cindices)
     
    # we want to compuate average & std validation c-index across all folds
    avg_val_cindex = np.mean(fold_cindices)
    std_val_cindex = np.std(fold_cindices)

    return avg_val_cindex, std_val_cindex, hidden_dim

def aggregate_attention_weights(patient_ids, all_attention_weights):
    """Average attention weights across folds for each patient"""
    from collections import defaultdict
    attention_dict = defaultdict(list)
    
    for fold_attention in all_attention_weights:
        for pid, attn in zip(patient_ids, fold_attention):
            attention_dict[pid].append(attn)
    
    # Average and normalize attention weights
    avg_attention = {}
    for pid, weights in attention_dict.items():
        stacked = np.stack(weights)
        avg = stacked.mean(axis=0)
        avg = (avg - avg.min()) / (avg.max() - avg.min())  # Normalize
        avg_attention[pid] = avg
        
    return avg_attention

# Function to predict risk score for all patients
def predict_risk_scores(model, data_loader):
    model.eval()

    all_risk_scores = []
    all_events = []
    all_durations = []
    all_patient_ids = []
    all_attention_weights = []
    all_treatments = []
    all_slide_ids = []
    all_image_embeds = []
    
    with torch.no_grad():
        for batch in data_loader:
            image_features = batch['image_features']
            treatments = batch['treatment'].to(device)
            events = batch['event'].numpy()
            durations = batch['duration'].numpy()
            patient_ids = batch['patient_id']
            slide_ids = batch['slide_id']

            clinical_features = batch['clinical'] if 'clinical' in batch.keys() else [None] * len(image_features)
            molecular_features = batch['molecular'] if 'molecular' in batch.keys() else [None] * len(image_features)
            
            batch_risk_scores = []
            batch_events = []
            batch_durations = []
            batch_patients = []
            batch_attention = []
            batch_treatments = []
            batch_slides = []
            batch_image_embeds = []
            
            # Process each image separately
            for i, (image_feature, treatment, clinical_feature, molecular_feature) in enumerate(zip(image_features, treatments, clinical_features, molecular_features)):
                image_feature = image_feature.to(device)
                
                if clinical_feature is not None:
                    clinical_feature = clinical_feature.to(device)
                if molecular_feature is not None:
                    molecular_feature = molecular_feature.to(device)

                # Get risk score for this patient
                risk_score, attention_weights, image_embed = model(image_feature, treatment, clinical_feature, molecular_feature)

                if torch.isnan(risk_score):
                    print(f"NaN risk score detected for patient {patient_ids[i]}, skipping")
                    continue

                batch_risk_scores.append(risk_score.item())
                batch_events.append(events[i])
                batch_durations.append(durations[i])
                batch_patients.append(patient_ids[i])
                batch_attention.append(attention_weights.cpu().numpy())
                batch_slides.append(slide_ids[i])
                batch_image_embeds.append(image_embed.cpu())

                values = [int(float(t)) for t in treatment.cpu()]
                # Define the mapping
                if values == [1, 0, 0]:
                    batch_treatments.append('io')
                elif values == [0, 1, 0]:
                    batch_treatments.append('chemo')
                elif values == [0, 0, 1]:
                    batch_treatments.append('combo')
            
            all_risk_scores.extend(batch_risk_scores)
            all_events.extend(batch_events)
            all_durations.extend(batch_durations)
            all_patient_ids.extend(batch_patients)
            all_attention_weights.extend(batch_attention)
            all_treatments.extend(batch_treatments)
            all_slide_ids.extend(batch_slides)
            all_image_embeds.extend(batch_image_embeds)
    
    # Create a dataframe with all results
    results_df = pd.DataFrame({
        'patient_id': all_patient_ids,
        'slide_id': all_slide_ids,
        'risk_score': all_risk_scores,
        'event': all_events,
        'duration': all_durations,
        'treatment': all_treatments
    })
    
    return results_df, all_attention_weights, all_image_embeds, all_slide_ids

def plot_KM_combined(treatment_data_dict, output_dir, feature, cutoffs=None, split='train', normalization_params=None):
    """
    Create a single Kaplan-Meier plot with 4 curves:
    - IO High Risk (dark blue)
    - IO Low Risk (light blue) 
    - Combo High Risk (dark red)
    - Combo Low Risk (light red)
    """
    
    def normalize_risk_scores(data, norm_params=None):
        if norm_params is None:
            min_score = data['avg_risk_score'].min()
            max_score = data['avg_risk_score'].max()
            norm_params = {'min_score': min_score, 'max_score': max_score}
        else:
            min_score = norm_params['min_score']
            max_score = norm_params['max_score']

        normalized = (data['avg_risk_score'] - min_score) / (max_score - min_score)
        return normalized, norm_params

    plt.clf()
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    colors = {
        'io_low': '#ff1816', #light red
        'io_high': '#8b0000', #dark red
        'combo_low': '#2020ff', #light blue
        'combo_high': '#02028c', #dark blue
    }
    
    median_vals = {}
    norm_params_dict = {}
    p_values = {}
    kmf_dict = {}
    
    # Process each treatment
    for treatment in ['io', 'combo']:
        results_df = treatment_data_dict[treatment]['risk_df'].copy()
        
        # Use test data if this is test split
        if split == 'test':
            results_df = treatment_data_dict[treatment]['test_risk_df'].copy()
        
        # Normalize risk scores
        if normalization_params and treatment in normalization_params:
            normalized_scores, norm_params = normalize_risk_scores(results_df, normalization_params[treatment])
        else:
            normalized_scores, norm_params = normalize_risk_scores(results_df, None)
            
        results_df.loc[:, 'normalized_risk_score'] = normalized_scores
        norm_params_dict[treatment] = norm_params
        
        # Determine cutoff
        if cutoffs and treatment in cutoffs:
            median_risk_score = cutoffs[treatment]
        else:
            median_risk_score = results_df['normalized_risk_score'].median()
        median_vals[treatment] = median_risk_score
        
        # Convert to months
        results_df['duration_months'] = results_df['duration'] / (365.25/12)
        
        # Split into high and low risk groups
        low_risk_df = results_df[results_df['normalized_risk_score'] < median_risk_score]
        high_risk_df = results_df[results_df['normalized_risk_score'] >= median_risk_score]
        
        # Create KM fitters
        kmf_low = KaplanMeierFitter()
        kmf_high = KaplanMeierFitter()
        
        # Treatment labels
        treatment_label = 'Pembro+Chemo' if treatment == 'combo' else 'Pembro'
        
        # Fit and plot
        kmf_low.fit(durations=low_risk_df['duration_months'], 
                   event_observed=low_risk_df['event'], 
                   label=f'{treatment_label} low risk')
        kmf_low.plot(ax=ax, ci_show=False, show_censors=True, 
                    color=colors[f'{treatment}_low'], linewidth=3)
        
        kmf_high.fit(durations=high_risk_df['duration_months'], 
                    event_observed=high_risk_df['event'], 
                    label=f'{treatment_label} high risk')
        kmf_high.plot(ax=ax, ci_show=False, show_censors=True, 
                    color=colors[f'{treatment}_high'], linewidth=3, linestyle='--')

        kmf_dict[f'{treatment}_low'] = kmf_low
        kmf_dict[f'{treatment}_high'] = kmf_high
        
        # Calculate p-value for this treatment
        results = logrank_test(low_risk_df['duration_months'], high_risk_df['duration_months'], 
                              event_observed_A=low_risk_df['event'], 
                              event_observed_B=high_risk_df['event'])
        p_values[treatment] = results.p_value
        
        # Save individual risk group files
        suffix = 'train&val' if split == 'train' else 'test'
        high_risk_df.to_csv(os.path.join(output_dir, f'high_risk_patients_{treatment}_{suffix}.csv'), index=False)
        low_risk_df.to_csv(os.path.join(output_dir, f'low_risk_patients_{treatment}_{suffix}.csv'), index=False)

    # Get the maximum time to determine xticks range
    max_time = max([kmf.timeline.max() for kmf in kmf_dict.values()])
    # Create xticks every 3 months from 0 to max_time
    xticks_list = list(range(0, int(max_time) + 3, 3))
    
    # Add at-risk counts for all 4 curves
    add_at_risk_counts(kmf_dict['io_low'], kmf_dict['io_high'], 
                       kmf_dict['combo_low'], kmf_dict['combo_high'],
                       ax=ax, fontsize=14, rows_to_show=['At risk'], xticks=xticks_list)
    
    # Calculate overall p-value (comparing all 4 groups)
    combo_data = treatment_data_dict['combo']['risk_df'] if split == 'train' else treatment_data_dict['combo']['test_risk_df']
    io_data = treatment_data_dict['io']['risk_df'] if split == 'train' else treatment_data_dict['io']['test_risk_df']
    
    overall_results = logrank_test(combo_data['duration'], io_data['duration'],
                                  event_observed_A=combo_data['event'], 
                                  event_observed_B=io_data['event'])
    overall_p_value = overall_results.p_value
    
    ax.set_xlabel('Time (months)', fontsize=18)
    ax.set_ylabel('PFS probability', fontsize=18)

    # Set x-axis ticks every 3 months
    from matplotlib.ticker import MultipleLocator
    ax.xaxis.set_major_locator(MultipleLocator(3))

    ax.tick_params(axis='both', labelsize=12)
    
    # Customize legend
    ax.legend(loc='center right', bbox_to_anchor=(0.98, 0.65), fontsize=14) # for imaging + molecular
    # ax.legend(loc='upper right', bbox_to_anchor=(1, 1), fontsize=14) # for imaging only
    
    plt.tight_layout()
    
    # Save plot
    suffix = 'train&val' if split == 'train' else 'test'
    plt.savefig(os.path.join(output_dir, f'KM_risk_score_combined_{suffix}.png'))
    print(p_values)
    
    # Return appropriate values based on split
    if split == 'train':
        return p_values, median_vals, norm_params_dict
    else:
        return p_values

def main(args):
    data_path = '' # TODO: add path to data folder here
    image_path = os.path.join(data_path, 'Images/processed_HE_images') # TODO: update to your H&E slides folder

    output_dir = './outputs/'
    output_test_df = pd.DataFrame(columns=['feature', 'treatment', 'c_index', 'p_value'])

    if args.use_clinical and args.use_molecular:
        ext = 'imaging+clinical+molecular'
    elif args.use_clinical:
        ext = 'imaging+clinical'
    elif args.use_molecular:
        ext = 'imaging+molecular'
    else:
        ext = 'imaging'
    output_dir += ext

    print("Output directory:", output_dir)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f'./outputs/checkpoints/{ext}', exist_ok=True)

    image_features = ['plip'] #['plip', 'uni', 'optimus', 'virchow']
    treatment_list = ['combo', 'io', 'chemo']

    all_data = []
    all_test_data = []

    # combine all data across treatments
    for treatment in treatment_list:
        print(f'processing {treatment} patients')
        data_train, data_test, _, _, clinical_attrs = load_pfs_data(data_path, image_path, treatment, args.use_clinical, args.use_molecular)
        
        # Add treatment column
        for df in [data_train, data_test]:
            df['treatment'] = treatment
        
        all_data.extend([data_train])
        all_test_data.extend([data_test])

    # Combine all data
    combined_data = pd.concat(all_data, ignore_index=True)
    all_test_data = pd.concat(all_test_data, ignore_index=True)

    def analyze_data_distribution(data):
        treatment_counts = data['treatment'].value_counts()
        treatment_event_counts = data.groupby('treatment')['event'].sum()
        treatment_event_rates = treatment_event_counts / treatment_counts
        
        print("\nTreatment counts:", treatment_counts.to_list())
        print("Events per treatment:", treatment_event_counts.to_list())
        print("Event rates:", treatment_event_rates.to_list())
        
    analyze_data_distribution(combined_data)
    analyze_data_distribution(all_test_data)

    # Using all patients (with treatment as a one-hot encoded feature), train 5 models on each kind of imaging features
    avg_val_cindices = []
    std_val_cindices = []

    for feature in image_features:
        print(f'training model, features from {feature}')
        feat_dir = os.path.join(data_path, f'Images/HE_features/{feature}') # TODO: update to your extracted features folder (output of extract_features.py)

        feature_output_dir = os.path.join(output_dir, feature)
        os.makedirs(feature_output_dir, exist_ok=True)

        if feature == 'resnet' or feature == 'plip':
            input_dim = 512
        elif feature == 'uni' or feature == 'optimus':
            input_dim = 1536
        elif feature == 'virchow':
            input_dim = 2560
        else:
            raise ValueError(f"Unknown feature extractor {feature}")
        
        avg_val_cindex, std_val_cindex, hidden_dim = train(combined_data, feature, input_dim, feat_dir, clinical_attrs, args.use_clinical, args.use_molecular, num_epochs=15)
        if args.use_clinical and args.use_molecular:
            hidden_dim = 96
            dropout_rate = 0.4
            learning_rate = 3e-4
        elif args.use_clinical:
            hidden_dim = 256
            dropout_rate = 0.5
            learning_rate = 7e-4
        elif args.use_molecular:
            hidden_dim = 160
            dropout_rate = 0.3
            learning_rate = 5e-4
        else:
            hidden_dim = 224
            dropout_rate = 0.2
            learning_rate = 8e-4

        # Now, we want to get average risk score for each patient across 5 folds - TREATMENT SPECIFIC
        print('evaluating model')
        train_val_dataset = MultimodalSurvivalDataset(feat_dir, combined_data, clinical_attrs, use_clinical=args.use_clinical, use_molecular=args.use_molecular)
        train_val_loader = DataLoader(train_val_dataset, batch_size=8, shuffle=False, collate_fn=custom_collate)

        test_dataset = MultimodalSurvivalDataset(feat_dir, all_test_data, clinical_attrs, use_clinical=args.use_clinical, use_molecular=args.use_molecular)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=custom_collate)

        risk_df = pd.DataFrame(columns=['slide_id'])
        test_risk_df = pd.DataFrame(columns=['slide_id'])

        all_attentions = []
        all_attentions_test = []
        for fold in range(5):

            MIL_model = AttentionMIL(input_dim=input_dim, hidden_dim=hidden_dim, use_clinical=args.use_clinical, use_molecular=args.use_molecular).to(device)
            MIL_model.load_state_dict(torch.load(f'./outputs/checkpoints/{ext}/best_fold{fold+1}_{feature}.pth'))

            results_df, fold_attention, fold_embeds, fold_slides = predict_risk_scores(MIL_model, train_val_loader)
            all_attentions.append(fold_attention) # fold_attention is a list of np arrays (shape is #patches,); all_attentions is a list of len 5 of this list of arrays

            results = results_df[['slide_id', 'risk_score']].rename(columns={'risk_score': f'fold{fold+1}_risk_score'})
            if risk_df.empty:
                risk_df = results
            else:
                risk_df = pd.merge(risk_df, results, on='slide_id')
            
            results_df_test, fold_attention_test, fold_embeds_test, fold_slides_test = predict_risk_scores(MIL_model, test_loader)
            all_attentions_test.append(fold_attention_test)

            results_test = results_df_test[['slide_id', 'risk_score']].rename(columns={'risk_score': f'fold{fold+1}_risk_score'})
            if test_risk_df.empty:
                test_risk_df = results_test
            else:
                test_risk_df = pd.merge(test_risk_df, results_test, on='slide_id')

        risk_df['treatment'] = results_df['treatment']
        risk_df['duration'] = results_df['duration']
        risk_df['event'] = results_df['event']
        risk_df['patient_id'] = results_df['patient_id']

        test_risk_df['treatment'] = results_df_test['treatment']
        test_risk_df['duration'] = results_df_test['duration']
        test_risk_df['event'] = results_df_test['event']
        test_risk_df['patient_id'] = results_df_test['patient_id']

        avg_attention = aggregate_attention_weights(risk_df['slide_id'], all_attentions)
        avg_attention_test = aggregate_attention_weights(test_risk_df['slide_id'], all_attentions_test)

        save_attn_path = f'./outputs/avg_attention/{ext}/{feature}'
        os.makedirs(save_attn_path, exist_ok=True)
        
        for pid, attn in avg_attention.items():
            new_pt_path = os.path.join(save_attn_path, f'{pid}.pt')
            torch.save({'avg_attention': torch.tensor(attn)}, new_pt_path)
        
        for pid, attn in avg_attention_test.items():
            new_pt_path = os.path.join(save_attn_path, f'{pid}.pt')
            torch.save({'avg_attention': torch.tensor(attn)}, new_pt_path)

        risk_score_cols = [f'fold{fold+1}_risk_score' for fold in range(5)]
        
        risk_df['avg_risk_score'] = risk_df[risk_score_cols].mean(axis=1, skipna=True)
        risk_df = risk_df.dropna()
        
        test_risk_df['avg_risk_score'] = test_risk_df[risk_score_cols].mean(axis=1, skipna=True)
        test_risk_df = test_risk_df.dropna()

        treatments = ['combo', 'io']
        all_treatment_data = {}
        all_test_data_dict = {}

        for treatment in treatments:
            risk_df_subset = risk_df.loc[risk_df['treatment'] == treatment]
            test_risk_df_subset = test_risk_df.loc[test_risk_df['treatment'] == treatment]
            
            combined_data_subset = combined_data.loc[combined_data['treatment'] == treatment]
            all_test_data_subset = all_test_data.loc[all_test_data['treatment'] == treatment]
            
            all_treatment_data[treatment] = {
                'risk_df': risk_df_subset,
                'test_risk_df': test_risk_df_subset,
                'combined_data': combined_data_subset,
                'all_test_data': all_test_data_subset
            }

        # Create the combined plot
        p_values, median_vals, norm_params_dict = plot_KM_combined(all_treatment_data, feature_output_dir, feature, split='train')

        # Test the model and create test plot
        p_values_test = plot_KM_combined(all_treatment_data, feature_output_dir, feature, 
                                        cutoffs=median_vals, split='test', 
                                        normalization_params=norm_params_dict)

        for treatment in ['combo', 'io']:
            risk_df_subset = risk_df.loc[risk_df['treatment'] == treatment]
            test_risk_df_subset = test_risk_df.loc[test_risk_df['treatment'] == treatment]

            p_value_test = p_values_test[treatment]
            cindex_test = concordance_index(test_risk_df_subset['duration'], -test_risk_df_subset['avg_risk_score'], test_risk_df_subset['event'])
            
            output_test_df = pd.concat([output_test_df, pd.DataFrame({
                'feature': [feature],
                'treatment': [treatment],
                'c_index': [cindex_test],
                'p_value': [p_value_test]
            })], ignore_index=True)

            imaging_df = pd.concat([risk_df_subset, test_risk_df_subset], ignore_index=True)
            imaging_df.to_csv(os.path.join(feature_output_dir, f'{treatment}_patient_risk_scores.csv'), index=False)

    print(output_test_df)
    output_test_df.to_csv(os.path.join(output_dir, 'test_results.csv'), index=False)

       
if __name__ == "__main__":
    set_seed()

    # example usage for imaging+molecular model: python imaging_model.py --use_molecular
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_clinical', action='store_true')
    parser.add_argument('--use_molecular', action='store_true')
    args = parser.parse_args()

    main(args)
