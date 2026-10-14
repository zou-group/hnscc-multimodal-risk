import os
import pandas as pd
import numpy as np
from sksurv.linear_model import CoxnetSurvivalAnalysis
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.plotting import add_at_risk_counts
from lifelines.utils import concordance_index

from utils import set_seed, load_sig_genes
from preprocessing import split_data, preprocess_data_pfs

def calculate_cindex(risk_df):
    """
    Calculate the concordance index (c-index) for a risk score model.
    
    Parameters:
    risk_df : pandas.DataFrame
        DataFrame containing 'risk_score', 'duration', and 'event' columns
    
    Returns:
    float
        The c-index value
    """
    # For c-index calculation, higher scores should indicate higher risk
    return concordance_index(
        event_times=risk_df['duration'],
        predicted_scores=-risk_df['risk_score'], 
        event_observed=risk_df['event']
    )

def run_lasso(data, alpha=0.1):
    gene_data = data.iloc[:, 2:].values
    gene_names = data.columns[2:].tolist()
    print("Length of gene_names:", len(gene_names))

    duration = data['duration'].values
    event = data['event'].values

    # Standardize gene data
    scaler = StandardScaler()
    gene_data_scaled = scaler.fit_transform(gene_data)

    # Create a structured array for survival analysis
    structured_data = np.array([(e,d) for e,d in zip(event, duration)], dtype=[('event', 'bool'),('duration', 'f8')])

    lasso_cox_model = CoxnetSurvivalAnalysis(l1_ratio=1.0, alphas=[alpha]) # L1 penalty

    try:
        lasso_cox_model.fit(gene_data_scaled, structured_data)
    except Exception as e:
        print(f"Error during fitting: {e}")
        raise

    coefficients = lasso_cox_model.coef_

    selected_indices = np.where(coefficients != 0)[0]
    selected_gene_names = [gene_names[i] for i in selected_indices]
    selected_gene_coefficients = coefficients[selected_indices].tolist()
    print(f'Number of selected genes: {len(selected_gene_names)}')

    return {'genes': selected_gene_names, 'coef': selected_gene_coefficients}

def calculate_risk_score(row, coefficients):
    return sum(coefficients[i] * row.iloc[i] for i in range(len(coefficients)))

def normalize_risk_scores(data, norm_params=None):
    if norm_params is None:
        # Training case - calculate from data
        min_score = data['risk_score'].min()
        max_score = data['risk_score'].max()
        norm_params = {'min_score': min_score, 'max_score': max_score}
    else:
        # Test case - use provided parameters
        min_score = norm_params['min_score']
        max_score = norm_params['max_score']
    
    normalized = (data['risk_score'] - min_score) / (max_score - min_score)
    return normalized, norm_params

def compute_risk_score(data_train, label_cols, unique_patients, alpha):
    lasso_dict = run_lasso(data_train, alpha=alpha)  
    data_lasso = data_train[lasso_dict['genes'] + label_cols].copy()
    lasso_coef = [x[0] for x in lasso_dict['coef']]
    data_lasso.loc[:, 'risk_score'] = data_lasso.apply(calculate_risk_score, coefficients=lasso_coef, axis=1)
    data_lasso.loc[:, 'normalized_risk_score'], norm_params = normalize_risk_scores(data_lasso)

    risk_df = pd.DataFrame({'risk_score': data_lasso['risk_score'], 'normalized_risk_score': data_lasso['normalized_risk_score'], 'event': data_lasso['event'], 'duration': data_lasso['duration']})
    
    cindex = calculate_cindex(risk_df)

    median_risk_score = risk_df['normalized_risk_score'].median()
    risk_df['median_risk_score'] = risk_df['normalized_risk_score'].apply(lambda x: 1 if x > median_risk_score else 0)

    return risk_df, median_risk_score, lasso_dict, cindex, norm_params

# Run risk score (created on train patients) on test patients
def test_risk_score(data_val, label_cols, lasso_dict, median_risk_score, norm_params):
    data_lasso = data_val[lasso_dict['genes'] + label_cols]  # based on train genes & coefs
    lasso_coef = [x[0] for x in lasso_dict['coef']]

    data_lasso = data_lasso.copy()
    data_lasso['risk_score'] = data_lasso.apply(calculate_risk_score, coefficients=lasso_coef, axis=1)
    data_lasso['normalized_risk_score'], _ = normalize_risk_scores(data_lasso, norm_params)

    risk_df = pd.DataFrame({'risk_score': data_lasso['risk_score'], 'normalized_risk_score': data_lasso['normalized_risk_score'], 'event': data_lasso['event'], 'duration': data_lasso['duration']})
    
    cindex = calculate_cindex(risk_df)

    risk_df['median_risk_score'] = risk_df['normalized_risk_score'].apply(lambda x: 1 if x > median_risk_score else 0) # based on train median_risk_score

    return risk_df, cindex

def plot_KM_combined_main(treatment_data_dict, alpha, use_clinical, split='train'):
    """
    Create a single Kaplan-Meier plot with 4 curves:
    - IO High Risk (dark blue)
    - IO Low Risk (light blue) 
    - Combo High Risk (dark red)
    - Combo Low Risk (light red)
    """
    
    plt.clf()
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    colors = {
        'io_low': '#ff1816', #light red
        'io_high': '#8b0000', #dark red
        'combo_low': '#2020ff', #light blue
        'combo_high': '#02028c', #dark blue
    }
    
    # Store all KM fitters for at-risk counts
    kmf_dict = {}
    p_values = {}
    
    # Process each treatment
    for treatment in ['io', 'combo']:
        if split == 'train':
            risk_df = treatment_data_dict[treatment]['train_risk_df'].copy()
            median_risk_score = treatment_data_dict[treatment]['median_risk_score']
            c_index = treatment_data_dict[treatment]['train_c_index']
        else:  # test
            risk_df = treatment_data_dict[treatment]['test_risk_df'].copy()
            median_risk_score = treatment_data_dict[treatment]['median_risk_score']
            c_index = treatment_data_dict[treatment]['test_c_index']
        
        # Convert to months
        risk_df['duration_months'] = risk_df['duration'] / (365.25/12)
        
        # Split into high and low risk groups
        low_risk_df = risk_df[risk_df['normalized_risk_score'] < median_risk_score]
        high_risk_df = risk_df[risk_df['normalized_risk_score'] >= median_risk_score]
        
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
        
        # Store KM fitters for at-risk counts
        kmf_dict[f'{treatment}_low'] = kmf_low
        kmf_dict[f'{treatment}_high'] = kmf_high
        
        # Calculate p-value for this treatment
        results = logrank_test(low_risk_df['duration_months'], high_risk_df['duration_months'], 
                              event_observed_A=low_risk_df['event'], 
                              event_observed_B=high_risk_df['event'])
        p_values[treatment] = results.p_value
        
        # Save individual risk group files
        suffix = 'train&val' if split == 'train' else 'test'
        save_path = './outputs/molecular+clinical' if use_clinical else './outputs/molecular'
        
        high_risk_df.to_csv(os.path.join(save_path, f'risk_score/high_risk_patients_{treatment}_{suffix}.csv'), index=False)
        low_risk_df.to_csv(os.path.join(save_path, f'risk_score/low_risk_patients_{treatment}_{suffix}.csv'), index=False)
    
    print(p_values)

    # Get the maximum time to determine xticks range
    max_time = max([kmf.timeline.max() for kmf in kmf_dict.values()])
    # Create xticks every 3 months from 0 to max_time
    xticks_list = list(range(0, int(max_time) + 3, 3))
    
    # Add at-risk counts for all 4 curves
    add_at_risk_counts(kmf_dict['io_low'], kmf_dict['io_high'], 
                       kmf_dict['combo_low'], kmf_dict['combo_high'],
                       ax=ax, fontsize=14, rows_to_show=['At risk'], xticks=xticks_list)
    
    ax.set_xlabel('Time (months)', fontsize=18)
    ax.set_ylabel('PFS probability', fontsize=18)

    # Set x-axis ticks every 3 months
    from matplotlib.ticker import MultipleLocator
    ax.xaxis.set_major_locator(MultipleLocator(3))

    ax.tick_params(axis='both', labelsize=12)
    
    # Customize legend
    ax.legend(loc='upper right', fontsize=14)
    
    plt.tight_layout()
    
    # Save plot
    suffix = 'train&val' if split == 'train' else 'test'
    save_path = './outputs/molecular+clinical' if use_clinical else './outputs/molecular'
    plt.savefig(os.path.join(save_path, f'risk_score/KM_risk_score_combined_{suffix}_alpha={alpha}.png'), dpi=300)

def run_risk_score(data_path, treatment, alpha, use_clinical):
    treatment_mapping = {'IO': 'io', 'IO+chemo': 'combo', 'IO+/-chemo': 'combo+io'}
    pfs = True

    lasso_results = []

    mode = 'rna+attr' if use_clinical else 'rna'

    if treatment in treatment_mapping.keys():
        t = treatment_mapping[treatment]
    else:
        t = treatment

    sig_genes = load_sig_genes(t, pfs, mode)  

    # process data
    data, clinical_attrs, label_cols, all_genes, unique_patients = preprocess_data_pfs(data_path = data_path, mode = mode, treatment_type = t)
    
    if 'pdl1_cps' in data.columns:
        data = data.drop('pdl1_cps', axis=1)
        if 'pdl1_cps' in clinical_attrs:
            clinical_attrs.remove('pdl1_cps')

    data_train, data_val, train_patients, val_patients = split_data(data = data, include_test=False, split = 'random', test_size=0.3)

    data_train = data_train[label_cols + sig_genes]
    data_val = data_val[label_cols + sig_genes]

    # compute the risk score on train data
    train_risk_df, median_risk_score, lasso_dict, train_c_index, norm_params = compute_risk_score(data_train, label_cols, unique_patients, alpha)
    # apply risk score to test data
    test_risk_df, test_c_index = test_risk_score(data_val, label_cols, lasso_dict, median_risk_score, norm_params)

    print(f"Training C-index: {train_c_index:.3f}")
    print(f"Testing C-index: {test_c_index:.3f}")
    
    res = {
        'train_risk_df': train_risk_df,
        'test_risk_df': test_risk_df,
        'median_risk_score': median_risk_score,
        'lasso_dict': lasso_dict,
        'train_c_index': train_c_index,
        'test_c_index': test_c_index,
        'train_patients': train_patients,
        'val_patients': val_patients,
        'norm_params': norm_params
    }

    # save information to a dict, which is later converted to a csv file
    result_dict = {
        'treatment_type': treatment,
        'PFS': pfs,
        'sig_genes': lasso_dict['genes'],
        'coefficients': lasso_dict['coef'],
        'median': median_risk_score,
        'train_cindex': train_c_index,
        'test_cindex': test_c_index
    }
    lasso_results.append(result_dict)

    train_risk_df['patient_id'] = train_patients
    test_risk_df['patient_id'] = val_patients

    combined_risk_df = pd.concat([train_risk_df, test_risk_df], ignore_index=True)

    output_dir = './outputs/molecular/risk_scores'
    os.makedirs(output_dir, exist_ok=True)

    combined_risk_df[['patient_id','normalized_risk_score','risk_score']].to_csv(
        f'{output_dir}/{t}_molecular_risk_scores.csv',
        index=False
    )

    print(f"Saved combined molecular risk scores to {t}_molecular_risk_scores.csv")

    if use_clinical:
        output_file = f'./outputs/molecular+clinical/lasso_results_alpha={alpha}.csv'
    else:
        output_file = f'./outputs/molecular/lasso_results_alpha={alpha}.csv'

    if os.path.exists(output_file):
        existing_df = pd.read_csv(output_file)
        new_row = pd.DataFrame([result_dict])
        updated_df = pd.concat([existing_df, new_row], ignore_index=True)
        updated_df.to_csv(output_file, index=False)
    else:
        new_df = pd.DataFrame([result_dict])
        new_df.to_csv(output_file, index=False)

    return res

def main(data_path):
    alpha = 0.1
    treatments = ['IO+chemo', 'IO', 'chemo'] # chemo risk score is needed for the imaging+molecular model
    use_clinical = [False]

    os.makedirs('./outputs/molecular/risk_score', exist_ok=True)
    os.makedirs('./outputs/molecular+clinical/risk_score', exist_ok=True)

    for c in use_clinical:
        treatment_data_dict = {}
        for treatment in treatments:
            print(f'use_clinical {c}, treatment: {treatment}, alpha = {alpha}')

            res = run_risk_score(data_path, treatment, alpha, c)
            t = 'combo' if treatment == 'IO+chemo' else treatment.lower()
            treatment_data_dict[t] = res

        plot_KM_combined_main(treatment_data_dict, alpha, c, split='train')
        plot_KM_combined_main(treatment_data_dict, alpha, c, split='test')
    
    print('done!')

if __name__ == "__main__":
    data_path = '' # TODO: add path to data folder here
    set_seed()

    main(data_path)
