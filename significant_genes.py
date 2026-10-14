from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

from lifelines import CoxPHFitter
from sklearn.linear_model import LinearRegression, LogisticRegression
from statsmodels.stats.multitest import fdrcorrection

from preprocessing import preprocess_data_pfs, split_data
from utils import set_seed


def process_HR(cph_summary: pd.DataFrame, name_feature: str) -> Tuple[str, float, float, float, float]:
    """
    Process result of CoxPH model to HR with 95% confidence intervals.
    
    Returns:
        HR_str: Formatted HR string
        HR: Hazard ratio value
        lower_ci: Lower confidence interval
        upper_ci: Upper confidence interval
        p_value: p-value
    """
    result = cph_summary.loc[name_feature, ['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'p']].values
    HR_str = '%.2f (%.2f, %.2f), %.4f' % tuple(result)
    HR, lower_ci, upper_ci, p_value = result
    return HR_str, HR, lower_ci, upper_ci, p_value


def fit_cox_model(data: pd.DataFrame, name_feature: str, mode: str, use_attr: bool) -> Tuple[str, float, float, float, float]:
    """
    Generalized function to fit Cox model with or without IPTW based on the feature and model type.
    
    Params:
        data (DataFrame): The data to fit. First two columns are label_cols (event + duration).
        name_feature (str): The feature (gene/treatment) to analyze.
        mode (str): 'rna', 'rna+attr', 'mutation', 'mutation+attr'
        use_attr (bool): Whether we have patient attributes and want to use IPTW (True) or not (False)
        
    Returns:
        HR_str: Formatted HR string
        HR: Hazard ratio value
        lower_ci: Lower confidence interval
        upper_ci: Upper confidence interval
        p_value: p-value
    """

    if use_attr:
        data_confounders = data.iloc[:, 2:].copy().drop(columns=[name_feature])
        df = pd.concat([data.iloc[:, :2], data[[name_feature]]], axis=1)
        X = data_confounders.values
        y = data[name_feature].values
        
        if 'rna' in mode:
            model = LinearRegression()
        elif 'mutation' in mode:
            model = LogisticRegression(solver='lbfgs', n_jobs=-1, class_weight='balanced')
        
        model.fit(X, y)
        
        if 'rna' in mode:
            df['weights'] = np.maximum(1e-10, model.predict(X))
        elif 'mutation' in mode:
            propensity_scores = model.predict_proba(X)[:, 1]
            df['weights'] = np.where(
                df[name_feature] == 1, 
                1 / propensity_scores, 
                1 / (1 - propensity_scores)
            )
        
        # Fit CoxPH model with weights
        cph = CoxPHFitter()
        cph.fit(df, 'duration', 'event', weights_col='weights', robust=True)
    else:
        # Fit CoxPH model without weights
        cph = CoxPHFitter()
        cph.fit(data, 'duration', 'event', robust=True)
    
    # Process Hazard Ratio
    HR_str, HR, lower_ci, upper_ci, p_value = process_HR(cph.summary, name_feature)
    return HR_str, HR, lower_ci, upper_ci, p_value


def analysis(data: pd.DataFrame, all_genes: List[str], patient_attrs: List[str], label_cols: List[str], mode: str, use_attr: bool) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, Dict]]:
    """
    Generalized function to perform Cox regression analysis for multiple genes.
    
    Params:
        data (DataFrame): The overall data containing genes and other columns.
        all_genes (list): List of genes to analyze.
        patient_attrs (list): Attribute columns (confounders).
        label_cols (list): Basic columns (e.g., 'duration', 'event').
        mode (str): 'rna', 'rna+attr', 'mutation', 'mutation+attr'
        use_attr (bool): Whether to use IPTW (True) or not (False).
    
    Returns:
        coef_dict: Dictionary of coefficients for each gene.
        p_dict: Dictionary of p-values for each gene.
        hr_dict: Dictionary with complete HR information for each gene.
    """
    coef_dict = {}
    p_dict = {}
    hr_dict = {}

    for g in tqdm(all_genes, total=len(all_genes)):
        if 'attr' in mode:
            data_fit = data[label_cols + patient_attrs + [g]]
        else:
            data_fit = data[label_cols + [g]] 

        if 'rna' not in mode:
            # Skip if gene contains all 1s or all 0s
            if (data_fit[g] != 1).all() or (data_fit[g] != 0).all():
                print(f'ERROR: gene {g} contains all 1s or all 0s')
                continue

            # Ensure events are not perfectly separable
            loc_0 = data_fit.index[data_fit[g] == 0]
            loc_1 = data_fit.index[data_fit[g] == 1]
            mask = (
                (data_fit.loc[loc_0, 'event'] == 0).all() |
                (data_fit.loc[loc_0, 'event'] == 1).all() |
                (data_fit.loc[loc_1, 'event'] == 0).all() |
                (data_fit.loc[loc_1, 'event'] == 1).all()
            )
            
            if mask:
                print(f'ERROR: event same for all patients with mutation in gene {g}')
                continue

        # Remove perfectly uniform columns
        data_fit = data_fit.loc[:, ~(data_fit == 1).all() & ~(data_fit == 0).all()]

        try:
            HR_str, HR, lower_ci, upper_ci, p_value = fit_cox_model(data_fit, g, mode, use_attr=use_attr)

            if HR_str:
                print(f'HR for Gene {g}: {HR_str}')
                
                coef_dict[g] = HR  # Store HR value as coefficient
                p_dict[g] = p_value
                # Store complete HR information
                hr_dict[g] = {
                    'HR': HR,
                    'lower_ci': lower_ci,
                    'upper_ci': upper_ci,
                    'p_value': p_value,
                    'HR_str': HR_str
                }

        except Exception as e:
            print(f"Error fitting Cox model for gene {g}: {e}")

    return coef_dict, p_dict, hr_dict


def FDR_correction(coef_dict: Dict[str, float], p_dict: Dict[str, float], hr_dict: Dict[str, Dict], p_val: float = 0.05) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, Dict]]:
    """
    Perform FDR correction on the p values
    
    Params:
        coef_dict: Dictionary of coefficients for each gene.
        p_dict: Dictionary of p-values for each gene.
        hr_dict: Dictionary with complete HR information for each gene.
        p_val: Significance cutoff to use (default 0.05)

    Returns:
        coef_dict: Dictionary of coefficients for each gene.
        new_p_dict: Updated dictionary of p-values for each gene after FDR correction
        hr_dict: Updated dictionary with complete HR information

    """

    p = list(p_dict.values())
    p = np.array(p)
    notnanidx = ~np.isnan(p)

    print("Num p significant: ", np.sum(p <= p_val))
    q = fdrcorrection(pvals=p[notnanidx], alpha=p_val, method='indep', is_sorted=False)[1]
    q_mat = np.ones(p.shape)
    q_mat[notnanidx] = q
    print("Num q significant: ", np.sum(q <= p_val))

    new_p_dict = {}
    
    for i, k in enumerate(coef_dict):
        new_p_dict[k] = q_mat[i]
        if q_mat[i] > p_val:
            coef_dict[k] = np.nan

            # Update the adjusted p-value in hr_dict
            if k in hr_dict:
                hr_dict[k]['adjusted_p_value'] = q_mat[i]
                hr_dict[k]['significant'] = False
        else:
            # Mark as significant in hr_dict
            if k in hr_dict:
                hr_dict[k]['adjusted_p_value'] = q_mat[i]
                hr_dict[k]['significant'] = True

    assert(coef_dict.keys() == new_p_dict.keys())
    return coef_dict, new_p_dict, hr_dict


def find_significant_genes(data_train: pd.DataFrame, all_genes: List[str], patient_attrs: List[str], label_cols: List[str], mode: str, p_val: float = 0.05) -> Tuple[List[str], Dict[str, float], Dict[str, float], Dict[str, Dict]]:
    """
    Find significant genes given mode and treatment_type

    Params:
        data_train: DataFrame with train data
        all_genes: optional list of gene names
        patient_attrs: list of attributes
        label_cols: list of labels
        mode (str): 'rna', 'rna+attr', 'mutation', 'mutation+attr'
        p_val: significance cutoff to use (default 0.05)

    Returns:
        sig_genes: Significant gene names (sorted)
        filtered_coef_dict: Filtered dictionary of coefficients for each gene (only ones with significant p-values)
        p_dict: Ddictionary of p-values for each gene after FDR correction
        hr_dict: Dictionary with complete HR information for each gene
    """
    print(f'p val = {p_val}')

    use_attr = True if 'attr' in mode else False
    coef_dict, p_dict, hr_dict = analysis(data_train, all_genes, patient_attrs, label_cols, mode, use_attr = use_attr)

    coef_dict, p_dict, hr_dict = FDR_correction(coef_dict, p_dict, hr_dict, p_val=p_val)

    filtered_coef_dict = {key: value for key, value in coef_dict.items() if not np.isnan(value)}
    sig_genes = sorted(set(key for key in filtered_coef_dict.keys()))

    return sig_genes, filtered_coef_dict, p_dict, hr_dict

def save_sig_genes():
    set_seed()

    treatment_type_list = ['combo', 'io', 'chemo']
    pfs_list = [True]
    p_val_list = [0.05]
    all_modes = ['rna']
    data_path = '' # TODO: add path to data folder here
    base_dir = './outputs/sig_genes'

    for p_val in p_val_list:
        sig_genes_df = pd.DataFrame(columns=['treatment_type', 'mode', 'pfs_bool', 'sig_genes'])
        
        for pfs in pfs_list:
            for t in treatment_type_list:
                for mode in all_modes:
                    print(f'Running pfs: {pfs}, treatment: {t}, mode: {mode}')

                    output_dir = os.path.join(base_dir, f'mode={mode}_treatment={t}')
                    os.makedirs(output_dir, exist_ok=True)

                    data, patient_attrs, label_cols, all_genes, unique_patients = preprocess_data_pfs(data_path = data_path, mode = mode, treatment_type = t)

                    if 'pdl1_cps' in data.columns:
                        data = data.drop('pdl1_cps', axis=1)
                        if 'pdl1_cps' in patient_attrs:
                            patient_attrs.remove('pdl1_cps')

                    data_train, data_val, train_patient_ids, val_patient_ids = split_data(data = data, include_test=False, split = 'random', test_size=0.3)

                    # Get significant genes and HR information
                    sig_genes, filtered_coef_dict, p_dict, hr_dict = find_significant_genes(data_train = data_train, all_genes = all_genes, patient_attrs = patient_attrs, label_cols = label_cols, mode = mode, p_val = p_val)
                    sig_genes_df = pd.concat([sig_genes_df, pd.DataFrame([[t, mode, pfs, sig_genes]], columns=sig_genes_df.columns)], ignore_index=True)

                    # Save HR information for all genes
                    hr_df = pd.DataFrame.from_dict(hr_dict, orient='index')
                    hr_df.to_csv(f"{output_dir}/hr_values_{t}_{mode}_pfs={pfs}_p={p_val}.csv")

        sig_genes_df.to_csv(f'./outputs/all_sig_genes_p={p_val}.csv')


if __name__ == "__main__":
    save_sig_genes()
