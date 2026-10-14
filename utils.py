import numpy as np
import random
import torch
import os
import pandas as pd
from typing import List
import ast
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def set_seed(seed: int = 42) -> None:
    """ Set random seed (default 42)"""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

def hpv_imputation(data_path: str) -> pd.DataFrame:
    """ Run binary HPV imputation for all patients without lab measurements """

    # get HPV measurements for patients with test values
    data = pd.read_csv(os.path.join(data_path, 'pfs_baseline_hpv_ecog_ihc_tnm_gender.tsv'), usecols = ['patient_id', 'isolate_analyte', 'rna_analysis_id', 'HPV_status', 'HPV_status_origin'], sep='\t', header=0) # TODO: update to your patient clinical and PFS outcome tsv file
    data = data.loc[data['isolate_analyte'] == 'rna']
    data = data.drop_duplicates(keep='first')

    hpv_data = data.copy()
    hpv_data['HPV_status'] = data['HPV_status'].map({'Negative': 0, 'Positive': 1})
    hpv_data = hpv_data.loc[hpv_data['HPV_status_origin'] == 'Known']
    hpv_data = hpv_data.drop_duplicates(keep='first')

    # Get CDKN2A expression values for all patients
    gene_df = pd.read_csv(os.path.join(data_path, 'g_rna_gene_expression_hnscc.tsv'), sep='\t', header=0) # TODO: update to your gene expression tsv file
    gene_df = gene_df.loc[gene_df['analysis_id'].isin(data['rna_analysis_id'])]
    gene_df = gene_df.pivot(index='patient_id', columns='gene_code', values='log2_gene_tpm_corrected')
    all_genes = list(gene_df.columns)
    unique_patients = list(gene_df.index)

    cdkn2a_id = 'ENSG00000147889'
    assert(cdkn2a_id in all_genes), "CDKN2A not in matrix"

    cdkn2a_df = gene_df[cdkn2a_id]

    hpv_test_patients = hpv_data['patient_id'].to_list()
    remaining_patients = list(set(unique_patients) - set(hpv_test_patients))

    hpv_test_df = cdkn2a_df.loc[hpv_test_patients]
    X_labeled = hpv_test_df.values.reshape(-1, 1)
    y_labeled = hpv_data['HPV_status'].values.reshape(-1, 1)

    # train logistic regression model on patients we have data for - predict binary hpv status from continuous expression value (essentially finding a threshold)
    X_train, X_test, y_train, y_test = train_test_split(X_labeled, y_labeled, test_size=0.3, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    X_unlabeled = cdkn2a_df.loc[remaining_patients].values.reshape(-1, 1)
    predicted_labels = model.predict(X_unlabeled)
    print(predicted_labels)

    hpv_all = hpv_data[['patient_id', 'HPV_status']].copy()

    predicted_df = pd.DataFrame({
        'patient_id': remaining_patients,
        'HPV_status': predicted_labels
    })

    hpv_all = pd.concat([hpv_all, predicted_df], ignore_index=True)
    print(hpv_all['HPV_status'].value_counts())

    hpv_all = hpv_all.rename(columns={'HPV_status': 'hpv'})

    os.makedirs('./outputs', exist_ok=True)
    hpv_all.to_csv('./outputs/hpv_imputed_pfs.csv', index=False)

    return hpv_all

def load_sig_genes(treatment_type: str, pfs: bool, mode: str, p_val: float = 0.05) -> List:
    """ Load precomputed significant genes for specific treatment type """

    df = pd.read_csv(f'./outputs/all_sig_genes_p={p_val}.csv', index_col=0)

    genes = df.loc[
        (df['treatment_type'] == treatment_type) &
        (df['mode'] == mode) &
        (df['pfs_bool'] == pfs),
        'sig_genes'
    ]

    return ast.literal_eval(genes.values[0])
