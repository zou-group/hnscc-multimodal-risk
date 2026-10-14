from typing import List, Tuple
import os
import pandas as pd

from utils import hpv_imputation

def preprocess_data_pfs(data_path: str, mode: str, treatment_type: str)-> Tuple[pd.DataFrame, List[str], List[str], List[str], List[str]]:
    """
    Preprocess the data and covariates.

    Params:
        data_path: path to the Data folder
        mode: rna, attr, or rna+attr
        treatment_type: 'chemo', 'io', 'combo', 'combo+io', or 'agnostic'

    Returns:
        data: DataFrame with patients, clinical attributes, optional genetic information, and labels
        clinical_attrs: list of clinical attributes
        label_cols: list of label columns
        all_genes: list of genes (None if mode is 'attr')
        unique_patients: list of patient IDs
    """

    clinical_attrs = ['pdl1_cps', 'hpv', 'gender', 'age', 'biopsy_hnscc', 'biopsy_lung']
    label_cols = ['duration', 'event']

    # load all PFS data from Tempus
    df = pd.read_csv(os.path.join(data_path, 'pfs_baseline_hpv_ecog_ihc_tnm_gender.tsv'), sep='\t', header=0) # TODO: update to your patient clinical and PFS outcome tsv file
    df = df.loc[df['isolate_analyte'] == 'rna']
    df = df.drop_duplicates(keep='first')

    # subset by patients who received specified treatment type as LOT1
    df['regimen_'] = df['regimen_'].str.lower()

    if treatment_type == 'agnostic':
        df = df
    elif treatment_type == 'combo+io':
        df = df[(df['regimen_'] == 'io') | (df['regimen_'] == 'combo')]
    else:
        df = df[df['regimen_'] == treatment_type]

    unique_patients = list(df['patient_id'].unique())
    df = df[['patient_id', 'rna_analysis_id', 'gender', 'age', 'biopsy_site', 'LOT1_start_date', 'PFS_Ti_days', 'PFS_status', 'pdl1_cps']]
    df = df.rename(columns={'LOT1_start_date': 'LOT1_start', 'PFS_Ti_days': 'duration', 'PFS_status': 'event'})

    df['LOT1_start'] = pd.to_datetime(df['LOT1_start'])

    # get genetic information
    if mode in ['rna', 'rna+attr']:
        gene_df = pd.read_csv(os.path.join(data_path, 'g_rna_gene_expression_hnscc.tsv'), sep='\t', header=0) # TODO: update to your gene expression tsv file
        gene_df = gene_df.loc[gene_df['analysis_id'].isin(df['rna_analysis_id'].to_list())]
        gene_df = gene_df.pivot(index='patient_id', columns='gene_code', values='log2_gene_tpm_corrected')
        all_genes = list(gene_df.columns)
    else:
        gene_df = None
        all_genes = None

    if 'attr' not in mode:
        label_df = df[['patient_id', 'LOT1_start', 'event', 'duration']]
        data = pd.merge(label_df, gene_df, on='patient_id')
        return data, None, label_cols, all_genes, unique_patients

    df = df.drop('rna_analysis_id', axis=1)

    # format and process clinical attributes
    df['gender'] = df['gender'].map({'Male': 0, 'Female': 1}) # female is positive class, male is negative
    df['age'] = df['age'].astype(int)

    HNSCC_sites = ["Accessory sinus", "Base of tongue","Cheek mucosa","Ethmoid sinus","Floor of mouth","Glottis","Gum",
               "Hard palate","Head, face or neck","Hypopharynx","Laryngeal cartilage","Larynx","Lateral wall of nasopharynx",
               "Lingual tonsil","Lip","Maxillary sinus","Mouth","Nasal cavity","Nasopharynx","Oropharynx","Palate",
               "Pharynx","Posterior wall of nasopharynx","Pyriform sinus","Retromolar area","Soft palate",
               "Sphenoid sinus","Subglottis","Supraglottis","Tongue","Tonsil","Uvula","Vallecula","Ventral surface of tongue"]
    Lung_sites = ["Heart","Lower lobe, lung","Lung","Main bronchus","Mediastinum","Middle lobe, lung","Pleura",
              "Thymus","Trachea","Upper lobe, lung"]

    df['biopsy_lung'] = df['biopsy_site'].isin(Lung_sites).astype(int)
    df['biopsy_hnscc'] = df['biopsy_site'].isin(HNSCC_sites).astype(int)
    df = df.drop('biopsy_site', axis=1)

    # load imputed hpv data
    hpv_path = './outputs/hpv_imputed_pfs.csv'
    if os.path.exists(hpv_path):
        hpv_df = pd.read_csv(hpv_path)
    else:
        hpv_df = hpv_imputation(data_path)

    data = pd.merge(df, hpv_df, on='patient_id')
    data = data[['patient_id', 'LOT1_start'] + clinical_attrs + label_cols]

    if gene_df is not None:
        data = pd.merge(data, gene_df, on='patient_id')

    return data, clinical_attrs, label_cols, all_genes, unique_patients


def split_data(data: pd.DataFrame, include_test: bool, split: str = 'temporal', test_size=0.15):
    """
    Split data in train/val/test sets

    Params:
        data: DataFrame with data
        include_test: if we want to create a test set (True) or not (False)
        split: random or temporal (default temporal)

    Returns:
        data_train: train DataFrame
        data_val: val DataFrame
        data_test (optional): test DataFrame
        train_patients: list of train patient IDs
        val_patients: list of val patient IDs
        test_patients (optional): list of test patient IDs
    """

    if split == 'random':
        from sklearn.model_selection import train_test_split
        data_train = data.drop('LOT1_start', axis=1)

        stratify_col = data_train['event']

        if include_test:

            data_train_val, data_test = train_test_split(
                data_train, test_size=test_size, stratify=stratify_col, random_state=42
            )

            test_patients = data_test['patient_id'].to_list()
            data_test = data_test.drop('patient_id', axis=1)

            adjusted_val_size = test_size / (1 - test_size)

            stratify_col = data_train_val['event']
            data_train, data_val = train_test_split(
                data_train_val, test_size=adjusted_val_size, stratify=stratify_col, random_state=42
            )

        else:
            stratify_col = data_train['event']
            data_train, data_val = train_test_split(
                data_train, test_size=test_size, stratify=stratify_col, random_state=42
            )

        # Shuffle the training data
        data_train = data_train.sample(frac=1, random_state=42)

        train_patients = data_train['patient_id'].to_list()
        val_patients = data_val['patient_id'].to_list()

        data_train = data_train.drop('patient_id', axis=1)
        data_val = data_val.drop('patient_id', axis=1)

    else:
        data_sorted = data.sort_values(by='LOT1_start').drop('LOT1_start', axis=1)

        total_rows = len(data_sorted)
        if include_test:
            train_rows = int(total_rows * (1 - 2*test_size))
            val_rows = int(total_rows * test_size)
        else:
            train_rows = int(total_rows * (1 - test_size))
            val_rows = int(total_rows * test_size)

        data_train = data_sorted.iloc[:train_rows]
        data_train = data_train.sample(frac=1).reset_index(drop=True) # shuffles data

        if include_test:
            data_val = data_sorted.iloc[train_rows:train_rows + val_rows]
            data_val = data_val.sample(frac=1).reset_index(drop=True)

            data_test = data_sorted.iloc[train_rows + val_rows:]
            data_test = data_test.sample(frac=1).reset_index(drop=True)

            test_patients = data_test['patient_id'].to_list()
            data_test = data_test.drop('patient_id', axis=1)
        else:
            data_val = data_sorted.iloc[train_rows:]
            data_val = data_val.sample(frac=1).reset_index(drop=True)

        train_patients = data_train['patient_id'].to_list()
        val_patients = data_val['patient_id'].to_list()

        data_train = data_train.drop('patient_id', axis=1)
        data_val = data_val.drop('patient_id', axis=1)

    if include_test:
        return data_train, data_val, data_test, train_patients, val_patients, test_patients
    else:
        return data_train, data_val, train_patients, val_patients
