from utils import set_seed
import pandas as pd

from preprocessing import preprocess_data_pfs, split_data
from multivariate import MultivariateModel

def main(data_path):
    treatment_list = ['io', 'combo']

    all_c_idx = []
    all_c_idx_val = []
    all_n = []

    for treatment in treatment_list:
        print(f'TREATMENT: {treatment}')
        data, clinical_attrs, label_cols, all_genes, unique_patients = preprocess_data_pfs(data_path,  mode='attr', treatment_type = treatment)

        if 'pdl1_cps' in data.columns:
            data = data.drop('pdl1_cps', axis=1)
        if 'pdl1_cps' in clinical_attrs:
            clinical_attrs.remove('pdl1_cps')

        data_train, data_val, train_patients, val_patients = split_data(data, include_test=False, split='random', test_size=0.3)
        data = pd.concat([data_train, data_val], ignore_index=True)
        all_n.append(len(data))

        # run multivariate CoxPH on clinical features
        multivariate_model = MultivariateModel(data_train=data_train, data_val=data_val, clinical_attrs=clinical_attrs, modality='clinical')
        c_idx, c_idx_val = multivariate_model.run_analysis()

        all_c_idx.append(c_idx)
        all_c_idx_val.append(c_idx_val)

    for treatment, n, c_idx_val in zip(['Pembro', 'Pembro+Chemo'], all_n, all_c_idx_val):
        print(f'{treatment} (n={n}) CoxPH test c-index: {c_idx_val[0]:.3f}')

if __name__ == "__main__":
    set_seed()

    data_path = '' # TODO: add path to data folder here
    main(data_path)
