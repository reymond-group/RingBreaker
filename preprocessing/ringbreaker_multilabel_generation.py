import os
import pandas as pd
import numpy as np
import swifter
import argparse
import pickle 
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import AllChem, rdChemReactions, rdmolfiles, rdmolops
from rdkit.DataStructs import cDataStructs

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, MultiLabelBinarizer
from scipy import sparse

def smiles_to_inchi(product):
    """Converts a single SMILES into an InChI Key

    Parameters:
        product (str): The SMILES string corresponing to the product/molecule of choice.
    
    Returns:
        InChI Key (str): The InChI Key corresponing to the product/molecule of choice.
    """
    
    m = Chem.MolFromSmiles(product)
    for atom in m.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToInchiKey(m)

def smiles_to_ecfp(product, size=2048):
    """Converts a single SMILES into an ECFP4

    Parameters:
        product (str): The SMILES string corresponing to the product/molecule of choice.
        size (int): Size (dimensions) of the ECFP4 vector to be calculated.
    
    Returns:
        ecfp4 (arr): An n dimensional ECFP4 vector of the molecule.
    """
    mol = Chem.MolFromSmiles(product)
    ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=size)
    arr = np.zeros((0,), dtype=np.int8)
    cDataStructs.ConvertToNumpyArray(ecfp, arr)
    return arr


def delete_row_csr(mat, i):
    if not isinstance(mat, sparse.csr_matrix):
        raise ValueError("works only for CSR format -- use .tocsr() first")
    n = mat.indptr[i+1] - mat.indptr[i]
    if n > 0:
        mat.data[mat.indptr[i]:-n] = mat.data[mat.indptr[i+1]:]
        mat.data = mat.data[:-n]
        mat.indices[mat.indptr[i]:-n] = mat.indices[mat.indptr[i+1]:]
        mat.indices = mat.indices[:-n]
    mat.indptr[i:-1] = mat.indptr[i+1:]
    mat.indptr[i:] -= n
    mat.indptr = mat.indptr[:-1]
    mat._shape = (mat._shape[0]-1, mat._shape[1])

"""
example script call:

ringbreaker_multilabel_generation.py -fp 2048 -d '../data/uspto_ringformations.csv' -o '../data/' -dn 'uspto' -fn 'uspto_ringbreaker'

"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate multilabel data with a pretrained network')
    parser.add_argument('-fp', '--fp_size', type = int, default = 2048,
                        help = 'Specify the fingerprint size, fingerprint type is ECFP4')
    parser.add_argument('-d', '--data', type = str, default = "../data/uspto_ringformations.csv",
                        help = 'Specify the absolute path to the dataset (.csv) of ringformations')
    parser.add_argument('-o', '--out_path', type = str, default = '../data/',
                        help = 'Specify the absolute path to the output, a folder will be created at the location containing the specified dataset name')
    parser.add_argument('-dn', '--dataset_name', type = str, default = 'uspto',
                        help = 'Specify the dataset name, this will be created in the ')
    parser.add_argument('-fn', '--filename', type = str, default = 'uspto_ringbreaker',
                        help = 'Specify the dataset name, this will be created in the ')
    args = parser.parse_args()

    fp_size = args.fp_size
    data = args.data
    out_path = args.out_path
    dataset_name = args.dataset_name
    if os.path.exists(out_path + dataset_name):
        print('Directory exists')
        pass
    else: 
        os.mkdir(out_path + dataset_name)

    output_name = args.filename


    df = pd.read_csv(data,
                    usecols=["reaction_hash", "products", "template_hash", "selectivity"])
    df = df.drop_duplicates(subset="reaction_hash")
    df['inchikey'] = df['products'].swifter.apply(smiles_to_inchi)

    processed_dataset = {}

    template_labels = LabelEncoder()
    df['template_code'] = template_labels.fit_transform(df['template_hash'])

    unique_products = df.groupby('inchikey')
    unique_products = unique_products.size().sort_values(ascending=False)

    for key in tqdm(unique_products.keys()):
        label_df = df[df['inchikey'] == key]
        smiles = label_df['products'].values[0]
        on_bits = list(set(df[df['inchikey'] == key]['template_code'].values))
        processed_dataset[smiles] = on_bits

    inputs = []
    mlb = MultiLabelBinarizer(sparse_output=True)
    labels = mlb.fit_transform(list(processed_dataset.values()))

    for i, key in enumerate(tqdm(processed_dataset.keys())):
        try:
            inputs.append(smiles_to_ecfp(key, fp_size))
        except:
            delete_row_csr(labels, i)

    inputs = sparse.lil_matrix(inputs)
    inputs = inputs.tocsr()

    train_labels, test_labels = train_test_split(labels, test_size = 0.1, random_state = 42, shuffle = True)
    val_labels, test_labels = train_test_split(test_labels, test_size = 0.5, random_state = 42, shuffle = True)
    sparse.save_npz(out_path + dataset_name + '/' + output_name + '_training_labels.npz', train_labels, compressed = True)
    sparse.save_npz(out_path + dataset_name + '/' + output_name + '_validation_labels.npz', val_labels, compressed = True)
    sparse.save_npz(out_path + dataset_name + '/' + output_name + '_testing_labels.npz', test_labels, compressed = True)

    train_inputs, test_inputs = train_test_split(inputs, test_size = 0.1, random_state = 42, shuffle = True)
    val_inputs, test_inputs = train_test_split(test_inputs, test_size = 0.5, random_state = 42, shuffle = True)
    sparse.save_npz(out_path + dataset_name + '/' + output_name + '_training_inputs.npz', train_inputs, compressed = True)
    sparse.save_npz(out_path + dataset_name + '/' + output_name + '_validation_inputs.npz', val_inputs, compressed = True)
    sparse.save_npz(out_path + dataset_name + '/' + output_name + '_testing_inputs.npz', test_inputs, compressed = True)










