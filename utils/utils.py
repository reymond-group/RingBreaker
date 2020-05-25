# +
import pandas as pd
import numpy as np
import pickle
import os
import sys
import swifter

from rdkit import Chem
from rdkit.Chem import AllChem, rdChemReactions, rdmolfiles, rdmolops, PandasTools
from rdkit import DataStructs
from rdkit.DataStructs import cDataStructs


# -

def remove_atom_maps(mol):
    """
    Iterates through the atoms in a Mol object and removes atom maps

    Parameters:
        mol (rdkit.Mol): Rdkit Mol Object

    Returns:
        mol (rdkit.Mol): Rdkit Mol Object without atom maps or 'nan' if error
    """
    try:
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        return mol
    except:
        return float('nan')


def calculate_number_rings(smiles):
    """
    Calculates the number of rings in a SMILES

    Parameters:
        smiles (str): SMILES of the target compound

    Returns:
        num_rings (int): The number of rings in the molecule or 'nan' if error
    """
    try:
        return AllChem.CalcNumRings(Chem.MolFromSmiles(smiles))
    except:
        return float('nan')

def smiles_to_binary(smiles):
    """
    Gets the binary representation of a molecule

    Parameters:
        smi (str): SMILES string of the component species

    Returns:
        binary_mol (bytes): byte string representation of the molecule or 'nan' if error
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        mol = remove_atom_maps(mol)
        return mol.ToBinary()
    except:
        return float('nan')

def smarts_to_binary(smarts):
    """
    Gets the binary representation of a molecule

    Parameters:
        smi (str): SMILES string of the component species

    Returns:
        binary_mol (bytes): byte string representation of the molecule or 'nan' if error
    """
    try:
        mol = Chem.MolFromSmarts(smarts)
        mol = remove_atom_maps(mol)
        return mol.ToBinary()
    except:
        return float('nan')

