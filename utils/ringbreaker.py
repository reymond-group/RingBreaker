import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pandas as pd
import numpy as np
import time
import functools
import pickle
from functools import partial

import keras
from keras.layers import Dense, Dropout, Activation
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras.metrics import top_k_categorical_accuracy
from keras.utils import Sequence
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint, Callback, TensorBoard, ReduceLROnPlateau

from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

from rdkit import Chem
from rdkit.Chem import AllChem, rdChemReactions, rdmolfiles, rdmolops, rdMolDescriptors
from rdkit.DataStructs import cDataStructs
from rdkit.Chem.Draw import IPythonConsole #Needed to show molecules
from rdkit.Chem.Draw.MolDrawing import MolDrawing, DrawingOptions #Only needed if modifying defaults

import sys
from rdchiral import main as rdc

class Model:
    """
    Class to facilate the predicition of synthetic steps for ring systems.
    Can only be used for the predicition of single synthetic steps for a given ring system.
    """
    def __init__(self, dataset="uspto_ringbreaker", mask=True):
        directory = os.getcwd()
        self.models = {"uspto_ringbreaker": os.path.join(directory, "models/checkpoints/weights.hdf5"),
                       "uspto_standard": os.path.join(directory,"models/uspto_standard/weights.hdf5"),
        }
        self.templates = {"uspto_ringbreaker": os.path.join(directory,"data/uspto_ringformations.csv"),
                          "uspto_standard": os.path.join(directory,"data/standard_uspto_template_library.csv"),
        }
        if 'filtered' in dataset:
            dataset = '_'.join(dataset.split('_')[:2])
            mask = True

        self.model = self.models[dataset]
        self.templates = self.templates[dataset]
        self.mask = mask
        
         #Initalise template library
        if 'ringbreaker' in dataset:
            lib = pd.read_csv(self.templates)
            lib = lib.drop(["Unnamed: 0", "index", "selectivity", "outcomes", "ring_change"], axis=1)
            template_labels = LabelEncoder()
            lib['template_code'] = template_labels.fit_transform(lib['template_hash'])
            lib = lib.drop(["reaction_hash", "reactants"], axis=1)
            self.lib = lib.set_index('template_code').T.to_dict('list')
            self.mask = False
        else:
            lib = pd.read_csv(self.templates, index_col=0, header=None, names=["index", "ID", "reaction_hash", "reactants", "products", "classification", "retro_template", "template_hash", "selectivity", "outcomes", "template_code"])
            lib = lib.drop(["reaction_hash", "reactants", "selectivity", "outcomes"], axis=1)
            self.lib = lib.set_index('template_code').T.to_dict('list')
        
        if mask == True and 'uspto' in dataset:
            with open("../data/uspto_filter_array.pkl", "rb") as f:
                self.filter_array = pickle.load(f)
        else:
            self.filter_array = None
        
        #Initialise policy
        # Define top k accuracies
        top10_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=10)
        top10_acc.__name__ = 'top10_acc'

        top50_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=50)
        top50_acc.__name__ = 'top50_acc'
        
        self.policy = load_model(self.model, custom_objects={'top10_acc': top10_acc, 'top50_acc': top50_acc})

    def smiles_to_ecfp(self, product, size=2048):
        """Converts a single SMILES into an ECFP4

        Parameters:
            product (str): The SMILES string corresponing to the product/molecule of choice.
            size (int): Size (dimensions) of the ECFP4 vector to be calculated.

        Returns:
            ecfp4 (arr): An n dimensional ECFP4 vector of the molecule.
        """
        mol = Chem.MolFromSmiles(product)
        ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits = 2048)
        arr = np.zeros((0,), dtype=np.int8)
        cDataStructs.ConvertToNumpyArray(ecfp, arr)
        return arr.reshape([1,2048])

    def get_templates(self, prediction, lib, topN=50):
        """Given a prediction and the template library, obtains the top N predicted templates

        Parameters:
            prediction (np.array): The prediction from the softmax layer as a numpy array
            lib (dict): The template library as a dictionary where the template hash is the key

        Returns:
            predicted_templates (dict): The predicted templates and associated meta-data
        """
        sort_pred = np.argsort(prediction)[::-1]

        predicted_templates = {}
        for i in range(1, topN+1):
            pred_temp = lib[sort_pred[-1][-i]]
            predicted_templates[i] = {'template': pred_temp[-2], 'classification': pred_temp[-3], 'ID': pred_temp[0]} 

        return predicted_templates

    def num_rings(self, reaction):
        """Given a reaction SMILES, obtains the change in the number of rings

        Parameters:
            reaction (SMILES): The reaction without agents in the form of SMILES

        Returns:
            num_rings (int): The change in the number of rings in the reaction
        """
        mol_r = Chem.MolFromSmiles(reaction.split('>>')[0])
        mol_p = Chem.MolFromSmiles(reaction.split('>>')[-1])

        return rdMolDescriptors.CalcNumRings(mol_p) - rdMolDescriptors.CalcNumRings(mol_r)
    
    def get_prediction(self, target):
        """Given a prediction and the template library, obtains the top N predicted templates

        Parameters:
            reaction (SMILES): The reaction without agents in the form of SMILES

        Returns:
            num_rings (int): The change in the number of rings in the reaction
        """
        prediction = self.policy.predict(self.smiles_to_ecfp(target))
        if self.mask:
            prediction = np.multiply(prediction, self.filter_array)
            return prediction
        else:
            return prediction

    def predict_ring_outcomes(self, target, cutoff=50):
        """Given a SMILES predicts the top N ring forming reactions and returns the results as a dictionary

        Parameters:
            target (SMILES): The target SMILES 
            cutoff (int): The Top N predictions 

        Returns:
            results (dict): The results in the form of a dictionary, contains the precursors, probability, cumulative probability,
                            ID, and the number of changed rings
        """
        prediction = self.get_prediction(target)
                
        predicted_templates = self.get_templates(prediction, self.lib, cutoff+1)
        sort_pred = np.sort(prediction)[::-1]

        results = {"prediction": [], 
                   "probability": [],
                   "cumulative_probability": [],
                   "id": [], 
                   "precursor": [],
                   "ring_change": []
                  }

        num_outcomes = 0
        for i in range(1, cutoff+1):
            template = predicted_templates[i]['template']
            outcomes = rdc.rdchiralRunText(template, target)
            if len(outcomes) == 0:
                continue
            if self.num_rings('>>'.join([outcomes[0], target])) >= 1:
                results["prediction"].append(i)
                results["probability"].append(sort_pred[-1][-i])
                results["cumulative_probability"].append(sum(sort_pred[-1][-i:]))
                results["id"].append(predicted_templates[i]['ID'])
                results["precursor"].append(outcomes[0])
                results["ring_change"].append(self.num_rings('>>'.join([outcomes[0], target])))
                num_outcomes += 1
            else:
                continue
        results["outcomes"] = num_outcomes

        return results

    def show_topN_predictions(self, target, topN=50):
        """Given a SMILES predicts the top N ring forming reactions and displays the predictions for visualisation in Jupyter Notebook

        Parameters:
            target (SMILES): The target SMILES 
        """
        prediction = self.get_prediction(target)
            
        predicted_templates = self.get_templates(prediction, self.lib, topN+1)
        sort_pred = np.sort(prediction)[::-1]

        for i in range(1, topN+1):
            template = predicted_templates[i]['template']
            outcomes = rdc.rdchiralRunText(template, target)
            if len(outcomes) != 0:
                print('###Prediction {}###'.format(str(i)))
                display(rdChemReactions.ReactionFromSmarts(template))
                print(sort_pred[-1][-i])
                print('Classification')
                print(predicted_templates[i]['classification'])
                print('Patent ID')
                print(predicted_templates[i]['ID'])
                print('---List of precursors---')
                print(outcomes)
                display(Chem.MolFromSmiles(outcomes[0]))
                print('\n')



