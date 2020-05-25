import pandas as pd
import numpy as np
import swifter
import pickle
from sklearn.preprocessing import LabelEncoder

"""
The filter is a vector, with a length corresponding to the size of the template library of the standard model. All positions in the vector that correspond to a template which is a ring formation are 1, and all others are set to zero. This is used to filter the standard policy network for ring forming reactions.
"""

ring_lib = pd.read_csv("../data/uspto_ringformations.csv")
print('Number of reactions corresponding to ringformations: {}'.format(len(ring_lib.template_hash.values)))
print('Number of ring forming templates: {}'.format(len(set(ring_lib.template_hash.values))))
ring_template_set = list(set(ring_lib.template_hash.values))

lib = pd.read_csv("../data/standard_uspto_template_library.csv", index_col=0, header=None, names=["index", "ID", "reaction_hash", "reactants", "products", "classification", "retro_template", "template_hash", "selectivity", "outcomes", "template_code"])
print('Number of reactions in dataset: {}'.format(len(lib)))
print('Number of templates in dataset: {}'.format(len(set(lib.template_hash.values))))

common_templates = set(lib.template_hash.values).intersection(set(ring_lib.template_hash.values))

filter_array = np.ones(len(set(lib.template_hash.values)))
print(len(filter_array))

"""
For each template that is common between the two models set the position in the vector = 1, such that all non-ring forming templates are zero and will be ignored at prediction. The resulting vector should be pickled
"""
def create_predictive_filter(row):
    if row['template_hash'] in ring_template_set:
        filter_array[row['template_code']] = 1
    else:
        filter_array[row['template_code']] = 0

lib.swifter.apply(create_predictive_filter, axis=1)

#Check number if on bits in filter array is equivalent to number of ringforming templates 
with open("../data/uspto_filter_array.pkl", "wb") as f:
	pickle.dump(filter_array, f)
