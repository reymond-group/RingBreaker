# "Ring Breaker": Neural Network Driven Synthesis Prediction of the Ring System Chemical Space

Thank you for your interest in the source code complementing our publication in the Journal of Medicinal Chemistry:
https://pubs.acs.org/doi/abs/10.1021/acs.jmedchem.9b01919


## Install conda environment

The requirements for the environment are given in the .yml file:

`conda create -f ringbreaker.yml`

These are also listed below:
- numpy
- pandas
- rdkit
- rdchiral
- scipy
- swifter 
- tensorflow
- keras
- scikit-learn


## Example

An example notebook *example_usage.ipynb* is included to demonstrate how the model can be used

As well as the Ring Breaker model our standard model trained on all templates is also included in the repository.

The models and underlying data must first be downloaded from the following location:

https://figshare.com/articles/_Ring_Breaker_Neural_Network_Driven_Synthesis_Prediction_of_the_Ring_System_Chemical_Space/12366551

These contents of these folders must be placed into a data and models folder respectively in order for the notebook to work.


## Retraining 

The Ring Breaker model can be retrained by first regenerating the training data from the preprocessing folder:

`python ringbreaker_multilabel_generation.py`

This yield a series of .npz files containin the training, validation, and test data as sparse matrices containing precomputed fingerprints. 

The model can be trained by running the following command from the training folder:

`python ringbreaker_training.py trin ../data/uspto/uspto_ringbreaker_training_inputs.npz -trout ../data/uspto/uspto_ringbreaker_training_labels.npz -vain ../data/uspto/uspto_ringbreaker_validation_inputs.npz -vaout ../data/uspto/uspto_ringbreaker_validation_labels.npz -od ../models/ -of uspto_ringbreaker -b 256 -e 200 -m 3 -fp 2048` 