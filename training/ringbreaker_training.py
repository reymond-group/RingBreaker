import os
from rdkit import Chem
from rdkit.Chem import AllChem, rdChemReactions, rdmolfiles, rdmolops
from rdkit.DataStructs import cDataStructs
import pandas as pd
import numpy as np
import keras
from keras.layers import Dense, Dropout, Activation
from keras.models import Sequential
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras.metrics import top_k_categorical_accuracy
from keras.utils import Sequence
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint, Callback, TensorBoard, ReduceLROnPlateau
from sklearn.utils import shuffle
import time
import functools
from functools import partial
from keras import regularizers
import argparse
from scipy import sparse

class RxnSequence(Sequence):
    """Custom keras.utils.Sequence object for generating data on the fly for use with model.fit_generator 

    - Shuffles all training data after epoch end
    - All training data held in memory 
    
    Parameters:
        input_matrix (scipy.sparse.csr_matrix): A CSR matrix containing the vectorised input data
        label_matrix (scipy.sparse.csr_matrix): A CSR matrix containing the vectorised label data
        batch_size (int): The number of samples to be loaded at any given time 
        fpsize (int): The length of the vector in the input sparse matrix
    """
    def __init__(self, input_matrix, label_matrix, batch_size, fpsize):
        self.batch_size = batch_size
        self.fpsize = fpsize
        self.input_matrix = input_matrix
        self.label_matrix = label_matrix

    def __len__(self):
        return int(np.ceil(self.label_matrix.shape[0] / float(self.batch_size)))

    def __getitem__(self, idx):
        #Get input data for the generator from the aparse matrix
        X_input = self.input_matrix[idx * self.batch_size:(idx + 1) * self.batch_size] 
        Y_input = self.label_matrix[idx * self.batch_size:(idx + 1) * self.batch_size] 

        return (X_input.toarray(), Y_input.toarray())
    
    def on_epoch_end(self):
        self.input_matrix, self.label_matrix = shuffle(self.input_matrix, self.label_matrix, random_state=0)

class TimeHistory(Callback):
    """Custom keras.callbacks.Callback object to keep track of time taken per epoch.
    """
    def on_train_begin(self, logs={}):
        self.times = []
        self.epoch = 0

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)
        with open(args.out + "/" + args.file + '_timing.txt', 'a') as t:
            t.write(str(self.epoch) + ',' + str(self.times[-1]) + '\n')
        self.epoch += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate multilabel data with a pretrained network')
    parser.add_argument('-te', '--templates', type = str, default = None,
                        help = 'Specify the absolute path to the template library .csv file')
    parser.add_argument('-trin', '--traininginput', type = str, default = None,
                        help = 'Specify the absolute path to the training set .csv file')
    parser.add_argument('-trout', '--trainingoutput', type = str, default = None,
                        help = 'Specify the absolute path to the training set .csv file')
    parser.add_argument('-vain', '--validationin', type = str, default = None,
                        help = 'Specify the absolute path to the validation set .csv file')
    parser.add_argument('-vaout', '--validationout', type = str, default = None,
                        help = 'Specify the absolute path to the validation set .csv file')
    parser.add_argument('-od', '--out', type = str, default = None,
                        help = 'Specify the absolute path to the folder to which the results should be written \n' +
                        'if the folder does not exist, it will be created')
    parser.add_argument('-of', '--file', type = str, default = None,
                        help = 'Specify the filename for the output file')
    parser.add_argument('-b', '--batchsize', type = int, default = 288,
                        help = 'Specify the batch size')
    parser.add_argument('-e', '--epochs', type = int, default = 50,
                        help = 'Specify the number of epochs')
    parser.add_argument('-m', '--mintemp', type = int, default = 3,
                        help = 'Specify the minimum number of templates to consider')
    parser.add_argument('-fp', '--fpsize', type = int, default = 2048,
                        help = 'Specify the size of the input vector (ECFP)')
    args = parser.parse_args()

    if os.path.exists(args.out):
        pass
    else: 
        os.mkdir(args.out)

training_in = sparse.load_npz(args.traininginput)
training_out = sparse.load_npz(args.trainingoutput)
validation_in = sparse.load_npz(args.validationin)
validation_out = sparse.load_npz(args.validationout)
output_size = training_out.shape[1]

batch_size = args.batchsize

rollout_policy = Sequential()
rollout_policy.add(Dense(512, input_shape = (args.fpsize,), activation='elu'))
rollout_policy.add(Dropout(0.5))
rollout_policy.add(Dense(output_size, activation = 'softmax'))
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)

early_stopping = EarlyStopping(monitor='val_loss', patience=10)

csv_logger = CSVLogger(args.out + "/" + args.file + '_training_log.log', append=True)

checkpoint_loc = args.out + "/checkpoints" 
os.mkdir(checkpoint_loc)
checkpoint = ModelCheckpoint(checkpoint_loc + "/" + "weights.hdf5", monitor='loss', save_best_only=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)

rollout_policy.compile(optimizer = adam, 
                    loss = 'categorical_crossentropy', 
                    metrics=["accuracy"])

time_callback = TimeHistory()
rollout_policy.fit_generator(RxnSequence(training_in, training_out, batch_size, args.fpsize), 
                            steps_per_epoch=None, 
                            epochs=args.epochs, 
                            verbose=1, 
                            callbacks=[early_stopping, csv_logger, checkpoint, reduce_lr, time_callback], 
                            validation_data=RxnSequence(validation_in, validation_out, batch_size, args.fpsize), 
                            validation_steps=None, 
                            class_weight=None, 
                            max_queue_size=15, 
                            workers=10, 
                            use_multiprocessing=False, 
                            shuffle=True, 
                            initial_epoch=0)

