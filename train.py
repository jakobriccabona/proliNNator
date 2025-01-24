import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import seaborn as sns

import pyrosetta
pyrosetta.init('-mute all')
from pyrosetta.rosetta.core.scoring import ScoreType
import menten_gcn as mg
import menten_gcn.decorators as decs

from spektral.layers import *
from tensorflow import keras
#from keras.regularizers import l2
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
#from tensorflow.keras import regularizers

from sklearn.utils import shuffle
#from imblearn.over_sampling import RandomOverSampler 
from imblearn.under_sampling import RandomUnderSampler

#from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, confusion_matrix
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_curve, roc_auc_score

plt.rcParams.update({'font.size': 15})
plt.rcParams['axes.linewidth'] = 2

def load_npz_files(path):
    data = np.load(f'{path}/merged_file.npz')
    Xs = data['Xs']
    As = data['As']
    Es = data['Es']
    outs = data['outs']
    return Xs, As, Es, outs

# Extract arrays from DataFrame
X_train, A_train, E_train, out_train = load_npz_files('/media/data/jri/cath_S40/train')
X_test, A_test, E_test, out_test = load_npz_files('/media/data/jri/cath_S40/test')

# Convert lists to numpy arrays
X_train = np.asarray(X_train)
A_train = np.asarray(A_train)
E_train = np.asarray(E_train)
out_train = np.asarray(out_train)

X_test = np.asarray(X_test)
A_test = np.asarray(A_test)
E_test = np.asarray(E_test)
out_test = np.asarray(out_test)

# Pick decorators
decorators = [decs.SequenceSeparation(ln = True), 
              decs.SimpleBBGeometry(use_nm = False), 
              decs.Rosetta_Ref2015_TwoBodyEneriges(individual=True, score_types=[ScoreType.fa_rep, 
                                                                                 ScoreType.fa_atr, 
                                                                                 ScoreType.fa_sol, 
                                                                                 ScoreType.lk_ball_wtd, 
                                                                                 ScoreType.fa_elec, 
                                                                                 ScoreType.hbond_sr_bb, 
                                                                                 ScoreType.hbond_lr_bb, 
                                                                                 ScoreType.hbond_bb_sc, 
                                                                                 ScoreType.hbond_sc])]

data_maker = mg.DataMaker(decorators=decorators,
                           edge_distance_cutoff_A=8.0,
                           max_residues=10,
                           nbr_distance_cutoff_A=10.0)

# Print summary
data_maker.summary()


def UnderSample(x_label, y_label):

    rus = RandomUnderSampler(sampling_strategy='auto', random_state=42)
    data_reshaped = x_label.reshape(x_label.shape[0], -1)
    data_new, y_rus = rus.fit_resample(data_reshaped, y_label)
    num_features = x_label.shape[1:]
    x_rus = data_new.reshape(-1, *num_features)
    
    return x_rus, y_rus

X_rus, y_rus = UnderSample(X_train, out_train)
A_rus, _ = UnderSample(A_train, out_train)
E_rus, _ = UnderSample(E_train, out_train)

# Define GCN model
X_in, A_in, E_in = data_maker.generate_XAE_input_layers()

L1 = ECCConv(64, activation=None)([X_in, A_in, E_in])
L1_bn = BatchNormalization()(L1)
L1_act = Activation('relu')(L1_bn)
L1_drop = Dropout(0.2)(L1_act)

L2 = ECCConv(32, activation=None)([L1_drop, A_in, E_in])
L2_bn = BatchNormalization()(L2)
L2_act = Activation('relu')(L2_bn)
L2_drop = Dropout(0.2)(L2_act)

L3 = ECCConv(16, activation=None)([L2_drop, A_in, E_in])
L3_bn = BatchNormalization()(L3)
L3_act = Activation('relu')(L3_bn)
L3_drop = Dropout(0.2)(L3_act)

L3 = GlobalMaxPool()(L3_drop)
L4 = Flatten()(L3)
output = Dense(1, name="out", activation="sigmoid")(L4)

model = Model(inputs=[X_in, A_in, E_in], outputs=output)
opt = keras.optimizers.Adam(learning_rate=1e-4)
model.compile(optimizer=opt, loss='binary_crossentropy')
model.summary()

# Early stopping callback
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    verbose=2,
    patience=20,
    mode='min',
    restore_best_weights=True
)

# Train the model
history = model.fit(x=[X_rus, A_rus, E_rus], y=y_rus, batch_size=50, epochs=500, validation_data=([X_test, A_test, E_test], out_test), callbacks=[early_stopping])
model.save("3D-model-v2.4.keras")

#further validation
y_pred_prob = model.predict([X_test, A_test, E_test])
y_pred = (y_pred_prob > 0.5).astype(int) 

mcc = matthews_corrcoef(out_test, y_pred)
print('Matthews Coefficient:', mcc)
fpr, tpr, thresholds = roc_curve(out_test, y_pred_prob)
auc = roc_auc_score(out_test, y_pred_prob)
print('auc:', auc)
precision, recall, thresholds = precision_recall_curve(out_test, y_pred_prob)
cm = confusion_matrix(out_test, y_pred)
cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

#Plottings
#plt.plot(history.history['loss'], label='Training Loss', color='royalblue')
#plt.plot(history.history['val_loss'], label='Validation Loss', color='goldenrod')
#plt.title('Model Loss Over Epochs')
#plt.xlabel('Epochs')
#plt.ylabel('Loss')
#plt.legend()
#plt.savefig('history.png')
#plt.clf()

#plt.plot(recall, precision)
#plt.xlabel('Recall')
#plt.ylabel('Precision')
#plt.title('Precision-Recall Curve')
#plt.savefig('precision-recall.png')
#plt.clf()

#plt.plot(fpr, tpr, label='ROC curve', color='royalblue')
#plt.xscale('log')  # Set x-axis to logarithmic scale
#plt.xlabel('Log False Positive Rate (FPR)')
#plt.ylabel('True Positive Rate (TPR)')
#plt.xlim([0.00001, 1])  # Set limits for the logarithmic scale
#plt.ylim([0.0, 1.05])
#plt.title('Log ROC Curve')
#plt.savefig('log-roc.png')
#plt.clf()

#sns.heatmap(cmn, annot=True, fmt='g', cmap='Blues', xticklabels=['Non-Proline', 'Proline'], yticklabels=['Non-Proline', 'Proline'])
#plt.xlabel('Predicted Label')
#plt.ylabel('True Label')
#plt.title('Confusion Matrix')
#plt.savefig('conf-m.png')
#plt.clf()