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
from keras.regularizers import l2
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers

from sklearn.utils import shuffle
from imblearn.over_sampling import RandomOverSampler 
from imblearn.under_sampling import RandomUnderSampler

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, confusion_matrix
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_curve, roc_auc_score

plt.rcParams.update({'font.size': 15})
plt.rcParams['axes.linewidth'] = 2

def load_npz_files():
    data = np.load('../data-generation/data/merged_file.npz')
    Xs = data['Xs']
    As = data['As']
    Es = data['Es']
    outs = data['outs']
    return Xs, As, Es, outs

# Extract arrays from DataFrame
Xs, As, Es, outs = load_npz_files()

# Convert lists to numpy arrays
Xs = np.asarray(Xs)
As = np.asarray(As)
Es = np.asarray(Es)
outs = np.asarray(outs)

# Pick decorators
decorators = [decs.SequenceSeparation(ln = True), decs.SimpleBBGeometry(use_nm = False), decs.Rosetta_Ref2015_TwoBodyEneriges(individual=True, score_types=[ScoreType.fa_rep, ScoreType.fa_atr, ScoreType.fa_sol, ScoreType.lk_ball_wtd, ScoreType.fa_elec, ScoreType.hbond_sr_bb, ScoreType.hbond_lr_bb, ScoreType.hbond_bb_sc, ScoreType.hbond_sc])]

data_maker = mg.DataMaker(decorators=decorators,
                           edge_distance_cutoff_A=8.0,
                           max_residues=10,
                           nbr_distance_cutoff_A=10.0)

# Print summary
data_maker.summary()

# Train Test split
X_train, X_val, A_train, A_val, E_train, E_val, y_train, y_val = train_test_split(Xs, As, Es, outs, test_size=0.2, random_state=42)

# Random Under Sampling of training split
ros = RandomUnderSampler(sampling_strategy='auto', random_state=42)
Xs_reshaped = X_train.reshape(X_train.shape[0], -1)
X_reshaped, y_ros = ros.fit_resample(Xs_reshaped, y_train)
num_features = X_train.shape[1:]
X_ros = X_reshaped.reshape(-1, *num_features)

As_reshaped = A_train.reshape(A_train.shape[0], -1)
A_reshaped, _ = ros.fit_resample(As_reshaped, y_train)
num_features = A_train.shape[1:]
A_ros = A_reshaped.reshape(-1, *num_features)

Es_reshaped = E_train.reshape(E_train.shape[0], -1)
E_reshaped, _ = ros.fit_resample(Es_reshaped, y_train)
num_features = E_train.shape[1:]
E_ros = E_reshaped.reshape(-1, *num_features)

# Define GCN model
X_in, A_in, E_in = data_maker.generate_XAE_input_layers()

L1 = ECCConv(30, activation=None)([X_in, A_in, E_in])
L1_bn = BatchNormalization()(L1)
L1_act = Activation('relu')(L1_bn)
L1_drop = Dropout(0.2)(L1_act)
L2 = GATConv(10, attn_heads=2, concat_heads=True, activation=None)([L1_drop, A_in])
L2_bn = BatchNormalization()(L2)
L2_act = Activation('relu')(L2_bn)
L2_drop = Dropout(0.2)(L2_act)
L3 = GlobalMaxPool()(L2_drop)
L4 = Flatten()(L3)
output = Dense(1, name="out", activation="sigmoid", kernel_regularizer=l2(0.01))(L4)

model = Model(inputs=[X_in, A_in, E_in], outputs=output)
opt = keras.optimizers.Adam(learning_rate=1e-4)
model.compile(optimizer=opt, loss='binary_crossentropy')
model.summary()

# Train the model
history = model.fit(x=[X_ros, A_ros, E_ros], y=y_ros, batch_size=200, epochs=100, validation_data=([X_val, A_val, E_val], y_val))
model.save("3D-model-new.keras")

#further validation
y_pred_prob = model.predict([X_val, A_val, E_val])
y_pred = (y_pred_prob > 0.5).astype(int) 

mcc = matthews_corrcoef(y_val, y_pred)
fpr, tpr, thresholds = roc_curve(y_val, y_pred_prob)
auc = roc_auc_score(y_val, y_pred_prob)
precision, recall, thresholds = precision_recall_curve(y_val, y_pred_prob)
cm = confusion_matrix(y_val, y_pred)
cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

#Plottings
plt.plot(history.history['loss'], label='Training Loss', color='royalblue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='goldenrod')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('history.png')
plt.clf()

plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.savefig('precision-recall.png')
plt.clf()

plt.plot(fpr, tpr, label='ROC curve', color='royalblue')
plt.xscale('log')  # Set x-axis to logarithmic scale
plt.xlabel('Log False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.xlim([0.00001, 1])  # Set limits for the logarithmic scale
plt.ylim([0.0, 1.05])
plt.title('Log ROC Curve')
plt.savefig('log-roc.png')
plt.clf()

sns.heatmap(cmn, annot=True, fmt='g', cmap='Blues', xticklabels=['Non-Proline', 'Proline'], yticklabels=['Non-Proline', 'Proline'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig('conf-m.png')
plt.clf()