import pyrosetta
pyrosetta.init('-mute all')
from pyrosetta.rosetta.core.scoring import ScoreType

import menten_gcn as mg
import menten_gcn.decorators as decs

from spektral.layers import *
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from sklearn.utils import shuffle
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split

import numpy as np
import glob
import pandas as pd
from multiprocessing import Pool, cpu_count

# Initialize PyRosetta
pyrosetta.init('-mute all')

# Pick decorators
decorators = [decs.SimpleBBGeometry(use_nm = False), decs.Rosetta_Ref2015_TwoBodyEneriges(individual=True, score_types=[ScoreType.fa_rep, ScoreType.fa_atr, ScoreType.fa_sol, ScoreType.lk_ball_wtd, ScoreType.fa_elec, ScoreType.hbond_sr_bb, ScoreType.hbond_lr_bb, ScoreType.hbond_bb_sc, ScoreType.hbond_sc])]

data_maker = mg.DataMaker(decorators=decorators,
                           edge_distance_cutoff_A=8.0,
                           max_residues=10,
                           nbr_distance_cutoff_A=10.0)

# Define function that picks prolines
def get_proline_positions(pose):
    proline_positions = []
    for i in range(1, pose.size() + 1):
        if pose.residue(i).name() == "PRO":
            proline_positions.append(i)
    return proline_positions

def get_non_proline_positions(pose, proline_positions):
    non_proline_positions = [i for i in range(1, pose.size() + 1) if i not in proline_positions]
    return non_proline_positions

def save_chunk(chunk_index, Xs, As, Es, outs):
    np.savez_compressed(f'data/data_chunk_{chunk_index}.npz', Xs=Xs, As=As, Es=Es, outs=outs)
    print(f'Chunk {chunk_index} saved!')

def process_and_save_chunk(chunk_index, pdb_chunk):
    Xs_chunk = []
    As_chunk = []
    Es_chunk = []
    outs_chunk = []

    for pdb in pdb_chunk:
        pose = pyrosetta.pose_from_pdb(pdb)
        proline_residues = get_proline_positions(pose)
        non_prolines = get_non_proline_positions(pose, proline_residues)
        wrapped_pose = mg.RosettaPoseWrapper(pose)
        cache = data_maker.make_data_cache(wrapped_pose)

        for resid in proline_residues:
            X, A, E, resids = data_maker.generate_input_for_resid(wrapped_pose, resid, data_cache=cache)
            Xs_chunk.append(X)
            As_chunk.append(A)
            Es_chunk.append(E)
            outs_chunk.append([1.0,])

        for resid in non_prolines:
            X, A, E, resids = data_maker.generate_input_for_resid(wrapped_pose, resid, data_cache=cache)
            Xs_chunk.append(X)
            As_chunk.append(A)
            Es_chunk.append(E)
            outs_chunk.append([0.0,])

    save_chunk(chunk_index, Xs_chunk, As_chunk, Es_chunk, outs_chunk)

def process_chunks(pdb_list, chunk_size):
        
    # Split the pdb_list into chunks
    chunks = [pdb_list[i:i + chunk_size] for i in range(0, len(pdb_list), chunk_size)]

    # Set up multiprocessing pool
    with Pool(processes=cpu_count()) as pool:
        pool.starmap(process_and_save_chunk, [(i + 1, chunk) for i, chunk in enumerate(chunks)])
        print("Processing and saving completed.")

# Process the pdb_list in chunks
pdb_list = glob.glob('/home/iwe14/Documents/database/cath_S40/*?.pdb')
chunk_size = 200
process_chunks(pdb_list, chunk_size)
