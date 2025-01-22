print("""
                     _  _  _   _  _   _         _                
 _ __   _ __   ___  | |(_)| \ | || \ | |  __ _ | |_   ___   _ __ 
| '_ \ | '__| / _ \ | || ||  \| ||  \| | / _` || __| / _ \ | '__|
| |_) || |   | (_) || || || |\  || |\  || (_| || |_ | (_) || |   
| .__/ |_|    \___/ |_||_||_| \_||_| \_| \__,_| \__| \___/ |_|   
|_|                                                                               
""")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

import pyrosetta
from pyrosetta.rosetta.core.scoring import ScoreType
from pyrosetta.rosetta.protocols.relax import FastRelax
from pyrosetta.rosetta.core.scoring.methods import EnergyMethodOptions

#import keras
from tensorflow.keras.models import load_model
#from sklearn.preprocessing import StandardScaler
from spektral.layers import ECCConv, GlobalMaxPool, GATConv

import menten_gcn as mg
import menten_gcn.decorators as decs

# main
def main():
    # Argument parser setup
    parser = argparse.ArgumentParser(description='ProliNNator is a tool that predicts Proline probabilties based on pretrained neural networks. \n Contact: Jakob.Riccabona@medizin.uni-leipzig.de')
    parser.add_argument('-i', '--input', type=str, required=True, help='Path to the input PDB file')
    parser.add_argument('-m', '--model', type=str, default='3D-model-v2.5.keras', help='Path to the model')
    parser.add_argument('-o', '--output', type=str, default='output.pdb', help='Name of the output PDB file (default: output.pdb)')
    parser.add_argument('--csv', type=str, default='output.csv', help='Filename to save a csv file with the probabilities')
    parser.add_argument('--ramachandran', type=str, help='Filename to save a Ramachandran plot with probabilities as a PNG')
    parser.add_argument('--fastrelax', action='store_true', help='Flag to perform a fast relax on the structure before analysis')
    args = parser.parse_args()

    # LOAD THE PDB
    pyrosetta.init('-mute all')
    pdb_file = args.input
    pose = pyrosetta.pose_from_pdb(pdb_file)

    # Perform FastRelax if flag is set
    if args.fastrelax:
        scfxn = pyrosetta.get_fa_scorefxn()
        fast_relax = FastRelax(scorefxn_in = scfxn, standard_repeats=1)
        fast_relax.set_scorefxn(scfxn)
        print('Executing fast relax...')
        fast_relax.apply(pose)
        print('fast relax finished')


    # Load the ML model
    mod = args.model
    # load model
    custom_objects = {'ECCConv': ECCConv, 'GlobalMaxPool': GlobalMaxPool}
    model = load_model(mod, custom_objects)

        # Pick some decorators to add to your network
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
    data_maker = mg.DataMaker(decorators=decorators, edge_distance_cutoff_A=8.0, max_residues=10, nbr_distance_cutoff_A=10.0)
    #data_maker.summary()

    wrapped_pose = mg.RosettaPoseWrapper(pose)
    cache = data_maker.make_data_cache(wrapped_pose)

    Xs = []
    As = []
    Es = []

    for resid in range(1, len(pose.sequence()) + 1):
        X, A, E, _ = data_maker.generate_input_for_resid(wrapped_pose, resid, data_cache=cache)
        Xs.append(X)
        As.append(A)
        Es.append(E)

    Xs = np.asarray(Xs)
    As = np.asarray(As)
    Es = np.asarray(Es) 
    y_pred = model.predict([Xs, As, Es])

    #create csv file
    if args.csv:
        rows = []
        for c in range(1, pose.pdb_info().num_chains() + 1):
            chain_start = pose.conformation().chain_begin(c)
            chain_end = pose.conformation().chain_end(c)
            for i in range(chain_start, chain_end + 1):
                if i-1 < len(y_pred):
                    row = {
                        'chain': pose.pdb_info().chain(i),
                        'amino_acid': pose.residue(i).name(),
                        'position_number': pose.pdb_info().number(i),
                        'probability': round(float(y_pred[i - 1]), 5)
                        }
                    rows.append(row)
        df = pd.DataFrame(rows)
        df.to_csv(args.csv, index=False, float_format='%.5f')
        print(f'Successfully generated {args.csv}')

    #save pdb
    if args.output:
    # Set all bfactors to zero
        for i in range(1, len(pose.sequence()) + 1):
            for j in range(1, pose.residue(i).natoms() + 1):
                pose.pdb_info().bfactor(i, j, 0)
    # Fill in bfactors
        counter = 0
        for c in range(1, pose.pdb_info().num_chains() + 1):
            chain_start = pose.conformation().chain_begin(c)
            chain_end = pose.conformation().chain_end(c)
            for i in range(chain_start, chain_end + 1):
                if counter < len(y_pred):
                    for j in range(1, pose.residue(i).natoms() + 1):
                        pose.pdb_info().bfactor(i, j, y_pred[counter])
                    counter += 1
        pose.dump_pdb(args.output)

    # plot the predicted positions in ramachandran space
    if args.ramachandran:
        phi = np.array([])
        psi = np.array([])
        weights = np.array([])
        counter = 0
        for c in range(1, pose.pdb_info().num_chains() + 1):
            chain_start = pose.conformation().chain_begin(c)
            chain_end = pose.conformation().chain_end(c)
            if '1D' in mod:
                for i in range(chain_start + 3, chain_end - 3):
                    if counter < len(y_pred):
                        phi = np.append(phi, pose.phi(i))
                        psi = np.append(psi, pose.psi(i))
                        weights = np.append(weights, y_pred[counter])
                        counter += 1
            elif '3D' in mod:
                for i in range(chain_start, chain_end + 1):
                    if counter < len(y_pred):
                        phi = np.append(phi, pose.phi(i))
                        psi = np.append(psi, pose.psi(i))
                        weights = np.append(weights, y_pred[counter])
                        counter += 1
        plotting_data = np.vstack((phi, psi, weights)).T
        df_plot = pd.DataFrame(plotting_data, columns=['phi', 'psi', 'weight'])
        sns.scatterplot(data=df_plot, x='phi', y='psi', hue='weight', hue_order= [0.0,0.25,0.5,0.75,1.0], palette='coolwarm')
        plt.xlim(-180, 180)
        plt.ylim(-180, 180)
        plt.xlabel(r'$ \phi $', fontsize=14)
        plt.ylabel(r'$ \psi $', fontsize=14)
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        plt.draw()
        plt.legend(title='weights', bbox_to_anchor=(1.05,1), loc=2, borderaxespad=0.)
        plt.tight_layout()
        plt.savefig(args.ramachandran, dpi=600)
       
if __name__ == "__main__":
    main()
