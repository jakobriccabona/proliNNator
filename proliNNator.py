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

from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from spektral.layers import ECCConv, GlobalSumPool

import menten_gcn as mg
import menten_gcn.decorators as decs

# functions
def extract_features(pose, indices):
    # Extract sequence
    sequence = pose.sequence()
    
    # Extract secondary structure using DSSP
    DSSP = pyrosetta.rosetta.protocols.moves.DsspMover()
    DSSP.apply(pose)
    dssp = pose.secstruct()
    
    # Extract SASA
    sasa = pyrosetta.rosetta.core.scoring.sasa.SasaCalc()
    sasa.calculate(pose)
    residue_sasa_list = sasa.get_residue_sasa()
    
    # Extract Scoring
    sfxn = pyrosetta.get_fa_scorefxn()
    emo = EnergyMethodOptions()
    emo.hbond_options().decompose_bb_hb_into_pair_energies(True)
    sfxn.set_energy_method_options(emo)
    sfxn(pose)
    scores_df = pd.DataFrame(pyrosetta.bindings.energies.residue_total_energies_array(pose.energies(), residue_selection=range(1, len(sequence)+1)))
    scores_df = scores_df.drop(columns=['fa_intra_rep','fa_intra_sol_xover4','dslf_fa13','rama_prepro','p_aa_pp','fa_dun','omega','pro_close', 'yhh_planarity', 'ref', 'total_score'])
     
    # Get chain information for each residue
    residue_chains = [pose.pdb_info().chain(i+1) for i in range(pose.total_residue())]
    
    filtered_data = []
    for i in indices:
        if i >= 2 and i + 2 < len(sequence):
            # Check if the residues are in the same chain
            if len(set(residue_chains[i-2:i+3])) == 1:
                seq_chunk = sequence[i-3:i+4]
                dssp_chunk = dssp[i-3:i+4]
                sasa_chunk = [residue_sasa_list[j] if j < len(residue_sasa_list) else None for j in range(i-2, i+5)]
                energy_chunk = scores_df.iloc[i-3:i+4].to_dict(orient='records')

                filtered_data.append((seq_chunk, dssp_chunk, sasa_chunk, energy_chunk))
    
    return filtered_data

def OneHotEncoder(df, column):
    seq = pd.DataFrame(df[column].apply(list).tolist())
    encoded = pd.get_dummies(seq)
    return encoded


# main
def main():
    # Argument parser setup
    parser = argparse.ArgumentParser(description='ProliNNator is a tool that predicts Proline probabilties based on pretrained neural networks. \n Contact: Jakob.Riccabona@medizin.uni-leipzig.de')
    parser.add_argument('-i', '--input', type=str, required=True, help='Path to the input PDB file')
    parser.add_argument('-m', '--model', type=str, default='3D-model.keras', help='Path to the model')
    parser.add_argument('-o', '--output', type=str, default='output.pdb', help='Name of the output PDB file (default: output.pdb)')
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

    # Calculate features from pdb
    features = extract_features(pose, [i for i in range(3, len(pose.sequence())-3)])

    # Load the ML model
    mod = args.model
    if '1D' in mod:
        model = load_model(mod)
         # Format data to relevant features
        data = pd.DataFrame(features, columns=['Sequence','DSSP','SASA','Scores'])
        sec_onehot = OneHotEncoder(data, "DSSP")
        converted_sasa = np.array([np.array(row) for row in data['SASA'].values])
        converted_sasa = converted_sasa.astype(float)

        # Parse the strings into lists of dictionaries
        parsed_scores = [row for row in data['Scores']]
        # Flatten each list of dictionaries into a single dictionary and then into a 1D array
        flattened_scores = []
        for score_list in parsed_scores:
            flattened_score = []
            for score_dict in score_list:
                flattened_score.extend(score_dict.values())
            flattened_scores.append(flattened_score)
        # Convert the list of flattened scores into a 2D numpy array
        data_array = np.array(flattened_scores)

        # Normalize SASA and data_array
        scaler = StandardScaler()
        converted_sasa_n = scaler.fit_transform(converted_sasa)
        data_array_n = scaler.fit_transform(data_array)

        # Final features
        features = np.concatenate((sec_onehot, converted_sasa_n, data_array_n), axis=-1)
        features = np.delete(features, [9,10,11,24,48,49,50,51,52,53,54,55,56], axis=1)

        # Run prediction
        y_pred = model.predict(features)

        # Map y_pred onto crystal structure and save pdb
        # Set all bfactors to zero
        for i in range(1, len(pose.sequence()) + 1):
            for j in range(1, pose.residue(i).natoms() + 1):
                pose.pdb_info().bfactor(i, j, 0)
        # Fill in bfactors
        counter = 0
        for c in range(1, pose.pdb_info().num_chains() + 1):
            chain_start = pose.conformation().chain_begin(c)
            chain_end = pose.conformation().chain_end(c)
            for i in range(chain_start + 3, chain_end - 3):
                if counter < len(y_pred):
                    for j in range(1, pose.residue(i).natoms() + 1):
                        pose.pdb_info().bfactor(i, j, y_pred[counter])
                    counter += 1
        # Save pdb
        pose.dump_pdb(args.output)

    elif '3D' in mod:
        # load model
        custom_objects = {'ECCConv': ECCConv, 'GlobalSumPool': GlobalSumPool}
        model = load_model(mod, custom_objects)

        # Pick some decorators to add to your network
        decorators = [decs.Rosetta_Ref2015_TwoBodyEneriges(individual=True, score_types=[ScoreType.fa_rep, ScoreType.fa_atr, ScoreType.fa_sol, ScoreType.lk_ball_wtd, ScoreType.fa_elec, ScoreType.hbond_sr_bb, ScoreType.hbond_lr_bb, ScoreType.hbond_bb_sc, ScoreType.hbond_sc])]
        data_maker = mg.DataMaker(decorators=decorators, edge_distance_cutoff_A=8.0, max_residues=10, nbr_distance_cutoff_A=10.0)
        data_maker.summary()

        wrapped_pose = mg.RosettaPoseWrapper(pose)
        cache = data_maker.make_data_cache(wrapped_pose)

        Xs = []
        As = []
        Es = []

        for resid in range(1, len(pose.sequence()) + 1):
            X, A, E, resids = data_maker.generate_input_for_resid(wrapped_pose, resid, data_cache=cache)
            Xs.append(X)
            As.append(A)
            Es.append(E)

        Xs = np.asarray(Xs)
        As = np.asarray(As)
        Es = np.asarray(Es) 
        y_pred = model.predict([Xs, As, Es])

        # Map y_pred onto crystal structure and save pdb
        # Set all bfactors to zero
        for i in range(1, len(pose.sequence()) + 1):
            for j in range(1, pose.residue(i).natoms() + 1):
                pose.pdb_info().bfactor(i, j, 0)
        # Fill in bfactors
        counter = 0
        for c in range(1, pose.pdb_info().num_chains() + 1):
            chain_start = pose.conformation().chain_begin(c)
            chain_end = pose.conformation().chain_end(c)
            for i in range(chain_start, chain_end):
                if counter < len(y_pred):
                    for j in range(1, pose.residue(i).natoms() + 1):
                        pose.pdb_info().bfactor(i, j, y_pred[counter])
                    counter += 1
        # Save pdb
        pose.dump_pdb(args.output)

    else:
        raise ValueError('Model name not known')

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
                for i in range(chain_start, chain_end):
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
