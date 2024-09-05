import numpy as np
import glob

def merge_npz_files(file_list, output_file):
    # Initialize a dictionary to hold all arrays
    combined_arrays = {}
    
    for file in file_list:
        with np.load(file) as data:
            # Iterate through each array in the file
            for key in data:
                # Append or update arrays in the combined dictionary
                if key in combined_arrays:
                    combined_arrays[key] = np.concatenate((combined_arrays[key], data[key]), axis=0)
                else:
                    combined_arrays[key] = data[key]
    
    # Save the combined arrays to a new .npz file
    np.savez(output_file, **combined_arrays)

# List of .npz files you want to merge
file_list = glob.glob('*?.npz')

# Output file name
output_file = 'merged_file.npz'

# Call the function to merge files
merge_npz_files(file_list, output_file)
