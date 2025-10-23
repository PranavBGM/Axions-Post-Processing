#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 11:27:40 2023

@author: pranavbharadwajgangrekalvemanoj
"""


import numpy as np
import os

# Path to the directory containing input files
input_directory = "/Volumes/Recent_Archives/P_files_he/"

# Path to the directory where .npy files will be saved
output_directory = os.path.join(input_directory, "npy_files")
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Number of frames
num_frames = 351  # Update this to match the number of frames you have

# Iterate through all input files and generate .npy files
for i in range(312,num_frames):
    try: 
        input_file_path = os.path.join(input_directory, f"fixed_zero_boost_{i}_debug.txt")
        
        # Load data from the input file
        array_1d = np.loadtxt(input_file_path)
        
        # col_0 = array_1d[:, 0]
        phi_0 = array_1d[:,0]
        phi_1 = array_1d[:,1]
        # col_1 = array_1d[:, 1]
        # col_2 = array_1d[:, 2]
        # col_3 = array_1d[:, 3]
        # col_4 = array_1d[:, 4]
        # col_5 = array_1d[:, 5]
        
        # Reshape the 1D array into a 3D array
        nx = 201
        ny = 201
        nz = 201
        # array_3d_col_0 = col_0.reshape(nx, ny, nz)
        array_3d_col_0 = phi_0.reshape(nx, ny, nz)
        array_3d_col_1 = phi_1.reshape(nx, ny, nz)
        
        # array_3d_col_1 = col_1.reshape(nx, ny, nz)
        # array_3d_col_2 = col_2.reshape(nx, ny, nz)
        # array_3d_col_3 = col_3.reshape(nx, ny, nz)
        # array_3d_col_4 = col_4.reshape(nx, ny, nz)
        # array_3d_col_5 = col_5.reshape(nx, ny, nz)
        
        # Save the 3D arrays with appropriate filenames
        np.save(os.path.join(output_directory, f"frame_{i}_positive_boost_0.npy"), array_3d_col_0)
        # np.save(os.path.join(output_directory, "frame_phi_1_evolved.npy"), array_3d_col_1)
        # np.save(os.path.join(output_directory, "frame_phi_2_evolved.npy"), array_3d_col_2)
        
        np.save(os.path.join(output_directory, f"frame_{i}_positive_boost_1.npy"), array_3d_col_1)
        # np.save(os.path.join(output_directory, f"col_2_frame_{i}.npy"), array_3d_col_2)
        # np.save(os.path.join(output_directory, f"negative_debug_col_3.npy"), array_3d_col_3)
        
        # np.save(os.path.join(output_directory, f"col_3_frame_{i}.npy"), array_3d_col_3)
        # np.save(os.path.join(output_directory, f"col_4_frame_{i}.npy"), array_3d_col_4)
        # np.save(os.path.join(output_directory, f"col_5_frame_{i}.npy"), array_3d_col_5)

        print(f"Arrays saved for frame {i}")
    


    except Exception as e:
    # Handle any exceptions (file not found, empty file, etc.) and continue to the next iteration
        print(f"Error processing file {i}: {str(e)}")
        continue