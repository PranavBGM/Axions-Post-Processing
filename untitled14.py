
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 21:48:35 2023

@author: pranavbharadwajgangrekalvemanoj
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# Path to the directory containing energy density frames (.npy files)
energy_density_directory = "/Users/pranavbharadwajgangrekalvemanoj/Desktop/Axions_Project/evolution_run_2/npy_frames"

# Path to the directory containing string position frames
string_pos_directory = "/Users/pranavbharadwajgangrekalvemanoj/Desktop/Axions_Project/evolution_run_2/string_pos_plots_v2"

# Output directory for combined frames
output_directory = "/Users/pranavbharadwajgangrekalvemanoj/Desktop/Axions_Project/evolution_run_2/combined_frames"
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Iterate through all frames and generate combined frames
num_frames = 161  # Number of frames assuming you have 161 string position frames
for i in range(num_frames):
    # Load energy density data from .npy file
    energy_density_file_path = os.path.join(energy_density_directory, f"frame_{i}.npy")
    energy_density = np.load(energy_density_file_path)

    # Load string position data from string position frame
    string_pos_file_path = os.path.join(string_pos_directory, f"frame_{i}.png")
    string_pos_image = plt.imread(string_pos_file_path)

    # Create a figure and 3D axis for energy density
    fig = plt.figure(figsize=(10, 8))
    ax_energy = plt.axes(projection="3d")
    ax_energy.set_xlim([0, energy_density.shape[0]])
    ax_energy.set_ylim([0, energy_density.shape[1]])
    ax_energy.set_zlim([0, energy_density.shape[2]])
    ax_energy.voxels(energy_density, edgecolor='k', facecolors='c', alpha=0.5)

    # Add string position frame as an overlay
    ax_energy.imshow(string_pos_image, extent=[0, energy_density.shape[0], 0, energy_density.shape[1]], origin='lower')

    # Customize your plot here (if needed)
    ax_energy.set_xlabel("X Axis")
    ax_energy.set_ylabel("Y Axis")
    ax_energy.set_zlabel("Z Axis")
    ax_energy.set_title(f"Combined Frame {i}")

    # Save the combined frame
    combined_frame_path = os.path.join(output_directory, f"combined_frame_{i}.png")
    plt.savefig(combined_frame_path)
    plt.close()

    print(f"Combined Frame {i} generated: {combined_frame_path}")
