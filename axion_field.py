#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 12:32:06 2023

@author: pranavbharadwajgangrekalvemanoj
"""

# import numpy as np
# import os
# import matplotlib.pyplot as plt
# import multiprocessing

# # Define a function to generate contour plots for a given input file and column index
# def generate_contour_plot(input_data):
#     j, column_index, input_directory, contour_frames_directory = input_data
#     try:
#         input_file_path = os.path.join(input_directory, 
f"small_gifData_{j}.txt")
#         array_1d = np.loadtxt(input_file_path)
#         column_array = array_1d[:, column_index]
#         nx, ny, nz = 100, 100, 100
#         array_3d = column_array.reshape(nx, ny, nz)

#         fig = plt.figure(figsize=(15, 15))
#         ax = fig.add_subplot(111, projection='3d')
#         X, Y, Z = np.meshgrid(np.arange(nx), np.arange(ny), -np.arange(nz))

#         kw = {
#             'vmin': array_3d.min(),
#             'vmax': array_3d.max(),
#             'levels': np.linspace(array_3d.min(), array_3d.max(), 10),
#         }

#         ax.contour(X, Y, array_3d, levels=[0.4], colors='k', linewidths=0.2)
#         ax.set(xlim=[0, 100], ylim=[0, 100], zlim=[-100, 0])
#         ax.set_xlabel("x")
#         ax.set_ylabel("y")
#         ax.set_zlabel("z")

#         frame_path = os.path.join(contour_frames_directory, f"contour_frame_{j}_col_{column_index}.png")
#         plt.savefig(frame_path, dpi=300)
#         plt.close()

#         print(f"Frame {j} column {column_index} generated: {frame_path}")

#     except Exception as e:
#         print(f"Error processing file {j} column {column_index}: {str(e)}")

# if __name__ == "__main__":
#     input_directory = "/Users/pranavbharadwajgangrekalvemanoj/Desktop/Axions_Project/evolution_run_5_field/"
#     contour_frames_directory = "/Users/pranavbharadwajgangrekalvemanoj/Desktop/Axions_Project/"
    
#     # Number of processes to run in parallel
#     num_processes = multiprocessing.cpu_count()

#     # Create a list of input data for parallel processing
#     input_data_list = [(j, i, input_directory, contour_frames_directory) for j in range(40) for i in range(6)]


# Number of frames
num_frames = 178  # Update this to match the number of frames you have

# Function to calculate total energy from the energy density grid
def calculate_total_energy(energy_density):
    # Assuming energy_density is a 3D NumPy array representing the energy density grid
    # Compute total energy as the sum of all grid points
    total_energy = np.sum(energy_density)
    return total_energy

# Initialize an empty list to store total energies for each frame
total_energies = []

# Iterate through all frames and calculate total energy
for i in range(num_frames):
    # Load energy density grid from the corresponding frame file
    frame_path = os.path.join(frames_directory, f"frame_{i}.npy")
    energy_density = np.load(frame_path)  # Load data as a NumPy array

    # Calculate total energy for the current frame
    total_energy = calculate_total_energy(energy_density)
    
    # Append the total energy to the list
    total_energies.append(total_energy)

# Plot total energy vs frame or time
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(range(num_frames), total_energies, marker='o', linestyle='-', color='b')
plt.xlabel('Frame or Time')
plt.ylabel('Total Energy')
plt.title('Total Energy vs Frame or Time')
plt.grid(True)
plt.show()
plt.savefig(os.path.join("/Users/pranavbharadwajgangrekalvemanoj/Desktop/Axions_Project/evolution_run_2", "total_energy_evol_run_2_178.png"))
#     # Create a multiprocessing pool with the specified number of processes
#     with multiprocessing.Pool(processes=num_processes) as pool:
#         # Use pool.starmap to pass multiple arguments to the function
#         pool.starmap(generate_contour_plot, input_data_list)