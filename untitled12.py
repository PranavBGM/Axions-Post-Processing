#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 19:02:23 2023

@author: pranavbharadwajgangrekalvemanoj
"""


import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
import imageio
import scipy

# Path to the directory containing input files
input_directory = "/Users/pranavbharadwajgangrekalvemanoj/Desktop/Axions_Project/evolution_run_5_field/"
contour_frames_directory = "/Users/pranavbharadwajgangrekalvemanoj/Desktop/Axions_Project/"
# Path to the directory where frames will be saved
frames_directory = os.path.join(input_directory, "framesdpi600v.2")
if not os.path.exists(frames_directory):
    os.makedirs(frames_directory)
dx = 0.5
dy = 0.5
dz = 0.5
# Define the sampling indices for each axis
sampling_interval_x = 1  # Sample every 10th point along X axis
sampling_interval_y = 1  # Sample every 10th point along Y axis
sampling_interval_z = 1  # Sample every 10th point along Z axis

# Iterate through all input files and generate frames
for j in range(300):  # Assuming you have 200 input files numbered from 0 to 200
    try:
        input_file_path = os.path.join(input_directory, f"small_gifData_{j}.txt")
    
        # Load data from the input file
        array_1d = np.loadtxt(input_file_path)
        
        # Separate the columns into individual arrays
        # column_arrays = [array_1d[:, 0] for i in range(array_1d.shape[1])]
        array_1d = array_1d[:, 0]

        # Iterate through each column array and generate contour plot
        # for idx, column_array in enumerate(column_arrays):
    
        # Reshape the 1D array into a 3D array
        nx = 100
        ny = 100
        nz = 100
        
        array_3d = array_1d.reshape(nx, ny, nz)
    
    
        # # Sample the 3D array
        # sampled_array_3d = array_3d[::sampling_interval_x, ::sampling_interval_y, ::sampling_interval_z]
    
        # # Create meshgrid for sampled data
        # X, Y, Z = np.meshgrid(
        #     np.arange(0, nx, sampling_interval_x),
        #     np.arange(0, ny, sampling_interval_y),
        #     np.arange(0, nz, sampling_interval_z)
        # )
    
        # # Plot the sampled 3D data using a scatter plot
        # threshold_value = 0.1  # Adjust this value based on your data
        
        # # Define the threshold value below which points will be considered transparent
        # # Adjust this value based on your data
        
        
        # # Create a mask to identify points above the threshold
        # mask = sampled_array_3d >= threshold_value
        
        # Plot the sampled 3D data using a scatter plot with transparency
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot points above the threshold with full opacity
        # ax.scatter(X[mask], Y[mask], Z[mask], c=sampled_array_3d[mask].flatten(), cmap="twilight", alpha=1)
        
        # # Plot points below the threshold with transparency (alpha=0)
        # ax.scatter(X[~mask], Y[~mask], Z[~mask], alpha=0)
        # Plot isosurfaces using contours
    
    
    
    # Assuming array_3d is your original 3D array with shape (300, 300, 100)
    # Slice the 3D array to match the desired dimensions (300, 100)
        array_3d_new = array_3d
    
        # # Create mesh grids for X, Y, and Z with consistent dimensions (100, 100, 100)
        # X, Y, Z = np.meshgrid(
        #     np.arange(0, array_3d.shape[1], sampling_interval_x),  # X-axis (0 to 100, sampling every 3rd point)
        #     np.arange(0, array_3d.shape[0], sampling_interval_y),  # Y-axis (0 to 100, sampling every 3rd point)
        #     np.arange(0, array_3d.shape[2], sampling_interval_z)  # Z-axis (0 to 100, sampling every 1st point)
        # )
    
        # # Separate the 3D array into 2D arrays along different axes
        # array_2d_x = array_3d.mean(axis=0)  # 2D array along X-axis (shape: (100, 100))
        # array_2d_y = array_3d.mean(axis=1)  # 2D array along Y-axis (shape: (100, 100))
        # array_2d_z = array_3d.mean(axis=2)  # 2D array along Z-axis (shape: (100, 100))
    
        # # Create a 3D contour plot with consistent dimensions along all axes
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.contour3D(array_2d_x, array_2d_y, array_2d_z, levels=50, cmap='Greys', alpha=0.5)
        
        # # Customize plot labels, title, etc. as needed
        # ax.set_xlabel('X Label')
        # ax.set_ylabel('Y Label')
        # ax.set_zlabel('Z Label')
        # ax.set_title('3D Contour Plot')
        nx = 100
        ny = 100
        nz = 100
        fig = plt.figure(figsize=(15,15))
        
        ax = fig.add_subplot(111, projection='3d')
    
        X, Y, Z = np.meshgrid(np.arange(nx), np.arange(ny), -np.arange(nz))
    
        kw = {
            'vmin': array_3d_new.min(),
            'vmax': array_3d_new.max(),
            'levels': np.linspace(array_3d_new.min(), array_3d_new.max(), 10),
        }
    
    
        for i in range(nz):
            ax.contour(
                X[:, :, i], Y[:, :, i], array_3d_new[:, :, i],
                levels=[0.4], colors='k', zdir='z', offset=-i,linewidths = 0.2,
            )
    
        for i in range(nx):
            ax.contour(
                X[i, :, :], Y[i, :, :], array_3d_new[i, :, :],
                levels=[0.4], colors='k', zdir='y', offset=0,linewidths = 0.2,
            )
    
        ax.contour(
            array_3d_new[:, -1, :], Y[:, -1, :], Z[:, -1, :],
            levels=[0.4], colors='k', zdir='x', offset=nx,linewidths = 0.2,)
    
        # Set limits of the plot from coord limits
        xmin, xmax = X.min(), X.max()
        ymin, ymax = Y.min(), Y.max()
        zmin, zmax = Z.min(), Z.max()
        ax.set(xlim=[xmin, xmax], ylim=[ymin, ymax], zlim=[zmin, zmax])
    
        # Plot edges
        edges_kw = dict(color='0.4', linewidth=0, zorder=1e3)
        ax.plot([xmax, xmax], [ymin, ymax], 0, **edges_kw)
        ax.plot([xmin, xmax], [ymin, ymin], 0, **edges_kw)
        ax.plot([xmax, xmax], [ymin, ymin], [zmin, zmax], **edges_kw)
        # ax.view_init(40, 30, 0)
        #ax.set_box_aspect(None, zoom=1)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
    
    
    
    # Separate the 3D array into 2D arrays along different axes
        # array_2d_x = array_3d.mean(axis=0)  # 2D array along X-axis (shape: (300, 100))
        # array_2d_y = array_3d.mean(axis=1)  # 2D array along Y-axis (shape: (300, 100))
        # array_2d_z = array_3d.mean(axis=2)  # 2D array along Z-axis (shape: (300, 300))
    
        # Separate the 3D array into 2D arrays along different axes
        # array_2d_x = array_3d.mean(axis=0)  # 2D array along X-axis (shape: (300, 100))
        # array_2d_y = array_3d.mean(axis=1)  # 2D array along Y-axis (shape: (300, 100))
        
        # # Slice the 3D array along the Z-axis to have dimensions (300, 100)
        # array_2d_z = array_3d[:, :, :100].mean(axis=2)  # 2D array along Z-axis (shape: (300, 100))
    
        # # Flatten the 3D arrays for contour plotting
        # # X_flat = X[mask].flatten()
        # # Y_flat = Y[mask].flatten()
        # # Z_flat = Z[mask].flatten()
        # ax.contour3D(array_2d_x, array_2d_y, array_2d_z, levels=10, cmap='twilight', alpha=0.5)
        
        # Set plot limits if necessary
        # ax.set_xlim(0, 100)
        # ax.set_ylim(0, 100)
        # ax.set_zlim(0, 100)
    
        # Save the plot as a frame
        
        frame_path = os.path.join(contour_frames_directory, f"contour_frame_{j}.png")
        plt.savefig(frame_path, dpi=300)
        plt.close()
    
        print(f"Frame {j} generated: {frame_path}")
    
    except Exception as e:
    # Handle any exceptions (file not found, empty file, etc.) and continue to the next iteration
        print(f"Error processing file {j}: {str(e)}")
        continue






