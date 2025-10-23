

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 20:15:55 2023

@author: pranavbharadwajgangrekalvemanoj
"""
    
# import numpy as np
# import os
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import imageio

# # Path to the directory containing gifs
# gif_directory = "/Users/pranavbharadwajgangrekalvemanoj/Desktop/Axions_Project/gifs/"

# # Path to the directory where frames will be loaded from
# contour_frames_directory = "/Users/pranavbharadwajgangrekalvemanoj/Desktop/Axions_Project/relevant_files_310124/npy_files_full_positive_boost_ev_1/"

# # Create a GIF using the saved frames
# frames = []

# for j in range(0,91,10):
#     frame_path = os.path.join(contour_frames_directory, f"pi_power_spec_xy_plane_12_radius_{j}.png")
#     try:
#         frame_image = imageio.imread(frame_path)
#         frames.append(frame_image)
#     except Exception as e:
#         # Handle any exceptions (file not found, unsupported format, etc.) and continue to the next iteration
#         print(f"Error processing frame {j}: {str(e)}")
#         continue

# # method to make faster gif

# # Path to the output GIF file
# output_gif_path = os.path.join(gif_directory, "pi_incremental_radius_circular_mask_power_spectra_comp.gif")

# # Save frames as a GIF
# try:
#     imageio.mimsave(output_gif_path, frames, duration=0.2)
#     print(f"GIF created and saved at: {output_gif_path}")
# except Exception as e:
#     print(f"Error creating GIF: {str(e)}")
    

# import numpy as np
# import os
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import imageio

# # Path to the directory containing input files lmao this sucks
# input_directory = "/Users/pranavbharadwajgangrekalvemanoj/Desktop/Axions_Project/evolution_run_2"

# # Path to the directory where frames will be saved
# contour_frames_directory = "/Users/pranavbharadwajgangrekalvemanoj/Desktop/Axions_Project/evolution_run_2/string_pos_plots_v2/combined_plots"

# # Create a GIF using every second saved frame
# frames = []
# for i in range(0, 156, 1):  # Include every second frame
#     frame_path = os.path.join(contour_frames_directory, f"plot_frame_{i}.png")
#     frames.append(imageio.imread(frame_path))

# # Path to the output GIF file
# output_gif_path = os.path.join(input_directory, "curvature_test_2.gif")

# # Save frames as a GIF with a reduced duration (playback speed)
# imageio.mimsave(output_gif_path, frames, duration=0.01)

# print(f"GIF created and saved at: {output_gif_path}")
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

# Path to the text file
file_path = "/Users/pranavbharadwajgangrekalvemanoj/Desktop/Axions_Project/vals_dx_0.7_cuboid.txt"
file_path_2 = "/Users/pranavbharadwajgangrekalvemanoj/Desktop/Axions_Project/vals_dx_0.7_199.txt"
file_path_3 = "/Users/pranavbharadwajgangrekalvemanoj/Desktop/Axions_Project/scp_from_centaurus/g_175/valsPerLoop.txt"
file_path_4 = "/Users/pranavbharadwajgangrekalvemanoj/Desktop/Axions_Project/vals_dx_0.7.txt"
file_path_5 = "/Users/pranavbharadwajgangrekalvemanoj/Desktop/Axions_Project/scp_from_centaurus/g_0_cuboid/valsPerLoop.txt"


file_path_g_0 = "/Users/pranavbharadwajgangrekalvemanoj/Desktop/Axions_Project/valsPerLoop_g_0.txt"
file_path_g_01 = "/Users/pranavbharadwajgangrekalvemanoj/Desktop/Axions_Project/valsPerLoop_g_0.1.txt"
file_path_g_0001= "/Users/pranavbharadwajgangrekalvemanoj/Desktop/Axions_Project/valsPerLoop_g_0.001_201.txt"
# Read the text file
data = np.loadtxt(file_path)
data2 = np.loadtxt(file_path_2)
data3 = np.loadtxt(file_path_3)
data4 = np.loadtxt(file_path_4)
data5 = np.loadtxt(file_path_5)

data_g_0 = np.loadtxt(file_path_g_0)
data_g_01 = np.loadtxt(file_path_g_01)
data_g_0001 = np.loadtxt(file_path_g_0001)

threshold_g0 = 3.2
threshold_g05 = 4

# Extract the columns
column1 = data[:, 0]
column2 = data[:, 1]
column3 = data[:, 3]

data2_column1 = data2[:, 0]
data2_column2 = data2[:, 1]
data2_column3 = data2[:, 3] 

data3_column1 = data3[:, 0]
data3_column2 = data3[:, 1]
data3_column3 = data3[:, 3] 

data4_column1 = data4[:, 0]
data4_column2 = data4[:, 1]
data4_column3 = data4[:, 3]

data5_column1 = data5[:, 0]
data5_column2 = data5[:, 1]
data5_column3 = data5[:, 3]

data_g0_column1 = data_g_0[:, 0]
data_g0_column2 = data_g_0[:, 1]
data_g0_column3 = data_g_0[:, 3]

data_g01_column1 = data_g_01[:, 0]
data_g01_column2 = data_g_01[:, 1]
data_g01_column3 = data_g_01[:, 3]

data_g0001_column1 = data_g_0001[:, 0]
data_g0001_column2 = data_g_0001[:, 1]
data_g0001_column3 = data_g_0001[:, 3]

# Mask for column3
mask_column3 = np.where(data4_column3 > threshold_g05, data4_column3, 0)
mask_g0_column3 = np.where(data5_column3 > threshold_g0, data5_column3, 0)

# Find peaks in masked column3
peaks, _ = find_peaks(mask_column3)
peaks_g0, _ = find_peaks(mask_g0_column3)

peak_x_values = peaks
peak_y_values = data4_column3[peaks]

peak_g0_x_values = peaks_g0
peak_g0_y_values = data5_column3[peaks_g0]


# print(peaks)
# print("---------------------------------")
# print(peaks_g0)
# Create an array of numbers from 0 to the number of points in the columns
# x = np.arange(len(column1))
# x = x* 0.3

x_cuboid = np.arange(len(column1)) * 0.3
x_199 = np.arange(len(data2_column1)) * 0.3
x_150 = np.arange(len(data3_column1)) * 0.3
x_101 = np.arange(len(data4_column1)) * 0.3
x_g0 = np.arange(len(data5_column1)) * 0.3


x_g0_801 = np.arange(len(data_g0_column1)) * 0.3
x_g01_801 = np.arange(len(data_g01_column1)) * 0.3
x_g0001_201 = np.arange(len(data_g0001_column1)) * 0.3
# Plot the columns
# plt.plot(x_101, data4_column1 / data4_column1[0]  , label = '201 whole grid ')
# plt.plot(x_199, data2_column1 / data2_column1[0] , label = '199 mini grid ')
# plt.plot(x_cuboid, column2 / column2[0] , label = 'g = 0.5 cuboid mini grid ')
# plt.plot(x_150, data3_column2 / data3_column2[0] , label = '150 mini grid ')
# plt.plot(x_101, data4_column2 / data4_column2[0] , label = 'g = 0.5 101 mini grid ')
# plt.plot(x_g0, data5_column2 / data5_column2[0], label = 'g = 0  mini grid ')

# plt.plot(x_g0_801, data_g0_column3 , label = 'whole grid g = 0')
# plt.plot(x_g01_801, data_g01_column3 , label = 'whole grid g = 0.1')
# plt.plot(x_g0001_201, data_g0001_column3 , label = 'whole grid g = 0.001')
# plt.plot(x_g0_801, data_g0_column2 , label = 'cuboid mini grid ')

# plt.plot(x_101, mask_column3)
# plt.plot(x_g0, mask_g0_column3)

# # Add labels and legend
# plt.xlabel('Time Step')
# plt.ylabel('Normalised Energy')
# plt.legend()

# # # Show the plot
# plt.show()


# fig, ax = plt.subplots()
# ax.scatter(peak_x_values, peak_y_values, label='g = 0.5', marker='s', s=200, facecolors='none', edgecolors='b')
# ax.scatter(peak_g0_x_values, peak_g0_y_values, label='g = 0', marker='s', s=200, facecolors='none', edgecolors='r')

# # Connect points within each scatter plot
# for i in range(len(peak_x_values) - 1):
#     ax.plot([peak_x_values[i], peak_x_values[i+1]], [peak_y_values[i], peak_y_values[i+1]], linestyle='--', alpha=0.3, color='blue')
#     ax.plot([peak_g0_x_values[i], peak_g0_x_values[i+1]], [peak_g0_y_values[i], peak_g0_y_values[i+1]], linestyle='--', alpha=0.3, color='red')

# # ax.set_ylim(4.26, 4.25)
# ax.set_xlabel('Time Step')
# ax.set_yscale('log')
# ax.set_ylabel('Amplitude')
# ax.legend()
# plt.show()