# #!/usr/bin/env python3

# # -*- coding: utf-8 -*-
# """
# Created on Thu Oct 12 19:02:23 2023

# @author: pranavbharadwajgangrekalvemanoj
# """


# import numpy as np
# import os
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from mpl_toolkits import mplot3d
# import imageio
# import scipy

# # Path to the directory containing input files
# input_directory = "/Users/pranavbharadwajgangrekalvemanoj/Desktop/Axions_Project/per_files/run_5_per/npy_files"
# contour_frames_directory = "/Users/pranavbharadwajgangrekalvemanoj/Desktop/Axions_Project/per_files/"
# # Path to the directory where frames will be saved
# frames_directory = os.path.join(contour_frames_directory, "contour_frames/")
# if not os.path.exists(frames_directory):
#     os.makedirs(frames_directory)
# dx = 0.5
# dy = 0.5
# dz = 0.5
# # Define the sampling indices for each axis
# sampling_interval_x = 1  # Sample every 10th point along X axis
# sampling_interval_y = 1  # Sample every 10th point along Y axis
# sampling_interval_z = 1  # Sample every 10th point along Z axis

# # Iterate through all input files and generate frames
# for j in range(201):  # Assuming you have 200 input files numbered from 0 to 200
#     try:
#         input_file_path = os.path.join(input_directory, f"col_0_frame_{j}.npy")
    
#         # Load data from the input file
#         array_3d = np.load(input_file_path)
        
#         # Separate the columns into individual arrays
#         # column_arrays = [array_1d[:, 0] for i in range(array_1d.shape[1])]
#         # array_1d = array_1d[:, 0]

#         # Iterate through each column array and generate contour plot
#         # for idx, column_array in enumerate(column_arrays):
    
#         # Reshape the 1D array into a 3D array

#         # array_3d = array_1d.reshape(nx, ny, nz)
    
    
#         # # Sample the 3D array
#         # sampled_array_3d = array_3d[::sampling_interval_x, ::sampling_interval_y, ::sampling_interval_z]
    
#         # # Create meshgrid for sampled data
#         # X, Y, Z = np.meshgrid(
#         #     np.arange(0, nx, sampling_interval_x),
#         #     np.arange(0, ny, sampling_interval_y),
#         #     np.arange(0, nz, sampling_interval_z)
#         # )
    
#         # # Plot the sampled 3D data using a scatter plot
#         # threshold_value = 0.1  # Adjust this value based on your data
        
#         # # Define the threshold value below which points will be considered transparent
#         # # Adjust this value based on your data
        
        
#         # # Create a mask to identify points above the threshold
#         # mask = sampled_array_3d >= threshold_value
        
#         # Plot the sampled 3D data using a scatter plot with transparency
#         fig = plt.figure()
#         ax = fig.add_subplot(111, projection='3d')
        
#         # Plot points above the threshold with full opacity
#         # ax.scatter(X[mask], Y[mask], Z[mask], c=sampled_array_3d[mask].flatten(), cmap="twilight", alpha=1)
        
#         # # Plot points below the threshold with transparency (alpha=0)
#         # ax.scatter(X[~mask], Y[~mask], Z[~mask], alpha=0)
#         # Plot isosurfaces using contours
    
    
    
#     # Assuming array_3d is your original 3D array with shape (300, 300, 100)
#     # Slice the 3D array to match the desired dimensions (300, 100)
#         array_3d_new = array_3d
    
#         # Create mesh grids for X, Y, and Z with consistent dimensions (100, 100, 100)
#         X, Y, Z = np.meshgrid(
#             np.arange(0, array_3d.shape[1], sampling_interval_x),  # X-axis (0 to 100, sampling every 3rd point)
#             np.arange(0, array_3d.shape[0], sampling_interval_y),  # Y-axis (0 to 100, sampling every 3rd point)
#             np.arange(0, array_3d.shape[2], sampling_interval_z)  # Z-axis (0 to 100, sampling every 1st point)
#         )
    
#         # Separate the 3D array into 2D arrays along different axes
#         array_2d_x = array_3d.mean(axis=0)  # 2D array along X-axis (shape: (100, 100))
#         array_2d_y = array_3d.mean(axis=1)  # 2D array along Y-axis (shape: (100, 100))
#         array_2d_z = array_3d.mean(axis=2)  # 2D array along Z-axis (shape: (100, 100))
    
#         # Create a 3D contour plot with consistent dimensions along all axes
#         fig = plt.figure()
#         ax = fig.add_subplot(111, projection='3d')
#         ax.contour3D(array_2d_x, array_2d_y, array_2d_z, levels=50, cmap='Greys', alpha=0.5)
        
#         # Customize plot labels, title, etc. as needed
#         ax.set_xlabel('X Label')
#         ax.set_ylabel('Y Label')
#         ax.set_zlabel('Z Label')
#         # ax.set_title('3D Contour Plot')
#         nx = 100
#         ny = 100
#         nz = 100
#         fig = plt.figure(figsize=(15,15))
        
#         ax = fig.add_subplot(111, projection='3d')
    
#         X, Y, Z = np.meshgrid(np.arange(nx), np.arange(ny), -np.arange(nz))
    
#         kw = {
#             'vmin': array_3d_new.min(),
#             'vmax': array_3d_new.max(),
#             'levels': np.linspace(array_3d_new.min(), array_3d_new.max(), 10),
#         }
    
    
#         for i in range(nz):
#             ax.contour(
#                 X[:, :, i], Y[:, :, i], array_3d_new[:, :, i],
#                 levels=[0.2], colors='k', zdir='z', offset=-i,linewidths = 0.2,
#             )
    
#         for i in range(nx):
#             ax.contour(
#                 X[i, :, :], Y[i, :, :], array_3d_new[i, :, :],
#                 levels=[0.2], colors='k', zdir='y', offset=0,linewidths = 0.2,
#             )
    
#         ax.contour(
#             array_3d_new[:, -1, :], Y[:, -1, :], Z[:, -1, :],
#             levels=[0.2], colors='k', zdir='x', offset=nx,linewidths = 0.2,)
    
#         # Set limits of the plot from coord limits
#         xmin, xmax = X.min(), X.max()
#         ymin, ymax = Y.min(), Y.max()
#         zmin, zmax = Z.min(), Z.max()
#         ax.set(xlim=[xmin, xmax], ylim=[ymin, ymax], zlim=[zmin, zmax])
    
#         # Plot edges
#         edges_kw = dict(color='0.4', linewidth=0, zorder=1e3)
#         ax.plot([xmax, xmax], [ymin, ymax], 0, **edges_kw)
#         ax.plot([xmin, xmax], [ymin, ymin], 0, **edges_kw)
#         ax.plot([xmax, xmax], [ymin, ymin], [zmin, zmax], **edges_kw)
#         # ax.view_init(40, 30, 0)
#         #ax.set_box_aspect(None, zoom=1)
#         ax.set_xlabel("x")
#         ax.set_ylabel("y")
#         ax.set_zlabel("z")
    
    
    
#     # Separate the 3D array into 2D arrays along different axes
#         # array_2d_x = array_3d.mean(axis=0)  # 2D array along X-axis (shape: (300, 100))
#         # array_2d_y = array_3d.mean(axis=1)  # 2D array along Y-axis (shape: (300, 100))
#         # array_2d_z = array_3d.mean(axis=2)  # 2D array along Z-axis (shape: (300, 300))
    
#         # Separate the 3D array into 2D arrays along different axes
#         # array_2d_x = array_3d.mean(axis=0)  # 2D array along X-axis (shape: (300, 100))
#         # array_2d_y = array_3d.mean(axis=1)  # 2D array along Y-axis (shape: (300, 100))
        
#         # # Slice the 3D array along the Z-axis to have dimensions (300, 100)
#         # array_2d_z = array_3d[:, :, :100].mean(axis=2)  # 2D array along Z-axis (shape: (300, 100))
    
#         # # Flatten the 3D arrays for contour plotting
#         # # X_flat = X[mask].flatten()
#         # # Y_flat = Y[mask].flatten()
#         # # Z_flat = Z[mask].flatten()
#         # ax.contour3D(array_2d_x, array_2d_y, array_2d_z, levels=10, cmap='twilight', alpha=0.5)
        
#         # Set plot limits if necessary
#         # ax.set_xlim(0, 100)
#         # ax.set_ylim(0, 100)
#         # ax.set_zlim(0, 100)
    
#         # Save the plot as a frame
        
#         frame_path = os.path.join(frames_directory, f"stat_contour_frame_{j}.png")
#         plt.savefig(frame_path, dpi=300)
#         plt.close()
    
#         print(f"Frame {j} generated: {frame_path}")
    
#     except Exception as e:
#     # Handle any exceptions (file not found, empty file, etc.) and continue to the next iteration
#         print(f"Error processing file {j}: {str(e)}")
#         continue

#----------------------------------------------------------------------------------------------------------------------------------------------------------------
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.fft import fftn, fftshift
# import os

# def fftfreq3D(shape, dx, dy, dz):
#     Nx, Ny, Nz = shape
#     freqs_x = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(Nx, dx))
#     freqs_y = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(Ny, dy))
#     freqs_z = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(Nz, dz))
#     freqs_3D = np.array(np.meshgrid(freqs_x, freqs_y, freqs_z, indexing='ij'))
#     return freqs_3D

# # Specify the directory where the data file is located
# dx = 0.25
# dy = 0.25
# dz = 0.25
# nx = 101  # Adjusted based on your data shape
# ny = 101
# nz = 101
# # bin_start = (((3)**(1/3))*2*np.pi)/(nx*dx)
# # bin_end = (2*np.pi)/(dx)
# bin_start = 0
# bin_end = 10
# n_bin = 400
# shape = (nx, ny, nz)  
# input_directory = "/Users/pranavbharadwajgangrekalvemanoj/Desktop/Axions_Project/trashh/npy_files/"  # Specify your directory

# # Load data from .npy file
# data_3d = np.load(os.path.join(input_directory, 'col_0_frame_299.npy'))
# data_real = np.load(os.path.join(input_directory, 'col_1_frame_299.npy'))
# data_im = np.load(os.path.join(input_directory, 'col_2_frame_299.npy'))

# # Assuming your grid is already in the right shape
# y_grid, x_grid, z_grid = np.meshgrid(np.arange(ny), np.arange(nx), np.arange(nz))

# data_3d = data_3d * np.sqrt(2)  # Assuming data is not squared already

# # Compute 3D Fourier Transform
# fft_data = np.fft.fftn(data_3d)
# fft_data = np.fft.fftshift(fft_data)
# fft_data = np.abs(fft_data)**2

# # Compute k-space frequencies
# k_space = fftfreq3D(shape, dx, dy, dz)

# dkx = k_space[0, 89, 60, 70] - k_space[0, 90, 60, 70]
# constant_factor = dkx**2 / ((2 * np.pi * nx * dkx)**3)

# k_bins = np.linspace(bin_start, bin_end, num=n_bin)

# epsilon = (bin_end - bin_start) / (n_bin * 2)
# storage = np.zeros(n_bin)

# for z in range(0, nz):
#     for y in range(0, ny):
#         for x in range(0, nx):
#             k = k_space[:, x, y, z]
#             kx, ky, kz = k[0], k[1], k[2]
#             mod_k = np.sqrt(kx**2 + ky**2 + kz**2)
#             for m in range(0, n_bin):
#                 if (mod_k >= (m * 2 * epsilon)) and (mod_k < 2 * epsilon * (m + 1)):
#                     pre_factor = constant_factor / mod_k
#                     solid_angle = (((kx**2 - ky**2) * kz) / (kx**2 + ky**2)) + ky - kx
#                     current_point = np.abs(pre_factor * solid_angle * fft_data[x, y, z])
#                     storage[m] = storage[m] + current_point

# plt.bar(k_bins, storage)
# plt.xlabel('|k|')
# plt.ylabel('Spectrum')
# plt.title('Spectrum histogram binning')
# plt.show()

# fig, ax = plt.subplots(figsize=(10, 7))
# plt.plot(k_bins - epsilon, storage)
# plt.yscale("log")
# plt.xscale("log")
# plt.xlabel("|k|")
# plt.xlim(bin_start + 1 * epsilon, bin_end - 1 * epsilon)
# plt.ylabel("Spherical Power Spectrum")
# plt.title("Spherical Power Spectrum")
# plt.show()


#----------------------------------------------------------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------------------------------------------------------



# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.fft import fftn, fftshift
# import os

# def fft_func(shape, dx, dy, dz):
#   Nx, Ny, Nz = shape
#   freqs_x = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(Nx, dx))
#   freqs_y = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(Ny, dy))
#   freqs_z = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(Nz, dz))
#   freqs_3D = np.array(np.meshgrid(freqs_x, freqs_y, freqs_z, indexing='ij'))
#   return freqs_3D

# # Specify the directory where the data file is located
# dx = 0.5
# dy = 0.5
# dz = 0.5
# nx = 101 # Adjusted based on your data shape
# ny = 101
# nz = 101
# bin_start = 0
# bin_end = 10
# n_bin = 500
# shape = (nx, ny, nz) 
# input_directory = "/Users/pranavbharadwajgangrekalvemanoj/Desktop/Axions_Project/comp/npy_files/" # Specify your directory

# # Load data from .npy file
# data_3d = np.load(os.path.join(input_directory, 'col_0_frame_157.npy'))
# data_real = np.load(os.path.join(input_directory, 'col_1_frame_157.npy'))
# data_im = np.load(os.path.join(input_directory, 'col_2_frame_157.npy'))

# #reference file
# data_3d_ref = np.load( '/Users/pranavbharadwajgangrekalvemanoj/Desktop/Axions_Project/npy_files/col_0_frame_20.npy')
# data_real_ref = np.load( '/Users/pranavbharadwajgangrekalvemanoj/Desktop/Axions_Project/npy_files/col_1_frame_20.npy')
# data_im_ref = np.load( '/Users/pranavbharadwajgangrekalvemanoj/Desktop/Axions_Project/npy_files/col_2_frame_20.npy')

# complex_number = data_real + 1j * data_im
# complex_number_ref = data_real_ref + 1j * data_im_ref
# comp_num = complex_number - complex_number_ref
# # Assuming your grid is already in the right shape
# y_grid, x_grid, z_grid = np.meshgrid(np.arange(ny), np.arange(nx), np.arange(nz))
# data_3d = np.abs(comp_num)
# # data_3d = data_3d **(2) # Assuming data is not squared already

# # Compute 3D Fourier Transform
# fft_data = np.fft.fftn(data_3d)
# fft_data = np.fft.fftshift(fft_data)
# fft_data = np.abs(fft_data)**2

# # Compute k-space frequencies
# k_space = fft_func(shape, dx, dy, dz)

# dkx = k_space[0, 0, 0, 0] - k_space[0, 1, 0, 0]
# constant_factor = dkx**2 / ((2 * np.pi * nx * dkx)**3)

# k_bins = np.linspace(bin_start, bin_end, num=n_bin)

# epsilon = (bin_end - bin_start) / (n_bin * 2)
# storage = np.zeros(n_bin)

# # Compute mod_k and indices for binning
# mod_k = np.sqrt(np.sum(k_space**2, axis=0))
# indices = np.digitize(mod_k.flatten(), bins=k_bins) - 1

# # Compute pre_factor and solid_angle
# pre_factor = constant_factor / mod_k.flatten()
# solid_angle = (((k_space[0]**2 - k_space[1]**2) * k_space[2]) / (k_space[0]**2 + k_space[1]**2)) + k_space[1] - k_space[0]
# current_point = np.abs(pre_factor.flatten() * solid_angle.flatten() * fft_data.flatten())


# # Accumulate current_point into storage bins
# np.add.at(storage, indices, current_point)

# # Plotting
# plt.figure(figsize=(10, 7))
# plt.bar(k_bins, storage, color='lightblue')
# plt.xlabel('|k|')
# plt.ylabel('Spherical Power Spectrum')
# plt.title('Spectrum histogram binning')
# plt.grid(True)
# plt.show()

# plt.figure(figsize=(10, 7))
# plt.plot(k_bins - epsilon, storage, color='skyblue', linewidth=2)
# plt.yscale("log")
# plt.xscale("log")
# plt.xlabel("|k|")
# plt.xlim(bin_start + 1 * epsilon, bin_end - 1 * epsilon)
# plt.ylabel("Spherical Power Spectrum")
# plt.title("Spherical Power Spectrum")
# plt.grid(True)
# plt.show()


#----------------------------------------------------------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------------------------------------------------------


import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import colors, gridspec
from matplotlib.ticker import FuncFormatter
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable

# import plotly.graph_objs as go

os.environ['PATH'] = '/usr/local/texlive/2023/bin:' + os.environ['PATH']
# matplotlib.use("pgf")

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'text.usetex': True,
    'pgf.texsystem': 'pdflatex',
    'pgf.rcfonts': False,
})

# Path to the directory containing input files
input_directory = "/Users/pranavbharadwajgangrekalvemanoj/Desktop/Axions_Project/sem_2_xy_abs_z_fix_101/npy_files_full_positive_boost_ev_1/"
reference_directory = "/Users/pranavbharadwajgangrekalvemanoj/Desktop/Axions_Project/sem_2_xy_abs_z_fix_101//npy_files_full_positive_boost_ev_1/"
frames_directory = "/Users/pranavbharadwajgangrekalvemanoj/Desktop/Axions_Project/sem_2_xy_abs_z_fix_101/npy_files_full_positive_boost_ev_1/"

if not os.path.exists(frames_directory):
    os.makedirs(frames_directory)

# Load the reference npy file
# reference_file_path = "/Users/pranavbharadwajgangrekalvemanoj/Desktop/Axions_Project/npy_file_singlular/col_0_frame_100.npy"
# reference_data = np.load(reference_file_path)

def calculate_power_spectrum(data_2d, delta):
    F_tilde = np.fft.fft2(data_2d) / np.sqrt(data_2d.shape[0] * data_2d.shape[1])
    F_inverse = np.fft.ifft2(F_tilde)
    # plt.contourf(F_inverse, cmap='RdPu', levels=200)
    # plt.show()
    F_tilde = np.fft.fftshift(F_tilde)
    P = np.abs(F_tilde)**2 / (data_2d.shape[0] * data_2d.shape[1])
    return P

def fftfreq2D(Nx, Ny, dx, dy):
    kx = np.fft.fftfreq(Nx, dx)
    kx = np.fft.fftshift(kx) * 2 * np.pi
    ky = np.fft.fftfreq(Ny, dy)
    ky = np.fft.fftshift(ky) * 2 * np.pi
    kx_2d, ky_2d = np.meshgrid(kx, ky, indexing='ij')
    return kx_2d, ky_2d

def plot_power_spectrum(Nx, Ny, delta, phi2t, custom_bins=None):
    # for z_plane in range(data_3d1.shape[2]):
        
        delta_t = 0.3
        # data_2d1 = data_3d1
        
        # mask_neg_2pi = np.isclose(data_2d1, -2 * np.pi, atol=tol)
        # mask_pos_2pi = np.isclose(data_2d1, 2 * np.pi, atol=tol)
        
        # data_2d1[mask_neg_2pi] += 2 * np.pi
        # data_2d1[mask_pos_2pi] -= 2 * np.pi
        
        # mask_neg_2pi = np.isclose(data_2d1, - np.pi, atol=tol)
        # mask_pos_2pi = np.isclose(data_2d1,  np.pi, atol=tol)
        
        # data_2d1[mask_neg_2pi] +=  np.pi
        # data_2d1[mask_pos_2pi] -= np.pi
        
        # data_2d2 = data_3d2
        
        # mask_neg_2pi = np.isclose(data_2d2, -2 * np.pi, atol=tol)
        # mask_pos_2pi = np.isclose(data_2d2, 2 * np.pi, atol=tol)
        
        # data_2d2[mask_neg_2pi] += 2 * np.pi
        # data_2d2[mask_pos_2pi] -= 2 * np.pi
        
        # mask_neg_2pi = np.isclose(data_2d2, - np.pi, atol=tol)
        # mask_pos_2pi = np.isclose(data_2d2,  np.pi, atol=tol)
        
        # data_2d2[mask_neg_2pi] +=  np.pi
        # data_2d2[mask_pos_2pi] -= np.pi
        
        # del_alpha = (data_2d2 - data_2d1)/delta_t
        
        # phi2t_plane = phi2t
        # phi2_del2 = phi2t_plane * np.abs(del_alpha)**2
        # phase_corrected = np.where(np.abs(data_2d) > 2*np.pi, data_2d - 2*np.pi*np.sign(data_2d), data_2d)
        # data_2d = np.where(np.abs(phase_corrected) > np.pi, phase_corrected + np.pi*np.sign(phase_corrected), phase_corrected)


        P_values = calculate_power_spectrum(phi2t, delta)

        Nz = 51
        beta1, beta2 = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing='ij')
        beta1_flat = beta1.ravel()
        beta2_flat = beta2.ravel()

        # k_values = np.sqrt((2 * np.pi * beta1_flat) / (Nx * delta))**2 + \
        #            np.sqrt((2 * np.pi * beta2_flat) / (Ny * delta))**2
        k_values_x, k_values_y = fftfreq2D(Nx, Ny, delta, delta) 
        mod_k = np.sqrt(k_values_x**2 + k_values_y**2)

        n_values = (mod_k * Nz * delta) / (2 * np.pi)
        
        # logic1 = (n_values > 0.5) & (n_values<1.5)
        # logic12= (n_values > 1.5) & (n_values<2.5)

        # print(np.sum(P_values[logic1]), np.sum(P_values[logic12]))

        plt.figure(figsize=(10, 5))
        # plt.style.use('seaborn-whitegrid')

        plt.subplot(1, 2, 1)
        plt.contourf(phi2t, cmap='plasma', levels=200)
        plt.title('Contour Plot')
        plt.colorbar()

        plt.subplot(1, 2, 2)
        plt.hist(x=n_values.ravel(), weights=P_values.ravel(), bins=np.linspace(0.5,10.5,11), color='skyblue', edgecolor='black')
        plt.xlabel('n values')
        plt.ylabel('Power Spectrum')
        plt.title('Binned Power Spectrum')
        plt.tight_layout()


        power_path = os.path.join(input_directory, f"power_spec_xy_plane_{+ 1}.png")
        plt.xlabel('n values')
        plt.ylabel('Power Spectrum')
        plt.title(f'Binned Power Spectrum - xy Plane { + 1}')
        plt.savefig(power_path, dpi=300)
        plt.show()

start_frame = 2751
end_frame = 2751
dt = 0.3
phi2_difference_list = []
phi_square_list = []
pseudo_square_list = []
phi2_difference_non_diff_phases = []
all_phi2_difference_list = []
all_phi_square_list = []
all_pseudo_square_list = []
all_phi2_difference_non_diff_phases = []

y, x = np.ogrid[-100:101, -100:101]

# Create a mask for a circle of radius 30
circular_mask = x**2 + y**2 < 40**2

# def adjust_to_zero(value):
#     # Define the thresholds for pi and 2*pi
#     pi_threshold = np.pi
#     two_pi_threshold = 2 * np.pi

#     # Calculate the differences from pi and 2*pi
#     diff_from_pi = abs(value - pi_threshold)
#     diff_from_two_pi = abs(value - two_pi_threshold)

#     # Check which threshold is closer and adjust accordingly
#     if diff_from_pi < diff_from_two_pi:
#         return value - np.sign(value) * diff_from_pi
#     else:
#         return value - np.sign(value) * diff_from_two_pi
# Iterate through all files and generate frames
for frame_number in range(start_frame, end_frame + 1):
    try:

        input_file_path_phi_0 = os.path.join(input_directory, f"col_0_frame_{frame_number}.npy")
        input_file_path_phi_1 = os.path.join(input_directory, f"col_1_frame_{frame_number}.npy")
        reference_file_path_1 = os.path.join(reference_directory, f"frame_{frame_number}_positive_boost_0.npy")
        reference_file_path_2 = os.path.join(reference_directory, f"frame_{frame_number}_positive_boost_1.npy")
    

# # Iterate through all input files and generate frames
# for j in range(201):  # Assuming you have 200 input files numbered from 0 to 200
#     try:
#         input_file_path = os.path.join(input_directory, f"col_0_frame_{j}.npy")
        
#         # Load data from the input file
#         input_data = np.load(input_file_path)
        
#         # Subtract corresponding points in the grids
#         energy_difference = input_data - reference_data
        
#         # Create 100 planes along the z-axis
#         nz = energy_difference.shape[2]
#         frames = 100
#         z_step = nz // frames
        
#         for k in range(frames):
#             z_start = k * z_step
#             z_end = (k + 1) * z_step
            
#             # Extract the plane along the z-axis
#             plane_data = energy_difference[:, :, z_start:z_end]
            
#             # Generate contour plot for the plane
#             plt.figure(figsize=(8, 8))
#             plt.contourf(plane_data[:, :, 0], cmap='viridis')  # Assuming z-axis step is 1
            
#             # Customize plot labels, title, etc. as needed
#             plt.xlabel('X Label')
#             plt.ylabel('Y Label')
#             plt.title(f'Energy Density Contour Plot - Frame {j}, Z-Plane {k}')
            
#             # Save the frame
#             frame_path = os.path.join(frames_directory, f"frame_{j}_plane_{k}.png")
#             plt.savefig(frame_path, dpi=300)
#             plt.close()
            
#             print(f"Frame {j}, Plane {k} generated: {frame_path}")

#     except Exception as e:
#         # Handle any exceptions (file not found, empty file, etc.) and continue to the next iteration
#         print(f"Error processing file {j}: {str(e)}")
#         continue

# reference_data = np.load(reference_file_path)  # Select the 50th plane along the z-axis
        reference_data_1 = np.load(reference_file_path_1)  # Select the 50th plane along the z-axis
        reference_data_2 = np.load(reference_file_path_2)  # Select the 50th plane along the z-axis

        input_data_phi_0 = np.load(input_file_path_phi_0)
        input_data_phi_1 = np.load(input_file_path_phi_1)
        mod = reference_data_1**2 + reference_data_2**2
        complex_data = input_data_phi_0 + 1j *input_data_phi_1
        complex_data_ref = reference_data_1 + 1j *reference_data_2
        phases = np.angle(complex_data)
        phases_ref = np.angle(complex_data_ref)
        diff_phases = phases - phases_ref
        # diff_phases = np.vectorize(adjust_to_zero)(diff_phases)
        evolved_mod = input_data_phi_0**2 + input_data_phi_1**2
        
        # phi_phase = np.angle(complex_data)
        
        # Subtract corresponding points in the grids
        phi2_differences = mod - evolved_mod
        # phases = np.angle(diff_comp)
        phases = np.abs(phases)
# mod_of_mod = np.abs(phi2_differences)
        # # energy_difference[energy_difference < 0] = 0 
        
        # # Generate contour plot for the 50th plane
        # plt.figure(figsize=(8, 8))
        # plt.contourf(phi2_difference, cmap='viridis')
        
        # # Customize plot labels, title, etc. as needed
        # plt.xlabel('X Label')
        # plt.ylabel('Y Label')
        # plt.title(f'Energy Density Contour Plot - Frame {p}, Z-Plane 50')
        
        # # Save the frame
        # frame_path = os.path.join(frames_directory, f"frame_{p}_plane_50.png")
        # plt.savefig(frame_path, dpi=300)
        # plt.close()
        
        # print(f"Frame {p}, Plane 50 generated: {frame_path}")
   
# for z_plane in range(12,51):
#     try:
        # xy_plane_data = input_data[:, :, z_plane]  # Extract XY plane data. was mod before
        # xy_plane_data = evolved_mod[:, :, z_plane]
        # Subtract corresponding points in the grids
        tol = 0.01
        z_slice = 12
        phi2_difference = diff_phases[:,:, z_slice]
        phi2_difference_non_diff_phase = phases[:, :, z_slice]
        phi2_difference_non_diff_phases.append(phi2_difference_non_diff_phase)
        # phi2_difference_corrected = np.where(np.abs(phi2_difference) > 2 * np.pi, phi2_difference + 2*np.pi*np.sign(phi2_difference), phi2_difference)
        # phi2_difference_corrected = np.where(np.abs(phi2_difference_corrected) > np.pi, phi2_difference_corrected + np.pi*np.sign(phi2_difference_corrected), phi2_difference_corrected)
        mask_neg_2pi = np.isclose(phi2_difference, -2 * np.pi, atol=tol)
        mask_pos_2pi = np.isclose(phi2_difference, 2 * np.pi, atol=tol)

        phi2_difference[mask_neg_2pi] += 2 * np.pi
        phi2_difference[mask_pos_2pi] -= 2 * np.pi
        
        mask_neg_pi = np.isclose(phi2_difference, - np.pi, atol=tol)
        mask_pos_pi = np.isclose(phi2_difference,  np.pi, atol=tol)

        phi2_difference[mask_neg_pi] += np.pi
        phi2_difference[mask_pos_pi] -= np.pi
        
        phi2_difference_corrected = phi2_difference
        mo = mod[:, :, z_slice]
        evolved_mo = evolved_mod[:, :, z_slice]
        phi2_difference_list.append(phi2_difference_corrected)

        phi2_difference_corrected_new = phi2_difference_corrected[50:150,50:150]
        phi_square = np.abs(evolved_mod[:, :, z_slice])**2
        phi_square_list.append(phi_square)
        
        pseudo_square = np.abs(mod[:, :, z_slice])**2
        pseudo_square_list.append(pseudo_square)
        
        for z_slice in range(24, 26):
            # Extract data for the current z slice
            phi2_difference = diff_phases[:,:, z_slice]
            phi2_difference_non_diff_phase = phases[:, :, z_slice]
            all_phi2_difference_non_diff_phases.append(phi2_difference_non_diff_phase)
            # phi2_difference_corrected = np.where(np.abs(phi2_difference) > 2 * np.pi, phi2_difference + 2*np.pi*np.sign(phi2_difference), phi2_difference)
            # phi2_difference_corrected = np.where(np.abs(phi2_difference_corrected) > np.pi, phi2_difference_corrected + np.pi*np.sign(phi2_difference_corrected), phi2_difference_corrected)
            mask_neg_2pi = np.isclose(phi2_difference, -2 * np.pi, atol=tol)
            mask_pos_2pi = np.isclose(phi2_difference, 2 * np.pi, atol=tol)

            phi2_difference[mask_neg_2pi] += 2 * np.pi
            phi2_difference[mask_pos_2pi] -= 2 * np.pi
            
            mask_neg_pi = np.isclose(phi2_difference, - np.pi, atol=tol)
            mask_pos_pi = np.isclose(phi2_difference,  np.pi, atol=tol)

            phi2_difference[mask_neg_pi] += np.pi
            phi2_difference[mask_pos_pi] -= np.pi
            
            phi2_difference_corrected = phi2_difference
            mo = mod[:, :, z_slice]
            evolved_mo = evolved_mod[:, :, z_slice]
            all_phi2_difference_list.append(phi2_difference_corrected)

            phi2_difference_corrected_new = phi2_difference_corrected[50:150,50:150]
            phi_square = np.abs(evolved_mod[:, :, z_slice])**2
            all_phi_square_list.append(phi_square)
            
            pseudo_square = np.abs(mod[:, :, z_slice])**2
            all_pseudo_square_list.append(pseudo_square)
            
            # Append the data to the accumulated lists

        
        # mod_of_mo = mod_of_mod[:, :, z_plane]
        # mo = mod[ :, :, z_plane]
        # evolved_mo = evolved_mod[ :, :, z_plane]
        # phase = phases[:,:,z_plane]
        # div1 = phi2_difference / evolved_mo
        # div2 = phi2_difference / mo
        # xy_plane_data 
        # - mod[:, :, z_plane] #was referece_data instead of mod
        # phi2_difference[phi2_difference < 0] = 0  # Optional: Set negative values to 0
        
        # Generate contour plot for the XY plane
        # plt.figure(figsize=(8, 8))
        # plt.contourf(evolved_mo, cmap='plasma', levels = 100)
        # plt.xlim(0,101)
        # plt.ylim(0,101)
        # # Customize plot labels, title, etc. as needed
        # # plt.xlabel('X Label')
        # # plt.ylabel('Y Label')
        # plt.title(f'Mod Contour Plot for frame {z_plane}')
        
        # # Save the frame
        # frame_path = os.path.join(frames_directory, f"test2_{z_plane}.png")
        # plt.savefig(frame_path, dpi=300)
        # plt.close()
        
        #  # Create a figure with three subplots
        # fig, axs = plt.subplots(1, 3, figsize=(30, 6), subplot_kw={'aspect': 'equal'})

        # # Plot mod on the left subplot
        # im1 = axs[0].contourf(mo, cmap='cool', levels=100)
        # axs[0].set_title(f'Pseudo Mod for frame {frame_number}', fontsize=16)

        # # Plot evolved_mod on the right subplot
        # im2 = axs[1].contourf(evolved_mo, cmap='cool', levels=100)
        # axs[1].set_title(f'Evolved Mod for frame {frame_number}', fontsize=16)

        # # Plot the difference on the third subplot
        # im3 = axs[2].contourf(phi2_difference_corrected_new, cmap='bwr', levels=np.linspace(-0.3, 0.3, 200), extent=[0, 100, 0, 100])
        # axs[2].set_title(f'Difference, frame {frame_number}', fontsize=16)

        # # Add colorbars
        # cbar1 = fig.colorbar(im1, ax=axs[0])
        # cbar2 = fig.colorbar(im2, ax=axs[1])
        # cbar3 = fig.colorbar(im3, ax=axs[2])

        # # Set common labels
        # for ax in axs:
        #     ax.set_xlabel('X Label')
        #     ax.set_ylabel('Y Label')

        # # Save the frame
        # time_step = frame_number
        # frame_path = os.path.join(frames_directory, f"mod_t{frame_number}_z12_corrected.png")
        # axs[2].set_ylim(0, 100)
        # axs[2].set_xlim(0, 100)
        # plt.savefig(frame_path, dpi=300)
        # plt.close()

        # print(f"Frame {frame_number}, Plane 12 generated: {frame_path}")
        print(f"{frame_number} Done")

        
        
    except Exception as e:
        # Handle any exceptions (file not found, empty file, etc.) and continue to the next iteration
        print(f"Error processing frame {frame_number}: {str(e)}")
        continue
for i in range(1, len(phi2_difference_list)):
    try:
        # Calculate the time derivative (phi_dot) and dt
        alpha_diff_phases_dot = (phi2_difference_list[i] - phi2_difference_list[i - 1]) / (dt )
        alpha_non_diff_phases_dot  = (phi2_difference_non_diff_phases[i] - phi2_difference_non_diff_phases[i-1]) / dt

        # Calculate the quantity |phi_dot|^2 * |phi|^2
        quantity_to_plot_1 = np.abs(alpha_diff_phases_dot)**2 * phi_square_list[i]
        quantity_to_plot_2 = np.abs(alpha_non_diff_phases_dot)**2 * phi_square_list[i]
        scale = colors.LogNorm(vmin = quantity_to_plot_1.min() , vmax = quantity_to_plot_1.max()  )
        
        sub_alpha = phi2_difference_list[i]
        # sub_alpha = sub_alpha[50:150,50:150]

        # Plotting code
                # Create a figure and a subplot
        # fig, axs2 = plt.subplots(figsize=(6, 4))

        # # Plot 3 - Centered in the middle row
        # im3 = axs2.contourf(sub_alpha, cmap='PuOr', levels=np.linspace(-0.3,0.3,200))
        # axs2.set_title(r'$\Delta \alpha $ ', fontsize=16)

        # axs2.set_aspect('equal')

        # # Add colorbar
        # cbar2 = fig.colorbar(im3, ax=axs2)

        # # Apply the scale to colorbars
        # mappable3 = im3
        # cbar2.mappable.set_norm(mappable3.norm)

        # fig.suptitle(f" Radiation field. Time Step : {(start_frame + i)}0", fontsize=12)
        # plt.text(0.95, 0.95, r'$\epsilon = 0.9$', horizontalalignment='right', verticalalignment='top', transform=fig.transFigure, fontsize=16)

        # # Save the frame
        # time_step = (start_frame + i)
        # frame_path = os.path.join(frames_directory, f"quad_lor_corr_5_plots_t{i}_z12.png")
        # plt.savefig(frame_path, dpi=300)
        # plt.close()

        fig, (axs0) = plt.subplots(1, 3, figsize=(6, 6))
        gs = gridspec.GridSpec(1, 1)
        # gs = gridspec.GridSpec(3, 3, height_ratios=[2, 2, 2], width_ratios=[2, 2, 2])
        
        # Plot 1
        # axs0 = plt.subplot(gs[0, 0])
        # im1 = axs0.contourf(quantity_to_plot_1, cmap='plasma', levels= np.linspace(0,0.003,200) )
        # axs0.set_title(r'$(\dot{\Delta \alpha})^2 |\phi|^2$ ', fontsize=16)

        # # Plot 2
        # axs1 = plt.subplot(gs[0, 1])
        # im2 = axs1.contourf(quantity_to_plot_2, cmap='plasma', levels= np.linspace(0,0.003,200) )
        # axs1.set_title(r'$(\dot{\alpha})^2 |\phi|^2$ ', fontsize=16)

        # # Plot 3 - Centered in the middle row
        axs0 = plt.subplot(gs[0, 0])
        im3 = axs0.contourf(sub_alpha, cmap='PuOr', levels=np.linspace(-0.025,0.025,200))
        axs0.set_title(r'$\Delta \alpha $ ', fontsize=16)

        # Plot 4
        # axs0 = plt.subplot(gs[0, 0])
        # im1 = axs0.contourf(phi_square_list[i], cmap='RdPu', levels=200)
        # axs0.set_title(r'Evolved $\phi$ ', fontsize=16)

        # # Plot 5
        # axs1 = plt.subplot(gs[0, 1])
        # im2 = axs1.contourf(pseudo_square_list[i], cmap='RdPu', levels=np.linspace(0, 1, 200))
        # axs1.set_title(r'Pseudo $\phi$ ', fontsize=16)
        
        for ax in [axs0]:
            ax.set_aspect('equal')

        divider = make_axes_locatable(axs0)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        # Add colorbars
        # cbar1 = fig.colorbar(im1, ax=axs0)
        cbar2 = fig.colorbar(im3, cax=cax)
        # cbar3 = fig.colorbar(im3, ax=axs2)
        # cbar4 = fig.colorbar(im4, ax=axs3)
        # cbar5 = fig.colorbar(im5, ax=axs4)

        # Apply the scale to colorbars
        # mappable1 = im1
        # mappable2 = im2
        mappable3 = im3

        # cbar1.mappable.set_norm(mappable1.norm)
        cbar2.mappable.set_norm(mappable3.norm)
        # cbar3.mappable.set_norm(mappable3.norm)

        fig.suptitle(f" Time Step : {(start_frame + i)}", fontsize=16)
        plt.text(0.95, 0.95, r'$\epsilon = 0.5$', horizontalalignment='right', verticalalignment='top', transform=fig.transFigure, fontsize=16)
        # Save the frame
        time_step = (start_frame + i)
        frame_path = os.path.join(frames_directory, f"eps_0.5_delta_alpha_attempts_t{start_frame + i}_z12.png")
        plt.savefig(frame_path, dpi=300)
        plt.close()
        
        # for k, (ax, quantity_to_plot, cmap, levels, title, colorbar_title) in enumerate([
        #     (axs0, quantity_to_plot_1, 'plasma', 200, r'$(\dot{\Delta \alpha})^2 |\phi|^2$', 'cbar1'),
        #     (axs1, quantity_to_plot_2, 'plasma', 200, r'$(\dot{\alpha})^2 |\phi|^2$', 'cbar2'),
        #     (axs2, sub_alpha, 'PuOr', np.linspace(-0.3,0.3,200), r'$\Delta \alpha $', 'cbar3'),
        #     (axs3, phi_square_list[i], 'RdPu', 200, r'Evolved $\phi$ ', 'cbar4'),
        #     (axs4, pseudo_square_list[i], 'RdPu', np.linspace(0, 1, 200), r'Secondary $\phi$ ', 'cbar5')
        # ]):
        #     fig, ax = plt.subplots(figsize=(6, 6))
        #     im = ax.contourf(quantity_to_plot, cmap=cmap, levels=levels)

        #     ax.set_title(title, fontsize=20)
        #     ax.tick_params(axis='both', which='major', labelsize=20)

        #     ax.set_xlabel('y axis', fontsize=15)  # Add x-axis label
        #     ax.set_ylabel('x axis', fontsize=15)  # Add y-axis label

        #     ax.set_aspect('equal')
            
        #     if i in [1, 2]:
        #         quantity_to_plot = np.ma.array(quantity_to_plot, mask=circular_mask)

        #     # Create a colorbar, except for the 4th plot
        #     if k not in [ 2,3,4]:
        #         cbar = fig.colorbar(im, ax=ax)
        #         # Define a function to format the colorbar labels
        #         def format_tick(x, pos):
        #             return f'{x:.3e}'
        #         # Apply the formatter to the colorbar
        #         formatter = FuncFormatter(format_tick)
        #         cbar.ax.yaxis.set_major_formatter(formatter)
        #         cbar.ax.tick_params(labelsize=15)
        #         # ax.set_title('')  # Remove title
        #         ax.set_xlabel('')  # Remove x-axis label
        #         ax.set_ylabel('')  # Remove y-axis label
        #     else:
        #         ax.set_title('')  # Remove title
        #         ax.set_xlabel('')  # Remove x-axis label
        #         ax.set_ylabel('')  # Remove y-axis label
        #         # ax.set_xticks([])  # Remove x-axis ticks
        #         # ax.set_yticks([])  # Remove y-axis ticks
        #     Nx = 201
        #     Ny = 201
        #     delta = 0.7
        #     # plot_power_spectrum(Nx, Ny, delta, quantity_to_plot_1)
        #     # if i in [1, 2]:
        #     #     ax.annotate("", xy=(100, 143.5), xytext=(100, 100), arrowprops=dict(arrowstyle="->"))
        #     #     ax.text(115, 120, "R", va='center', ha='right', backgroundcolor='w', fontsize=12)



        #     frame_path = os.path.join(frames_directory, f"alpha_16_5_plots_t{i}_z12_{colorbar_title}.png")
        #     plt.savefig(frame_path, dpi=300)
        #     plt.close()



        print(f"Frame {start_frame + i}")
        print("Done")


    except Exception as e:
        print(f"Error processing frame {start_frame + i}: {str(e)}")
        continue
    


# Create a figure and an axis for 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Iterate over each element in all_phi2_difference_list
for z_slice, phi2_difference_corrected in enumerate(all_phi2_difference_list):
    # Create X, Y grids
    Y, X = np.meshgrid(np.arange(phi2_difference_corrected.shape[1]), np.arange(phi2_difference_corrected.shape[0]))
    
    # Plot the 3D surface for the current z slice
    ax.plot_surface(X, Y, np.full_like(X, z_slice), facecolors=plt.cm.viridis(phi2_difference_corrected), rstride=1, cstride=1, shade=False)

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Surface Plot of Z Slices')

# Show the plot
plt.show()

# for i in range(1, len(phi2_difference_list)):
#     try:
#         # Calculate the time derivative (phi_dot) and dt
#         alpha_dot = (phi2_difference_list[i] - phi2_difference_list[i - 1]) / (dt)

#         # Calculate the quantity |phi_dot|^2 * |phi|^2
#         quantity_to_plot = phi_square_list[i]

#         # Additional quantities
#         quantity_1 = np.abs(alpha_dot)**2 * phi_square_list[i]
#         quantity_2 = np.abs(alpha_dot)**2 * phi_square_list[i - 1]
#         quantity_3 = phi2_difference_list[i]
#         quantity_4 = phi2_difference_list[i - 1]

#         # Plotting code
#         fig, axs = plt.subplots(3, 3, figsize=(18, 18), subplot_kw={'aspect': 'equal'})

#         # Plot phi_square_list[i]
#         im1 = axs[0, 0].contourf(quantity_to_plot, cmap='cool', levels = np.linspace(0,1,200) )
#         axs[0, 0].set_title(f'|phi|^2 - Frame {start_frame + i}', fontsize=16)

#         # Plot np.abs(alpha_dot)**2 * phi_square_list[i]
#         im2 = axs[0, 1].contourf(quantity_1, cmap='cool', levels=np.linspace(0,1e-5,200))
#         axs[0, 1].set_title(f'|alpha_dot|^2 * |phi|^2 - Frame {start_frame + i}', fontsize=16)

#         # Plot phi2_difference_list[i]
#         im3 = axs[1, 0].contourf(quantity_3, cmap='bwr', levels=200, extent=[0, 100, 0, 100])
#         axs[1, 0].set_title(f'Difference - Frame {start_frame + i}', fontsize=16)

#         # Plot phi2_difference_list[i - 1]
#         im4 = axs[1, 1].contourf(quantity_4, cmap='bwr', levels=200, extent=[0, 100, 0, 100])
#         axs[1, 1].set_title(f'Difference - Frame {start_frame + i - 1}', fontsize=16)

#         # Leave the center plot blank
#         # axs[1, 2].axis('off')

#         # Plot additional quantities in the bottom row
#         im5 = axs[2, 0].contourf(quantity_2, cmap='cool', levels=200)
#         axs[2, 0].set_title(f'|alpha_dot|^2 * |phi|^2 - Frame {start_frame + i - 1}', fontsize=16)

#         im6 = axs[2, 1].contourf(np.abs(alpha_dot)**2 * phi_square_list[i - 2], cmap='cool', levels=np.linspace(0, 1e-5, 200))
#         axs[2, 1].set_title(f'|alpha_dot|^2 * |phi|^2 - Frame {start_frame + i - 2}', fontsize=16)

#         im7 = axs[2, 2].contourf(phi2_difference_list[i - 2], cmap='bwr', levels=np.linspace(-0.03, 0.03, 200), extent=[0, 100, 0, 100])
#         axs[2, 2].set_title(f'Difference - Frame {start_frame + i - 2}', fontsize=16)

#         # Add colorbars
#         for ax, im in zip(axs.flat, [im1, im2, im3, im4, im5, im6, im7]):
#             cbar = fig.colorbar(im, ax=ax)
#             ax.set_xlabel('X Axis')
#             ax.set_ylabel('Y Axis')

#         # Save the frame
#         frame_path = os.path.join(frames_directory, f"combined_plots_t{i}.png")
#         plt.savefig(frame_path, dpi=300)
#         plt.close()

#         print(f"Combined plots for Frame {start_frame + i} generated: {frame_path}")
#         print("Done")

#     except Exception as e:
#         print(f"Error processing frame {start_frame + i}: {str(e)}")
#         continue
    # except Exception as e:
    #     # Handle any exceptions (file not found, empty file, etc.) and continue to the next iteration
    #     print(f"Error processing file {p}: {str(e)}")
    #     continue
# sampling_interval_x = 1
# sampling_interval_y = 1
# sampling_interval_z = 1

# # for j in range(201):  # Assuming you have 200 input files numbered from 0 to 200
# #     try:
# # input_file_path = os.path.join(input_directory, f"col_0_frame_{j}.npy")

# # Load data from the input file
# # array_3d_raw = np.load(input_file_path)
# # array_3d = array_3d_raw - reference_data


# # nx, ny, nz = array_3d.shape
# array_3d = evolved_mod 
# nx, ny, nz = array_3d.shape

# fig = plt.figure(figsize=(15, 15))
# ax = fig.add_subplot(111, projection='3d')

# # Create meshgrid for sampled data
# X, Y, Z = np.meshgrid(
#     np.arange(0, nx, sampling_interval_x),
#     np.arange(0, ny, sampling_interval_y),
#     np.arange(0, nz, sampling_interval_z)
# )

# # Plot contours
# for i in range(nz):
#     ax.contour(
#         X[:, :, i], Y[:, :, i], array_3d[:, :, i],
#         levels=[0.09], colors='r', zdir='z', offset=i, linewidths=1,
#     )

# for i in range(nx):
#     ax.contour(
#         X[i, :, :], Y[i, :, :], array_3d[i, :, :],
#         levels=[0.09], colors='r', zdir='y', offset=0, linewidths=1,
#     )

# ax.contour(
#     array_3d[:, -1, :], Y[:, -1, :], Z[:, -1, :],
#     levels=[0.09], colors='r', zdir='x', offset=nx, linewidths=1,
# )

# # Set limits of the plot from coord limits
# xmin, xmax = X.min(), X.max()
# ymin, ymax = Y.min(), Y.max()
# zmin, zmax = Z.min(), Z.max()
# ax.set(xlim=[xmin, xmax], ylim=[ymin, ymax], zlim=[zmin, zmax])

# # Plot edges
# edges_kw = dict(color='0.4', linewidth=0, zorder=1e3)
# ax.plot([xmax, xmax], [ymin, ymax], [zmin, zmin], **edges_kw)
# ax.plot([xmin, xmax], [ymin, ymin], [zmin, zmin], **edges_kw)
# ax.plot([xmax, xmax], [ymin, ymin], [zmin, zmax], **edges_kw)

# ax.set_xlabel("x")
# ax.set_ylabel("y")
# ax.set_zlabel("z")

# # Save the frame
# frame_path = os.path.join(frames_directory, f"frame_176.png")
# plt.savefig(frame_path, dpi=300)
# plt.close()

    #     print(f"Frame {j} generated: {frame_path}")

    # except Exception as e:
    #     # Handle any exceptions (file not found, empty file, etc.) and continue to the next iteration
    #     print(f"Error processing file {j}: {str(e)}")
    #     continue
