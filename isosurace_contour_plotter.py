

# import numpy as np

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import os


# # Path to the input file
# input_file_path = "/Users/pranavbharadwajgangrekalvemanoj/Desktop/Axions_Project/sem_2_xy_abs_z_fix_101/npy_files_full_positive_boost_ev_1/"

# # # Path to the directory where plots will be saved
# output_directory = "/Users/pranavbharadwajgangrekalvemanoj/Desktop/Axions_Project/"

# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import os
# tol = 1e-8
# # Load data from .npy files
# # Load data for time step 2758
# data_real_t1 = np.load(os.path.join(input_file_path, 'col_0_frame_2758.npy'))
# data_im_t1 = np.load(os.path.join(input_file_path, 'col_1_frame_2758.npy'))

# # Create complex number for time step 2758
# complex_number_t1 = data_real_t1 + 1j * data_im_t1

# # Calculate phase for time step 2758
# phi_phase_t1 = np.angle(complex_number_t1)

# # Load data for time step 2759
# data_real_t2 = np.load(os.path.join(input_file_path, 'frame_2759_positive_boost_0.npy'))
# data_im_t2 = np.load(os.path.join(input_file_path, 'frame_2759_positive_boost_1.npy'))

# # Create complex number for time step 2759
# complex_number_t2 = data_real_t2 + 1j * data_im_t2

# # Calculate phase for time step 2759
# phi_phase_t2 = np.angle(complex_number_t2)

# # Calculate phase difference
# phi_phase_diff = phi_phase_t2 - phi_phase_t1

# # Correct the phase difference data
# for z_plane in range(phi_phase_diff.shape[2]):
#     data_2d1 = phi_phase_diff[:, :, z_plane]

#     mask_neg_2pi_1 = np.isclose(data_2d1, -2 * np.pi, atol=tol)
#     mask_pos_2pi_1 = np.isclose(data_2d1, 2 * np.pi, atol=tol)

#     data_2d1[mask_neg_2pi_1] += 2 * np.pi
#     data_2d1[mask_pos_2pi_1] -= 2 * np.pi

#     mask_neg_2pi_1_ = np.isclose(data_2d1, - np.pi, atol=tol)
#     mask_pos_2pi_1_ = np.isclose(data_2d1,  np.pi, atol=tol)

#     data_2d1[mask_neg_2pi_1_] +=  np.pi
#     data_2d1[mask_pos_2pi_1_] -= np.pi

#     phi_phase_diff[:, :, z_plane] = data_2d1

# nx = 101

# ny = 101
# nz = 101
# # Plot the corrected phase data
# # Define the sampling indices for each axis
# sampling_interval_x = 2  # Sample every 2nd point along X axis
# sampling_interval_y = 2  # Sample every 2nd point along Y axis
# sampling_interval_z = 2  # Sample every point along Z axis

# # Sample the 3D array
# sampled_phi_phase_t = phi_phase_diff[::sampling_interval_x, ::sampling_interval_y, ::sampling_interval_z]

# # Mask the 0th z slice
# sampled_phi_phase_t[:, :, 0] = np.nan

# # Create meshgrid for sampled data
# # X, Y, Z = np.meshgrid(
# #     np.arange(0, nx, sampling_interval_x)[:sampled_phi_phase_t.shape[0]],
# #     np.arange(0, ny, sampling_interval_y)[:sampled_phi_phase_t.shape[1]],
# #     np.arange(0, nz, sampling_interval_z)[:sampled_phi_phase_t.shape[2]]
# # )

# # Plot the sampled 3D data using a scatter plot
# # sampled_phi_phase_t[:75, :, :] = np.nan
# # sampled_phi_phase_t[125:, :, :] = np.nan
# # sampled_phi_phase_t[:, :75, :] = np.nan
# # sampled_phi_phase_t[:, 125:, :] = np.nan

# # Plot the sampled 3D data using a scatter plot
# # Plot the sampled 3D data using a scatter plot
# # Create meshgrid for sampled data
# # -----------------------------------------isosurfaces---------------------------------------------

# X, Y, Z = np.meshgrid(
#     np.arange(0, nx, sampling_interval_x)[:sampled_phi_phase_t.shape[0]],
#     np.arange(0, ny, sampling_interval_y)[:sampled_phi_phase_t.shape[1]],
#     np.arange(0, nz, sampling_interval_z)[:sampled_phi_phase_t.shape[2]]
# )

# # Flatten the arrays for scatter plot
# X_flat = X.flatten()
# Y_flat = Y.flatten()
# Z_flat = Z.flatten()
# C_flat = sampled_phi_phase_t.flatten()

# # Create a 3D figure
# fig = plt.figure(figsize=(15,15))
# ax = fig.add_subplot(111, projection='3d')

# # Define levels for contour plot
# levels = np.linspace(0.002,0.003, 100)

# # Create contour plot (isosurfaces)
# # Create contour plot (isosurfaces)
# for i in range(sampled_phi_phase_t.shape[2]-1):
#     ax.contour(X[:, :, i], Y[:, :, i], sampled_phi_phase_t[:, :, i],
#                levels=levels, colors='k', zdir='z', offset=0, linewidths=0.2)

# for i in range(sampled_phi_phase_t.shape[0]-1):
#     ax.contour(X[i, :, :], Y[i, :, :], sampled_phi_phase_t[i, :, :],
#                levels=levels, colors='k', zdir='y', offset=0, linewidths=0.2)

# ax.contour(sampled_phi_phase_t[:, -1, :], Y[:, -1, :], Z[:, -1, :],
#            levels=levels, colors='k', zdir='x', offset=0, linewidths=0.2)

# # Set limits of the plot from coord limits
# xmin, xmax = X.min(), X.max()
# ymin, ymax = Y.min(), Y.max()
# zmin, zmax = Z.min(), Z.max()
# ax.set(xlim=[xmin, xmax], ylim=[ymin, ymax], zlim=[zmin, zmax])

# # Plot edges
# edges_kw = dict(color='0.4', linewidth=0, zorder=1e3)
# ax.plot([xmax, xmax], [ymin, ymax], 0, **edges_kw)
# ax.plot([xmin, xmax], [ymin, ymin], 0, **edges_kw)
# ax.plot([xmax, xmax], [ymin, ymin], [zmin, zmax], **edges_kw)

# ax.set_xlabel("x")
# ax.set_ylabel("y")
# ax.set_zlabel("z")

# plt.show()# Create contour plot (isosurfaces)
# for i in range(nz-1):
#     ax.contour(X[:, :, i], Y[:, :, i], sampled_phi_phase_t[:, :, i],
#                levels=levels, colors='k', zdir='z', offset=-i, linewidths=0.2)

# for i in range(nx-1):
#     ax.contour(X[i, :, :], Y[i, :, :], sampled_phi_phase_t[i, :, :],
#                levels=levels, colors='k', zdir='y', offset=0, linewidths=0.2)

# ax.contour(sampled_phi_phase_t[:, -1, :], Y[:, -1, :], Z[:, -1, :],
#            levels=levels, colors='k', zdir='x', offset=nx, linewidths=0.2)
# if not os.path.exists(output_directory):
#     os.makedirs(output_directory)


# array_1d_real, array_1d_imag = np.loadtxt(input_file_path, unpack=True)
# print("data loaded")
# # Reshape the 1D array into a 3D array

# phi_real_3d = array_1d_real.reshape(nx, ny, nz)
# phi_imag_3d = array_1d_imag.reshape(nx, ny, nz)
# phi = np.sqrt(phi_real_3d**2 + phi_imag_3d**2)
# print("reshape successful")
# array_3d = np.zeros((nx,ny,nz))
# for i in range(0, len(array_1d)):
#     a = i//(nz*ny)
#     b = (i-nz*ny*a)//nz
#     c = ((i-nz*ny*a)-nz*b)
#     array_3d[a,b,c] = array_1d[i]



# X,Y,Z = np.meshgrid(array_3d[0],array_3d[1],array_3d[2])
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(X, Y, Z, cmap='viridis')
# plt.show()
# plt.savefig("test_mexhgrid_plot.png")


# Define the sampling indices for each axis
# sampling_interval_x = 2  # Sample every 10th point along X axis
# sampling_interval_y =  2 # Sample every 10th point along Y axis
# sampling_interval_z = 2  # Sample every 10th point along Z axis

# # Reshape the 1D array into a 3D array


# # Sample the 3D array
# # sampled_array_3d = phi[::sampling_interval_x, ::sampling_interval_y, ::sampling_interval_z]

# # # Create meshgrid for sampled data
# X, Y, Z = np.meshgrid(
#     np.arange(0, nx, sampling_interval_x),
#     np.arange(0, ny, sampling_interval_y),
#     np.arange(0, nz, sampling_interval_z)
# )

# # Plot the sampled 3D data using a scatter plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(X, Y, Z, c=sampled_phi_phase_t.flatten(), cmap="PuRd")
# plt.show()
# plt.savefig(os.path.join(output_directory, "sampled_3d_plot_5sampling_rdbu_allz_0.2.png"))

# Define the threshold value below which points will be considered transparent
# threshold_value = 0.1  # Adjust this value based on your data

# # Define the threshold value below which points will be considered transparent
# # Adjust this value based on your data


# # Create a mask to identify points above the threshold
# mask = sampled_array_3d >= threshold_value

# # Plot the sampled 3D data using a scatter plot with transparency
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Plot points above the threshold with full opacity
# ax.scatter(X[mask], Y[mask], Z[mask], c=sampled_array_3d[mask].flatten(), cmap="twilight", alpha=1)

# # Plot points below the threshold with transparency (alpha=0)
# ax.scatter(X[~mask], Y[~mask], Z[~mask], alpha=0)

# plt.show()
# plt.savefig(os.path.join(output_directory, "test_plot_0.png"))

# num_frames = nz

# for frame in range(num_frames):
#     plt.figure()
#     z_slice = phi[:, :, frame]
#     mask_slice = z_slice >= threshold_value
#     plt.imshow(mask_slice, cmap="twilight", alpha=1, extent=[0, nx, 0, ny])
#     plt.imshow(~mask_slice, cmap="twilight", alpha=0, extent=[0, nx, 0, ny])
#     plt.xlim(0, nx)
#     plt.ylim(0, ny)
#     plt.xlabel('X')
#     plt.ylabel('Y')
#     plt.title(f'2D Grid at Z = {frame} (Threshold: 0.1)')
#     plt.savefig(os.path.join(output_directory, f"frame_{frame:03d}.png"))
#     plt.close()

# print("Frames saved as images.")

# After running this code, you can use ImageMagick to convert the frames to a GIF:
# Open your terminal and navigate to the output directory containing the frames.
# Run the following command in the terminal:
# convert -delay 10 -loop 0 frame_*.png output.gif

# ---------------------------------------fresh attempt at isosurfaces using plot_surface------------------------------
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# Path to the input file
input_file_path = "/Users/pranavbharadwajgangrekalvemanoj/Desktop/Axions_Project/sem_2_xy_abs_z_fix_101/npy_files_full_positive_boost_ev_1/"

# Path to the directory where plots will be saved
output_directory = "/Users/pranavbharadwajgangrekalvemanoj/Desktop/Axions_Project/"

tol = 1e-8

# Load data from .npy files
# Load data for time step 2758
data_real_t1 = np.load(os.path.join(input_file_path, 'col_0_frame_2758.npy'))
data_im_t1 = np.load(os.path.join(input_file_path, 'col_1_frame_2758.npy'))

# Create complex number for time step 2758
complex_number_t1 = data_real_t1 + 1j * data_im_t1

# Calculate phase for time step 2758
phi_phase_t1 = np.angle(complex_number_t1)

# Load data for time step 2759
data_real_t2 = np.load(os.path.join(input_file_path, 'frame_2758_positive_boost_0.npy'))
data_im_t2 = np.load(os.path.join(input_file_path, 'frame_2758_positive_boost_1.npy'))

# Create complex number for time step 2759
complex_number_t2 = data_real_t2 + 1j * data_im_t2

# Calculate phase for time step 2759
phi_phase_t2 = np.angle(complex_number_t2)

# Calculate phase difference
phi_phase_diff = phi_phase_t2 - phi_phase_t1

# Correct the phase difference data
for z_plane in range(phi_phase_diff.shape[2]):
    data_2d1 = phi_phase_diff[:, :, z_plane]

    mask_neg_2pi_1 = np.isclose(data_2d1, -2 * np.pi, atol=tol)
    mask_pos_2pi_1 = np.isclose(data_2d1, 2 * np.pi, atol=tol)

    data_2d1[mask_neg_2pi_1] += 2 * np.pi
    data_2d1[mask_pos_2pi_1] -= 2 * np.pi

    mask_neg_2pi_1_ = np.isclose(data_2d1, - np.pi, atol=tol)
    mask_pos_2pi_1_ = np.isclose(data_2d1,  np.pi, atol=tol)

    data_2d1[mask_neg_2pi_1_] +=  np.pi
    data_2d1[mask_pos_2pi_1_] -= np.pi

    phi_phase_diff[:, :, z_plane] = data_2d1

nx = 101
ny = 101
nz = 101

# Define the sampling indices for each axis
sampling_interval_x = 1  # Sample every 2nd point along X axis
sampling_interval_y = 1  # Sample every 2nd point along Y axis
sampling_interval_z = 1  # Sample every 2nd point along Z axis

# Sample the 3D array
sampled_phi_phase_diff = phi_phase_diff[::sampling_interval_x, ::sampling_interval_y, ::sampling_interval_z]

# Create meshgrid for sampled data
X, Y, Z = np.meshgrid(
    np.arange(0, nx, sampling_interval_x)[:sampled_phi_phase_diff.shape[0]],
    np.arange(0, ny, sampling_interval_y)[:sampled_phi_phase_diff.shape[1]],
    np.arange(0, nz, sampling_interval_z)[:sampled_phi_phase_diff.shape[2]],
    indexing='ij'  # Use indexing='ij'
)
val = np.linspace(-0.0015,0.0015,51)
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(111, projection='3d')
levels = np.linspace(-0.0015, 0.0015, 21)[::20]  # Select every second value

# Plot contours
for p in range(len(levels)):
    val = levels[p]  # Get the contour level

    if val < 0:
        color = 'orange'
    else:
        color = 'purple'

    for i in range(sampled_phi_phase_diff.shape[2]):
        z_coord = Z[0, 0, i * sampling_interval_z]  # Get the actual z-coordinate
        ax.contour(X[:, :, i], Y[:, :, i], sampled_phi_phase_diff[:, :, i],
                   levels=[val], colors=color, zdir='z', offset=z_coord, linewidths=0.2)

    for i in range(sampled_phi_phase_diff.shape[0]):
        ax.contour(X[i, :, :], Y[i, :, :], sampled_phi_phase_diff[i, :, :],
                   levels=[val], colors=color, zdir='y', offset=0, linewidths=0.2)

    ax.contour(sampled_phi_phase_diff[:, -1, :], Y[:, -1, :], Z[:, -1, :],
               levels=[val], colors=color, zdir='x', offset=nx-1, linewidths=0.2)

# Set limits of the plot from coord limits
xmin, xmax = X.min(), X.max()
ymin, ymax = Y.min(), Y.max()
zmin, zmax = Z.min(), Z.max()
ax.set(xlim=[xmin, xmax], ylim=[ymin, ymax], zlim=[zmin, zmax])

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.title("Contour Plot of Phi Phase Differences")

# Save or show the plot
plt.savefig("contour_plot_phi_phase_diff.jpg")  # Save the plot
plt.show()  # Show the plot
plt.close(fig)  # Close the figure
    
