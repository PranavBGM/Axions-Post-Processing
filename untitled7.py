

# # import numpy as np

# # import matplotlib.pyplot as plt
# # from mpl_toolkits.mplot3d import Axes3D
# # import os


# # # Path to the input file
# # input_file_path = "/Users/pranavbharadwajgangrekalvemanoj/Desktop/Axions_Project/sem_2_xy_abs_z_fix_101/npy_files_full_positive_boost_ev_1/"

# # # # Path to the directory where plots will be saved
# # output_directory = "/Users/pranavbharadwajgangrekalvemanoj/Desktop/Axions_Project/"

# # import numpy as np
# # import matplotlib.pyplot as plt
# # from mpl_toolkits.mplot3d import Axes3D
# # import os
# # tol = 1e-8
# # # Load data from .npy files
# # # Load data for time step 2758
# # data_real_t1 = np.load(os.path.join(input_file_path, 'col_0_frame_2758.npy'))
# # data_im_t1 = np.load(os.path.join(input_file_path, 'col_1_frame_2758.npy'))

# # # Create complex number for time step 2758
# # complex_number_t1 = data_real_t1 + 1j * data_im_t1

# # # Calculate phase for time step 2758
# # phi_phase_t1 = np.angle(complex_number_t1)

# # # Load data for time step 2759
# # data_real_t2 = np.load(os.path.join(input_file_path, 'frame_2759_positive_boost_0.npy'))
# # data_im_t2 = np.load(os.path.join(input_file_path, 'frame_2759_positive_boost_1.npy'))

# # # Create complex number for time step 2759
# # complex_number_t2 = data_real_t2 + 1j * data_im_t2

# # # Calculate phase for time step 2759
# # phi_phase_t2 = np.angle(complex_number_t2)

# # # Calculate phase difference
# # phi_phase_diff = phi_phase_t2 - phi_phase_t1

# # # Correct the phase difference data
# # for z_plane in range(phi_phase_diff.shape[2]):
# #     data_2d1 = phi_phase_diff[:, :, z_plane]

# #     mask_neg_2pi_1 = np.isclose(data_2d1, -2 * np.pi, atol=tol)
# #     mask_pos_2pi_1 = np.isclose(data_2d1, 2 * np.pi, atol=tol)

# #     data_2d1[mask_neg_2pi_1] += 2 * np.pi
# #     data_2d1[mask_pos_2pi_1] -= 2 * np.pi

# #     mask_neg_2pi_1_ = np.isclose(data_2d1, - np.pi, atol=tol)
# #     mask_pos_2pi_1_ = np.isclose(data_2d1,  np.pi, atol=tol)

# #     data_2d1[mask_neg_2pi_1_] +=  np.pi
# #     data_2d1[mask_pos_2pi_1_] -= np.pi

# #     phi_phase_diff[:, :, z_plane] = data_2d1

# # nx = 101

# # ny = 101
# # nz = 101
# # # Plot the corrected phase data
# # # Define the sampling indices for each axis
# # sampling_interval_x = 2  # Sample every 2nd point along X axis
# # sampling_interval_y = 2  # Sample every 2nd point along Y axis
# # sampling_interval_z = 2  # Sample every point along Z axis

# # # Sample the 3D array
# # sampled_phi_phase_t = phi_phase_diff[::sampling_interval_x, ::sampling_interval_y, ::sampling_interval_z]

# # # Mask the 0th z slice
# # sampled_phi_phase_t[:, :, 0] = np.nan

# # # Create meshgrid for sampled data
# # # X, Y, Z = np.meshgrid(
# # #     np.arange(0, nx, sampling_interval_x)[:sampled_phi_phase_t.shape[0]],
# # #     np.arange(0, ny, sampling_interval_y)[:sampled_phi_phase_t.shape[1]],
# # #     np.arange(0, nz, sampling_interval_z)[:sampled_phi_phase_t.shape[2]]
# # # )

# # # Plot the sampled 3D data using a scatter plot
# # # sampled_phi_phase_t[:75, :, :] = np.nan
# # # sampled_phi_phase_t[125:, :, :] = np.nan
# # # sampled_phi_phase_t[:, :75, :] = np.nan
# # # sampled_phi_phase_t[:, 125:, :] = np.nan

# # # Plot the sampled 3D data using a scatter plot
# # # Plot the sampled 3D data using a scatter plot
# # # Create meshgrid for sampled data
# # # -----------------------------------------isosurfaces---------------------------------------------

# # X, Y, Z = np.meshgrid(
# #     np.arange(0, nx, sampling_interval_x)[:sampled_phi_phase_t.shape[0]],
# #     np.arange(0, ny, sampling_interval_y)[:sampled_phi_phase_t.shape[1]],
# #     np.arange(0, nz, sampling_interval_z)[:sampled_phi_phase_t.shape[2]]
# # )

# # # Flatten the arrays for scatter plot
# # X_flat = X.flatten()
# # Y_flat = Y.flatten()
# # Z_flat = Z.flatten()
# # C_flat = sampled_phi_phase_t.flatten()

# # # Create a 3D figure
# # fig = plt.figure(figsize=(15,15))
# # ax = fig.add_subplot(111, projection='3d')

# # # Define levels for contour plot
# # levels = np.linspace(0.002,0.003, 100)

# # # Create contour plot (isosurfaces)
# # # Create contour plot (isosurfaces)
# # for i in range(sampled_phi_phase_t.shape[2]-1):
# #     ax.contour(X[:, :, i], Y[:, :, i], sampled_phi_phase_t[:, :, i],
# #                levels=levels, colors='k', zdir='z', offset=0, linewidths=0.2)

# # for i in range(sampled_phi_phase_t.shape[0]-1):
# #     ax.contour(X[i, :, :], Y[i, :, :], sampled_phi_phase_t[i, :, :],
# #                levels=levels, colors='k', zdir='y', offset=0, linewidths=0.2)

# # ax.contour(sampled_phi_phase_t[:, -1, :], Y[:, -1, :], Z[:, -1, :],
# #            levels=levels, colors='k', zdir='x', offset=0, linewidths=0.2)

# # # Set limits of the plot from coord limits
# # xmin, xmax = X.min(), X.max()
# # ymin, ymax = Y.min(), Y.max()
# # zmin, zmax = Z.min(), Z.max()
# # ax.set(xlim=[xmin, xmax], ylim=[ymin, ymax], zlim=[zmin, zmax])

# # # Plot edges
# # edges_kw = dict(color='0.4', linewidth=0, zorder=1e3)
# # ax.plot([xmax, xmax], [ymin, ymax], 0, **edges_kw)
# # ax.plot([xmin, xmax], [ymin, ymin], 0, **edges_kw)
# # ax.plot([xmax, xmax], [ymin, ymin], [zmin, zmax], **edges_kw)

# # ax.set_xlabel("x")
# # ax.set_ylabel("y")
# # ax.set_zlabel("z")

# # plt.show()# Create contour plot (isosurfaces)
# # for i in range(nz-1):
# #     ax.contour(X[:, :, i], Y[:, :, i], sampled_phi_phase_t[:, :, i],
# #                levels=levels, colors='k', zdir='z', offset=-i, linewidths=0.2)

# # for i in range(nx-1):
# #     ax.contour(X[i, :, :], Y[i, :, :], sampled_phi_phase_t[i, :, :],
# #                levels=levels, colors='k', zdir='y', offset=0, linewidths=0.2)

# # ax.contour(sampled_phi_phase_t[:, -1, :], Y[:, -1, :], Z[:, -1, :],
# #            levels=levels, colors='k', zdir='x', offset=nx, linewidths=0.2)
# # if not os.path.exists(output_directory):
# #     os.makedirs(output_directory)


# # array_1d_real, array_1d_imag = np.loadtxt(input_file_path, unpack=True)
# # print("data loaded")
# # # Reshape the 1D array into a 3D array

# # phi_real_3d = array_1d_real.reshape(nx, ny, nz)
# # phi_imag_3d = array_1d_imag.reshape(nx, ny, nz)
# # phi = np.sqrt(phi_real_3d**2 + phi_imag_3d**2)
# # print("reshape successful")
# # array_3d = np.zeros((nx,ny,nz))
# # for i in range(0, len(array_1d)):
# #     a = i//(nz*ny)
# #     b = (i-nz*ny*a)//nz
# #     c = ((i-nz*ny*a)-nz*b)
# #     array_3d[a,b,c] = array_1d[i]



# # X,Y,Z = np.meshgrid(array_3d[0],array_3d[1],array_3d[2])
# # fig = plt.figure()
# # ax = fig.add_subplot(111, projection='3d')
# # ax.scatter(X, Y, Z, cmap='viridis')
# # plt.show()
# # plt.savefig("test_mexhgrid_plot.png")


# # Define the sampling indices for each axis
# # sampling_interval_x = 2  # Sample every 10th point along X axis
# # sampling_interval_y =  2 # Sample every 10th point along Y axis
# # sampling_interval_z = 2  # Sample every 10th point along Z axis

# # # Reshape the 1D array into a 3D array


# # # Sample the 3D array
# # # sampled_array_3d = phi[::sampling_interval_x, ::sampling_interval_y, ::sampling_interval_z]

# # # # Create meshgrid for sampled data
# # X, Y, Z = np.meshgrid(
# #     np.arange(0, nx, sampling_interval_x),
# #     np.arange(0, ny, sampling_interval_y),
# #     np.arange(0, nz, sampling_interval_z)
# # )

# # # Plot the sampled 3D data using a scatter plot
# # fig = plt.figure()
# # ax = fig.add_subplot(111, projection='3d')
# # ax.scatter(X, Y, Z, c=sampled_phi_phase_t.flatten(), cmap="PuRd")
# # plt.show()
# # plt.savefig(os.path.join(output_directory, "sampled_3d_plot_5sampling_rdbu_allz_0.2.png"))

# # Define the threshold value below which points will be considered transparent
# # threshold_value = 0.1  # Adjust this value based on your data

# # # Define the threshold value below which points will be considered transparent
# # # Adjust this value based on your data


# # # Create a mask to identify points above the threshold
# # mask = sampled_array_3d >= threshold_value

# # # Plot the sampled 3D data using a scatter plot with transparency
# # fig = plt.figure()
# # ax = fig.add_subplot(111, projection='3d')

# # # Plot points above the threshold with full opacity
# # ax.scatter(X[mask], Y[mask], Z[mask], c=sampled_array_3d[mask].flatten(), cmap="twilight", alpha=1)

# # # Plot points below the threshold with transparency (alpha=0)
# # ax.scatter(X[~mask], Y[~mask], Z[~mask], alpha=0)

# # plt.show()
# # plt.savefig(os.path.join(output_directory, "test_plot_0.png"))

# # num_frames = nz

# # for frame in range(num_frames):
# #     plt.figure()
# #     z_slice = phi[:, :, frame]
# #     mask_slice = z_slice >= threshold_value
# #     plt.imshow(mask_slice, cmap="twilight", alpha=1, extent=[0, nx, 0, ny])
# #     plt.imshow(~mask_slice, cmap="twilight", alpha=0, extent=[0, nx, 0, ny])
# #     plt.xlim(0, nx)
# #     plt.ylim(0, ny)
# #     plt.xlabel('X')
# #     plt.ylabel('Y')
# #     plt.title(f'2D Grid at Z = {frame} (Threshold: 0.1)')
# #     plt.savefig(os.path.join(output_directory, f"frame_{frame:03d}.png"))
# #     plt.close()

# # print("Frames saved as images.")

# # After running this code, you can use ImageMagick to convert the frames to a GIF:
# # Open your terminal and navigate to the output directory containing the frames.
# # Run the following command in the terminal:
# # convert -delay 10 -loop 0 frame_*.png output.gif

# # ---------------------------------------fresh attempt at isosurfaces using plot_surface------------------------------
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import os

# # Path to the input file
# input_file_path = "/Users/pranavbharadwajgangrekalvemanoj/Desktop/Axions_Project/to_scp_270224_401/npy_files_full_positive_boost_ev_1/"

# # Path to the directory where plots will be saved
# output_directory = "/Users/pranavbharadwajgangrekalvemanoj/Desktop/Axions_Project/"

# tol = 1e-8

# # Load data from .npy files
# # Load data for time step 2758
# data_real_t1 = np.load(os.path.join(input_file_path, 'col_0_frame_2751.npy'))
# data_im_t1 = np.load(os.path.join(input_file_path, 'col_1_frame_2751.npy'))

# # Create complex number for time step 2758
# complex_number_t1 = data_real_t1 + 1j * data_im_t1

# # Calculate phase for time step 2758
# phi_phase_t1 = np.angle(complex_number_t1)

# # Load data for time step 2759
# data_real_t2 = np.load(os.path.join(input_file_path, 'frame_2751_positive_boost_0.npy'))
# data_im_t2 = np.load(os.path.join(input_file_path, 'frame_2751_positive_boost_1.npy'))

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

# nx = 401
# ny = 401
# nz = 401

# # Define the sampling indices for each axis
# sampling_interval_x = 4  # Sample every 2nd point along X axis
# sampling_interval_y = 4  # Sample every 2nd point along Y axis
# sampling_interval_z = 4  # Sample every 2nd point along Z axis

# # Sample the 3D array
# sampled_phi_phase_diff = phi_phase_diff[::sampling_interval_x, ::sampling_interval_y, ::sampling_interval_z]

# # Create meshgrid for sampled data
# X, Y, Z = np.meshgrid(
#     np.arange(0, nx, sampling_interval_x)[:sampled_phi_phase_diff.shape[0]],
#     np.arange(0, ny, sampling_interval_y)[:sampled_phi_phase_diff.shape[1]],
#     np.arange(0, nz, sampling_interval_z)[:sampled_phi_phase_diff.shape[2]],
#     indexing='ij'  # Use indexing='ij'
# )
# val = np.linspace(-0.0015,0.0015,51)
# fig = plt.figure(figsize=(15, 15))
# ax = fig.add_subplot(111, projection='3d')
# levels = np.linspace(-0.0015, 0.0015, 21)[::20]  # Select every second value

# # Plot contours
# for p in range(len(levels)):
#     val = levels[p]  # Get the contour level

#     if val < 0:
#         color = 'orange'
#     else:
#         color = 'purple'

#     for i in range(sampled_phi_phase_diff.shape[2]):
#         z_coord = Z[0, 0, i]  # Get the actual z-coordinate
#         ax.contour(X[:, :, i], Y[:, :, i], sampled_phi_phase_diff[:, :, i],
#                    levels=[val], colors=color, zdir='z', offset=z_coord, linewidths=0.2)

#     for i in range(sampled_phi_phase_diff.shape[0]):
#         ax.contour(X[i, :, :], Y[i, :, :], sampled_phi_phase_diff[i, :, :],
#                    levels=[val], colors=color, zdir='y', offset=0, linewidths=0.2)

#     ax.contour(sampled_phi_phase_diff[:, -1, :], Y[:, -1, :], Z[:, -1, :],
#                levels=[val], colors=color, zdir='x', offset=nx-1, linewidths=0.2)

# # Set limits of the plot from coord limits
# xmin, xmax = X.min(), X.max()
# ymin, ymax = Y.min(), Y.max()
# zmin, zmax = Z.min(), Z.max()
# ax.set(xlim=[xmin, xmax], ylim=[ymin, ymax], zlim=[zmin, zmax])

# ax.set_xlabel("x axis")
# ax.set_ylabel("y axis")
# ax.set_zlabel("z axis")
# plt.title(r"Contours of $\Delta \alpha$ ")

# # Save or show the plot
# # plt.savefig("contour_plot_phi_phase_diff.jpg")  # Save the plot
# plt.show()  # Show the plot
# # plt.close(fig)  # Close the figure
    
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import matplotlib
os.environ['PATH'] = '/usr/local/texlive/2023/bin:' + os.environ['PATH']
# matplotlib.use("pgf")

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'text.usetex': True,
    'pgf.texsystem': 'pdflatex',
    'pgf.rcfonts': False,
})
plt.style.use('dark_background')

def apply_circular_mask(data_3d, radius):
    Nx, Ny, Nz = data_3d.shape
    masked_slices = []  # List to store masked 2D slices
    masked_data_3d = np.copy(data_3d)
    for z_plane in range(Nz):
        y, x = np.ogrid[-100:101, -100:101]
        circular_mask = x**2 + y**2 < radius**2
        data_2d = masked_data_3d[:, :, z_plane]
        masked_slice = np.where(circular_mask, 0, data_2d)
        masked_slices.append(masked_slice)  # Append masked 2D slice to the list
    # Convert the list of masked slices to a 3D NumPy array
    masked_data_3d = np.stack(masked_slices, axis=2)
    return masked_data_3d

# Path to the input file
input_file_path = "/Users/pranavbharadwajgangrekalvemanoj/Desktop/Axions_Project/to_scp_280224_201/npy_files_full_positive_boost_ev_1/"
input_ref_file_path = "/Users/pranavbharadwajgangrekalvemanoj/Desktop/Axions_Project/to_scp_290224_pseudo_201/npy_files_full_positive_boost_ev_1/"
tol = 1e-8
# Load data from .npy files

data_real_t1 = np.load(os.path.join(input_file_path, 'col_0_frame_145.npy'))
data_im_t1 = np.load(os.path.join(input_file_path, 'col_1_frame_145.npy'))


complex_number_t1 = data_real_t1 + 1j * data_im_t1
mod_t1 = np.abs((data_real_t1)**2 + (data_im_t1)**2)


phi_phase_t1 = np.angle(complex_number_t1)

data_real_t1_ref = np.load(os.path.join(input_ref_file_path, 'frame_145_positive_boost_0.npy'))
data_im_t1_ref = np.load(os.path.join(input_ref_file_path, 'frame_145_positive_boost_1.npy'))


complex_number_t1_ref = data_real_t1_ref + 1j * data_im_t1_ref


phi_phase_t1_ref = np.angle(complex_number_t1_ref)

# Calculate phase difference
phi_phase_diff_t1 = phi_phase_t1 - phi_phase_t1_ref

data_real_t2 = np.load(os.path.join(input_file_path, 'col_0_frame_146.npy'))
data_im_t2 = np.load(os.path.join(input_file_path, 'col_1_frame_146.npy'))

complex_number_t2 = data_real_t2 + 1j * data_im_t2
mod_t2 = np.abs((data_real_t2)**2 + (data_im_t2)**2)

phi_phase_t2 = np.angle(complex_number_t2)

data_real_t2_ref = np.load(os.path.join(input_ref_file_path, 'frame_146_positive_boost_0.npy'))
data_im_t2_ref = np.load(os.path.join(input_ref_file_path, 'frame_146_positive_boost_1.npy'))

# data_real_t2_ref = np.load(os.path.join(input_ref_file_path, 'frame_1352_positive_boost_0.npy'))
# data_im_t2_ref = np.load(os.path.join(input_ref_file_path, 'frame_1352_positive_boost_1.npy'))

complex_number_t2_ref = data_real_t2_ref + 1j * data_im_t2_ref
phi_phase_t2_ref = np.angle(complex_number_t2_ref)

# Calculate phase difference
phi_phase_diff_t2 = phi_phase_t2 - phi_phase_t2_ref


# Correct the phase difference data
for z_plane in range(phi_phase_diff_t1.shape[2]):
    data_2d1 = phi_phase_diff_t1[:, :, z_plane]

    mask_neg_2pi_1 = np.isclose(data_2d1, -2 * np.pi, atol=tol)
    mask_pos_2pi_1 = np.isclose(data_2d1, 2 * np.pi, atol=tol)

    data_2d1[mask_neg_2pi_1] += 2 * np.pi
    data_2d1[mask_pos_2pi_1] -= 2 * np.pi

    mask_neg_2pi_1_ = np.isclose(data_2d1, - np.pi, atol=tol)
    mask_pos_2pi_1_ = np.isclose(data_2d1,  np.pi, atol=tol)

    data_2d1[mask_neg_2pi_1_] +=  np.pi
    data_2d1[mask_pos_2pi_1_] -= np.pi

    phi_phase_diff_t1[:, :, z_plane] = data_2d1
    
for z_plane in range(phi_phase_diff_t2.shape[2]):
    data_2d1 = phi_phase_diff_t2[:, :, z_plane]

    mask_neg_2pi_1 = np.isclose(data_2d1, -2 * np.pi, atol=tol)
    mask_pos_2pi_1 = np.isclose(data_2d1, 2 * np.pi, atol=tol)

    data_2d1[mask_neg_2pi_1] += 2 * np.pi
    data_2d1[mask_pos_2pi_1] -= 2 * np.pi

    mask_neg_2pi_1_ = np.isclose(data_2d1, - np.pi, atol=tol)
    mask_pos_2pi_1_ = np.isclose(data_2d1,  np.pi, atol=tol)

    data_2d1[mask_neg_2pi_1_] +=  np.pi
    data_2d1[mask_pos_2pi_1_] -= np.pi

    phi_phase_diff_t2[:, :, z_plane] = data_2d1

nx = 201
ny = 201
nz = 201

# # Define the sampling indices for each axis
sampling_interval_x = 4  # Sample every 2nd point along X axis
sampling_interval_y = 4  # Sample every 2nd point along Y axis
sampling_interval_z = 4  # Sample every 2nd point along Z axis

radius = 0  # Define the radius of the circular mask

phi_del_alpha = (phi_phase_diff_t2 - phi_phase_diff_t1)/0.3 * mod_t1

phi_diff_alpha = (phi_phase_t2 - phi_phase_t1)
for z_plane in range(phi_diff_alpha.shape[2]):
    data_2d1 = phi_diff_alpha[:, :, z_plane]

    mask_neg_2pi_1 = np.isclose(data_2d1, -2 * np.pi, atol=tol)
    mask_pos_2pi_1 = np.isclose(data_2d1, 2 * np.pi, atol=tol)

    data_2d1[mask_neg_2pi_1] += 2 * np.pi
    data_2d1[mask_pos_2pi_1] -= 2 * np.pi

    mask_neg_2pi_1_ = np.isclose(data_2d1, - np.pi, atol=tol)
    mask_pos_2pi_1_ = np.isclose(data_2d1,  np.pi, atol=tol)

    data_2d1[mask_neg_2pi_1_] +=  np.pi
    data_2d1[mask_pos_2pi_1_] -= np.pi

    phi_diff_alpha[:, :, z_plane] = data_2d1

phi_alpha = (phi_diff_alpha)/0.3 * mod_t1

# masked_sampled_phi_phase_diff = apply_circular_mask(phi_alpha, radius)
# # Sample the 3D array
# masked_sampled_phi_phase_diff = masked_sampled_phi_phase_diff[::sampling_interval_x, ::sampling_interval_y, ::sampling_interval_z]

# Scalar projection

Nx = 201
Ny = 201
Nz = 201
delta = 0.7
delta_t = 0.3

y, x = np.ogrid[-100:101, -100:101]

# Create a mask for a circle of radius 30
circular_radius = radius
circular_mask = x**2 + y**2 < circular_radius**2

# Load data from .npy file

# Frame number
# user_input = input("Do you want to enter the loop? (yes/no): ")

# if user_input.lower() == "yes":
#     for frame_t in range(320,335):

frame_t = 155
# frame_t1 = frame_t + 1

# Load data from .npy file
data_real_t = np.load(os.path.join(input_file_path, f'col_0_frame_{frame_t}.npy'))
data_im_t = np.load(os.path.join(input_file_path, f'col_1_frame_{frame_t}.npy'))
# data_real_t1 = np.load(os.path.join(input_file_path, f'col_0_frame_{frame_t1}.npy'))
# data_im_t1 = np.load(os.path.join(input_file_path, f'col_1_frame_{frame_t1}.npy'))

# test stationary file
# data_real_test = np.load(os.path.join(input_file_path, f'col_0_frame_1.npy'))
# data_im_test = np.load(os.path.join(input_file_path, f'col_1_frame_1.npy'))
# complex_number_test = data_real_test + 1j * data_im_test
# mod_test = np.abs((data_real_test)**2 + (data_im_test)**2)

# Reference file
# data_real_ref_t = np.load(os.path.join(input_ref_file_path, f'frame_{frame_t}_positive_boost_0.npy'))
# data_real_ref_t1 = np.load(os.path.join(input_ref_file_path, f'frame_{frame_t1}_positive_boost_0.npy'))

# data_im_ref_t = np.load(os.path.join(input_ref_file_path, f'frame_{frame_t}_positive_boost_1.npy'))
# data_im_ref_t1 = np.load(os.path.join(input_ref_file_path, f'frame_{frame_t1}_positive_boost_1.npy'))

complex_number_t = data_real_t + 1j * data_im_t
# complex_number_ref_t = data_real_ref_t + 1j * data_im_ref_t
# complex_conjugate_ref_t = data_real_ref_t - 1j * data_im_ref_t
# new = complex_number_t * complex_conjugate_ref_t

# complex_number_t1 = data_real_t1 + 1j * data_im_t1
# complex_number_ref_t1 = data_real_ref_t1 + 1j * data_im_ref_t1
# comp_num = complex_number - complex_number_ref
# # Assuming your grid is already in the right shape
# y_grid, x_grid, z_grid = np.meshgrid(np.arange(N), np.arange(N), np.arange(N))
mod_t_squared = np.abs((data_real_t)**2 + (data_im_t)**2)**2
mod_t = np.abs((data_real_t)**2 + (data_im_t)**2)
# mod_t1_squared = np.abs((data_real_t1)**2 + (data_im_t1)**2)**2

# phi_phase_ref_t = np.angle(complex_number_ref_t)

# phi_phase_ref_t1 = np.angle(complex_number_ref_t1)
# data_3d = phi_phase
# # Set the parameters


# Assuming your grid is already in the right shape
y_grid, x_grid, z_grid = np.meshgrid(np.arange(Ny), np.arange(Nx), np.arange(data_real_t.shape[2]))

# phi_phase_t = np.angle(complex_number_t)
# phase_t = phi_phase_t - phi_phase_ref_t
# phi_phase_t1 = np.angle(complex_number_t1)
# phase_t1 = phi_phase_t1 - phi_phase_ref_t1

# delta_alpha = (phase_t1 - phase_t)/delta_t
# phase_multiplied = np.angle(new)
# Set the parameters for the Fourier Transform

tol = 0.005
z_slice_to_plot = 0  # Adjust as needed

# diagnostic_t = (data_real_t * np.gradient * data_real_t + data_im_t * np.gradient * data_im_t)/(mod_t) 

grad_x_real_t = (np.roll(data_real_t, -1, axis=0) - np.roll(data_real_t, 1, axis=0)) / (2 * delta)
grad_x_im_t = (np.roll(data_im_t, -1, axis=0) - np.roll(data_im_t, 1, axis=0)) / (2 * delta)

grad_y_real_t = (np.roll(data_real_t, -1, axis=1) - np.roll(data_real_t, 1, axis=1)) / (2 * delta)
grad_y_im_t = (np.roll(data_im_t, -1, axis=1) - np.roll(data_im_t, 1, axis=1)) / (2 * delta)


grad_z_real_t = (np.roll(data_real_t, -1, axis=2) - np.roll(data_real_t, 1, axis=2)) / (2 * delta)
grad_z_im_t = (np.roll(data_im_t, -1, axis=2) - np.roll(data_im_t, 1, axis=2)) / (2 * delta)

D_x_alpha = (data_real_t * grad_x_im_t - data_im_t * grad_x_real_t)/(mod_t) 
D_y_alpha = (data_real_t * grad_y_im_t - data_im_t * grad_y_real_t)/(mod_t) 
D_z_alpha = (data_real_t * grad_z_im_t - data_im_t * grad_z_real_t)/(mod_t) 


# Define the center point (I_0, J_0, K_0)
I_0 = Nx // 2  # Assuming Nx, Ny, Nz are the dimensions of your grid
J_0 = Ny // 2
K_0 = Nz // 2

# Define the grid spacing along each dimension
delta_x = 0.7 
delta_y = 0.7
delta_z = 0.7

# Compute the coordinates relative to the center point
I_relative = np.arange(Nx) - I_0
J_relative = np.arange(Ny) - J_0
K_relative = np.arange(Nz) - K_0

# Create 3D grids of relative coordinates
I_grid, J_grid, K_grid = np.meshgrid(I_relative, J_relative, K_relative, indexing='ij')

# Compute the radial vector components
radial_vector_x = I_grid * delta_x
radial_vector_y = J_grid * delta_y
radial_vector_z = 0

# Compute the magnitude of the radial vector
magnitude_r = np.sqrt(radial_vector_x**2 + radial_vector_y**2 + radial_vector_z**2)

# Compute the unit vector components
unit_vector_x = radial_vector_x / magnitude_r
unit_vector_y = radial_vector_y / magnitude_r
unit_vector_z = radial_vector_z / magnitude_r

# Compute the dot product of D with the components of the unit vector
D_dot_unit_x = D_x_alpha * unit_vector_x
D_dot_unit_y = D_y_alpha * unit_vector_y
D_dot_unit_z = D_z_alpha * unit_vector_z

D_dot_unit_x = np.nan_to_num(D_dot_unit_x)
D_dot_unit_y = np.nan_to_num(D_dot_unit_y)
D_dot_unit_z = np.nan_to_num(D_dot_unit_z)


scalar_projection = D_dot_unit_x + D_dot_unit_y + D_dot_unit_z

masked_sampled_phi_phase_diff = apply_circular_mask(scalar_projection, radius)
# Sample the 3D array
masked_sampled_phi_phase_diff = masked_sampled_phi_phase_diff[::sampling_interval_x, ::sampling_interval_y, ::sampling_interval_z]



# Create meshgrid for sampled and masked data
X, Y, Z = np.meshgrid(
    np.arange(0, nx, sampling_interval_x)[:masked_sampled_phi_phase_diff.shape[0]],
    np.arange(0, ny, sampling_interval_y)[:masked_sampled_phi_phase_diff.shape[1]],
    np.arange(0, nz, sampling_interval_z)[:masked_sampled_phi_phase_diff.shape[2]],
    indexing='ij'  # Use indexing='ij'
)

# Plot 3D contours of the masked data
fig = plt.figure(figsize=(5, 5))
fig.set_facecolor('black')
ax = fig.add_subplot(111, projection='3d')

diagnostic_value = 0.0015
levels = np.linspace(-1 * diagnostic_value, diagnostic_value, 21)[::20]  # Select every second value
ax.grid(False)

# Set background color to black
ax.set_facecolor('black')
ax.w_xaxis.pane.fill = False
ax.w_yaxis.pane.fill = False
ax.w_zaxis.pane.fill = False
# Plot contours
for p in range(len(levels)):
    val = levels[p]  # Get the contour level

    if val < 0:
        color = 'magenta'
    else:
        color = 'cyan'

    for i in range(masked_sampled_phi_phase_diff.shape[2] - 0):  # Exclude the last layer in zx direction
        z_coord = Z[0, 0, i]  # Get the actual z-coordinate
        ax.contour(X[:, :, i], Y[:, :, i], masked_sampled_phi_phase_diff[:, :, i],
                   levels=[val], colors=color, zdir='z', offset=z_coord, linewidths=0.2)

    for i in range(masked_sampled_phi_phase_diff.shape[1] - 0):  # Exclude the last layer in zy direction
        ax.contour(X[:, i, :], Y[:, i, :], masked_sampled_phi_phase_diff[:, i, :],
                   levels=[val], colors=color, zdir='y', offset=0, linewidths=0.2)

    ax.contour(masked_sampled_phi_phase_diff[:, -1, :], Y[:, -1, :], Z[:, -1, :],
               levels=[val], colors=color, zdir='x', offset=nx-1, linewidths=0.2)

# Set limits of the plot from coord limits
xmin, xmax = X.min(), X.max()
ymin, ymax = Y.min(), Y.max()
zmin, zmax = Z.min(), Z.max()
ax.set(xlim=[xmin, xmax], ylim=[ymin, ymax], zlim=[zmin, zmax])
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
# ax.set_xlabel("x axis")
# ax.set_ylabel("y axis")
# ax.set_zlabel("z axis")
# plt.title(r"Contours of $ \dot{\Delta \alpha} |\phi|$" + f" ± {diagnostic_value} with Circular Mask of Radius = {circular_radius} grid squares. (Time Step {frame_t})")
plt.title(f"(Time Step {frame_t})")
# with Circular Mask of Radius = {circular_radius} grid squares.
# Save or show the plot
ax.view_init(elev=90, azim=0)
filename = f"contour_plot_diagnostic_{frame_t}.jpg"
frame_path = os.path.join(input_file_path, filename)
plt.savefig(frame_path, dpi=600)  # Save the plot


# plt.show()  # Show the plot
# plt.close(fig)  # Close the figure
