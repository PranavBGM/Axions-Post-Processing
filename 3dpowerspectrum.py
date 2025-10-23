# import numpy as np
# import matplotlib.pyplot as plt

# def discrete_function(alpha1, alpha2, alpha3):
#     # Define your function F_alpha here
#     # For example, let's use a simple function for demonstration
#     return np.sin(alpha1) + np.cos(alpha2) + np.exp(alpha3)

# def fourier_transform(F, N, delta, beta1, beta2, beta3):
#     # Create arrays for alpha values
#     alpha1 = np.arange(N)
#     alpha2 = np.arange(N)
#     alpha3 = np.arange(N)

#     # Create a grid of alpha values
#     alpha1, alpha2, alpha3 = np.meshgrid(alpha1, alpha2, alpha3, indexing='ij')

#     # Compute the Fourier transform using np.fft.fftn
#     exponent_term = np.exp(-2j * np.pi * (
#         (alpha1 * beta1 / N) + (alpha2 * beta2 / N) + (alpha3 * beta3 / N)
#     ))
#     F_values = F(alpha1 * delta, alpha2 * delta, alpha3 * delta)
#     F_tilde = np.fft.fftn(F_values) / np.sqrt(N**3)

#     return F_tilde

# def power_spectrum(N, delta, beta1, beta2, beta3):
#     # Compute the one-sided power spectrum
#     F_tilde = fourier_transform(discrete_function, N, delta, beta1, beta2, beta3)
#     P = np.abs(F_tilde)**2 / (N**2)

#     return P

# def plot_power_spectrum(N, delta):
#     # Define the range of beta values
#     beta_values = np.arange(N)

#     # Create a grid of beta values with indexing='ij'
#     beta1, beta2, beta3 = np.meshgrid(beta_values, beta_values, beta_values, indexing='ij')

#     # Vectorized computation of power spectrum
#     P_values = power_spectrum(N, delta, beta1, beta2, beta3)

#     # Flatten the beta arrays and use ravel()
#     beta1_flat = beta1.ravel()
#     beta2_flat = beta2.ravel()
#     beta3_flat = beta3.ravel()

#     # Print the sizes of the arrays for debugging
#     print("Sizes - beta1_flat:", len(beta1_flat), "beta2_flat:", len(beta2_flat), "beta3_flat:", len(beta3_flat), "P_values.ravel():", len(P_values.ravel()))

#     # Ensure the sizes match
#     assert len(beta1_flat) == len(beta2_flat) == len(beta3_flat) == len(P_values.ravel()), "Size mismatch!"

#     # Plot the power spectrum
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(beta1_flat, beta2_flat, beta3_flat, c=P_values.ravel(), cmap='Greys', marker='.')

#     ax.set_xlabel('Beta1')
#     ax.set_ylabel('Beta2')
#     ax.set_zlabel('Beta3')
#     ax.set_title('3D Power Spectrum')

#     plt.show()

# # Set the parameters
# N = 101  # Adjust the value based on your requirement
# delta = 0.5  # Adjust the value based on your requirement

# # Plot the power spectrum
# plot_power_spectrum(N, delta)

# import numpy as np
# import matplotlib.pyplot as plt

# def discrete_function(alpha1, alpha2, alpha3):
#     # Define your function F_alpha here
#     # For example, let's use a simple function for demonstration
#     return np.sin(alpha1) + np.cos(alpha2) + np.exp(alpha3)

# def fourier_transform(F, N, delta, beta1, beta2, beta3):
#     # Create arrays for alpha values
#     alpha1 = np.arange(N)
#     alpha2 = np.arange(N)
#     alpha3 = np.arange(N)

#     # Create a grid of alpha values
#     alpha1, alpha2, alpha3 = np.meshgrid(alpha1, alpha2, alpha3, indexing='ij')

#     # Compute the Fourier transform using np.fft.fftn
#     exponent_term = np.exp(-2j * np.pi * (
#         (alpha1 * beta1 / N) + (alpha2 * beta2 / N) + (alpha3 * beta3 / N)
#     ))
#     F_values = F(alpha1 * delta, alpha2 * delta, alpha3 * delta)
#     F_tilde = np.fft.fftn(F_values) / np.sqrt(N**3)

#     return F_tilde

# def power_spectrum(N, delta, beta1, beta2, beta3):
#     # Compute the one-sided power spectrum
#     F_tilde = fourier_transform(discrete_function, N, delta, beta1, beta2, beta3)
#     P = np.abs(F_tilde)**2 / (N**2)

#     return P

# def plot_power_spectrum(N, delta):
#     # Define the range of beta values
#     beta_values = np.arange(N)

#     # Create a grid of beta values with indexing='ij'
#     beta1, beta2, beta3 = np.meshgrid(beta_values, beta_values, beta_values, indexing='ij')

#     # Vectorized computation of power spectrum
#     P_values = power_spectrum(N, delta, beta1, beta2, beta3)

#     # Flatten the beta arrays and use ravel()
#     beta1_flat = beta1.ravel()
#     beta2_flat = beta2.ravel()
#     beta3_flat = beta3.ravel()

#     # Calculate the wavenumber for each point
#     k_values = np.sqrt((2 * np.pi * beta1_flat) / (N * delta))**2 + \
#                np.sqrt((2 * np.pi * beta2_flat) / (N * delta))**2 + \
#                np.sqrt((2 * np.pi * beta3_flat) / (N * delta))**2

#     # Plot the modes
#     fig, ax = plt.subplots()
#     ax.scatter(k_values, P_values.ravel(), marker='.')
#     ax.set_xlabel('Wavenumber (k)')
#     ax.set_ylabel('Power Spectrum')
#     ax.set_title('Modes in Power Spectrum')

#     plt.show()

# # Set the parameters
# N = 101  # Adjust the value based on your requirement
# delta = 0.5  # Adjust the value based on your requirement

# # Plot the power spectrum and modes
# plot_power_spectrum(N, delta)

# import os
# import numpy as np
# import matplotlib.pyplot as plt


# def fourier_transform(data, N, delta, beta1, beta2, beta3):
#     # Create arrays for alpha values
#     alpha1 = np.arange(N)
#     alpha2 = np.arange(N)
#     alpha3 = np.arange(N)

#     # Create a grid of alpha values
#     alpha1, alpha2, alpha3 = np.meshgrid(alpha1, alpha2, alpha3, indexing='ij')

#     # Compute the Fourier transform using np.fft.fftn
#     exponent_term = np.exp(-2j * np.pi * (
#         (alpha1 * beta1 / N) + (alpha2 * beta2 / N) + (alpha3 * beta3 / N)
#     ))
#     F_values = data[alpha1, alpha2, alpha3]
#     F_tilde = np.fft.fftn(F_values) / np.sqrt(N**3)

#     return F_tilde

# def power_spectrum(data, N, delta, beta1, beta2, beta3):
#     # Compute the one-sided power spectrum
#     F_tilde = fourier_transform(data, N, delta, beta1, beta2, beta3)
#     P = np.abs(F_tilde)**2 / (N**2)

#     return P

# def plot_power_spectrum(N, delta, data_3d, custom_bins=None):
#     # Define the range of beta values
#     beta_values = np.arange(N)

#     # Create a grid of beta values with indexing='ij'
#     beta1, beta2, beta3 = np.meshgrid(beta_values, beta_values, beta_values, indexing='ij')

#     # Vectorized computation of power spectrum
#     P_values = power_spectrum(data_3d, N, delta, beta1, beta2, beta3)

#     # Flatten the beta arrays and use ravel()
#     beta1_flat = beta1.ravel()
#     beta2_flat = beta2.ravel()
#     beta3_flat = beta3.ravel()

#     # Calculate the wavenumber for each point
#     k_values = np.sqrt((2 * np.pi * beta1_flat) / (N * delta))**2 + \
#                np.sqrt((2 * np.pi * beta2_flat) / (N * delta))**2 + \
#                np.sqrt((2 * np.pi * beta3_flat) / (N * delta))**2

#     # Create bins for the histogram with custom edges
#     if custom_bins is not None:
#         bins = custom_bins
#     else:
#         bins = 'auto'  # or specify the number of bins: bins=50

#     n_values = (k_values * N * delta) / (2* np.pi)
#     # Plot the histogram
    
#     plt.figure(figsize=(20, 10))
#     plt.style.use('seaborn-whitegrid')

#     plt.hist(x=n_values, weights=P_values.ravel(), bins=bins, color = 'skyblue', edgecolor='black')
    
#     power_path = os.path.join(input_directory, f"power_spec_phase_ev_10.png")
#     plt.xlabel('Wavenumber (k)')
#     plt.ylabel('Power Spectrum')
#     plt.title('Binned Power Spectrum')
#     plt.savefig(power_path, dpi = 300)
#     plt.show()



# # Specify your directory
# input_directory = "/Users/pranavbharadwajgangrekalvemanoj/Desktop/Axions_Project/npy_files_ev_10/"

# N = 51
# delta = 0.7
# # Plot the binned power spectrum with custom bin edges
# custom_bins = np.linspace(0, 10, 10)  # Specify your custom bin edges



# # Load data from .npy file
# # data_3d = np.load(os.path.join(input_directory, 'col_0_frame_319.npy'))
# data_real = np.load(os.path.join(input_directory, 'col_0_frame_16.npy'))
# data_im = np.load(os.path.join(input_directory, 'col_1_frame_16.npy'))

# #reference file
# # data_3d_ref = np.load( '/Users/pranavbharadwajgangrekalvemanoj/Desktop/Axions_Project/npy_files/col_0_frame_20.npy')
# data_real_ref = np.load( '/Users/pranavbharadwajgangrekalvemanoj/Desktop/Axions_Project/npy_files_ev_10/frame_pseudo_0.npy')
# data_im_ref = np.load( '/Users/pranavbharadwajgangrekalvemanoj/Desktop/Axions_Project/npy_files_ev_10/frame_pseudo_1.npy')

# complex_number = data_real + 1j * data_im
# complex_number_ref = data_real_ref + 1j * data_im_ref
# comp_num = complex_number - complex_number_ref
# # Assuming your grid is already in the right shape
# y_grid, x_grid, z_grid = np.meshgrid(np.arange(N), np.arange(N), np.arange(N))


# phi_phase = np.angle(comp_num)
# data_3d = phi_phase
# # Set the parameters

# # Plot the binned power spectrum
# plot_power_spectrum(N, delta, data_3d, custom_bins)

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import BoundaryNorm
from matplotlib.cm import get_cmap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import FuncFormatter, MaxNLocator
import matplotlib.ticker as ticker
cmap = plt.get_cmap('viridis')
import matplotlib
matplotlib.use('Qt5Agg') # Use Agg backend (non-interactive)

# Disable TeX rendering for text elements
# matplotlib.rcParams['text.usetex'] = False
import matplotlib.pyplot as plt
os.environ['PATH'] = '/usr/local/texlive/2023/bin:' + os.environ['PATH']
# matplotlib.use("pgf")

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'text.usetex': True,
    'pgf.texsystem': 'pdflatex',
    'pgf.rcfonts': False,
})
# import matplotlib as mpl

# # Increase the path chunk size to avoid OverflowError
# mpl.rcParams['agg.path.chunksize'] = 10000

def calculate_power_spectrum(data_2d, delta):
    F_tilde = np.fft.fft2(data_2d) / np.sqrt(data_2d.shape[0] * data_2d.shape[1])
    F_inverse = np.fft.ifft2(F_tilde)
    # plt.contourf(F_inverse, cmap='RdPu', levels=200)
    # plt.show()
    F_tilde = np.fft.fftshift(F_tilde)
    P = np.abs(F_tilde)**2 / (data_2d.shape[0] * data_2d.shape[1])
    return P

def calculate_fourier_transform(data_2d, delta):

    F_tilde = np.fft.fft2(data_2d) / np.sqrt(data_2d.shape[0] * data_2d.shape[1])
    # F_inverse = np.fft.ifft2(F_tilde)
    # plt.contourf(F_inverse, cmap='RdPu', levels=200)
    # plt.show()
    F_tilde = np.fft.fftshift(F_tilde)
    return F_tilde

def calculate_inv_fourier_transform(data_2d, delta):

    F_tilde = np.fft.fft2(data_2d) / np.sqrt(data_2d.shape[0] * data_2d.shape[1])
    F_inverse = np.fft.ifft2(F_tilde)
    plt.contourf(F_inverse, cmap='RdPu', levels=200)
    plt.show()

def fftfreq2D(Nx, Ny, dx, dy):
    kx = np.fft.fftfreq(Nx, dx)
    kx = np.fft.fftshift(kx) * 2 * np.pi
    ky = np.fft.fftfreq(Ny, dy)
    ky = np.fft.fftshift(ky) * 2 * np.pi

    kx_2d, ky_2d = np.meshgrid(kx, ky, indexing='ij')
    return kx_2d, ky_2d

# def plot_power_spectrum(Nx, Ny, delta, data_3d1, data_3d2, phi2t, alpha_t,alpha_t1,frame_number,custom_bins=None):
#     radius = 50
#     theta = np.linspace(0, 2 * np.pi, 500)  # Angular positions
#     x_circle = radius * np.cos(theta)  # x coordinates on the circle
#     y_circle = radius * np.sin(theta)  # y coordinates on the circle
#     circle_values = np.zeros((data_3d1.shape[2], len(theta)))
    
#     bin_values_1 = []
#     bin_values_2 = []
#     bin_values_3 = []
    
#     for z_plane in range(1,data_3d1.shape[2]-1):
#         x = np.arange(data_3d1.shape[0])
#         data_2d1 = data_3d1[:, :, z_plane]
        
#         mask_neg_2pi_1 = np.isclose(data_2d1, -2 * np.pi, atol=tol)
#         mask_pos_2pi_1 = np.isclose(data_2d1, 2 * np.pi, atol=tol)
        
#         data_2d1[mask_neg_2pi_1] += 2 * np.pi
#         data_2d1[mask_pos_2pi_1] -= 2 * np.pi
        
#         mask_neg_2pi_1_ = np.isclose(data_2d1, - np.pi, atol=tol)
#         mask_pos_2pi_1_ = np.isclose(data_2d1,  np.pi, atol=tol)
        
#         data_2d1[mask_neg_2pi_1_] +=  np.pi
#         data_2d1[mask_pos_2pi_1_] -= np.pi
        
#         # line1 = cos1(x)
#         # line2 = cos2(x)
        
#         y_index = 125  # The y index for which you want to plot the data

# #         plt.figure(figsize=(10, 6))

# # # Loop over the z slices
# #         for z in range(1, data_3d1.shape[2], 5):
# #             # Extract the data for y=125 for the current z slice
# #             data_2d1 = data_3d1[:, y_index, z]

# #             # Apply phase correction
# #             mask_neg_2pi_1 = np.isclose(data_2d1, -2 * np.pi, atol=tol)
# #             mask_pos_2pi_1 = np.isclose(data_2d1, 2 * np.pi, atol=tol)

# #             data_2d1[mask_neg_2pi_1] += 2 * np.pi
# #             data_2d1[mask_pos_2pi_1] -= 2 * np.pi

# #             mask_neg_2pi_1_ = np.isclose(data_2d1, - np.pi, atol=tol)
# #             mask_pos_2pi_1_ = np.isclose(data_2d1,  np.pi, atol=tol)

# #             data_2d1[mask_neg_2pi_1_] +=  np.pi
# #             data_2d1[mask_pos_2pi_1_] -= np.pi
# #             x = np.arange(data_3d1.shape[0])
# #             # Plot the data against x
# #             plt.plot(x, data_2d1, label=f'z={z}')

# #         plt.plot(x, line1, color='red', linestyle='--', marker='o', label='cos(4 pi x / L)')
# #         plt.plot(x, line2, color='blue', linestyle=':', marker='x', label='cos(8 pi x / L)')
# #         plt.xlabel('x')
# #         plt.ylabel('delta alpha')
# #         plt.legend()
# #         plt.show()

#         data_2d2 = data_3d2[:, :, z_plane]
        
#         mask_neg_2pi_2 = np.isclose(data_2d2, -2 * np.pi, atol=tol)
#         mask_pos_2pi_2 = np.isclose(data_2d2, 2 * np.pi, atol=tol)
        
#         data_2d2[mask_neg_2pi_2] += 2 * np.pi
#         data_2d2[mask_pos_2pi_2] -= 2 * np.pi
        
#         mask_neg_2pi_2_ = np.isclose(data_2d2, - np.pi, atol=tol)
#         mask_pos_2pi_2_ = np.isclose(data_2d2,  np.pi, atol=tol)
        
#         data_2d2[mask_neg_2pi_2_] +=  np.pi
#         data_2d2[mask_pos_2pi_2_] -= np.pi
        
        
        
        
#         phi2t_plane = phi2t[:, :, z_plane]
        
#         # phase_corrected = np.where(np.abs(data_2d) > 2*np.pi, data_2d - 2*np.pi*np.sign(data_2d), data_2d)
#         # data_2d = np.where(np.abs(phase_corrected) > np.pi, phase_corrected + np.pi*np.sign(phase_corrected), phase_corrected)
#         # if z_plane == 62:
#             # bin_values_1 = []
#             # bin_values_2 = []
#             # bin_values_3 = []
#             # radius = 50
#         theta = np.linspace(0, 2 * np.pi, 500)  # Angular positions
#         x_circle = radius * np.cos(theta)  # x coordinates on the circle
#         y_circle = radius * np.sin(theta)  # y coordinates on the circle

#         #     # Interpolate data_2d1 values along the circle
#         x_indices = np.round(x_circle).astype(int) + data_2d1.shape[1] // 2
#         y_indices = np.round(y_circle).astype(int) + data_2d1.shape[0] // 2
#         circle_values[z_plane, :] = data_2d1[y_indices, x_indices]
#         circle_values_12 = None

#         #     # # Plotting
#         # if z_plane == 12:
#         #     circle_values_12 = data_2d1[y_indices, x_indices]
#         #     plt.figure()
#         #     plt.plot(theta, circle_values_12)
#         #     plt.xlabel('Angular Position (radians)')
#         #     plt.ylabel('Data Values on Circle')
#         #     plt.title(f'Data Values on Circle of Radius 50 units and z slice of {z_plane}')
#         #     plt.show()
#         #     for radius in range(0, 199, 10):
#                 # y, x = np.ogrid[-200:201, -200:201]
#                 # circular_mask = x**2 + y**2 < radius**2

#                 # data_2d1 = np.where(circular_mask, 0, data_2d1)
#                 # data_2d2 = np.where(circular_mask, 0, data_2d2)
#                 # phi2t_plane = np.where(circular_mask, 0, phi2t_plane)


                
                
#         del_alpha = (data_2d2 - data_2d1)/delta_t
#         alpha = (alpha_t1 - alpha_t)/delta_t
#         alpha_slice = alpha[:,:,z_plane]
#         phi2_del2 = phi2t_plane * np.abs(del_alpha)**2
#         phi_del = np.sqrt(phi2_del2)
#         phi2_alpha2 = phi2t_plane * np.abs(alpha_slice)**2
#         phi_alpha = np.sqrt(phi2_alpha2)
        
        
#     #     # P_values1 = calculate_power_spectrum(phi_del, delta)
#     #     # P_values2 = calculate_power_spectrum(phi_alpha, delta)
#         P_values1 = calculate_power_spectrum(data_2d1, delta)
#         P_values2 = calculate_power_spectrum(phi_del, delta)
#         P_values3 = calculate_power_spectrum(phi_alpha, delta)


#         beta1, beta2 = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing='ij')
#         beta1_flat = beta1.ravel()
#         beta2_flat = beta2.ravel()

#         # k_values = np.sqrt((2 * np.pi * beta1_flat) / (Nx * delta))**2 + \
#         #            np.sqrt((2 * np.pi * beta2_flat) / (Ny * delta))**2
#         k_values_x, k_values_y = fftfreq2D(Nx, Ny, delta, delta) 
#         mod_k = np.sqrt(k_values_x**2 + k_values_y**2)

#         n_values = (mod_k * Nx * delta) / (4 * np.pi) # factor of 2 is added to account for the fact that the first bin took up both n = 1 and n = 2 modes.
        
#     #     # logic1 = (n_values > 0.5) & (n_values<1.5)
#     #     # logic12= (n_values > 1.5) & (n_values<2.5)

#     #     # print(np.sum(P_values[logic1]), np.sum(P_values[logic12]))
        # fig, axs = plt.subplots(2, 3, figsize=(9, 9))  # Create a 2x2 grid of subplots

        # # First plot
        # im1 = axs[0, 0].contourf(data_2d1, cmap='PuOr', levels=np.linspace(-0.1,0.1,200))  # Fix the colorbar range to [0, 1]
        # # axs[0, 0].set_title(r'$(\dot{\Delta \alpha}) |\phi|$ ')
        # axs[0, 0].set_title(r'$({\Delta \alpha}) $ ')

        # # Second plot
        # axs[1, 0].hist(x=n_values.ravel(), weights=P_values1.ravel(), bins=np.linspace(0.5,10.5,11), color='purple', edgecolor='black')
        # # axs[1, 0].set_ylim([0, 2e-6])
        # # axs[1, 0].set_yscale('log')
        # # axs[1, 0].set_ylim([1e-7, 3.5e-6]) 
        # # Third plot
        # im2 = axs[0, 1].contourf(phi_del, cmap='plasma', levels=np.linspace(0,0.01,200)) # Fix the colorbar range to [0, 1]
        # axs[0, 1].set_title(r'$(\dot{\Delta \alpha}) |\phi|$ ')

        # # Fourth plot
        # axs[1, 1].hist(x=n_values.ravel(), weights=P_values2.ravel(), bins=np.linspace(0.5,10.5,11), color='purple', edgecolor='black')
        # # axs[1, 1].set_ylim([0, 0.5e-6])
        # # axs[1, 1].set_yscale('log')
        # # axs[1, 1].set_ylim([1e-7, 3.5e-6])               
        # axs[1, 1].set_xlabel('n values')
        # axs[1, 0].set_ylabel('Power Spectrum')
        # # axs[1, 1].set_title(f' Power Spectrum - xy Plane {z_plane + 1}')

        # im3 = axs[0, 2].contourf(phi_alpha, cmap='plasma', levels=np.linspace(0,0.01,200) ) # Fix the colorbar range to [0, 1]
        # axs[0, 2].set_title(r'$(\dot{\alpha}) |\phi|$ ')
        # axs[1, 2].hist(x=n_values.ravel(), weights=P_values3.ravel(), bins=np.linspace(0.5,10.5,11), color='purple', edgecolor='black')   
        # # axs[1, 2].set_ylim([0, 0.5e-6])             

#         # # Create a new axes for the colorbar that spans both subplots
#         # for ax in axs.flat:
#         #     box = ax.get_position()
#         #     ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])

#         # # Create a new axes for the colorbar that spans both subplots
#         # # cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])
#         # # fig.colorbar(im1, cax=cbar_ax)

#         # plt.tight_layout(pad=3.0)  # Increase the padding between the plots
#         # plt.suptitle("Radius = " + str(radius) + " - xy Plane " + str(z_plane + 1))
#         # power_path = os.path.join(input_directory, f"4pi_power_spec_xy_plane_{z_plane}.png")
#         # plt.savefig(power_path, dpi=300)
#         # plt.show()
                



#                 # P_values1 = calculate_power_spectrum(data_2d1, delta)
#                 # P_values2 = calculate_power_spectrum(phi_del, delta)
#                 # P_values3 = calculate_power_spectrum(phi_alpha, delta)

#                 # Extract histogram values for the first three bins
#     #     hist1, _ = np.histogram(n_values.ravel(), weights=P_values1.ravel(), bins=np.linspace(0.5, 10.5, 11))
#     #     hist2, _ = np.histogram(n_values.ravel(), weights=P_values2.ravel(), bins=np.linspace(0.5, 10.5, 11))
#     #     hist3, _ = np.histogram(n_values.ravel(), weights=P_values3.ravel(), bins=np.linspace(0.5, 10.5, 11))

#     #     # Append the first three bin values to the respective lists
#     #     bin_values_1.append(hist1[:3])
#     #     bin_values_2.append(hist2[:3])
#     #     bin_values_3.append(hist3[:3])
#     #     print(z_plane)

#     #         # # Convert lists to numpy arrays for easier manipulation
#     # bin_values_1 = np.array(bin_values_1)
#     # bin_values_2 = np.array(bin_values_2)
#     # bin_values_3 = np.array(bin_values_3)

#     #         # # Plotting
#     # radii = np.arange(0, 199, 1)

#     #         # # Plot for the first power spectrum
#     # fig, axs = plt.subplots(3, 1, figsize=(8, 12))

#     # # Plot for the first power spectrum
#     # for i in range(3):
#     #     axs[0].plot(range(1,data_3d1.shape[2]-1), bin_values_1[:, i], label=f'Bin {i+1}')
#     # axs[0].set_ylabel('Bin Values')
#     # axs[0].set_title(r'$({\Delta \alpha}) $ Power Spectrum')
#     # axs[0].legend()

#     # # Plot for the second power spectrum
#     # for i in range(3):
#     #     axs[1].plot(range(1,data_3d1.shape[2]-1), bin_values_2[:, i], label=f'Bin {i+1}')
#     # axs[1].set_ylabel('Bin Values')
#     # axs[1].set_title(r'$(\dot{\Delta \alpha}) |\phi|$ Power Spectrum')
#     # axs[1].legend()

#     # # Plot for the third power spectrum
#     # for i in range(3):
#     #     axs[2].plot(range(1,data_3d1.shape[2]-1), bin_values_3[:, i], label=f'Bin {i+1}')
#     # axs[2].set_xlabel('Z Slice')
#     # axs[2].set_ylabel('Bin Values')
#     # axs[2].set_title(r'$(\dot{\alpha}) |\phi|$ Power Spectrum')
#     # axs[2].legend()

#     # plt.tight_layout()
#     # plt.show()
                
#     # Plotting for the differnt z slices and angular position for delta alpha
#     plt.figure()
#     boundaries = np.linspace(-0.025, 0.025, 200)

#     # Create a colormap
#     cmap = get_cmap('viridis')

#     # Create a normalization object
#     norm = BoundaryNorm(boundaries, cmap.N)

#     plt.imshow(circle_values, aspect='auto', cmap=cmap, extent=[0, 2 * np.pi, 0, data_3d1.shape[2]])
#     plt.colorbar(label=r'$\Delta \alpha$ Values on Circle')
#     plt.xlabel('Angular Position (radians)')
#     plt.ylabel('Z Slice')
#     plt.title(r'$\Delta \alpha$ Values on Circle of Radius 50 units' + '\n' + 'All Z Slices for the ' + f'{frame_t}th time step')
    
#         # Set custom ticks and tick labels for the x-axis
#     plt.xticks(np.linspace(0, 2 * np.pi, 5), ['0', '$\pi/2$', '$\pi$', '$3\pi/2$', '$2\pi$'])
#     plt.savefig(os.path.join(input_directory, f"plot_{frame_t}.png"))
#     # plt.show()             

# --------------------------------------------------------------------------------------------------------------------------------------------------

def calculate_power_spectrum(data_2d, delta):
    F_tilde = np.fft.fft2(data_2d) / np.sqrt(data_2d.shape[0] * data_2d.shape[1])
    F_inverse = np.fft.ifft2(F_tilde)
    # plt.contourf(F_inverse, cmap='RdPu', levels=200)
    # plt.show()
    F_tilde = np.fft.fftshift(F_tilde)
    P = np.abs(F_tilde)**2 / (data_2d.shape[0] * data_2d.shape[1])
    return P

def calculate_fourier_transform(data_2d, delta):

    F_tilde = np.fft.fft2(data_2d) / np.sqrt(data_2d.shape[0] * data_2d.shape[1])
    # F_inverse = np.fft.ifft2(F_tilde)
    # plt.contourf(F_inverse, cmap='RdPu', levels=200)
    # plt.show()
    F_tilde = np.fft.fftshift(F_tilde)
    return F_tilde

def calculate_inv_fourier_transform(data_2d, delta):

    F_tilde = np.fft.fft2(data_2d) / np.sqrt(data_2d.shape[0] * data_2d.shape[1])
    F_inverse = np.fft.ifft2(F_tilde)
    plt.contourf(F_inverse, cmap='RdPu', levels=200)
    plt.show()

def fftfreq2D(Nx, Ny, dx, dy):
    kx = np.fft.fftfreq(Nx, dx)
    kx = np.fft.fftshift(kx) * 2 * np.pi
    ky = np.fft.fftfreq(Ny, dy)
    ky = np.fft.fftshift(ky) * 2 * np.pi

    kx_2d, ky_2d = np.meshgrid(kx, ky, indexing='ij')
    return kx_2d, ky_2d

# def plot_power_spectrum(Nx, Ny, delta, data_3d1, data_3d2, phi2t, alpha_t,alpha_t1,custom_bins=None):
#     for radius in range(1, 51):
#         theta = np.linspace(0, 2 * np.pi, 500)  # Angular positions
#         x_circle = radius * np.cos(theta)  # x coordinates on the circle
#         y_circle = radius * np.sin(theta)  # y coordinates on the circle
#         circle_values = np.zeros((data_3d1.shape[2], len(theta)))
        
#         bin_values_1 = []
#         bin_values_2 = []
#         bin_values_3 = []
        
#         for z_plane in range(1,data_3d1.shape[2]-1):
#             x = np.arange(data_3d1.shape[0])
#             data_2d1 = data_3d1[:, :, z_plane]
            
#             mask_neg_2pi_1 = np.isclose(data_2d1, -2 * np.pi, atol=tol)
#             mask_pos_2pi_1 = np.isclose(data_2d1, 2 * np.pi, atol=tol)
            
#             data_2d1[mask_neg_2pi_1] += 2 * np.pi
#             data_2d1[mask_pos_2pi_1] -= 2 * np.pi
            
#             mask_neg_2pi_1_ = np.isclose(data_2d1, - np.pi, atol=tol)
#             mask_pos_2pi_1_ = np.isclose(data_2d1,  np.pi, atol=tol)
            
#             data_2d1[mask_neg_2pi_1_] +=  np.pi
#             data_2d1[mask_pos_2pi_1_] -= np.pi
            
#             # line1 = cos1(x)
#             # line2 = cos2(x)
            
#             y_index = 125  # The y index for which you want to plot the data

#     #         plt.figure(figsize=(10, 6))

#     # # Loop over the z slices
#     #         for z in range(1, data_3d1.shape[2], 5):
#     #             # Extract the data for y=125 for the current z slice
#     #             data_2d1 = data_3d1[:, y_index, z]

#     #             # Apply phase correction
#     #             mask_neg_2pi_1 = np.isclose(data_2d1, -2 * np.pi, atol=tol)
#     #             mask_pos_2pi_1 = np.isclose(data_2d1, 2 * np.pi, atol=tol)

#     #             data_2d1[mask_neg_2pi_1] += 2 * np.pi
#     #             data_2d1[mask_pos_2pi_1] -= 2 * np.pi

#     #             mask_neg_2pi_1_ = np.isclose(data_2d1, - np.pi, atol=tol)
#     #             mask_pos_2pi_1_ = np.isclose(data_2d1,  np.pi, atol=tol)

#     #             data_2d1[mask_neg_2pi_1_] +=  np.pi
#     #             data_2d1[mask_pos_2pi_1_] -= np.pi
#     #             x = np.arange(data_3d1.shape[0])
#     #             # Plot the data against x
#     #             plt.plot(x, data_2d1, label=f'z={z}')

#     #         plt.plot(x, line1, color='red', linestyle='--', marker='o', label='cos(4 pi x / L)')
#     #         plt.plot(x, line2, color='blue', linestyle=':', marker='x', label='cos(8 pi x / L)')
#     #         plt.xlabel('x')
#     #         plt.ylabel('delta alpha')
#     #         plt.legend()
#     #         plt.show()

#             data_2d2 = data_3d2[:, :, z_plane]
            
#             mask_neg_2pi_2 = np.isclose(data_2d2, -2 * np.pi, atol=tol)
#             mask_pos_2pi_2 = np.isclose(data_2d2, 2 * np.pi, atol=tol)
            
#             data_2d2[mask_neg_2pi_2] += 2 * np.pi
#             data_2d2[mask_pos_2pi_2] -= 2 * np.pi
            
#             mask_neg_2pi_2_ = np.isclose(data_2d2, - np.pi, atol=tol)
#             mask_pos_2pi_2_ = np.isclose(data_2d2,  np.pi, atol=tol)
            
#             data_2d2[mask_neg_2pi_2_] +=  np.pi
#             data_2d2[mask_pos_2pi_2_] -= np.pi
            
            
            
            
#             phi2t_plane = phi2t[:, :, z_plane]
            
#             # phase_corrected = np.where(np.abs(data_2d) > 2*np.pi, data_2d - 2*np.pi*np.sign(data_2d), data_2d)
#             # data_2d = np.where(np.abs(phase_corrected) > np.pi, phase_corrected + np.pi*np.sign(phase_corrected), phase_corrected)
#             # if z_plane == 62:
#                 # bin_values_1 = []
#                 # bin_values_2 = []
#                 # bin_values_3 = []
#                 # radius = 50
#             theta = np.linspace(0, 2 * np.pi, 500)  # Angular positions
#             x_circle = radius * np.cos(theta)  # x coordinates on the circle
#             y_circle = radius * np.sin(theta)  # y coordinates on the circle

#             #     # Interpolate data_2d1 values along the circle
#             x_indices = np.round(x_circle).astype(int) + data_2d1.shape[1] // 2
#             y_indices = np.round(y_circle).astype(int) + data_2d1.shape[0] // 2
#             circle_values[z_plane, :] = data_2d1[y_indices, x_indices]
#             circle_values_12 = None

#             #     # # Plotting
#             # if z_plane == 12:
#             #     circle_values_12 = data_2d1[y_indices, x_indices]
#             #     plt.figure()
#             #     plt.plot(theta, circle_values_12)
#             #     plt.xlabel('Angular Position (radians)')
#             #     plt.ylabel('Data Values on Circle')
#             #     plt.title(f'Data Values on Circle of Radius 50 units and z slice of {z_plane}')
#             #     plt.show()
#             #     for radius in range(0, 199, 10):
#                     # y, x = np.ogrid[-200:201, -200:201]
#                     # circular_mask = x**2 + y**2 < radius**2

#                     # data_2d1 = np.where(circular_mask, 0, data_2d1)
#                     # data_2d2 = np.where(circular_mask, 0, data_2d2)
#                     # phi2t_plane = np.where(circular_mask, 0, phi2t_plane)


                    
                    
#             del_alpha = (data_2d2 - data_2d1)/delta_t
#             alpha = (alpha_t1 - alpha_t)/delta_t
#             alpha_slice = alpha[:,:,z_plane]
#             phi2_del2 = phi2t_plane * np.abs(del_alpha)**2
#             phi_del = np.sqrt(phi2_del2)
#             phi2_alpha2 = phi2t_plane * np.abs(alpha_slice)**2
#             phi_alpha = np.sqrt(phi2_alpha2)
            
            
#         #     # P_values1 = calculate_power_spectrum(phi_del, delta)
#         #     # P_values2 = calculate_power_spectrum(phi_alpha, delta)
#             P_values1 = calculate_power_spectrum(data_2d1, delta)
#             P_values2 = calculate_power_spectrum(phi_del, delta)
#             P_values3 = calculate_power_spectrum(phi_alpha, delta)


#             beta1, beta2 = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing='ij')
#             beta1_flat = beta1.ravel()
#             beta2_flat = beta2.ravel()

#             # k_values = np.sqrt((2 * np.pi * beta1_flat) / (Nx * delta))**2 + \
#             #            np.sqrt((2 * np.pi * beta2_flat) / (Ny * delta))**2
#             k_values_x, k_values_y = fftfreq2D(Nx, Ny, delta, delta) 
#             mod_k = np.sqrt(k_values_x**2 + k_values_y**2)

#             n_values = (mod_k * Nx * delta) / (4 * np.pi) # factor of 2 is added to account for the fact that the first bin took up both n = 1 and n = 2 modes.
            
#         #     # logic1 = (n_values > 0.5) & (n_values<1.5)
#         #     # logic12= (n_values > 1.5) & (n_values<2.5)

#         #     # print(np.sum(P_values[logic1]), np.sum(P_values[logic12]))
#             # fig, axs = plt.subplots(2, 3, figsize=(9, 9))  # Create a 2x2 grid of subplots

#             # # First plot
#             # im1 = axs[0, 0].contourf(data_2d1, cmap='PuOr', levels=np.linspace(-0.1,0.1,200))  # Fix the colorbar range to [0, 1]
#             # # axs[0, 0].set_title(r'$(\dot{\Delta \alpha}) |\phi|$ ')
#             # axs[0, 0].set_title(r'$({\Delta \alpha}) $ ')

#             # # Second plot
#             # axs[1, 0].hist(x=n_values.ravel(), weights=P_values1.ravel(), bins=np.linspace(0.5,10.5,11), color='purple', edgecolor='black')
#             # # axs[1, 0].set_ylim([0, 2e-6])
#             # # axs[1, 0].set_yscale('log')
#             # # axs[1, 0].set_ylim([1e-7, 3.5e-6]) 
#             # # Third plot
#             # im2 = axs[0, 1].contourf(phi_del, cmap='plasma', levels=np.linspace(0,0.01,200)) # Fix the colorbar range to [0, 1]
#             # axs[0, 1].set_title(r'$(\dot{\Delta \alpha}) |\phi|$ ')

#             # # Fourth plot
#             # axs[1, 1].hist(x=n_values.ravel(), weights=P_values2.ravel(), bins=np.linspace(0.5,10.5,11), color='purple', edgecolor='black')
#             # # axs[1, 1].set_ylim([0, 0.5e-6])
#             # # axs[1, 1].set_yscale('log')
#             # # axs[1, 1].set_ylim([1e-7, 3.5e-6])               
#             # axs[1, 1].set_xlabel('n values')
#             # axs[1, 0].set_ylabel('Power Spectrum')
#             # # axs[1, 1].set_title(f' Power Spectrum - xy Plane {z_plane + 1}')

#             # im3 = axs[0, 2].contourf(phi_alpha, cmap='plasma', levels=np.linspace(0,0.01,200) ) # Fix the colorbar range to [0, 1]
#             # axs[0, 2].set_title(r'$(\dot{\alpha}) |\phi|$ ')
#             # axs[1, 2].hist(x=n_values.ravel(), weights=P_values3.ravel(), bins=np.linspace(0.5,10.5,11), color='purple', edgecolor='black')   
#             # axs[1, 2].set_ylim([0, 0.5e-6])             

#             # # Create a new axes for the colorbar that spans both subplots
#             # for ax in axs.flat:
#             #     box = ax.get_position()
#             #     ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])

#             # # Create a new axes for the colorbar that spans both subplots
#             # # cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])
#             # # fig.colorbar(im1, cax=cbar_ax)

#             # plt.tight_layout(pad=3.0)  # Increase the padding between the plots
#             # plt.suptitle("Radius = " + str(radius) + " - xy Plane " + str(z_plane + 1))
#             # power_path = os.path.join(input_directory, f"4pi_power_spec_xy_plane_{z_plane}.png")
#             # plt.savefig(power_path, dpi=300)
#             # plt.show()
                    



#                     # P_values1 = calculate_power_spectrum(data_2d1, delta)
#                     # P_values2 = calculate_power_spectrum(phi_del, delta)
#                     # P_values3 = calculate_power_spectrum(phi_alpha, delta)

#                     # Extract histogram values for the first three bins
#         #     hist1, _ = np.histogram(n_values.ravel(), weights=P_values1.ravel(), bins=np.linspace(0.5, 10.5, 11))
#         #     hist2, _ = np.histogram(n_values.ravel(), weights=P_values2.ravel(), bins=np.linspace(0.5, 10.5, 11))
#         #     hist3, _ = np.histogram(n_values.ravel(), weights=P_values3.ravel(), bins=np.linspace(0.5, 10.5, 11))

#         #     # Append the first three bin values to the respective lists
#         #     bin_values_1.append(hist1[:3])
#         #     bin_values_2.append(hist2[:3])
#         #     bin_values_3.append(hist3[:3])
#         #     print(z_plane)

#         #         # # Convert lists to numpy arrays for easier manipulation
#         # bin_values_1 = np.array(bin_values_1)
#         # bin_values_2 = np.array(bin_values_2)
#         # bin_values_3 = np.array(bin_values_3)

#         #         # # Plotting
#         # radii = np.arange(0, 199, 1)

#         #         # # Plot for the first power spectrum
#         # fig, axs = plt.subplots(3, 1, figsize=(8, 12))

#         # # Plot for the first power spectrum
#         # for i in range(3):
#         #     axs[0].plot(range(1,data_3d1.shape[2]-1), bin_values_1[:, i], label=f'Bin {i+1}')
#         # axs[0].set_ylabel('Bin Values')
#         # axs[0].set_title(r'$({\Delta \alpha}) $ Power Spectrum')
#         # axs[0].legend()

#         # # Plot for the second power spectrum
#         # for i in range(3):
#         #     axs[1].plot(range(1,data_3d1.shape[2]-1), bin_values_2[:, i], label=f'Bin {i+1}')
#         # axs[1].set_ylabel('Bin Values')
#         # axs[1].set_title(r'$(\dot{\Delta \alpha}) |\phi|$ Power Spectrum')
#         # axs[1].legend()

#         # # Plot for the third power spectrum
#         # for i in range(3):
#         #     axs[2].plot(range(1,data_3d1.shape[2]-1), bin_values_3[:, i], label=f'Bin {i+1}')
#         # axs[2].set_xlabel('Z Slice')
#         # axs[2].set_ylabel('Bin Values')
#         # axs[2].set_title(r'$(\dot{\alpha}) |\phi|$ Power Spectrum')
#         # axs[2].legend()

#         # plt.tight_layout()
#         # plt.show()
                    
#         # Plotting for the differnt z slices and angular position for delta alpha
#         plt.figure()
#         boundaries = np.linspace(-0.0125, 0.0125, 200)

#         # Create a colormap
#         cmap = get_cmap('viridis')

#         # Create a normalization object
#         norm = BoundaryNorm(boundaries, cmap.N)

#         plt.imshow(circle_values, aspect='auto', cmap=cmap, extent=[0, 2 * np.pi, 0, data_3d1.shape[2]])
#         plt.colorbar(label=r'$\Delta \alpha$ Values on Circle')
#         plt.xlabel('Angular Position (radians)')
#         plt.ylabel('Z Slice')
#         plt.title(r'$\Delta \alpha$ Values on Circle of Radius' + f' of {radius} grid lengths'+ '\n' + 'All Z Slices for the ' + f'{frame_t}th time step')
        
#             # Set custom ticks and tick labels for the x-axis
#         plt.xticks(np.linspace(0, 2 * np.pi, 5), ['0', '$\pi/2$', '$\pi$', '$3\pi/2$', '$2\pi$'])
#         plt.savefig(os.path.join(input_directory, f"plot_{radius}_{frame_t}_wit_norm.png"))
#         # plt.show()   
#         plt.close()          

# --------------------------------------------------------------------------------------------------------------------------------------------------




# def plot_power_spectrum(Nx, Ny, delta, data_3d1, data_3d2, phi2t, alpha_t,alpha_t1,custom_bins=None):
#     bin_values_1 = []
#     bin_values_2 = []
#     bin_values_3 = []
#     threshold_values = np.linspace(0.1, 0.9, 9)

# # Initialize lists to store the bin values for each threshold value
#     bin_values_del_alpha = []
#     bin_values_alpha_slice = []
#     for z_plane in range(data_3d1.shape[2]):
#         x = np.arange(data_3d1.shape[0])
#         data_2d1 = data_3d1[:, :, z_plane]
#         # boundary_width = 1  # Define the width of the boundary to be masked
#         # data_2d1[:boundary_width, :] = 0
#         # data_2d1[-boundary_width:, :] = 0
#         # data_2d1[:, :boundary_width] = 0
#         # data_2d1[:, -boundary_width:] = 0
        
#         mask_neg_2pi_1 = np.isclose(data_2d1, -2 * np.pi, atol=tol)
#         mask_pos_2pi_1 = np.isclose(data_2d1, 2 * np.pi, atol=tol)
        
#         data_2d1[mask_neg_2pi_1] += 2 * np.pi
#         data_2d1[mask_pos_2pi_1] -= 2 * np.pi
        
#         mask_neg_2pi_1_ = np.isclose(data_2d1, - np.pi, atol=tol)
#         mask_pos_2pi_1_ = np.isclose(data_2d1,  np.pi, atol=tol)
        
#         data_2d1[mask_neg_2pi_1_] +=  np.pi
#         data_2d1[mask_pos_2pi_1_] -= np.pi
        
#         # line1 = cos1(x)
#         # line2 = cos2(x)
        
#         y_index = 125  # The y index for which you want to plot the data

# #         plt.figure(figsize=(10, 6))

# # # Loop over the z slices
# #         for z in range(1, data_3d1.shape[2], 5):
# #             # Extract the data for y=125 for the current z slice
# #             data_2d1 = data_3d1[:, y_index, z]

# #             # Apply phase correction
# #             mask_neg_2pi_1 = np.isclose(data_2d1, -2 * np.pi, atol=tol)
# #             mask_pos_2pi_1 = np.isclose(data_2d1, 2 * np.pi, atol=tol)

# #             data_2d1[mask_neg_2pi_1] += 2 * np.pi
# #             data_2d1[mask_pos_2pi_1] -= 2 * np.pi

# #             mask_neg_2pi_1_ = np.isclose(data_2d1, - np.pi, atol=tol)
# #             mask_pos_2pi_1_ = np.isclose(data_2d1,  np.pi, atol=tol)

# #             data_2d1[mask_neg_2pi_1_] +=  np.pi
# #             data_2d1[mask_pos_2pi_1_] -= np.pi
# #             x = np.arange(data_3d1.shape[0])
# #             # Plot the data against x
# #             plt.plot(x, data_2d1, label=f'z={z}')

# #         plt.plot(x, line1, color='red', linestyle='--', marker='o', label='cos(4 pi x / L)')
# #         plt.plot(x, line2, color='blue', linestyle=':', marker='x', label='cos(8 pi x / L)')
# #         plt.xlabel('x')
# #         plt.ylabel('delta alpha')
# #         plt.legend()
# #         plt.show()

#         data_2d2 = data_3d2[:, :, z_plane]
#         # boundary_width = 1  # Define the width of the boundary to be masked
#         # data_2d2[:boundary_width, :] = 0
#         # data_2d2[-boundary_width:, :] = 0
#         # data_2d2[:, :boundary_width] = 0
#         # data_2d2[:, -boundary_width:] = 0
        
#         mask_neg_2pi_2 = np.isclose(data_2d2, -2 * np.pi, atol=tol)
#         mask_pos_2pi_2 = np.isclose(data_2d2, 2 * np.pi, atol=tol)
        
#         data_2d2[mask_neg_2pi_2] += 2 * np.pi
#         data_2d2[mask_pos_2pi_2] -= 2 * np.pi
        
#         mask_neg_2pi_2_ = np.isclose(data_2d2, - np.pi, atol=tol)
#         mask_pos_2pi_2_ = np.isclose(data_2d2,  np.pi, atol=tol)
        
#         data_2d2[mask_neg_2pi_2_] +=  np.pi
#         data_2d2[mask_pos_2pi_2_] -= np.pi
        
        
        
        
#         phi2t_plane = phi2t[:, :, z_plane]
        
#         # phase_corrected = np.where(np.abs(data_2d) > 2*np.pi, data_2d - 2*np.pi*np.sign(data_2d), data_2d)
#         # data_2d = np.where(np.abs(phase_corrected) > np.pi, phase_corrected + np.pi*np.sign(phase_corrected), phase_corrected)
#         if z_plane == 12:
#             # for radius in range(0, 49, 1):

#             #     y, x = np.ogrid[-50:51, -50:51]
#             #     circular_mask = x**2 + y**2 < radius**2

#             #     data_2d1 = np.where(circular_mask, 0, data_2d1)
#             #     data_2d2 = np.where(circular_mask, 0, data_2d2)
#             #     phi2t_plane = np.where(circular_mask, 0, phi2t_plane)


                
                
#             del_alpha = (data_2d2 - data_2d1)/delta_t
#             alpha = (alpha_t1 - alpha_t)/delta_t
#             alpha_slice = alpha[:,:,z_plane]
#             phi2_del2 = phi2t_plane * np.abs(del_alpha)**2
#             phi_del = np.sqrt(phi2_del2)
#             phi2_alpha2 = phi2t_plane * np.abs(alpha_slice)**2
#             phi_alpha = np.sqrt(phi2_alpha2)
#             for threshold in threshold_values: #this bit is new
                
#                 # ---------------------changed stuff after this point---------------------------
#                 mask = phi2t_plane < threshold

#                 # Apply the mask to the del_alpha and alpha_slice arrays
#                 del_alpha_masked = np.where(mask, 0, del_alpha)
#                 alpha_slice_masked = np.where(mask, 0, alpha_slice)
#                 phi_del_masked = np.where(mask, 0, phi_del)
#                 phi_alpha_masked = np.where(mask, 0, phi_alpha)

#                 # Calculate the power spectrum for the masked arrays
#                 P_values_del_alpha = calculate_power_spectrum(phi_del_masked, delta)
#                 P_values_alpha_slice = calculate_power_spectrum(phi_alpha_masked, delta)


                
#                 # P_values1 = calculate_power_spectrum(phi_del, delta)
#                 # P_values2 = calculate_power_spectrum(phi_alpha, delta)
#                 P_values1 = calculate_power_spectrum(data_2d1, delta)
#                 P_values2 = calculate_power_spectrum(phi_del, delta)
#                 P_values3 = calculate_power_spectrum(phi_alpha, delta)
                
#                 k_values_x, k_values_y = fftfreq2D(Nx, Ny, delta, delta) 
#                 mod_k = np.sqrt(k_values_x**2 + k_values_y**2)
                
                
#                 n_values = (mod_k * Nx * delta) / (2 * np.pi) # factor of 2 is added to account for the fact that the first bin took up both n = 1 and n = 2 modes.
#                 # plotting with power spectra
#                 fig, axs = plt.subplots(2, 2, figsize=(9, 9))  # Create a 2x2 grid of subplots

#                 # First plot
#                 axs[0, 0].hist(x=n_values.ravel(), weights=P_values_del_alpha.ravel(), bins=np.linspace(0.5,10.5,11), color='purple', edgecolor='black')
#                 # axs[0, 0].set_ylim([0, 0.2e-6])

#                 # Second plot
#                 im1 = axs[0, 1].contourf(phi_del_masked, cmap='plasma', levels=200 ) # Fix the colorbar range to [0, 1]
#                 axs[0, 1].set_title(r'$(\dot{\Delta \alpha}) |\phi|$ ')

#                 # Third plot
#                 axs[1, 0].hist(x=n_values.ravel(), weights=P_values_alpha_slice.ravel(), bins=np.linspace(0.5,10.5,11), color='purple', edgecolor='black')
#                 axs[1, 0].set_xlabel('n values')
#                 axs[1, 0].set_ylabel('Power Spectrum')
#                 # axs[1, 0].set_ylim([0, 0.2e-6])

#                 # Fourth plot
#                 im2 = axs[1, 1].contourf(phi_alpha_masked, cmap='plasma', levels=200 ) # Fix the colorbar range to [0, 1]
#                 axs[1, 1].set_title(r'$(\dot{\alpha}) |\phi|$ ')
#                 plt.suptitle(f'Threshold of phi mask : {threshold} ')
#                 plt.tight_layout()
#                 filename = "12th_z_plot_threshold_{:.4f}.png".format(threshold)
#                 frame_path = os.path.join(input_directory, filename)
#                 plt.savefig(frame_path)
#                 plt.show()
                
#                 # plotting without power spectra and only colorbars
#                 # fig, axs = plt.subplots(2, 2, figsize=(9, 9))  # Create a 2x2 grid of subplots

#                 # # First plot
#                 # axs[0, 0].remove()  # Remove the histogram plot
#                 # divider = make_axes_locatable(axs[0, 1])  # Create a divider for the colorbar
#                 # cax = divider.append_axes("right", size="5%", pad=0.05)  # Add an axis for the colorbar
#                 # im1 = axs[0, 1].contourf(data_2d1, cmap='PuOr', levels=np.linspace(-0.1,0.1,200))  # Create the contour plot
#                 # # axs[0, 1].set_title(r'$(\dot{\Delta \alpha}) |\phi|$ ')
#                 # fig.colorbar(im1, cax=cax)  # Add colorbar to the contour plot

#                 # # Second plot
#                 # axs[1, 0].remove()  # Remove the histogram plot
#                 # divider = make_axes_locatable(axs[1, 1])  # Create a divider for the colorbar
#                 # cax = divider.append_axes("right", size="5%", pad=0.05)  # Add an axis for the colorbar
#                 # im2 = axs[1, 1].contourf(data_2d2, cmap='PuOr', levels=np.linspace(-0.1,0.1,200))  # Create the contour plot
#                 # # axs[1, 1].set_title(r'$(\dot{\alpha}) |\phi|$ ')
#                 # fig.colorbar(im2, cax=cax)  # Add colorbar to the contour plot

#                 # plt.suptitle(f'Radius of circular mask : {radius} grid points')
#                 # plt.tight_layout()
#                 # plt.show()
                

#                 # Save the figure
#                 # filename = "12th_z_plot_radius_{:03d}.png".format(radius)
#                 # frame_path = os.path.join(input_directory, filename)
#                 # plt.savefig(frame_path)

#                 # # Close the figure to free up memory
#                 # plt.close(fig)
#                 # # # Extract histogram values for the first three bins
#                 # hist1, _ = np.histogram(n_values.ravel(), weights=P_values1.ravel(), bins=np.linspace(0.5, 10.5, 11))
#                 # hist2, _ = np.histogram(n_values.ravel(), weights=P_values2.ravel(), bins=np.linspace(0.5, 10.5, 11))
#                 # hist3, _ = np.histogram(n_values.ravel(), weights=P_values3.ravel(), bins=np.linspace(0.5, 10.5, 11))

#                 # # # Append the first three bin values to the respective lists
#                 # bin_values_1.append(hist1[:3])
#                 # bin_values_2.append(hist2[:3])
#                 # bin_values_3.append(hist3[:3])
                
#                                 # Extract histogram values for the first three bins
#                 hist_del_alpha, _ = np.histogram(n_values.ravel(), weights=P_values_del_alpha.ravel(), bins=np.linspace(0.5, 10.5, 11))
#                 hist_alpha_slice, _ = np.histogram(n_values.ravel(), weights=P_values_alpha_slice.ravel(), bins=np.linspace(0.5, 10.5, 11))

#                 # Append the first three bin values to the respective lists
#                 bin_values_del_alpha.append(hist_del_alpha[:3])
#                 bin_values_alpha_slice.append(hist_alpha_slice[:3])

#             # Convert lists to numpy arrays for easier manipulation
#             # bin_values_1 = np.array(bin_values_1)
#             # bin_values_2 = np.array(bin_values_2)
#             # bin_values_3 = np.array(bin_values_3)
#             bin_values_del_alpha = np.array(bin_values_del_alpha)
#             bin_values_alpha_slice = np.array(bin_values_alpha_slice)

#             # Plotting
#             radii = threshold_values

#             # Plot for the first power spectrum
#             fig, axs = plt.subplots(1, 2, figsize=(10, 5))

#             # Plot for the first power spectrum
#             # axs[0].plot(radii, bin_values_1[:, 0], label='Bin 1')
#             # axs[0].plot(radii, bin_values_1[:, 1], label='Bin 2')
#             # axs[0].plot(radii, bin_values_1[:, 2], label='Bin 3')
#             # axs[0].set_xlabel('Radius')
#             # axs[0].set_ylabel('Bin Values')
#             # axs[0].set_title(r'$(\dot{\Delta \alpha}) |\phi|$ Power Spectrum')
#             # axs[0].legend()

#             # Plot for the second power spectrum
#             # axs[1].plot(radii, bin_values_2[:, 0], label='Bin 1')
#             axs[0].plot(radii, bin_values_del_alpha[:, 1], label=r'$(\dot{\Delta \alpha}) |\phi|$ Power Spectrum')
#             axs[0].plot(radii, bin_values_alpha_slice[:, 1], label=r'$(\dot{ \alpha}) |\phi|$ Power Spectrum')
#             axs[0].set_xlabel('Radius')
#             axs[0].set_ylabel('Bin Values')
#             axs[0].set_title('Bin 2')
#             axs[0].legend()

#             # Plot for the third power spectrum
#             # axs[2].plot(radii, bin_values_3[:, 0], label='Bin 1')
#             axs[1].plot(radii, bin_values_del_alpha[:, 2], label=r'$(\dot{\Delta \alpha}) |\phi|$ Power Spectrum')
#             axs[1].plot(radii, bin_values_alpha_slice[:, 2], label=r'$(\dot{ \alpha}) |\phi|$ Power Spectrum')
#             axs[1].set_xlabel('Radius')
#             axs[1].set_ylabel('Bin Values')
#             axs[1].set_title('Bin 3')
#             axs[1].legend()

#             plt.tight_layout()
#             plt.show()



# # --------------------------------------------------------------------------------------------------------------------------------------------------
# # def plot_power_spectrum(Nx, Ny, delta, data_3d1, data_3d2, phi2t, alpha_t,alpha_t1,custom_bins=None):
# #     for z_plane in range(data_3d1.shape[2]):
# #         x = np.arange(data_3d1.shape[0])
# #         data_2d1 = data_3d1[:, :, z_plane]
        
# #         mask_neg_2pi_1 = np.isclose(data_2d1, -2 * np.pi, atol=tol)
# #         mask_pos_2pi_1 = np.isclose(data_2d1, 2 * np.pi, atol=tol)
        
# #         data_2d1[mask_neg_2pi_1] += 2 * np.pi
# #         data_2d1[mask_pos_2pi_1] -= 2 * np.pi
        
# #         mask_neg_2pi_1_ = np.isclose(data_2d1, - np.pi, atol=tol)
# #         mask_pos_2pi_1_ = np.isclose(data_2d1,  np.pi, atol=tol)
        
# #         data_2d1[mask_neg_2pi_1_] +=  np.pi
# #         data_2d1[mask_pos_2pi_1_] -= np.pi
        
# #         # line1 = cos1(x)
# #         # line2 = cos2(x)
# #         Nx, Ny = data_3d1.shape[0], data_3d1.shape[1]

# #         # Create a 2D grid of x values
# #         x_grid = np.tile(np.arange(Nx), (Ny, 1))

# #         # Calculate line1 and line2 for all y values
# #         line1_2d = cos1(x_grid)
# #         line2_2d = cos2(x_grid)
# #         line3_2d = cos3(x_grid)
# #         line4_2d = cos4(x_grid)
# #         line5_2d = cos5(x_grid)
# #         line6_2d = cos6(x_grid)
# #         y_index = 125  # The y index for which you want to plot the data

# # #         plt.figure(figsize=(10, 6))

# # # # Loop over the z slices
# # #         for z in range(1, data_3d1.shape[2], 5):
# # #             # Extract the data for y=125 for the current z slice
# # #             data_2d1 = data_3d1[:, y_index, z]

# # #             # Apply phase correction
# # #             mask_neg_2pi_1 = np.isclose(data_2d1, -2 * np.pi, atol=tol)
# # #             mask_pos_2pi_1 = np.isclose(data_2d1, 2 * np.pi, atol=tol)

# # #             data_2d1[mask_neg_2pi_1] += 2 * np.pi
# # #             data_2d1[mask_pos_2pi_1] -= 2 * np.pi

# # #             mask_neg_2pi_1_ = np.isclose(data_2d1, - np.pi, atol=tol)
# # #             mask_pos_2pi_1_ = np.isclose(data_2d1,  np.pi, atol=tol)

# # #             data_2d1[mask_neg_2pi_1_] +=  np.pi
# # #             data_2d1[mask_pos_2pi_1_] -= np.pi
# # #             x = np.arange(data_3d1.shape[0])
# # #             # Plot the data against x
# # #             plt.plot(x, data_2d1, label=f'z={z}')

# # #         plt.plot(x, line1, color='red', linestyle='--', marker='o', label='cos(4 pi x / L)')
# # #         plt.plot(x, line2, color='blue', linestyle=':', marker='x', label='cos(8 pi x / L)')
# # #         plt.xlabel('x')
# # #         plt.ylabel('delta alpha')
# # #         plt.legend()
# # #         plt.show()

# #         data_2d2 = data_3d2[:, :, z_plane]
        
# #         mask_neg_2pi_2 = np.isclose(data_2d2, -2 * np.pi, atol=tol)
# #         mask_pos_2pi_2 = np.isclose(data_2d2, 2 * np.pi, atol=tol)
        
# #         data_2d2[mask_neg_2pi_2] += 2 * np.pi
# #         data_2d2[mask_pos_2pi_2] -= 2 * np.pi
        
# #         mask_neg_2pi_2_ = np.isclose(data_2d2, - np.pi, atol=tol)
# #         mask_pos_2pi_2_ = np.isclose(data_2d2,  np.pi, atol=tol)
        
# #         data_2d2[mask_neg_2pi_2_] +=  np.pi
# #         data_2d2[mask_pos_2pi_2_] -= np.pi
        
        
        
        
# #         phi2t_plane = phi2t[:, :, z_plane]
        
# #         # phase_corrected = np.where(np.abs(data_2d) > 2*np.pi, data_2d - 2*np.pi*np.sign(data_2d), data_2d)
# #         # data_2d = np.where(np.abs(phase_corrected) > np.pi, phase_corrected + np.pi*np.sign(phase_corrected), phase_corrected)

# #         data_2d1 = np.where(circular_mask, 0, data_2d1)
# #         data_2d2 = np.where(circular_mask, 0, data_2d2)
# #         phi2t_plane = np.where(circular_mask, 0, phi2t_plane)
        
        
# #         del_alpha = (data_2d2 - data_2d1)/delta_t
# #         phi2_del2 = phi2t_plane * np.abs(del_alpha)**2
        
        
# #         P_values1 = calculate_power_spectrum(line1_2d, delta)
# #         P_values2 = calculate_power_spectrum(line2_2d, delta)
# #         P_values3 = calculate_power_spectrum(line3_2d, delta)
# #         P_values4 = calculate_power_spectrum(line4_2d, delta)
# #         P_values5 = calculate_power_spectrum(line5_2d, delta)
# #         P_values6 = calculate_power_spectrum(line6_2d, delta)
        
# #         # Calculate the Fourier transform
        


# #         beta1, beta2 = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing='ij')
# #         beta1_flat = beta1.ravel()
# #         beta2_flat = beta2.ravel()

# #         # k_values = np.sqrt((2 * np.pi * beta1_flat) / (Nx * delta))**2 + \
# #         #            np.sqrt((2 * np.pi * beta2_flat) / (Ny * delta))**2
# #         k_values_x, k_values_y = fftfreq2D(Nx, Ny, delta, delta) 
# #         mod_k = np.sqrt(k_values_x**2 + k_values_y**2)

# #         n_values = (mod_k * Nx * delta) / (2 * np.pi)
        
# #         F_values1 = calculate_fourier_transform(line1_2d, delta)
# #         F_values2 = calculate_fourier_transform(line2_2d, delta)
# #         F_values3 = calculate_fourier_transform(line3_2d, delta)
# #         F_values4 = calculate_fourier_transform(line4_2d, delta)
# #         F_values5 = calculate_fourier_transform(line5_2d, delta)

# #         # Create a 2D grid
# #         nx, ny = n_values.shape
# # # Create a 2D grid for each Fourier transform
# #         fig, axs = plt.subplots(1, 5, figsize=(15, 3))

# #         # Plot the Fourier transform values
# #         axs[0].imshow(np.abs(F_values1), cmap='hot')
# #         axs[0].set_title('4')
# #         axs[1].imshow(np.abs(F_values2), cmap='hot')
# #         axs[1].set_title('8')
# #         axs[2].imshow(np.abs(F_values3), cmap='hot')
# #         axs[2].set_title('16')
# #         axs[3].imshow(np.abs(F_values4), cmap='hot')
# #         axs[3].set_title('32')
# #         axs[4].imshow(np.abs(F_values5), cmap='hot')
# #         axs[4].set_title('64')

# #         plt.show()


        
# #         # logic1 = (n_values > 0.5) & (n_values<1.5)
# #         # logic12= (n_values > 1.5) & (n_values<2.5)

# #         # print(np.sum(P_values[logic1]), np.sum(P_values[logic12]))


# #         plt.figure(figsize=(15, 10))

# #         # Plot the first histogram
# #         plt.subplot(2, 3, 1)
# #         plt.hist(x=n_values.ravel(), weights=P_values1.ravel(), bins=np.linspace(0.5,10.5,11), color='yellow', edgecolor='black', alpha=0.5)
# #         plt.xlabel('n values')
# #         plt.ylabel('Power Spectrum')
# #         plt.title('cos(4 pi x / L)')

# #         # Plot the second histogram
# #         plt.subplot(2, 3, 2)
# #         plt.hist(x=n_values.ravel(), weights=P_values2.ravel(), bins=np.linspace(0.5,10.5,11), color='purple', edgecolor='black', alpha=0.5)
# #         plt.xlabel('n values')
# #         plt.ylabel('Power Spectrum')
# #         plt.title('cos(8 pi x / L)')

# #         # Plot the third histogram
# #         plt.subplot(2, 3, 3)
# #         plt.hist(x=n_values.ravel(), weights=P_values3.ravel(), bins=np.linspace(0.5,10.5,11), color='pink', edgecolor='black', alpha=0.5)
# #         plt.xlabel('n values')
# #         plt.ylabel('Power Spectrum')
# #         plt.title('cos(16 pi x / L)')

# #         # Plot the fourth histogram
# #         plt.subplot(2, 3, 4)
# #         plt.hist(x=n_values.ravel(), weights=P_values4.ravel(), bins=np.linspace(0.5,10.5,11), color='blue', edgecolor='black', alpha=0.5)
# #         plt.xlabel('n values')
# #         plt.ylabel('Power Spectrum')
# #         plt.title('cos(32 pi x / L)')

# #         # Plot the fifth histogram
# #         plt.subplot(2, 3, 6)
# #         plt.hist(x=n_values.ravel(), weights=P_values5.ravel(), bins=np.linspace(0.5,10.5,11), color='green', edgecolor='black', alpha=0.5)
# #         plt.xlabel('n values')
# #         plt.ylabel('Power Spectrum')
# #         plt.title('cos(64 pi x / L)')

# #         # Plot the sixth histogram
# #         # plt.subplot(2, 3, 6)
# #         # plt.hist(x=n_values.ravel(), weights=P_values6.ravel(), bins=np.linspace(0.5,10.5,11), color='red', edgecolor='black', alpha=0.5)
# #         # plt.xlabel('n values')
# #         # plt.ylabel('Power Spectrum')
# #         # plt.title('cos(128 pi x / L)')

# #         plt.tight_layout()

# #         power_path = os.path.join(input_directory, f"power_spec_xy_plane_{z_plane + 1}.png")
# #         # plt.savefig(power_path, dpi=300)
# #         plt.show()

# multiple z plane plotting ----------------------------------------------------------


def plot_power_spectrum(Nx, Ny, delta, data_3d1, data_3d2, phi2t,phi2reft, alpha_t,alpha_t1, time,custom_bins=None):
    bin_values_1 = []
    bin_values_2 = []
    bin_values_3 = []
    threshold_values = np.linspace(0.991, 1, 10)

# Initialize lists to store the bin values for each threshold value
    bin_values_del_alpha = []
    bin_values_alpha_slice = []
    bin_values_del_alpha_all = []
    bin_values_alpha_slice_all = []
    z_planes_list = [12]
    for z_plane in z_planes_list:
        x = np.arange(data_3d1.shape[0])
        data_2d1 = data_3d1[:, :, z_plane]
        boundary_width = 1  # Define the width of the boundary to be masked
        data_2d1[:boundary_width, :] = 0
        data_2d1[-boundary_width:, :] = 0
        data_2d1[:, :boundary_width] = 0
        data_2d1[:, -boundary_width:] = 0
        

        
        # mask_neg_2pi_1 = np.isclose(data_2d1, -2 * np.pi, atol=tol)
        # mask_pos_2pi_1 = np.isclose(data_2d1, 2 * np.pi, atol=tol)
        
        # data_2d1[mask_neg_2pi_1] += 2 * np.pi
        # data_2d1[mask_pos_2pi_1] -= 2 * np.pi
        
        # mask_neg_2pi_1_ = np.isclose(data_2d1, - np.pi, atol=tol)
        # mask_pos_2pi_1_ = np.isclose(data_2d1,  np.pi, atol=tol)
        
        # data_2d1[mask_neg_2pi_1_] +=  np.pi
        # data_2d1[mask_pos_2pi_1_] -= np.pi
        
        # line1 = cos1(x)
        # line2 = cos2(x)
        
        y_index = 125  # The y index for which you want to plot the data

#         plt.figure(figsize=(10, 6))

# # Loop over the z slices
#         for z in range(1, data_3d1.shape[2], 5):
#             # Extract the data for y=125 for the current z slice
#             data_2d1 = data_3d1[:, y_index, z]

#             # Apply phase correction
#             mask_neg_2pi_1 = np.isclose(data_2d1, -2 * np.pi, atol=tol)
#             mask_pos_2pi_1 = np.isclose(data_2d1, 2 * np.pi, atol=tol)

#             data_2d1[mask_neg_2pi_1] += 2 * np.pi
#             data_2d1[mask_pos_2pi_1] -= 2 * np.pi

#             mask_neg_2pi_1_ = np.isclose(data_2d1, - np.pi, atol=tol)
#             mask_pos_2pi_1_ = np.isclose(data_2d1,  np.pi, atol=tol)

#             data_2d1[mask_neg_2pi_1_] +=  np.pi
#             data_2d1[mask_pos_2pi_1_] -= np.pi
#             x = np.arange(data_3d1.shape[0])
#             # Plot the data against x
#             plt.plot(x, data_2d1, label=f'z={z}')

#         plt.plot(x, line1, color='red', linestyle='--', marker='o', label='cos(4 pi x / L)')
#         plt.plot(x, line2, color='blue', linestyle=':', marker='x', label='cos(8 pi x / L)')
#         plt.xlabel('x')
#         plt.ylabel('delta alpha')
#         plt.legend()
#         plt.show()

        data_2d2 = data_3d2[:, :, z_plane]
        boundary_width = 1  # Define the width of the boundary to be masked
        data_2d2[:boundary_width, :] = 0
        data_2d2[-boundary_width:, :] = 0
        data_2d2[:, :boundary_width] = 0
        data_2d2[:, -boundary_width:] = 0
        
        # mask_neg_2pi_2 = np.isclose(data_2d2, -2 * np.pi, atol=tol)
        # mask_pos_2pi_2 = np.isclose(data_2d2, 2 * np.pi, atol=tol)
        
        # data_2d2[mask_neg_2pi_2] += 2 * np.pi
        # data_2d2[mask_pos_2pi_2] -= 2 * np.pi
        
        # mask_neg_2pi_2_ = np.isclose(data_2d2, - np.pi, atol=tol)
        # mask_pos_2pi_2_ = np.isclose(data_2d2,  np.pi, atol=tol)
        
        # data_2d2[mask_neg_2pi_2_] +=  np.pi
        # data_2d2[mask_pos_2pi_2_] -= np.pi
        
        
        
        
        phi2t_plane = phi2t[:, :, z_plane]
        phi2reft_plane = phi2reft[:, :, z_plane]

        
        # phase_corrected = np.where(np.abs(data_2d) > 2*np.pi, data_2d - 2*np.pi*np.sign(data_2d), data_2d)
        # data_2d = np.where(np.abs(phase_corrected) > np.pi, phase_corrected + np.pi*np.sign(phase_corrected), phase_corrected)
        # if z_plane == 87:
        

    # Initialize lists to store bin contributions for the current z plane


        for radius in range(10, 25, 1):

            y, x = np.ogrid[-100:101, -100:101]
            circular_mask = x**2 + y**2 < radius**2

            data_2d1 = np.where(circular_mask, 0, data_2d1)
            data_2d2 = np.where(circular_mask, 0, data_2d2)
            phi2t_plane = np.where(circular_mask, 0, phi2t_plane)
            phi2t_plane_small = phi2t_plane[Nx//2-25:Nx//2+25, Ny//2-25:Ny//2+25]
            
            phi2reft_plane = np.where(circular_mask, 0, phi2reft_plane)
            phi2reft_plane_small = phi2reft_plane[Nx//2-25:Nx//2+25, Ny//2-25:Ny//2+25]
            
        # for threshold in threshold_values: #this bit is new

            
            
            del_alpha = (data_2d2 - data_2d1)/delta_t
            alpha_diff_slice = alpha_t1[:, :, z_plane] - alpha_t[:, :, z_plane]
            mask_neg_2pi_2 = np.isclose(alpha_diff_slice, -2 * np.pi, atol=tol)
            mask_pos_2pi_2 = np.isclose(alpha_diff_slice, 2 * np.pi, atol=tol)
            
            alpha_diff_slice[mask_neg_2pi_2] += 2 * np.pi
            alpha_diff_slice[mask_pos_2pi_2] -= 2 * np.pi
            
            mask_neg_2pi_2_ = np.isclose(alpha_diff_slice, - np.pi, atol=tol)
            mask_pos_2pi_2_ = np.isclose(alpha_diff_slice,  np.pi, atol=tol)
            
            alpha_diff_slice[mask_neg_2pi_2_] +=  np.pi
            alpha_diff_slice[mask_pos_2pi_2_] -= np.pi
            
            alpha_slice = alpha_diff_slice/delta_t
            # alpha = (alpha_t1 - alpha_t)/delta_t
            # alpha_slice = alpha[:,:,z_plane]
            phi2_del2 = phi2t_plane * np.abs(del_alpha)**2
            phi_del = np.sqrt(phi2_del2)
            phi2_alpha2 = phi2t_plane * np.abs(alpha_slice)**2
            phi_alpha = np.sqrt(phi2_alpha2)
            
            # ---------------------changed stuff after this point---------------------------
            # mask = phi2t_plane < threshold

            # # Apply the mask to the del_alpha and alpha_slice arrays
            # del_alpha_masked = np.where(mask, 0, phi_del)
            # alpha_slice_masked = np.where(mask, 0, phi_alpha)

            # # Calculate the power spectrum for the masked arrays
            # P_values_del_alpha = calculate_power_spectrum(del_alpha_masked, delta)
            # P_values_alpha_slice = calculate_power_spectrum(alpha_slice_masked, delta)


            
            # P_values1 = calculate_power_spectrum(data_2d1, delta)
            # P_values2 = calculate_power_spectrum(data_2d2, delta)
            # P_values1 = calculate_power_spectrum(data_2d1, delta)
            # P_values2 = calculate_power_spectrum(data_2d2, delta)
            
            P_values1 = calculate_power_spectrum(phi2t_plane, delta)
            P_values2 = calculate_power_spectrum(phi2reft_plane, delta)
            
            k_values_x, k_values_y = fftfreq2D(Nx, Ny, delta, delta) 
            mod_k = np.sqrt(k_values_x**2 + k_values_y**2)
            
            
            n_values = (mod_k * Nx * delta) / (2 * np.pi) # factor of 2 is added to account for the fact that the first bin took up both n = 1 and n = 2 modes.
            # plotting with power spectra
            # -------------------------4 plot-version-------------------------------------------------------------
            # fig, axs = plt.subplots(2, 2, figsize=(9, 9))  # Create a 2x2 grid of subplots

            # # First plot
            # axs[0, 0].hist(x=n_values.ravel(), weights=P_values1.ravel(), bins=np.linspace(0.5,10.5,11), color='purple', edgecolor='black')
            # axs[0, 0].set_ylim([0, 0.5e-5])

            # # Second plot
            # im1 = axs[0, 1].contourf(phi_del, cmap='plasma', levels=np.linspace(0,0.02,200) ) # Fix the colorbar range to [0, 1]
            # # axs[0, 1].set_title(r'$|\phi| \partial_i \Delta \alpha \hat{r_i}$ ')
            # axs[0, 1].set_title(r'$|\phi| \dot{\Delta \alpha }$ ')

            # # Third plot
            # axs[1, 0].hist(x=n_values.ravel(), weights=P_values2.ravel(), bins=np.linspace(0.5,10.5,11), color='purple', edgecolor='black')
            # axs[1, 0].set_xlabel('n values')
            # axs[1, 0].set_ylabel('Power Spectrum')
            # axs[1, 0].set_ylim([0, 0.5e-5])

            # # Fourth plot
            # im2 = axs[1, 1].contourf(phi_alpha, cmap='plasma', levels=np.linspace(0,0.02,200) ) # Fix the colorbar range to [0, 1]
            # axs[1, 1].set_title(r'$ |\phi| \dot{\alpha} $ ')
            # -------------------------3 plot-version-------------------------------------------------------------
            
            fig, axs = plt.subplots(1, 2, figsize=(12,5))  # Create a 2x2 grid of subplots

            # First plot


            # Second plot
            # im1 = axs[1].contourf(del_alpha_masked, cmap='plasma', levels=np.linspace(0,0.02,200) ) # Fix the colorbar range to [0, 1]
            
            # Flipped axis


            
            umin, umax = -0.05, 0.05  # Ensure the range is symmetric around 0
            im1 = axs[0].contourf(data_2d1.T, cmap='PuOr', levels=np.linspace(umin,umax,200))
            axs[0].set_title(r'$\hat{r}^i \mathcal{D}_i \alpha$ ')
            # Add the colorbar
            cbar = plt.colorbar(im1, ax=axs[1], orientation='vertical', pad=0.05)

            # Define the tick positions as multiples of pi
            tick_positions = np.linspace(umin, umax, 11)  # Adjust the number of ticks as needed



            # Apply the custom tick positions and formatter
            cbar.set_ticks(tick_positions)

            
            def custom_ticks(x, pos):
                return f"{(x  - 100)*0.7:.1f}"
            def custom_zoom_ticks(x, pos):
                return f"{(x  - 25)*0.7:.1f}"
            
            def custom_zoom_ticks(x, pos):
                return f"{(x - 25)*0.7:.1f}" 

            # Apply the custom tick formatter to x and y axes
            axs[0].xaxis.set_major_locator(MaxNLocator(nbins=4))
            axs[0].yaxis.set_major_locator(MaxNLocator(nbins=4))
            axs[0].xaxis.set_major_formatter(FuncFormatter(custom_ticks))
            axs[0].yaxis.set_major_formatter(FuncFormatter(custom_ticks))
            
            # axs[0].remove()  # Remove the histogram plot
            # axs[1].remove()
            # divider = make_axes_locatable(axs[1])  # Create a divider for the colorbar
            # cax = divider.append_axes("right", size="5%", pad=0.05)  # Add an axis for the colorbar

            axs[0].set_xlabel(r'$x$', fontsize=14)
            axs[0].set_ylabel(r'$y$', fontsize=14)
            # axs[0, 1].set_title(r'$|\phi| \partial_i \Delta \alpha \hat{r_i}$ ')
            # axs[1].set_title(r'$\mathcal{P}_{\alpha}$ ')
            # axs[2].set_title(r'$\hat{r}^i \mathcal{D} \alpha$ ')
            # axs[1].set_title(r'$|\phi| \dot{\Delta \alpha }$')
            # axs[2].set_title(r'$|\phi| \dot{\alpha }$ ')

            # Third plot
            # axs[0].hist(x=n_values.ravel(), weights=P_values1.ravel(), bins=np.linspace(0.5,10.5,11), color='blue', edgecolor='blue', alpha = 0.3, label = r'$|\phi| \dot{\Delta \alpha}$ ')
            # # axs[0].hist(x=n_values.ravel(), weights=P_values1.ravel(), bins=np.linspace(0.5,10.5,11), color='blue', edgecolor='blue', alpha = 0.3, label = r'$\mathcal{P}_{\alpha}$ ')
            # axs[0].set_ylim([0, 2.5e-5])
            # axs[0].hist(x=n_values.ravel(), weights=P_values2.ravel(), bins=np.linspace(0.5,10.5,11), color='red', edgecolor='red', alpha = 0.1, label = r'$|\phi| \dot{\alpha}$ ')
            # # axs[0].hist(x=n_values.ravel(), weights=P_values2.ravel(), bins=np.linspace(0.5,10.5,11), color='red', edgecolor='red', alpha = 0.1, label = r'$\hat{r}^i \mathcal{D} \alpha$ ')
            # axs[0].legend(fontsize=14)
            # axs[0].set_xlabel('n values', fontsize=14)
            # axs[0].set_ylabel('Power Spectrum', fontsize=14)
            # axs[0].tick_params(axis='both', which='major', labelsize=14)
            axs[0].tick_params(axis='both', which='major', labelsize=14)
            axs[1].tick_params(axis='both', which='major', labelsize=14)
            # axs[1, 0].set_ylim([0, 0.5e-5])

            # Fourth plot

            
            vmin, vmax = -0.05, 0.05  # Ensure the range is symmetric around 0
            im2 = axs[1].contourf(data_2d2.T, cmap='PuOr', levels=np.linspace(vmin, vmax, 200))

            
            axs[1].set_title(r'$\phi \hat{r}^i  \partial_i \Delta \alpha $')
            # Add the colorbar
            # cbar = plt.colorbar(im2, ax=axs[1], orientation='vertical', pad=0.05)

            # Define the tick positions as multiples of pi
            # tick_positions = np.linspace(vmin, vmax, 11)  # Adjust the number of ticks as needed

            # Define a custom tick formatter for multiples of pi
            def pi_formatter(x, pos):
                return f"{x/np.pi:.1f}" + r"$\pi$"

            # Apply the custom tick positions and formatter
            # cbar.set_ticks(tick_positions)
            # cbar.set_ticklabels([pi_formatter(tick, None) for tick in tick_positions])
                        
            axs[1].xaxis.set_major_locator(MaxNLocator(nbins=4))
            axs[1].yaxis.set_major_locator(MaxNLocator(nbins=4))
            axs[1].xaxis.set_major_formatter(FuncFormatter(custom_ticks))
            axs[1].yaxis.set_major_formatter(FuncFormatter(custom_ticks))
            # axs[0].set_xlim(5, 45)
            # axs[0].set_ylim(5, 45)
            # axs[1].set_xlim(5, 45)
            # axs[1].set_ylim(5, 45)
            axs[1].set_xlabel(r'$x$', fontsize=14)
            axs[1].set_ylabel(r'$y$', fontsize=14)

            axs[0].set_aspect('equal')
            axs[1].set_aspect('equal')
            # axs[1].set_title(r'$ |\phi| \dot{\alpha} $ ')
            # cbar = fig.colorbar(im2, cax=cax)
            # ticks=[-0.30, -0.20, -0.10, 0.00, 0.10, 0.20, 0.30 ]
            # cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
            # fig.colorbar(im1, cax=cbar_ax)
            
            # plt.suptitle(f'Threshold Mask value : {threshold:.4f} physical spatial units')
            # plt.suptitle(f'Radius Mask value : {(radius*0.7):.3f} physical spatial units')
            plt.tight_layout()
    
            
            # plotting without power spectra and only colorbars
            # fig, axs = plt.subplots(2, 2, figsize=(9, 9))  # Create a 2x2 grid of subplots

            # # First plot
            # axs[0, 0].remove()  # Remove the histogram plot
            # divider = make_axes_locatable(axs[0, 1])  # Create a divider for the colorbar
            # cax = divider.append_axes("right", size="5%", pad=0.05)  # Add an axis for the colorbar
            # im1 = axs[0, 1].contourf(data_2d1, cmap='PuOr', levels=np.linspace(-0.1,0.1,200))  # Create the contour plot
            # # axs[0, 1].set_title(r'$(\dot{\Delta \alpha}) |\phi|$ ')
            # fig.colorbar(im1, cax=cax)  # Add colorbar to the contour plot

            # # Second plot
            # axs[1, 0].remove()  # Remove the histogram plot
            # divider = make_axes_locatable(axs[1, 1])  # Create a divider for the colorbar
            # cax = divider.append_axes("right", size="5%", pad=0.05)  # Add an axis for the colorbar
            # im2 = axs[1, 1].contourf(data_2d2, cmap='PuOr', levels=np.linspace(-0.1,0.1,200))  # Create the contour plot
            # # axs[1, 1].set_title(r'$(\dot{\alpha}) |\phi|$ ')
            # fig.colorbar(im2, cax=cax)  # Add colorbar to the contour plot
            # # Update the levels parameter in the contourf function to specify the desired colorbar ticks
            # im2 = axs[1, 1].contourf(data_2d2, cmap='PuOr', levels=np.linspace(-0.1, 0.1, 200))
            # # Set the colorbar ticks to the desired values
            # cbar = fig.colorbar(im2, cax=cax, ticks=[0.10, 0.06, 0.02, -0.02, -0.06, -0.10])

            # # plt.suptitle(f'Radius of circular mask : {radius} grid points')
            # plt.tight_layout()
            plt.show()
            

            # Save the figure
            # filename = "12th_z_plot_radius_{:03f}.png".format(time)
            # frame_path = os.path.join(input_directory, filename)
            # # plt.show()
            # plt.savefig(frame_path)
            

#             # # Close the figure to free up memory
            # plt.close(fig)
#             # # # Extract histogram values for the first three bins
#             # hist1, _ = np.histogram(n_values.ravel(), weights=P_values1.ravel(), bins=np.linspace(0.5, 10.5, 11))
#             # hist2, _ = np.histogram(n_values.ravel(), weights=P_values2.ravel(), bins=np.linspace(0.5, 10.5, 11))
#             # hist3, _ = np.histogram(n_values.ravel(), weights=P_values3.ravel(), bins=np.linspace(0.5, 10.5, 11))

#             # # # Append the first three bin values to the respective lists
#             # bin_values_1.append(hist1[:3])
#             # bin_values_2.append(hist2[:3])
#             # bin_values_3.append(hist3[:3])
            
#                             # Extract histogram values for the first three bins
#         #     hist_del_alpha, _ = np.histogram(n_values.ravel(), weights=P_values_del_alpha.ravel(), bins=np.linspace(0.5, 10.5, 11))
#         #     hist_alpha_slice, _ = np.histogram(n_values.ravel(), weights=P_values_alpha_slice.ravel(), bins=np.linspace(0.5, 10.5, 11))

#         # #     # Append the first three bin values to the respective lists
#         #     bin_values_del_alpha.append(hist_del_alpha[:3])
#         #     bin_values_alpha_slice.append(hist_alpha_slice[:3])
            
#         #                     # Convert lists to numpy arrays and append to the overall lists
#         # bin_values_del_alpha_all.append(bin_values_del_alpha)
#         # bin_values_alpha_slice_all.append(bin_values_alpha_slice)

#         # Convert lists to numpy arrays for easier manipulation
#         # bin_values_1 = np.array(bin_values_1)
#         # bin_values_2 = np.array(bin_values_2)
#         # bin_values_3 = np.array(bin_values_3)
#     bin_values_del_alpha = np.array(bin_values_del_alpha)
#     bin_values_alpha_slice = np.array(bin_values_alpha_slice)


#     # Plotting


#     # Plot for the first power spectrum


#     # Plot for the first power spectrum
#     # axs[0].plot(radii, bin_values_1[:, 0], label='Bin 1')
#     # axs[0].plot(radii, bin_values_1[:, 1], label='Bin 2')
#     # axs[0].plot(radii, bin_values_1[:, 2], label='Bin 3')
#     # axs[0].set_xlabel('Radius')
#     # axs[0].set_ylabel('Bin Values')
#     # axs[0].set_title(r'$(\dot{\Delta \alpha}) |\phi|$ Power Spectrum')
#     # axs[0].legend()

#     # Plot for the second power spectrum
#     # axs[1].plot(radii, bin_values_2[:, 0], label='Bin 1')
# # Plot for the first power spectrum
#     # bin_values_del_alpha_all = np.array(bin_values_del_alpha_all)
#     # bin_values_alpha_slice_all = np.array(bin_values_alpha_slice_all)

#     # Plotting
#     radii = np.arange(0, 99, 1)

#     # Plot for the second and third bin contributions from all z planes
#     # fig, axs = plt.subplots(1, 2, figsize=(10, 5))

#     # Plot for the second bin contributions
#     for i in range(len(z_planes_list)):
#         block_size = 99 # 49 for 101, 99 for 201 and so on....
        
#         # Extract data for the current z plane
#         data_del_alpha = bin_values_del_alpha[i * block_size: (i + 1) * block_size]
#         data_alpha_slice = bin_values_alpha_slice[i *block_size: (i + 1) * block_size]
#         color = cmap(i / len(z_planes_list))

#         # Plot the second bin contributions for the current z plane
#         for j in range(1,2):  # Plot each column separately
#             # axs[0].plot(radii, data_del_alpha[:, j] - data_alpha_slice[:, j], label=f'Z Plane {z_planes_list[i]} ' + r'$(\dot{\Delta \alpha}) |\phi| - (\dot{ \alpha}) |\phi|$', color=color)

#             # axs[0].plot(radii*0.35, data_del_alpha[:, j], label=f'Z Plane {z_planes_list[i]} ' + r'$(\dot{\Delta \alpha}) |\phi|$', color=color)
#             # axs[0].plot(radii*0.35, data_alpha_slice[:, j], linestyle='--', label=f'Z Plane {z_planes_list[i]} ' + r'$(\dot{ \alpha}) |\phi|$', color=color)
#             np.savetxt(f"{input_directory}201_data_del_alpha_{j}.txt", data_del_alpha[:,j])
#             np.savetxt(f"{input_directory}201_data_alpha_slice_{j}.txt", data_alpha_slice[:,j])
        
#         for j in range(2,3):  # Plot each column separately
#             # axs[0].plot(radii, data_del_alpha[:, j] - data_alpha_slice[:, j], label=f'Z Plane {z_planes_list[i]} ' + r'$(\dot{\Delta \alpha}) |\phi| - (\dot{ \alpha}) |\phi|$', color=color)

#             # axs[0].plot(radii*0.35, data_del_alpha[:, j], label=f'Z Plane {z_planes_list[i]} ' + r'$(\dot{\Delta \alpha}) |\phi|$', color=color)
#             # axs[0].plot(radii*0.35, data_alpha_slice[:, j], linestyle='--', label=f'Z Plane {z_planes_list[i]} ' + r'$(\dot{ \alpha}) |\phi|$', color=color)
#             np.savetxt(f"{input_directory}201_data_del_alpha_{j}.txt", data_del_alpha[:,j])
#             np.savetxt(f"{input_directory}201_data_alpha_slice_{j}.txt", data_alpha_slice[:,j])

#     # axs[0].set_xlabel('Radius')
#     # axs[0].set_ylabel('Bin Values')
#     # axs[0].set_title('Second Bin Contributions for 201 grid')
#     # axs[0].legend()

#     # Plot for the third bin contributions
#     for i in range(len(z_planes_list)):
#         # Extract data for the current z plane
#         data_del_alpha = bin_values_del_alpha[i * block_size: (i + 1) * block_size]
#         data_alpha_slice = bin_values_alpha_slice[i * block_size: (i + 1) * block_size]

#         color = cmap(i / len(z_planes_list))

#         # Plot the third bin contributions for the current z plane
#         for j in range(0,1):  # Plot each column separately
#             # axs[1].plot(radii, data_del_alpha[:, j] - data_alpha_slice[:, j], label=f'Z Plane {z_planes_list[i]} ' + r'$(\dot{\Delta \alpha}) |\phi| - (\dot{ \alpha}) |\phi|$', color=color)

#             # axs[1].plot(radii*0.35, data_del_alpha[:, j], label=f'Z Plane {z_planes_list[i]} ' + r'$(\dot{\Delta \alpha}) |\phi|$', color=color)
#             # axs[1].plot(radii*0.35, data_alpha_slice[:, j], linestyle='--', label=f'Z Plane {z_planes_list[i]} ' + r'$(\dot{ \alpha}) |\phi|$', color=color)
#             np.savetxt(f"{input_directory}201_data_del_alpha_{j}.txt", data_del_alpha[:,j])
#             np.savetxt(f"{input_directory}201_data_alpha_slice_{j}.txt", data_alpha_slice[:,j])

#     # axs[1].set_xlabel('Radius')
#     # axs[1].set_ylabel('Bin Values')
#     # axs[1].set_title('First Bin Contributions for 201 grid')
#     # axs[1].legend()

#     # plt.tight_layout()
#     # plt.show()
            
#             # ---------------------plotting code for radii varying across two simulations---------------------------
            
#             # data_loaded = np.loadtxt("/Users/pranavbharadwajgangrekalvemanoj/Desktop/Axions_Project/relevant_files_060224/2000+/npy_files_full_positive_boost_ev_1/4_column_bin_contributions_401.txt")

#             # # Create a new radii array for the loaded data
#             # radii_loaded = np.arange(0, 199, 1)

#             # # Plot the loaded data
#             # plt.plot(radii_loaded, data_loaded[:, 0], label=r'$(\dot{\Delta \alpha}) |\phi|$ Bin 2, nx,ny = 401', color='purple', linestyle=' ', marker='o', markersize=3)
#             # plt.plot(radii_loaded, data_loaded[:, 1], label=r'$(\dot{\Delta \alpha}) |\phi|$ Bin 3, nx,ny = 401', color='gold', linestyle=' ',marker='o', markersize=3)
#             # plt.plot(radii_loaded, data_loaded[:, 2], label=r'$(\dot{\alpha}) |\phi|$ Bin 2, nx,ny = 401', color='purple', linestyle='-' )
#             # plt.plot(radii_loaded, data_loaded[:, 3], label=r'$(\dot{\alpha}) |\phi|$ Bin 3, nx,ny = 401', color='gold', linestyle='-')

#             # # # Plot the current data
#             # plt.plot(radii, bin_values_2[:, 1], label=r'$(\dot{\Delta \alpha}) |\phi|$ Bin 2, nx,ny = 201', color='purple', linestyle=' ' ,marker='+', markersize=3)
#             # plt.plot(radii, bin_values_2[:, 2], label=r'$(\dot{\Delta \alpha}) |\phi|$ Bin 3, nx,ny = 201', color='gold', linestyle=' ' ,marker='+', markersize=3)
#             # plt.plot(radii, bin_values_3[:, 1], label=r'$(\dot{\alpha}) |\phi|$ Bin 2, nx,ny = 201', color='purple', linestyle='--')
#             # plt.plot(radii, bin_values_3[:, 2], label=r'$(\dot{\alpha}) |\phi|$ Bin 3, nx,ny = 201', color='gold', linestyle='--')

#             # plt.xlabel('Radius')
#             # plt.ylabel('Bin Values')
#             # plt.title('Power Spectra contributions for Nx,Ny = 201 and 401')
#             # plt.legend()

#             # plt.tight_layout()
#             # plt.show()
            
#             # ----------------------------------------------------------------------------
#             # plt.plot(threshold_values, bin_values_del_alpha[:, 1], label=r'$(\dot{\Delta \alpha}) |\phi|$ Bin 2', color='purple', linestyle=' ' ,marker='+', markersize=3)
#             # plt.plot(threshold_values, bin_values_del_alpha[:, 2], label=r'$(\dot{\Delta \alpha}) |\phi|$ Bin 3', color='gold', linestyle=' ' ,marker='+', markersize=3)
#             # plt.plot(threshold_values, bin_values_alpha_slice[:, 1], label=r'$(\dot{\alpha}) |\phi|$ Bin 2', color='purple', linestyle='--')
#             # plt.plot(threshold_values, bin_values_alpha_slice[:, 2], label=r'$(\dot{\alpha}) |\phi|$ Bin 3', color='gold', linestyle='--')

#             # plt.xlabel('Threshold Value')
#             # plt.ylabel('Bin Contributions')
#             # plt.title('Bin Contributions vs Threshold Value')
#             # plt.legend()

#             # plt.tight_layout()
#             # plt.show()

# --------------------------------------------------------------------------------------------------

def apply_circular_mask(data_3d, radius):
    Nx, Ny, Nz = data_3d.shape
    masked_slices = []  # List to store masked 2D slices
    masked_data_3d = np.copy(data_3d)
    for z_plane in range(Nz):
        y, x = np.ogrid[-50:51, -50:51]
        circular_mask = x**2 + y**2 < radius**2
        data_2d = masked_data_3d[:, :, z_plane]
        masked_slice = np.where(circular_mask, 0, data_2d)
        masked_slices.append(masked_slice)  # Append masked 2D slice to the list
    # Convert the list of masked slices to a 3D NumPy array
    masked_data_3d = np.stack(masked_slices, axis=2)
    return masked_data_3d

def calculate_power_spectrum_3d(data_3d, delta, radius):
    masked_data_3d = apply_circular_mask(data_3d, radius)  # Apply circular mask
    masked_data_3d = data_3d
    mask_neg_2pi = np.isclose(masked_data_3d, -2 * np.pi, atol=tol)
    mask_pos_2pi = np.isclose(masked_data_3d, 2 * np.pi, atol=tol)
    
    masked_data_3d[mask_neg_2pi] += 2 * np.pi
    masked_data_3d[mask_pos_2pi] -= 2 * np.pi
    
    mask_neg_2pi = np.isclose(masked_data_3d, - np.pi, atol=tol)
    mask_pos_2pi = np.isclose(masked_data_3d,  np.pi, atol=tol)
    
    masked_data_3d[mask_neg_2pi] +=  np.pi
    masked_data_3d[mask_pos_2pi] -= np.pi
    
    F_tilde = np.fft.fftn(masked_data_3d[:,:,1:100]) / np.sqrt(masked_data_3d.size)
    F_tilde_shifted = np.fft.fftshift(F_tilde)
    power_spectrum_3d = np.abs(F_tilde_shifted)**2 / masked_data_3d.size
    return power_spectrum_3d

def calculate_power_spectrum_3d_for_phi_masked_quantities(data_3d1, data_3d2, alpha_t, alpha_t1, mod_phi, delta, radius):
    masked_data_3d = apply_circular_mask(data_3d1, radius)  # Apply circular mask
    masked_data_3d = data_3d1
    mask_neg_2pi = np.isclose(masked_data_3d, -2 * np.pi, atol=tol)
    mask_pos_2pi = np.isclose(masked_data_3d, 2 * np.pi, atol=tol)
    
    masked_data_3d[mask_neg_2pi] += 2 * np.pi
    masked_data_3d[mask_pos_2pi] -= 2 * np.pi
    
    mask_neg_2pi = np.isclose(masked_data_3d, - np.pi, atol=tol)
    mask_pos_2pi = np.isclose(masked_data_3d,  np.pi, atol=tol)
    
    masked_data_3d[mask_neg_2pi] +=  np.pi
    masked_data_3d[mask_pos_2pi] -= np.pi
    
    masked_data_3d2 = apply_circular_mask(data_3d2, radius)  # Apply circular mask
    masked_data_3d2 = data_3d2
    mask_neg_2pi = np.isclose(masked_data_3d2, -2 * np.pi, atol=tol)
    mask_pos_2pi = np.isclose(masked_data_3d2, 2 * np.pi, atol=tol)
    
    masked_data_3d2[mask_neg_2pi] += 2 * np.pi
    masked_data_3d2[mask_pos_2pi] -= 2 * np.pi
    
    mask_neg_2pi = np.isclose(masked_data_3d2, - np.pi, atol=tol)
    mask_pos_2pi = np.isclose(masked_data_3d2,  np.pi, atol=tol)
    
    masked_data_3d2[mask_neg_2pi] +=  np.pi
    masked_data_3d2[mask_pos_2pi] -= np.pi
    
    del_alpha = (masked_data_3d - masked_data_3d2)/delta_t
    
    alpha_t = apply_circular_mask(alpha_t, radius)  # Apply circular mask
    alpha_t1 = apply_circular_mask(alpha_t1, radius)  # Apply circular mask
    mod_phi = apply_circular_mask(mod_phi, radius)  # Apply circular mask
    alpha = (alpha_t1 - alpha_t)/delta_t

    phi2_del2 = mod_phi * np.abs(del_alpha)**2
    phi_del = np.sqrt(phi2_del2)
    phi2_alpha2 = mod_phi * np.abs(alpha)**2
    phi_alpha = np.sqrt(phi2_alpha2)
    
    
    F_tilde1 = np.fft.fftn(phi_del[:,:,1:100]) / np.sqrt(phi_del.size)
    F_tilde_shifted1 = np.fft.fftshift(F_tilde1)
    power_spectrum_3d1 = np.abs(F_tilde_shifted1)**2 / phi_del.size
    
    F_tilde2 = np.fft.fftn(phi_alpha[:,:,1:100]) / np.sqrt(phi_alpha.size)
    F_tilde_shifted2 = np.fft.fftshift(F_tilde2)
    power_spectrum_3d2 = np.abs(F_tilde_shifted2)**2 / phi_alpha.size
    return power_spectrum_3d1, power_spectrum_3d2


def plot_and_save_power_spectra_3d(data_3d, data_3d2, alpha_t, alpha_t1, mod_phi, delta, input_directory):
    Nx = 101
    Ny = 101
    Nz = 101
    bin_values_del_alpha = []
    bin_values_alpha_slice = []
    bin_values_del_alpha_all = []
    bin_values_alpha_slice_all = []
    for radius in range(0, 49):
        # power_spectrum_3d = calculate_power_spectrum_3d(data_3d, delta, radius)
        power_spectrum_3d1, power_spectrum_3d2 = calculate_power_spectrum_3d_for_phi_masked_quantities(data_3d, data_3d2, alpha_t, alpha_t1, mod_phi, delta, radius)
        
        freq_x = np.fft.fftshift(np.fft.fftfreq(Nx, delta)) * 2 * np.pi
        freq_y = np.fft.fftshift(np.fft.fftfreq(Ny, delta)) * 2 * np.pi
        freq_z = np.fft.fftshift(np.fft.fftfreq(Nz, delta)) * 2 * np.pi
        
        k_x, k_y, k_z = np.meshgrid(freq_x, freq_y, freq_z, indexing='ij')

        k_modulus = np.sqrt(k_x**2 + k_y**2 + k_z**2)

        L = Nx * delta
        n_values = k_modulus * L / (2*np.pi)
        n_values = n_values[:,:,1:100]
        # fig, ax = plt.subplots(figsize=(8, 6))
        # hist = ax.hist(x=n_values.ravel(), weights=power_spectrum_3d.ravel(), bins=np.linspace(0.5,10.5,11), color='purple', edgecolor='black')
        # ax.set_title(f'3D Power Spectrum for radius: {radius} grid points')
        # ax.set_xlabel('n values')
        # ax.set_ylabel('Power Spectrum')
        # # ax.set_ylim([0, 2e-5])
        # ax.legend()
        # ax.grid(True)
        # plt.tight_layout()
        # plt.show()
        
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))  # Create a 2x2 grid of subplots

        # First plot
        axs[0].hist(x=n_values.ravel(), weights=power_spectrum_3d1.ravel(), bins=np.linspace(0.5,10.5,11), color='purple', edgecolor='black')
        axs[0].set_ylim([0, 0.5e-5])
        axs[0].set_xlabel('n values')
        axs[0].set_ylabel(r'3D Power Spectrum for $(\dot{\Delta \alpha}) |\phi|$')

        # Third plot
        axs[1].hist(x=n_values.ravel(), weights=power_spectrum_3d2.ravel(), bins=np.linspace(0.5,10.5,11), color='purple', edgecolor='black')
        axs[1].set_xlabel('n values')
        axs[1].set_ylabel(r'3D Power Spectrum for $(\dot{\alpha}) |\phi|$')
        axs[1].set_ylim([0, 0.5e-5])

        plt.figtext(0.5, 0.01, f'Mask radius :{radius * 0.35:.4f} physical spatial units', ha='center', va='bottom')
        plt.tight_layout()
        # plt.show()

        # Save the figure
        filename = f"3D_power_spectrum_radius_{radius:03d}.png"
        frame_path = os.path.join(input_directory, filename)
        plt.savefig(frame_path)

        # Close the figure to free up memory
        plt.close(fig)
        
                                    # Extract histogram values for the first three bins
    #     hist_del_alpha, _ = np.histogram(n_values.ravel(), weights=power_spectrum_3d.ravel(), bins=np.linspace(0.5, 10.5, 11))
       
    #     # Append the first three bin values to the respective lists
    #     bin_values_del_alpha.append(hist_del_alpha[:3])

            
    # bin_values_del_alpha_all.append(bin_values_del_alpha)



    # bin_values_del_alpha = np.array(bin_values_del_alpha)

    # bin_values_del_alpha_all = np.array(bin_values_del_alpha_all)


def plot_3d_power_spectrum(power_spectrum_3d, delta):
    freq_x = np.fft.fftshift(np.fft.fftfreq(power_spectrum_3d.shape[0], delta))  * 2 * np.pi
    freq_y = np.fft.fftshift(np.fft.fftfreq(power_spectrum_3d.shape[1], delta))  * 2 * np.pi
    freq_z = np.fft.fftshift(np.fft.fftfreq(power_spectrum_3d.shape[2], delta))  * 2 * np.pi
    
    # freq_x_ = np.fft.fftshift(np.fft.fftfreq(power_spectrum_3d.shape[0], delta))  
    # freq_y_ = np.fft.fftshift(np.fft.fftfreq(power_spectrum_3d.shape[1], delta))  
    # freq_z_ = np.fft.fftshift(np.fft.fftfreq(power_spectrum_3d.shape[2], delta)) 

    k_x, k_y, k_z = np.meshgrid(freq_x, freq_y, freq_z, indexing='ij')
    # k_x_, k_y_, k_z_ = np.meshgrid(freq_x_, freq_y_, freq_z_, indexing='ij')

    # Calculate modulus of wave vectors
    k_modulus = np.sqrt(k_x**2 + k_y**2 + k_z**2)
    # k_modulus_ = np.sqrt(k_x_**2 + k_y_**2 + k_z_**2)

    # Calculate n values
    L = Nx * delta
    n_values = k_modulus * L / ( 2*np.pi)
    # n_values_ = k_modulus_ * L / (2 * np.pi)

    # Plot the power spectrum against n values
    plt.figure(figsize=(8, 6))
    hist = plt.hist(x=n_values.ravel(), weights=power_spectrum_3d.ravel(), bins=np.linspace(0.5,10.5,11), color='purple', edgecolor='black')
    # plt.figure(figsize=(8, 6))
    # hist_ = plt.hist(x=n_values_.ravel(), weights=power_spectrum_3d.ravel(), bins=np.linspace(0.05,4.5,11), color='skyblue', edgecolor='black')
    plt.title('3D Power Spectrum for length scale associated with nz = 51')
    plt.xlabel('n values')
    plt.ylabel('Power Spectrum')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Print the number of data points per bin
    # for i in range(len(hist)):
    #     print(f"Bin {i+1}: {int(hist[i])} data points")

    # for i in range(len(hist_)):
    #     print(f"Bin {i+1}: {int(hist_[i])} data points")



def filter_power_spectrum(power_spectrum, freq_x, freq_y, freq_z, low_cutoff, high_cutoff):
    # Create a filter in the frequency domain
    filter_mask_x = np.logical_and(np.abs(freq_x) >= low_cutoff, np.abs(freq_x) <= high_cutoff)[:, np.newaxis, np.newaxis]
    filter_mask_y = np.logical_and(np.abs(freq_y) >= low_cutoff, np.abs(freq_y) <= high_cutoff)[np.newaxis, :, np.newaxis]
    filter_mask_z = np.logical_and(np.abs(freq_z) >= low_cutoff, np.abs(freq_z) <= high_cutoff)[np.newaxis, np.newaxis, :]

    # Apply the filter to the power spectrum
    filter_mask = filter_mask_x * filter_mask_y * filter_mask_z
    filtered_spectrum = power_spectrum * filter_mask

    return filtered_spectrum

def inverse_fourier_transform(filtered_spectrum):
    # Perform inverse Fourier transform
    filtered_data = np.fft.ifftn(np.fft.ifftshift(filtered_spectrum)).real

    return filtered_data

def cos1(x):
    return 0.01*np.cos(4*np.pi*x/201)

def cos2(x):
    return 0.01*np.cos(8*np.pi*x/201)

def cos3(x):
    return 0.01*np.cos(16*np.pi*x/201)

def cos4(x):
    return 0.01*np.cos(32*np.pi*x/201)

def cos5(x):
    return 0.01*np.cos(64*np.pi*x/201)

def cos6(x):
    return 0.01*np.cos(128*np.pi*x/201)
    

# Specify your directory!
# input_directory = "/Users/pranavbharadwajgangrekalvemanoj/Desktop/Axions_Project/to_scp_120324_201_all/npy_files_full_positive_boost_ev_1/"
# input_ref_directory = "/Users/pranavbharadwajgangrekalvemanoj/Desktop/Axions_Project/to_scp_120324_201_all/npy_files_full_positive_boost_ev_1/"

input_directory = "/Volumes/Recent_Archives/P_files_he/npy_files/"
input_ref_directory = "/Volumes/Recent_Archives/P_files_he/npy_files/"

Nx = 201
Ny = 201
Nz = 201
delta = 0.7
delta_t = 0.3
tol = 0.005

y, x = np.ogrid[-50:51, -50:51]

# Create a mask for a circle of radius 30
circular_mask = x**2 + y**2 < 70**2

# Load data from .npy file

# Frame number
# user_input = input("Do you want to enter the loop? (yes/no): ")

# if user_input.lower() == "yes":
#     for frame_t in range(320,335):

# frame_t = 150 
# frame_t1 = frame_t + 1
frame_range = range(330, 351)

for frame_t in frame_range:
    frame_t1 = frame_t + 1

    # Load data from .npy file
    data_real_t = np.load(os.path.join(input_directory, f'col_0_frame_{frame_t}.npy'))
    data_im_t = np.load(os.path.join(input_directory, f'col_1_frame_{frame_t}.npy'))
    data_real_t1 = np.load(os.path.join(input_directory, f'col_0_frame_{frame_t1}.npy'))
    data_im_t1 = np.load(os.path.join(input_directory, f'col_1_frame_{frame_t1}.npy'))

    print("First set of data loaded")
    # test stationary file
    # data_real_test = np.load(os.path.join(input_directory, f'col_0_frame_1.npy'))
    # data_im_test = np.load(os.path.join(input_directory, f'col_1_frame_1.npy'))
    # complex_number_test = data_real_test + 1j * data_im_test
    # mod_test = np.abs((data_real_test)**2 + (data_im_test)**2)

    # Reference file
    data_real_ref_t = np.load(os.path.join(input_ref_directory, f'frame_{frame_t}_positive_boost_0.npy'))
    data_real_ref_t1 = np.load(os.path.join(input_ref_directory, f'frame_{frame_t1}_positive_boost_0.npy'))

    data_im_ref_t = np.load(os.path.join(input_ref_directory, f'frame_{frame_t}_positive_boost_1.npy'))
    data_im_ref_t1 = np.load(os.path.join(input_ref_directory, f'frame_{frame_t1}_positive_boost_1.npy'))
    print("Second set of data loaded")
    complex_number_t = data_real_t + 1j * data_im_t
    complex_number_ref_t = data_real_ref_t + 1j * data_im_ref_t
    # complex_conjugate_ref_t = data_real_ref_t - 1j * data_im_ref_t
    # new = complex_number_t * complex_conjugate_ref_t

    complex_number_t1 = data_real_t1 + 1j * data_im_t1
    complex_number_ref_t1 = data_real_ref_t1 + 1j * data_im_ref_t1
    print("complex numbers defined")
    # comp_num = complex_number - complex_number_ref
    # # Assuming your grid is already in the right shape
    # y_grid, x_grid, z_grid = np.meshgrid(np.arange(N), np.arange(N), np.arange(N))
    mod_t_squared = np.abs((data_real_t)**2 + (data_im_t)**2)**2
    
    
    mod_t = ((data_real_t)**2 + (data_im_t)**2)**0.5
    mod_ref_t = ((data_real_ref_t)**2 + (data_im_ref_t)**2)**0.5
    
    
    mod_t1_squared = np.abs((data_real_t1)**2 + (data_im_t1)**2)**2

    phi_phase_ref_t = np.angle(complex_number_ref_t)

    phi_phase_ref_t1 = np.angle(complex_number_ref_t1)
    # data_3d = phi_phase
    # # Set the parameters

    print("phases calculated")
    # Assuming your grid is already in the right shape
    y_grid, x_grid, z_grid = np.meshgrid(np.arange(Ny), np.arange(Nx), np.arange(data_real_t.shape[2]))
    print("meshgrid created")
    phi_phase_t = np.angle(complex_number_t)
    phase_t = phi_phase_t - phi_phase_ref_t
    phi_phase_t1 = np.angle(complex_number_t1)
    phase_t1 = phi_phase_t1 - phi_phase_ref_t1

    delta_alpha = (phase_t1 - phase_t)/delta_t
    # phase_multiplied = np.angle(new)
    # Set the parameters for the Fourier Transform

    tol = 0.005
    z_slice_to_plot = 0  # Adjust as needed
    print("plotting about to start")
    
    # np.save(os.path.join(input_directory, 'phase_t.npy'), phase_t)
    # np.save(os.path.join(input_directory, 'phase_t1.npy'), phase_t1)
    # np.save(os.path.join(input_directory, 'mod_t.npy'), mod_t)
    # np.save(os.path.join(input_directory, 'phi_phase_t.npy'), np.abs(phi_phase_t))
    # np.save(os.path.join(input_directory, 'phi_phase_t1.npy'), np.abs(phi_phase_t1))
    
    
# --------------------------------load the files--------------------------------------------
    # phase_t = np.load(os.path.join(input_directory, 'phase_t.npy'))
    # phase_t1 = np.load(os.path.join(input_directory, 'phase_t1.npy'))
    # mod_t = np.load(os.path.join(input_directory, 'mod_t.npy'))
    # phi_phase_t = np.load(os.path.join(input_directory, 'phi_phase_t.npy'))
    # phi_phase_t1 = np.load(os.path.join(input_directory, 'phi_phase_t1.npy'))


    # diagnostic_t = (data_real_t * np.gradient * data_real_t + data_im_t * np.gradient * data_im_t)/(mod_t) 
# ---------------------------------- uncomment after trying the 401 loading--------------------------------------------------
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
    
    for z_plane in range(phase_t.shape[2]):
        data_2d1 = phase_t[:, :, z_plane]

        mask_neg_2pi_1 = np.isclose(data_2d1, -2 * np.pi, atol=tol)
        mask_pos_2pi_1 = np.isclose(data_2d1, 2 * np.pi, atol=tol)

        data_2d1[mask_neg_2pi_1] += 2 * np.pi
        data_2d1[mask_pos_2pi_1] -= 2 * np.pi

        mask_neg_2pi_1_ = np.isclose(data_2d1, - np.pi, atol=tol)
        mask_pos_2pi_1_ = np.isclose(data_2d1,  np.pi, atol=tol)

        data_2d1[mask_neg_2pi_1_] +=  np.pi
        data_2d1[mask_pos_2pi_1_] -= np.pi

        phase_t[:, :, z_plane] = data_2d1
    
    grad_x_t_comp = (np.roll(phase_t, -1, axis=0) - np.roll(phase_t, 1, axis=0)) / (2 * delta)

    grad_y_t_comp = (np.roll(phase_t, -1, axis=1) - np.roll(phase_t, 1, axis=1)) / (2 * delta)

    grad_z_t_comp = (np.roll(phase_t, -1, axis=2) - np.roll(phase_t, 1, axis=2)) / (2 * delta)


    diagnostic_comp = mod_t * (grad_x_t_comp * unit_vector_x + grad_y_t_comp * unit_vector_y + grad_z_t_comp * unit_vector_z)
    diagnostic_comp = np.nan_to_num(diagnostic_comp)
    
# -----------------------------------------------uncomment till here-----------------------------------------------------
 
    # test scalar projection
    # grad_x_real_test = (np.roll(data_real_test, -1, axis=0) - np.roll(data_real_test, 1, axis=0)) / (2 * delta)
    # grad_x_im_test = (np.roll(data_im_test, -1, axis=0) - np.roll(data_im_test, 1, axis=0)) / (2 * delta)

    # grad_y_real_test = (np.roll(data_real_test, -1, axis=1) - np.roll(data_real_test, 1, axis=1)) / (2 * delta)
    # grad_y_im_test = (np.roll(data_im_test, -1, axis=1) - np.roll(data_im_test, 1, axis=1)) / (2 * delta)


    # grad_z_real_test = (np.roll(data_real_test, -1, axis=2) - np.roll(data_real_test, 1, axis=2)) / (2 * delta)
    # grad_z_im_test = (np.roll(data_im_test, -1, axis=2) - np.roll(data_im_test, 1, axis=2)) / (2 * delta)

    # D_x_alpha_test = (data_real_test * grad_x_im_test - data_im_test * grad_x_real_test)/(mod_test) 
    # D_y_alpha_test = (data_real_test * grad_y_im_test - data_im_test * grad_y_real_test)/(mod_test) 
    # D_z_alpha_test = (data_real_test * grad_z_im_test - data_im_test * grad_z_real_test)/(mod_test) 


    # # Define the center point (I_0, J_0, K_0)
    # I_0 = Nx // 2  # Assuming Nx, Ny, Nz are the dimensions of your grid
    # J_0 = Ny // 2
    # K_0 = Nz // 2

    # # Define the grid spacing along each dimension
    # delta_x = 0.7 
    # delta_y = 0.7
    # delta_z = 0.7

    # # Compute the coordinates relative to the center point
    # I_relative = np.arange(Nx) - I_0
    # J_relative = np.arange(Ny) - J_0
    # K_relative = np.arange(Nz) - K_0

    # # Create 3D grids of relative coordinates
    # I_grid, J_grid, K_grid = np.meshgrid(I_relative, J_relative, K_relative, indexing='ij')

    # # Compute the radial vector components
    # radial_vector_x = I_grid * delta_x
    # radial_vector_y = J_grid * delta_y
    # radial_vector_z = 0

    # # Compute the magnitude of the radial vector
    # magnitude_r = np.sqrt(radial_vector_x**2 + radial_vector_y**2 + radial_vector_z**2)

    # # Compute the unit vector components
    # unit_vector_x = radial_vector_x / magnitude_r
    # unit_vector_y = radial_vector_y / magnitude_r
    # unit_vector_z = radial_vector_z / magnitude_r

    # # Compute the dot product of D with the components of the unit vector
    # D_dot_unit_x_test = D_x_alpha_test * unit_vector_x
    # D_dot_unit_y_test = D_y_alpha_test * unit_vector_y
    # D_dot_unit_z_test = D_z_alpha_test * unit_vector_z

    # D_dot_unit_x_test = np.nan_to_num(D_dot_unit_x_test)
    # D_dot_unit_y_test = np.nan_to_num(D_dot_unit_y_test)
    # D_dot_unit_z_test = np.nan_to_num(D_dot_unit_z_test)


    # scalar_projection_test = D_dot_unit_x_test + D_dot_unit_y_test + D_dot_unit_z_test



    # Extract the specific z-slice from the 3D array
    # phase_slice1 = phase_t[:,:,z_slice_to_plot]
    # phase_slice = phase_t1[:,:,z_slice_to_plot]


            # data_3d = phase_slice[:,:,z_slice_to_plot]
            # for i in data_3d[:,:,]
            #     plt.contourf(i, cmap='RdPu', levels=200)      
            #     plt.show() 


            # plot_input = input("Do you want to plot the 2D or 3D power spectrum? (2d/3d): ")
            # # Select the z-slice you want to work with (assuming 0-based indexing)
            # if plot_input.lower() == "2d":
    plot_power_spectrum(Nx, Ny, delta, scalar_projection, diagnostic_comp, mod_t, mod_ref_t,phi_phase_ref_t, phi_phase_t, frame_t)
    # z_planes = [12,25,37,50,62,75,87]
    # plot_power_spectrum(Nx, Ny, delta, phase_t, phase_t1, mod_t, np.abs(phi_phase_t), np.abs(phi_phase_t1))
    # plot_3d_power_spectrum(power_spectrum_3d, delta)

    # plot_and_save_power_spectra_3d(phase_t, phase_t1, np.abs(phi_phase_t), np.abs(phi_phase_t1), mod_t, delta, input_directory)   

        # elif plot_input.lower() == "3d":
# power_spectrum_3d = calculate_power_spectrum_3d(phase_t, delta)
# plot_3d_power_spectrum(power_spectrum_3d, delta)

# -----------------------------------------------3d power spectrum known function test------------------------------------
# x = np.arange(Nx)
# y = np.arange(Ny)
# z = np.arange(Nz)
# X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# data_3d = cos1(X) + cos2(X) + cos3(X) + cos4(X) + cos5(X) + cos6(X)

# # # Calculate power spectrum
# power_spectrum_3d = calculate_power_spectrum_3d(data_3d, delta)

# # # Plot power spectrum
# plot_3d_power_spectrum(power_spectrum_3d, delta)

# plot_and_save_power_spectra_3d(phase_t, delta, input_directory)

# freq_x = np.fft.fftshift(np.fft.fftfreq(power_spectrum_3d.shape[0], delta)) * 2 * np.pi
# freq_y = np.fft.fftshift(np.fft.fftfreq(power_spectrum_3d.shape[1], delta)) * 2 * np.pi
# freq_z = np.fft.fftshift(np.fft.fftfreq(power_spectrum_3d.shape[2], delta)) * 2 * np.pi

# # Define frequency cutoffs (adjust these values as needed)
# low_cutoff = 0.3
# high_cutoff = 3.5

# # Filter power spectrum based on frequency cutoffs
# low_freq_spectrum = filter_power_spectrum(power_spectrum_3d, freq_x, freq_y, freq_z, 0, low_cutoff)
# mid_freq_spectrum = filter_power_spectrum(power_spectrum_3d, freq_x, freq_y, freq_z, low_cutoff, high_cutoff)
# high_freq_spectrum = filter_power_spectrum(power_spectrum_3d, freq_x, freq_y, freq_z, high_cutoff, 4.4)

# # Inverse Fourier transform
# low_freq_data = inverse_fourier_transform(low_freq_spectrum)
# mid_freq_data = inverse_fourier_transform(mid_freq_spectrum)
# high_freq_data = inverse_fourier_transform(high_freq_spectrum)

# # Visualization
# fig = plt.figure(figsize=(15, 5))

# # Low Frequency Region
# ax1 = fig.add_subplot(131, projection='3d')
# x_grid_3d, y_grid_3d = np.meshgrid(x_grid, y_grid)
# ax1.plot_surface(x_grid_3d, y_grid_3d, low_freq_data[:, :, z_slice_to_plot], cmap='viridis')
# ax1.set_title('Low Frequency Region')

# # Mid Frequency Region
# ax2 = fig.add_subplot(132, projection='3d')
# ax2.plot_surface(x_grid_3d, y_grid_3d, mid_freq_data[:, :, z_slice_to_plot], cmap='viridis')
# ax2.set_title('Mid Frequency Region')

# # High Frequency Region
# ax3 = fig.add_subplot(133, projection='3d')
# ax3.plot_surface(x_grid_3d, y_grid_3d, high_freq_data[:, :, z_slice_to_plot], cmap='viridis')
# ax3.set_title('High Frequency Region')

# plt.show()
# # 2D FFT
# # fft_result = np.fft.fftshift(np.fft.fft2(data_3d))

# # # Inverse FFT to obtain the spatial domain representation
# # inverse_fft_result = np.fft.ifft2(np.fft.ifftshift(fft_result))

# # # Plot the result
# # fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# # # Plot power spectrum
# # axs[0].pcolormesh(np.fft.fftshift(np.fft.fftfreq(Nx, delta)),
# #                   np.fft.fftshift(np.fft.fftfreq(Ny, delta)),
# #                   np.log10(np.abs(fft_result)**2))
# # axs[0].set_title(f'2D Power Spectrum for z-slice {z_slice_to_plot}')
# # axs[0].set_xlabel('Frequency (cycles per unit distance)')
# # axs[0].set_ylabel('Frequency (cycles per unit distance)')
# # axs[0].set_aspect('equal')
# # axs[0].grid(True)

# # # Plot inverse FFT result
# # axs[1].pcolormesh(inverse_fft_result.real, cmap='viridis')
# # axs[1].set_title(f'Inverse FFT Result for z-slice {z_slice_to_plot}')
# # axs[1].set_xlabel('X-axis')
# # axs[1].set_ylabel('Y-axis')
# # axs[1].set_aspect('equal')
# # axs[1].grid(True)

# # plt.show()

# # freq_x = np.fft.fftshift(np.fft.fftfreq(Nx, delta))
# # freq_y = np.fft.fftshift(np.fft.fftfreq(Ny, delta))

# # k_x = 2 * np.pi * freq_x
# # k_y = 2 * np.pi * freq_y

# # # Calculate modulus of wave vectors
# # k_modulus = np.sqrt(k_x**2 + k_y**2)

# # # Calculate n values
# # L = Nx * delta
# # n_values = k_modulus * L / (2 * np.pi)


# # power_spectrum = np.abs(fft_result)**2
# # power_spectrum_1d = np.sum(power_spectrum, axis=1)
# # n_values_2d, _ = np.meshgrid(n_values, np.arange(Ny))
# # power_spectrum_2d, _ = np.meshgrid(power_spectrum, np.arange(Ny))

# # # Flatten the arrays
# # n_values_flat = n_values_2d.flatten()
# # power_spectrum_flat = power_spectrum_2d.flatten()

# # # # Plot the power spectrum against n values
# # # plt.figure(figsize=(8, 6))
# # # plt.plot(n_values, (power_spectrum_1d), label=f'z-slice {z_slice_to_plot}')
# # # plt.title('Power Spectrum in terms of k values')
# # # plt.xlim(0,10)
# # # plt.xlabel('n values')
# # # plt.ylabel('log10(Power Spectrum)')
# # # plt.legend()
# # # plt.grid(True)
# # # plt.show()

# # num_bins = 50  # Adjust as needed

# # # Plot the power spectrum against n values
# # plt.figure(figsize=(8, 6))
# # plt.hist(n_values, bins=num_bins, weights=power_spectrum_1d, label=f'z-slice {z_slice_to_plot}')
# # plt.title('Power Spectrum in terms of k values')
# # plt.xlabel('k values')
# # plt.ylabel('Power Spectrum')
# # plt.legend()
# # plt.grid(True)
# # plt.show()

# # # Calculate 2D FFT and shift the zero-frequency component to the center
# # fft_result = (np.fft.fftn((abs(data_3d))))

# # # Calculate power spectrum
# # power_spectrum = np.abs(fft_result)**2

# # # Calculate mode numbers
# # Ny, Nx = data_3d.shape
# # Y, X = np.ogrid[:Ny, :Nx]
# # center = np.array([[(Nx - 1) / 2.0], [(Ny - 1) / 2.0]])
# # n_values = np.sqrt((X - center[0])**2 + (Y - center[1])**2)

# # # Flatten power spectrum and mode numbers for plotting
# # power_spectrum_flat = power_spectrum.flatten()
# # n_values_flat = n_values.flatten()

# # # Plot power spectrum against mode numbers
# # plt.figure(figsize=(8, 6))
# # plt.hist(n_values_flat, bins=50, weights=power_spectrum_flat, range=(0, 10))
# # plt.title('Power Spectrum in terms of n values')
# # plt.xlabel('n values')
# # plt.ylabel('Power Spectrum')
# # plt.grid(True)
# # plt.show()

# # freq_x = np.fft.fftfreq(Nx, delta)
# # freq_y = np.fft.fftfreq(Ny, delta)

# # # Calculate wave vectors
# # k_x = 2 * np.pi * freq_x
# # k_y = 2 * np.pi * freq_y

# # # Create a 2D grid of k_x and k_y values
# # k_x_2d, k_y_2d = np.meshgrid(k_x, k_y)

# # # Calculate modulus of wave vectors
# # k_modulus = np.sqrt(k_x_2d**2 + k_y_2d**2)

# # # Calculate n values
# # L = Nx * delta
# # n_values = k_modulus * L / (2 * np.pi)

# # # Calculate power spectrum
# # power_spectrum = np.abs(fft_result)**2

# # # Flatten the arrays
# # n_values_flat = n_values.flatten()
# # power_spectrum_flat = power_spectrum.flatten()

# # # Check the shapes
# # print(n_values_flat.shape)
# # print(power_spectrum_flat.shape)

# # # Plot power spectrum against n values
# # plt.figure(figsize=(8, 6))
# # plt.hist(n_values_flat, bins=50, weights=power_spectrum_flat, range=(0, 20))
# # plt.title('Power Spectrum in terms of n values')
# # plt.xlabel('n values')
# # plt.ylabel('Power Spectrum')
# # plt.grid(True)
# # plt.show()




# # # Calculate 2D FFT and shift the zero-frequency component to the center
# # fft_result = np.fft.fftshift(np.fft.fft2(data_3d))

# # # Calculate power spectrum
# # power_spectrum = np.abs(fft_result)**2

# # # Calculate mode numbers
# # Ny, Nx = data_3d.shape
# # Y, X = np.ogrid[:Ny, :Nx]
# # center = np.array([[(Nx - 1) / 2.0], [(Ny - 1) / 2.0]])
# # n_values = np.sqrt((X - center[0])**2 + (Y - center[1])**2)

# # # Flatten power spectrum and mode numbers for plotting
# # power_spectrum_flat = power_spectrum.flatten()
# # n_values_flat = n_values.flatten()

# # # Plot power spectrum against mode numbers
# # plt.figure(figsize=(8, 6))
# # plt.hist(n_values_flat, bins=10, weights=power_spectrum_flat, range=(0, 10))
# # plt.title('Power Spectrum in terms of Mode Numbers (n)')
# # plt.xlabel('Mode Number (n)')
# # plt.ylabel('Power Spectrum')
# # plt.grid(True)
# # plt.show()
