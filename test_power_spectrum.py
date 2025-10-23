# import numpy as np
# import matplotlib.pyplot as plt
# import os

# input_directory = "/Users/pranavbharadwajgangrekalvemanoj/Desktop/Axions_Project/ev10data/npy_files_ev_10/"

# Nx = 201
# Ny = 201
# Nz = 51
# delta = 0.7

# # Load data from .npy file
# data_real = np.load(os.path.join(input_directory, 'col_0_frame_51.npy'))
# data_im = np.load(os.path.join(input_directory, 'col_1_frame_51.npy'))

# # #reference file
# # # data_3d_ref = np.load( '/Users/pranavbharadwajgangrekalvemanoj/Desktop/Axions_Project/npy_files/col_0_frame_20.npy')
# data_real_ref = np.load( '/Users/pranavbharadwajgangrekalvemanoj/Desktop/Axions_Project/somethings/npy_files_ev_10/frame_51_pseudo_0.npy')
# data_im_ref = np.load( '/Users/pranavbharadwajgangrekalvemanoj/Desktop/Axions_Project/somethings/npy_files_ev_10/frame_51_pseudo_1.npy')

# complex_number = data_real + 1j * data_im
# complex_number_ref = data_real_ref + 1j * data_im_ref
# # comp_num = complex_number - complex_number_ref
# # # Assuming your grid is already in the right shape
# # y_grid, x_grid, z_grid = np.meshgrid(np.arange(N), np.arange(N), np.arange(N))


# phi_phase_ref = np.angle(complex_number_ref)
# # data_3d = phi_phase
# # # Set the parameters


# # Assuming your grid is already in the right shape
# y_grid, x_grid, z_grid = np.meshgrid(np.arange(Ny), np.arange(Nx), np.arange(data_real.shape[2]))

# phi_phase = np.angle(complex_number)
# phase = phi_phase - phi_phase_ref
# tol = 0.1
# z_slice_to_plot = 11  # Adjust as needed

# # Extract the specific z-slice from the 3D array
# phase_slice = phase[:, :, z_slice_to_plot]
# mask_neg_2pi = np.isclose(phase_slice, -2 * np.pi, atol=tol)
# mask_pos_2pi = np.isclose(phase_slice, 2 * np.pi, atol=tol)

# phase_slice[mask_neg_2pi] += 2 * np.pi
# phase_slice[mask_pos_2pi] -= 2 * np.pi

# mask_neg_2pi = np.isclose(phase_slice, - np.pi, atol=tol)
# mask_pos_2pi = np.isclose(phase_slice,  np.pi, atol=tol)

# phase_slice[mask_neg_2pi] +=  np.pi
# phase_slice[mask_pos_2pi] -= np.pi

# data_3d = phase_slice

# length = 201
# # Calculate the power spectrum
# power_spectrum = np.abs(np.fft.fftshift(np.fft.fft2(data_3d)))**2

# # Calculate the wave vectors


# modes = np.fft.fftshift(np.fft.fftfreq(length, d=1/Nx))

# # Plot the power spectrum vs modes
# plt.plot(modes, power_spectrum)
# plt.xlabel('Modes')
# plt.ylabel('Power Spectrum')
# plt.title('Power Spectrum of data_3d')
# plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Define parameters
Nx = 201
Ny = 201
dx = dy = 0.7

# Create a 2D array of phases (assuming you already have this)
phases = np.random.uniform(0, 2 * np.pi, size=(Nx, Ny))

# Compute the Fourier Transform of the phases
fft_result = np.fft.fft2(phases)
power_spectrum = np.abs(fft_result)**2

# Compute wave numbers in x and y directions
kx = 2 * np.pi * np.fft.fftfreq(Nx, dx)
ky = 2 * np.pi * np.fft.fftfreq(Ny, dy)

# Create a grid of wave numbers
kx_grid, ky_grid = np.meshgrid(kx, ky)

# Avoid division by zero by setting small values to a small constant
epsilon = 1e-10
denominator = np.sqrt(kx_grid**2 + ky_grid**2)
denominator[denominator < epsilon] = epsilon

# Compute wavelengths corresponding to wave numbers
wavelengths = 1 / denominator

# Plot the power spectrum
plt.imshow(np.log(power_spectrum), extent=(kx.min(), kx.max(), ky.min(), ky.max()), origin='lower', cmap='viridis')
plt.colorbar(label='Log Power Spectrum')
plt.xlabel('Wave Number (kx)')
plt.ylabel('Wave Number (ky)')
plt.title('2D Power Spectrum')

# Plot modes based on wavelengths
modes = plt.scatter(2 * np.pi / wavelengths, np.ones_like(wavelengths), color='red', marker='o', label='Modes')

plt.legend()
plt.show()
