
# import numpy as np
# import os
# import matplotlib.pyplot as plt
# from scipy.fft import fftn, fftshift
# import math

# # Path to the directory containing input files
# input_directory = "/Users/pranavbharadwajgangrekalvemanoj/Desktop/Axions_Project/trash_copy/npy_files/"

# # Load the reference npy file
# reference_file_path = "/Users/pranavbharadwajgangrekalvemanoj/Desktop/Axions_Project/evolution_run_5_stat_v_2/npy_files/col_0_frame_27.npy"
# reference_data = np.load(reference_file_path)[:, :, 50]  # Select the 50th plane along the z-axis

# # Number of frames
# num_frames = 300  # Update this to match the number of frames you have

# # Grid parameters (assuming dx, dy, dz = 0.5 and nx, ny, nz = 100)
# dx, dy, dz = 0.5, 0.5, 0.5
# nx, ny, nz = 100, 100, 100
# shape = (nx,ny,nz)

# def fftfreq3D(shape,dx,dy,dz):
    
#     Nx, Ny, Nz = shape
    
#     freqs_x = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(Nx, dx))
#     freqs_y = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(Ny, dy))
#     freqs_z = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(Nz, dz))
                
#     freqs_3D = np.array(np.meshgrid(freqs_x, freqs_y, freqs_z, indexing='ij'))
        
#     return freqs_3D


# # Initialize arrays to store power spectrum data
# power_spectrum_data_x = np.zeros(nx)
# power_spectrum_data_y = np.zeros(ny)
# power_spectrum_data_z = np.zeros(nz)

# # Iterate through all input files and calculate power spectrum along different wave numbers
# for p in range(num_frames):
#     try:
#         input_file_path_phi_0 = os.path.join(input_directory, f"col_1_frame_{p}.npy")
#         input_file_path_phi_1 = os.path.join(input_directory, f"col_2_frame_{p}.npy")

#         # Load data from the input file
#         input_data_phi_0 = np.load(input_file_path_phi_0)  # Select the 50th plane along the z-axis
#         input_data_phi_1 = np.load(input_file_path_phi_1)  # Select the 50th plane along the z-axis

#         # Calculate the complex number field
#         complex_data = input_data_phi_0 + 1j * input_data_phi_1

#         # Calculate the Fourier transform
#         fourier_transform = np.fft.fftn(complex_data)

#         # Compute power spectrum along x, y, z directions
#         power_spectrum_x = np.abs(fourier_transform) ** 2 * (dx * dy * dz)  # Multiply by grid spacing for normalization
#         power_spectrum_y = np.abs(fourier_transform) ** 2 * (dx * dy * dz)
#         power_spectrum_z = np.abs(fourier_transform) ** 2 * (dx * dy * dz)

#         # Integrate power spectrum over y and z to get 1D power spectrum along x
#         power_spectrum_data_x += np.sum(np.sum(power_spectrum_x, axis=2), axis=1)
#         # Integrate power spectrum over x and z to get 1D power spectrum along y
#         power_spectrum_data_y += np.sum(np.sum(power_spectrum_y, axis=0), axis=1)
#         # Integrate power spectrum over x and y to get 1D power spectrum along z
#         power_spectrum_data_z += np.sum(np.sum(power_spectrum_z, axis=0), axis=0)

#         print(f"Power spectrum calculated for frame {p}")

#     except Exception as e:
#         # Handle any exceptions (file not found, empty file, etc.) and continue to the next iteration
#         print(f"Error processing file {p}: {str(e)}")
#         continue

# # Generate wave number arrays for plotting
# kx_values = 2 * np.pi * np.fft.fftfreq(nx, dx)
# ky_values = 2 * np.pi * np.fft.fftfreq(ny, dy)
# kz_values = 2 * np.pi * np.fft.fftfreq(nz, dz)

# # Plot power spectrum vs modes for x, y, and z directions
# plt.figure(figsize=(10, 6))
# plt.plot(kx_values, power_spectrum_data_x, label='Power Spectrum along X', color='r')
# plt.plot(ky_values, power_spectrum_data_y, label='Power Spectrum along Y', color='g')
# plt.plot(kz_values, power_spectrum_data_z, label='Power Spectrum along Z', color='b')
# plt.xlabel('Wave Number')
# plt.ylabel('Power Spectrum')
# plt.title('Power Spectrum vs Modes')
# plt.legend()
# plt.grid(True)
# plt.show()

import numpy as np

# Constants
nx = 401
ny = 401
nz = 401
dx = 0.5
dy = 0.5
dz = 0.5
dt = 0.05
n = 1
g = 0
pi = 4 * np.arctan(1)
SORnx = 20001
SORa = 0.01

# Arrays
phi = np.zeros((2, nx, ny, nz))
A = np.zeros((3, nx, ny, nz))

# SOR Fields array
SOR_Fields = np.zeros((SORnx, 2))

# File paths
dir_path = "/Users/pranavbharadwajgangrekalvemanoj/Desktop/Axions_Project"
icPath = f"{dir_path}/Data/ic.txt"
test1sPath = f"{dir_path}/Data/test1s.txt"
test1aPath = f"{dir_path}/Data/test1a.txt"
test2sPath = f"{dir_path}/Data/test2s.txt"
test2aPath = f"{dir_path}/Data/test2a.txt"
SOR_inputPath = f"{dir_path}/SOR_Fields.txt"

# Open files
ic = open(icPath, "w")
test1s = open(test1sPath, "w")
test1a = open(test1aPath, "w")
test2s = open(test2sPath, "w")
test2a = open(test2aPath, "w")

# Read SOR Fields data
with open(SOR_inputPath, "r") as SOR_input:
    for i in range(SORnx):
        SOR_Fields[i, 0], SOR_Fields[i, 1] = map(float, SOR_input.readline().split())

# Function to calculate phiMag and AMag
def calculate_mags(distance, pClosest):
    if pClosest == 0:
        phiMag = (SOR_Fields[pClosest + 1, 0] * distance - SOR_Fields[pClosest, 0] * (distance - SORa)) / SORa
        AMag = (SOR_Fields[pClosest + 1, 1] * distance - SOR_Fields[pClosest, 1] * (distance - SORa)) / SORa
    elif 0 < pClosest < SORnx:
        phiMag = (
            (SOR_Fields[pClosest - 1, 0] * (distance - pClosest * SORa) * (distance - (pClosest + 1) * SORa)
             - 2 * SOR_Fields[pClosest, 0] * (distance - (pClosest - 1) * SORa) * (distance - (pClosest + 1) * SORa)
             + SOR_Fields[pClosest + 1, 0] * (distance - (pClosest - 1) * SORa) * (distance - pClosest * SORa)
             ) / (2 * SORa * SORa)
        )
        AMag = (
            (SOR_Fields[pClosest - 1, 1] * (distance - pClosest * SORa) * (distance - (pClosest + 1) * SORa)
             - 2 * SOR_Fields[pClosest, 1] * (distance - (pClosest - 1) * SORa) * (distance - (pClosest + 1) * SORa)
             + SOR_Fields[pClosest + 1, 1] * (distance - (pClosest - 1) * SORa) * (distance - pClosest * SORa)
             ) / (2 * SORa * SORa)
        )
    else:
        phiMag = 1
        AMag = n / g if g != 0 else 0
        print("Off straight string solution grid")
    return phiMag, AMag

# Loop over the grid
for i in range(nx):
    x = (i - 0.162651) * dx
    for j in range(ny):
        y = j * dy
        for k in range(nz):
            z = -20
            xdist = x - 0.1 * np.sin(2 * np.pi * z) / (2 * np.pi)
            distance = np.sqrt(xdist ** 2 + y ** 2)  # x-y plane distance from string
            pClosest = round(distance / SORa)

            if xdist == 0:
                phi[0, i, j, k] = 0
                A[1, i, j, k] = 0
            else:
                phiMag, AMag = calculate_mags(distance, pClosest)
                phi[0, i, j, k] = phiMag * xdist / distance
                A[1, i, j, k] = AMag * xdist / (distance ** 2)

            if y == 0:
                phi[1, i, j, k] = 0
                A[0, i, j, k] = 0
                A[2, i, j, k] = 0
            else:
                phiMag, _ = calculate_mags(distance, pClosest)
                phi[1, i, j, k] = phiMag * y / distance
                A[0, i, j, k] = -AMag * y / (distance ** 2)
                A[2, i, j, k] = 0

# Write data to files
for i in range(nx):
    for j in range(ny):
        for k in range(nz):
            ic.write(f"{phi[0, i, j, k]} {phi[1, i, j, k]} {dx * g * A[0, i, j, k]} {dy * g * A[1, i, j, k]} {dz * g * A[2, i, j, k]}\n")

# Close files
ic.close()

print("Code execution completed.")

