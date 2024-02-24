import re
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

os.environ['PATH'] = '/usr/local/texlive/2023/bin:' + os.environ['PATH']
# matplotlib.use("pgf")

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'text.usetex': True,
    'pgf.texsystem': 'pdflatex',
    'pgf.rcfonts': False,
})
# Read the content of the txt file
txt_directory = "/Users/pranavbharadwajgangrekalvemanoj/Desktop/Axions_Project/pos_files_2_omega/"

with open(os.path.join(txt_directory,"energy_x_max_timestep_output.txt"), 'r') as file:
    data = file.read()
# data = np.loadtxt(os.path.join(txt_directory,"SOR_Fields.txt"))

# y = data[:, 0]

# # Create a range object for the x-axis
# x = range(len(y))

# # Create a figure with two subplots side by side
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# # Create the plot for the full range of x values on the first subplot
# ax1.plot(x, y, color='black', label='simulated radial profile')
# ax1.axhline(y=1, linestyle='--', color='gray', label=r'$\phi = 1$')
# ax1.legend()


# # Create the plot for x values up to 5000 on the second subplot
# ax2.plot(x[:5000], y[:5000], color='black', label='simulated radial profile')
# ax2.axhline(y=1, linestyle='--', color='gray', label=r'$\phi = 1$')
# ax2.legend()


# # Display the plots
# plt.tight_layout()
# plt.show()
# Extracting time steps, energy, and maximum x using regular expressions
time_steps = re.findall(r'Time Step: (\d+)', data)
energy = re.findall(r'Energy: (\d+\.\d+)', data)
max_x = re.findall(r'maximum x: (\d+\.\d+)', data)

# # Convert strings to numerical values
time_steps = np.array(list(map(int, time_steps)), dtype=float)
energy = list(map(float, energy))
max_x = list(map(float, max_x))

# Calculate Amplitude (maximum x translated to eps)
amplitude = np.array([(x * 2 * np.pi) / (0.7 * 51) for x in max_x])


# # Define the |cosx| function
# def cos_func(x, amplitude, period):
#     return amplitude * np.abs(np.cos(2 * np.pi * x / period))

# def exp_decay(x, amplitude, decay_rate):
#     return amplitude * np.exp(-decay_rate * x)

# # Fit the data to the exponential decay function with an initial guess
# initial_guess_exp = [max(amplitude), 0.001]  # You may need to adjust the initial guess values
# popt_exp, pcov_exp = curve_fit(exp_decay, time_steps, amplitude, p0=initial_guess_exp)

# # Extract the fitted parameters for the exponential decay
# fitted_amplitude_exp, fitted_decay_rate_exp = popt_exp

# # Find stationary points using find_peaks with adjusted height
# derivative_exp = np.gradient(exp_decay(time_steps, fitted_amplitude_exp, fitted_decay_rate_exp), time_steps)
# peaks_exp, _ = find_peaks(-derivative_exp, height=0.005)  # Adjust the height value as needed

# initial_guess_cos = [max(amplitude), 50]  # You may need to adjust the initial guess values
# popt_cos, pcov_cos = curve_fit(cos_func, time_steps, amplitude, p0=initial_guess_cos)

# # Extract the fitted parameters for the cosine function
# fitted_amplitude_cos, fitted_period_cos = popt_cos

# # Find stationary points using find_peaks with adjusted height
# derivative_cos = np.gradient(cos_func(time_steps, fitted_amplitude_cos, fitted_period_cos), time_steps)
# peaks_cos, _ = find_peaks(-derivative_cos, height=0.005)  # Adjust the height value as needed
# # Plotting
# ------------------------------------------------------------------------------------------------
plt.figure(figsize=(12, 5))

# Plot Energy vs Time
plt.subplot(1, 2, 1)
plt.plot(time_steps, energy, color='black')
# plt.title('Energy vs Time')
plt.xlabel('Time Step')
plt.ylabel('Energy')

plt.subplot(1, 2, 2)
plt.plot(time_steps, amplitude, color='black')
# # # # plt.plot(time_steps, cos_func(time_steps, fitted_amplitude_cos, fitted_period_cos), label='Fitted Curve (Cosine)')
# # # # plt.plot(time_steps[peaks_cos], amplitude[peaks_cos], 'ro', label='Stationary Points (Cosine)')

# # # # # Plot Amplitude vs Time with exponential decay
# # # # plt.plot(time_steps, amplitude, label='Original Data')
# # # # plt.plot(time_steps, exp_decay(time_steps, fitted_amplitude_exp, fitted_decay_rate_exp), label='Fitted Curve (Exp. Decay)')
# # # # plt.plot(time_steps[peaks_exp], amplitude[peaks_exp], 'ro', label='Stationary Points (Exp. Decay)')

# # # plt.title('Amplitude vs Time')
plt.xlabel('Time Step')
plt.ylabel(r'$\epsilon$')
plt.legend()

# # # Show the plots
# # plt.tight_layout()
plt.show()
# plt.savefig('energy_vs_amp.pgf')
# import matplotlib.pyplot as plt
# import numpy as np
# from scipy.optimize import curve_fit

# # Generate simulated data for energy decay and amplitude decay
# time_steps = np.arange(0, 500, 1)

# # Define energy up to the first 50 time steps
# initial_energy = 1200
# energy_decay_rate = 1000
# energy = np.zeros_like(time_steps)
# energy[:50] = initial_energy

# # Decay energy from time step 50 onwards
# energy[50:] = (initial_energy-1000) + energy_decay_rate / (1 + (time_steps[50:] - 50))

# # Define initial amplitude and decay
# initial_amplitude = 2.5
# amplitude_decay_rate = 0.05
# amplitude = initial_amplitude * (1 - amplitude_decay_rate * time_steps)

# # Model for energy decay
# def energy_decay_model(t, initial_energy, decay_rate):
#     return initial_energy + decay_rate / (1 + t)

# # Model for linear amplitude decay
# def amplitude_decay_model(t, initial_amplitude):
#     return initial_amplitude * (1 - amplitude_decay_rate * t)

# # Fit the models to the data after the 50th time step
# popt_energy, _ = curve_fit(energy_decay_model, time_steps[50:], energy[50:])
# popt_amplitude, _ = curve_fit(amplitude_decay_model, time_steps[50:], amplitude[50:])

# # Create a range for plotting
# t_range = np.linspace(0, max(time_steps), 1000)

# # Plotting
# plt.figure(figsize=(12, 5))

# # Plot Energy Decay
# plt.subplot(1, 2, 1)
# plt.plot(time_steps, energy, color='black')
# plt.axvline(x=time_steps[50], linestyle='--', color='red', label='Light crossing time', ymax=1200/plt.ylim()[1]) 
# # plt.plot(t_range, energy_decay_model(t_range, *popt_energy), label='Energy Decay Model', linestyle='--', color='red')
# # plt.title('Energy Decay vs Time')
# plt.xlabel('Time Step')
# plt.ylabel('Energy')
# plt.legend()




# # Function for exponentially decaying |cosine|
# def decayed_cosine(t, decay_rate=0.01):
#     return np.exp(-decay_rate * t) * np.abs(np.cos(0.2 * t))

# # Generate time values
# time_values = np.linspace(0, 500, 1000)

# # Calculate function values
# function_values = decayed_cosine(time_values)

# # Plot Amplitude Decay
# plt.subplot(1, 2, 2)
# plt.plot(time_values, function_values, color='black')
# plt.xlabel(r'Time $(t)$')
# plt.ylabel(r'Decayed $|cos(t)|$')
# # plt.title('Exponentially Decaying |cosine| Function vs Time')
# plt.legend()


# plt.tight_layout()


# # plt.savefig("model_energy_epsilon.pgf")
# plt.show()

# -------------------------------------------------------------------------------------------------------------------

# import re
# import matplotlib.pyplot as plt
# import numpy as np
# import os
# plt.rcParams['text.usetex'] = False

# # Define the directory paths
# directory_paths = [
#     "/Users/pranavbharadwajgangrekalvemanoj/Desktop/Axions_Project/sem_2_xy_abs_z_fix_101/",
#     "/Users/pranavbharadwajgangrekalvemanoj/Desktop/Axions_Project/energy_301/",
#     "/Users/pranavbharadwajgangrekalvemanoj/Desktop/Axions_Project/energy_501/",
#     "/Users/pranavbharadwajgangrekalvemanoj/Desktop/Axions_Project/energy_601/",
#     "/Users/pranavbharadwajgangrekalvemanoj/Desktop/Axions_Project/energy_701/",
#     "/Users/pranavbharadwajgangrekalvemanoj/Desktop/Axions_Project/energy_801/"
# ]

# # Function to extract data from a file
# def extract_data(file_path):
#     with open(file_path, 'r') as file:
#         data = file.read()
#     time_steps = re.findall(r'Time Step: (\d+)', data)
#     energy = re.findall(r'Energy: (\d+\.\d+)', data)
#     max_x = re.findall(r'maximum x: (\d+\.\d+)', data)
#     time_steps = np.array(list(map(int, time_steps)), dtype=float)
#     energy = list(map(float, energy))
#     max_x = list(map(float, max_x))
#     amplitude = np.array([(x * 2 * np.pi) / (0.7 * 51) for x in max_x])
#     return time_steps, energy, amplitude

# custom_labels = ['101', '301', '501', '601', '701', '801']

# # Iterate over directory paths, read files, and plot
# plt.figure(figsize=(12, 5))
# for i, directory_path in enumerate(directory_paths):
#     file_path = os.path.join(directory_path, "energy_x_max_timestep_output.txt")
#     if os.path.isfile(file_path):
#         time_steps, energy, amplitude = extract_data(file_path)
        
#         # Plot Energy vs Time
#         plt.subplot(1, 2, 1)
#         plt.plot(time_steps, energy, label=custom_labels[i])

#         # Plot Amplitude vs Time
#         plt.subplot(1, 2, 2)
#         plt.plot(time_steps, amplitude, label=custom_labels[i])


# plt.subplot(1, 2, 1)
# plt.xlabel('Time Step')
# plt.ylabel('Energy')
# plt.legend()

# plt.subplot(1, 2, 2)
# plt.xlabel('Time Step')
# plt.ylabel(r'$\epsilon$')
# plt.legend()

# plt.tight_layout()
# plt.show()