# import os
# import imageio

# # # Path to the directory containing frames
# frames_directory_4 = "/Volumes/Ancient Archives/evolution_run_9_v_2/npy_files_ev_9_v_2/tot_en___periodic_z_abs_xy.npy"
# frames_directory_1 = "/Users/pranavbharadwajgangrekalvemanoj/Desktop/Axions_Project/fixed_bc/npy_files_fixed_bc_z_only/tot_en_16000_fixed_z_abs_xy.npy"
# frames_directory_2 = "/Users/pranavbharadwajgangrekalvemanoj/Desktop/Axions_Project/fixed_bc_all/npy_files_fixed_bc/tot_en_5500_fixed_xyz.npy"
# frames_directory_3 = "/Volumes/Ancient Archives/evolution_run_9/npy_files_ev_9/tot_en_8600_fixed_z_absorbing_xy.npy"
# # Path to save the output GIF
# output_gif_path = "/Users/pranavbharadwajgangrekalvemanoj/Desktop/Axions_Project/evolution_run_2/gifs"

# if not os.path.exists(output_gif_path):
#     os.makedirs(output_gif_path)

# # Get a list of frames in the frames directory
# frames = []
# for filename in sorted(os.listdir(frames_directory), key=lambda x: int(x.split("_")[1].split(".")[0])):
#     if filename.startswith("frame") and filename.endswith(".png"):
#         frame_path = os.path.join(frames_directory, filename)
#         frames.append(imageio.imread(frame_path))

# # Output GIF file path
# output_gif_file = os.path.join(output_gif_path, "norm_output.gif")

# # Create the GIF from frames using imageio

# imageio.mimsave(output_gif_file, frames, duration=0.1)
# print(f"GIF created and saved at: {output_gif_file}")

# import numpy as np
# import matplotlib.pyplot as plt

# te1 = np.load(frames_directory_1)
# te2 = np.load(frames_directory_2)
# te3 = np.load(frames_directory_3)
# te4 = np.load(frames_directory_4)

# num_frames_1 = len(te1)
# num_frames_2 = len(te2)
# num_frames_3 = len(te3)
# num_frames_4 = len(te4)

# max_frames = max(num_frames_1, num_frames_2, num_frames_3, num_frames_4)
# te_1_padded = np.pad(te1, (0, max_frames - num_frames_1), mode='constant', constant_values=0)
# te_2_padded = np.pad(te2, (0, max_frames - num_frames_2), mode='constant', constant_values=0)
# te_3_padded = np.pad(te3, (0, max_frames - num_frames_3), mode='constant', constant_values=0)
# te_4_padded = np.pad(te4, (0, max_frames - num_frames_4), mode='constant', constant_values=0)

# mask_1 = te_1_padded != 0
# mask_2 = te_2_padded != 0
# mask_3 = te_3_padded != 0
# mask_4 = te_4_padded != 0

# # Filter out zero values using the masks
# te_1_filtered = te_1_padded[mask_1]
# te_2_filtered = te_2_padded[mask_2]
# te_3_filtered = te_3_padded[mask_3]
# te_4_filtered = te_4_padded[mask_4]

# # Plot total energy vs frame or time for different runs
# plt.figure(figsize=(10, 6))
# plt.plot(np.arange(len(te_1_filtered)), te_1_filtered, marker='.', linestyle='dashed', color='k', label='fixed x,y,z run 1')
# plt.plot(np.arange(len(te_2_filtered)), te_2_filtered, marker='.', linestyle='dashed', color='r', label='fixed x,y,z run 2')
# plt.plot(np.arange(len(te_3_filtered)), te_3_filtered, marker='.', linestyle='dashed', color='purple', label='fixed z, absorbing xy')
# plt.plot(np.arange(len(te_4_filtered)), te_4_filtered, marker='.', linestyle='dashed', color='pink', label='periodic z, absorbing xy')
# plt.legend()
# plt.xlabel('Time step')
# plt.ylabel('Total energy')
# plt.title('Total energy for different runs vs Frame Number')
# Path to the directory containing frames


# # Number of frames
# num_frames = 161  # Update this to match the number of frames you have
# nx,ny,nz = 201,201,201
# # Function to calculate total energy from the energy density grid
# def calculate_total_energy(energy_density):
#     # Assuming energy_density is a 3D NumPy array representing the energy density grid
#     # Compute total energy as the sum of all grid points
#     total_energy = np.sum(energy_density)
#     return total_energy

# def calculate_average_magnitude(magnitude):
#     # Assuming energy_density is a 3D NumPy array representing the energy density grid
#     # Compute total energy as the sum of all grid points
#     total_magnitude = np.sum(magnitude)
#     average_magnitude = total_magnitude / (nx*ny*nz)
#     return average_magnitude

# # Initialize an empty list to store total energies for each frame
# included_frame_numbers = []
# total_energies = []
# average_magnitudes = []
# # actual_energies = [1063.21, 1075.48, 1079.94, 1081.17, 1082.27, 1083.73, 1091.09, 1095.95]
# # Iterate through all frames and calculate total energy
# for i in range(num_frames):
#     try:
#         # Load energy density grid from the corresponding frame file
#         frame_path = os.path.join(frames_directory, f"col_2_frame_{i}.npy")
#         energy_density = np.load(frame_path)  # Load data as a NumPy array
#         # magnitude = np.load(frame_path)
#         # Calculate total energy for the current frame
#         total_energy = calculate_total_energy(energy_density)
#         # average_magnitude = calculate_average_magnitude(magnitude)
#         # Append the total energy and frame number to the lists
#         total_energies.append(total_energy)
#         # average_magnitudes.append(average_magnitude)
#         included_frame_numbers.append(i * 100)
#         print(f"processed file {i}")
#     except Exception as e:
#         # Handle any exceptions (file not found, empty file, etc.) and continue to the next iteration
#         # print(f"Error processing file {i}: {str(e)}")
#         continue
# average_magnitudes = np.array(total_energies) 
# np.save(os.path.join(frames_directory, "tot_en___periodic_z_abs_xy.npy"), average_magnitudes)
# # Plot total energy vs frame or time
# plt.figure(figsize=(10, 6))
# plt.plot(included_frame_numbers, average_magnitudes , marker='o', linestyle='-', color='b')
# # plt.plot(included_frame_numbers, actual_energies , marker='o', linestyle='-', color='r')
# plt.xlabel('Time step')
# plt.ylabel('Total Energy')
# plt.title('Total energy vs Frame Number')

# # # Display frame numbers as labels on data points
# # # for frame_number, total_energy in zip(included_frame_numbers, total_energies):
# # #     plt.text(frame_number, total_energy, f'{frame_number}', ha='right', va='bottom')

# plt.grid(True)

# plt.savefig(os.path.join(frames_directory, "tot_energy_fixed_z_abs_xy_without_labels.png"))
# plt.close()
#--------------------------------------------------------------------------------------------------------------------------------------------------------

#npy_file gen

#--------------------------------------------------------------------------------------------------------------------------------------------------------


# import numpy as np
# import os

# # Path to the directory containing input files
# input_directory = "
# somethings/"

# # Path to the directory where .npy files will be saved
# output_directory = os.path.join(input_directory, "npy_files_ev_10")
# if not os.path.exists(output_directory):
#     os.makedirs(output_directory)

# # Number of frames
# num_frames = 52  # Update this to match the number of frames you have

# # Iterate through all input files and generate .npy files
# for i in range(num_frames):
#     try: 
#         input_file_path = os.path.join(input_directory, f"something_{i}_debug.txt")
        
#         # Load data from the input file
#         array_1d = np.loadtxt(input_file_path)
        
#         # col_0 = array_1d[:, 0]
#         phi_0 = array_1d[:,0]
#         phi_1 = array_1d[:,1]
#         # col_1 = array_1d[:, 1]
#         # col_2 = array_1d[:, 2]
#         # col_3 = array_1d[:, 3]
#         # col_4 = array_1d[:, 4]
#         # col_5 = array_1d[:, 5]
        
#         # Reshape the 1D array into a 3D array
#         nx = 201
#         ny = 201
#         nz = 51
#         # array_3d_col_0 = col_0.reshape(nx, ny, nz)
#         array_3d_col_0 = phi_0.reshape(nx, ny, nz)
#         array_3d_col_1 = phi_1.reshape(nx, ny, nz)
        
#         # array_3d_col_1 = col_1.reshape(nx, ny, nz)
#         # array_3d_col_2 = col_2.reshape(nx, ny, nz)
#         # array_3d_col_3 = col_3.reshape(nx, ny, nz)
#         # array_3d_col_4 = col_4.reshape(nx, ny, nz)
#         # array_3d_col_5 = col_5.reshape(nx, ny, nz)
        
#         # Save the 3D arrays with appropriate filenames
#         np.save(os.path.join(output_directory, f"frame_{i}_pseudo_0.npy"), array_3d_col_0)
#         # np.save(os.path.join(output_directory, "frame_phi_1_evolved.npy"), array_3d_col_1)
#         # np.save(os.path.join(output_directory, "frame_phi_2_evolved.npy"), array_3d_col_2)
        
#         np.save(os.path.join(output_directory, f"frame_{i}_pseudo_1.npy"), array_3d_col_1)
#         # np.save(os.path.join(output_directory, f"col_2_frame_{i}.npy"), array_3d_col_2)
#         # np.save(os.path.join(output_directory, f"negative_debug_col_3.npy"), array_3d_col_3)
        
#         # np.save(os.path.join(output_directory, f"col_3_frame_{i}.npy"), array_3d_col_3)
#         # np.save(os.path.join(output_directory, f"col_4_frame_{i}.npy"), array_3d_col_4)
#         # np.save(os.path.join(output_directory, f"col_5_frame_{i}.npy"), array_3d_col_5)

#         print(f"Arrays saved for frame {i}")
    


#     except Exception as e:
#     # Handle any exceptions (file not found, empty file, etc.) and continue to the next iteration
#         print(f"Error processing file {i}: {str(e)}")
#         continue



#--------------------------------------------------------------------------------------------------------------------------------------------------------

#npy_file gen

#--------------------------------------------------------------------------------------------------------------------------------------------------------


# Initialize empty lists to store total energies for each column
# import os
# import numpy as np
# import matplotlib.pyplot as plt

# # Path to the directory containing frames
# frames_directory = "/Users/pranavbharadwajgangrekalvemanoj/Desktop/Axions_Project/evolution_run_5_field/npy_files"

# # Number of frames
# num_frames = 300  # Update this to match the number of frames you have

# # Initialize empty lists to store total energies for each column
# total_energies_col_0 = []
# total_energies_col_1 = []
# total_energies_col_2 = []
# total_energies_col_3 = []
# total_energies_col_4 = []
# total_energies_col_5 = []

# # Iterate through all frames and calculate total energy for each column
# for i in range(num_frames):
#     try:
#         # Load column data from the corresponding frame files
#         col_0_data = np.load(os.path.join(frames_directory, f"col_0_frame_{i}.npy"))
#         col_1_data = np.load(os.path.join(frames_directory, f"col_1_frame_{i}.npy"))
#         col_2_data = np.load(os.path.join(frames_directory, f"col_2_frame_{i}.npy"))
#         col_3_data = np.load(os.path.join(frames_directory, f"col_3_frame_{i}.npy"))
#         # col_4_data = np.load(os.path.join(frames_directory, f"col_4_frame_{i}.npy"))
#         # col_5_data = np.load(os.path.join(frames_directory, f"col_5_frame_{i}.npy"))
    
#         # Calculate total energy for each column for the current frame
#         total_energy_col_0 = np.sum(col_0_data)
#         total_energy_col_1 = np.sum(col_1_data)
#         total_energy_col_2 = np.sum(col_2_data)
#         total_energy_col_3 = np.sum(col_3_data)
#         # total_energy_col_4 = np.sum(col_4_data)
#         # total_energy_col_5 = np.sum(col_5_data)
        
#         # Append the total energies to the respective lists
#         total_energies_col_0.append(total_energy_col_0)
#         total_energies_col_1.append(total_energy_col_1)
#         total_energies_col_2.append(total_energy_col_2)
#         total_energies_col_3.append(total_energy_col_3)
#         # total_energies_col_4.append(total_energy_col_4)
#         # total_energies_col_5.append(total_energy_col_5)
#     except Exception as e:
#         # Handle any exceptions (file not found, empty file, etc.) and continue to the next iteration
#         print(f"Error processing file {i}: {str(e)}")
#         continue

# # Plot total energies for each column vs frame or time
# plt.figure(figsize=(10, 6))
# plt.plot(range(34), total_energies_col_0, marker='o', linestyle='-', color='b', label='Column 0')
# plt.plot(range(34), total_energies_col_1, marker='o', linestyle='-', color='g', label='Column 1')
# plt.plot(range(34), total_energies_col_2, marker='o', linestyle='-', color='r', label='Column 2')
# plt.plot(range(34), total_energies_col_3, marker='o', linestyle='-', color='c', label='Column 3')
# # plt.plot(range(34), total_energies_col_4, marker='o', linestyle='-', color='m', label='Column 4')
# # plt.plot(range(34), total_energies_col_5, marker='o', linestyle='-', color='y', label='Column 5')
# plt.xlabel('Frame or Time', fontsize=16)
# plt.ylabel('Total Energy', fontsize=16)
# plt.title('Total Energy vs Frame or Time', fontsize=18)
# plt.legend()
# plt.grid(True)
# plt.tight_layout()

# # Save the plot
# plt.savefig("/Users/pranavbharadwajgangrekalvemanoj/Desktop/Axions_Project/evolution_run_5/tot_energy_columns.png")
# print("Plot saved ")
# plt.show()


# import os
# import numpy as np
# import matplotlib.pyplot as plt

#--------------------------------------------------------------------------------------------------------------------------------------------------------

#total energy

#--------------------------------------------------------------------------------------------------------------------------------------------------------




# # Path to the directory containing frames
# frames_directory = "/Users/pranavbharadwajgangrekalvemanoj/Desktop/Axions_Project/evolution_run_5_field/npy_files"

# # Number of frames
# num_frames = 300  # Update this to match the number of frames you have

# # Function to calculate total energy from the energy density grid
# def calculate_total_energy(energy_density):
#     # Assuming energy_density is a 3D NumPy array representing the energy density grid
#     # Compute total energy as the sum of all grid points
#     total_energy = np.sum(energy_density)
#     return total_energy

# # Initialize an empty list to store total energies for each frame
# total_energies = []

# # Iterate through all frames and calculate total energy
# for i in range(num_frames):
#     try:
#     # Load energy density grid from the corresponding frame file
#         frame_path = os.path.join(frames_directory, f"frame_{i}.npy")
#         energy_density = np.load(frame_path)  # Load data as a NumPy array
    
#         # Calculate total energy for the current frame
#         total_energy = calculate_total_energy(energy_density)
        
#         # Append the total energy to the list
#         total_energies.append(total_energy)
#     except Exception as e:
#     # Handle any exceptions (file not found, empty file, etc.) and continue to the next iteration
#         print(f"Error processing file {i}: {str(e)}")
#         continue

# # Plot total energy vs frame or time
# plt.figure(figsize=(10, 6))
# plt.plot(range(34), total_energies, marker='o', linestyle='-', color='b')
# plt.xlabel('Frame or Time', fontsize=16)
# plt.ylabel('Total Energy', fontsize=16)
# plt.title('Total Energy vs Frame or Time', fontsize=18)
# plt.grid(True)
# plt.tight_layout()

# # # Save the plot with LaTeX font
# # output_path = "/Users/pranavbharadwajgangrekalvemanoj/Desktop/Axions_Project/evolution_run_2/total_energy_evol_run_2_178.pdf"
# plt.savefig("/Users/pranavbharadwajgangrekalvemanoj/Desktop/Axions_Project/evolution_run_5/tot_energy.png")
# print("Plot saved ")
# plt.show()
#--------------------------------------------------------------------------------------------------------------------------------------------------------

#string position

#--------------------------------------------------------------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# Set up TeX for rendering
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'text.usetex': True,
    'pgf.texsystem': 'pdflatex',
    'pgf.rcfonts': False,
})

# Function to generate frames
def generate_frame(i, data, output_directory):
    # Create a 3D scatter plot
    # tolerance = 1e-5  # Set a low tolerance for the division remainder
    # data = data[np.abs(data[:, 2] % 0.3) < tolerance]
    data = data[np.abs(data[:, 1]) <= 0.4]
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot with purple markers
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=10, color='#800080', alpha=0.7)

    # Set axis limits
    ax.set_xlim([-30, 30])
    ax.set_ylim([-30, 30])
    ax.set_zlim([-30, 30])

    # Customize appearance
    ax.set_xlabel("X", color='#FF1493')  # Pink color
    ax.set_ylabel("Y", color='#008000')  # Green color
    ax.set_zlabel("Z", color='#FF8C00')  # Dark orange color
    # ax.set_title(f"Frame {i}")

    # Set a different viewing perspective
    ax.view_init(elev=10, azim=60)

    # Remove the background grid
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # Remove grid lines
    ax.grid(False)

    # Set transparent grid lines
    ax.xaxis.pane.set_edgecolor('gray')
    ax.yaxis.pane.set_edgecolor('gray')
    ax.zaxis.pane.set_edgecolor('gray')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # Draw arrows for the axes
    arrow_length = 30
    ax.quiver(0, 0, 0, arrow_length, 0, 0, color='#FF1493', arrow_length_ratio=0.1)  # Pink color
    ax.quiver(0, 0, 0, 0, arrow_length, 0, color='#008000', arrow_length_ratio=0.1)  # Green color
    ax.quiver(0, 0, 0, 0, 0, arrow_length, color='#FF8C00', arrow_length_ratio=0.1)  # Dark orange color

    # Save the plot as a frame
    frame_path = os.path.join(output_directory, f"frame_{i}.png")
    plt.savefig(frame_path)
    plt.close()

    # print(f"Frame {i} generated: {frame_path}")

# Path to the directory containing input files
input_directory = "/Users/pranavbharadwajgangrekalvemanoj/Desktop/Axions_Project/sem_2_xy_abs_z_fix_101/"

# Path to the directory where frames will be saved
output_directory = os.path.join(input_directory, "pos_plots")
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Total number of files
total_files = 5000  # Assuming the files are numbered from 0 to 700

# Iterate through all input files and generate frames
for i in range(total_files):
    input_file_path = os.path.join(input_directory, f"fixed_pergifStringPosData_{i}.txt")

    # Attempt to load data from the input file
    try:
        data = np.loadtxt(input_file_path)  # Load data from file

        # Generate frame
        generate_frame(i, data, output_directory)

    except Exception as e:
        # Handle any exceptions (file not found, empty file, etc.) and continue to the next iteration
        print(f"Error processing file {i}: {str(e)}")
        continue



#--------------------------------------------------------------------------------------------------------------------------------------------------------



#--------------------------------------------------------------------------------------------------------------------------------------------------------

# import os
# import re
# import imageio

# frames_directory = "/Users/pranavbharadwajgangrekalvemanoj/Desktop/Axions_Project/relevant_files_310124/npy_files_full_positive_boost_ev_1/"

# # Get a list of frames in the frames directory
# frames = []

# # Get a list of files in the input directory
# files = sorted(os.listdir(frames_directory))

# # Iterate over the sorted files and append them to the frames list
# for filename in files:
#     if filename.startswith("3D_power_spectrum_radius_") and filename.endswith(".png"):
#         frame_path = os.path.join(frames_directory, filename)
#         frames.append(imageio.imread(frame_path))

# # Now the 'frames' list contains all frames in ascending order: 'frame_00.png', 'frame_01.png', ...


# gif_directory = "/Users/pranavbharadwajgangrekalvemanoj/Desktop/Axions_Project/gifs/"
# if not os.path.exists(gif_directory):
#     os.makedirs(gif_directory)

# output_gif_file = os.path.join(gif_directory, "power_spectrum_vs_radius_3d_Nz_length_scale_2751st_time_step_201.gif")

# # Check if frames exist
# try:
#     imageio.mimsave(output_gif_file, frames, duration=0.1)
#     print(f"GIF created and saved at: {output_gif_file}")
# except Exception as e:
#     print(f"Error creating GIF: {e}")