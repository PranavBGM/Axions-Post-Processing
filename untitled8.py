import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# Path to the directory containing input files
input_directory = "/Users/pranavbharadwajgangrekalvemanoj/Desktop/Axions_Project/evolution_run_2/string_pos_plots_v2/"

# Path to the directory where frames will be saved
output_directory = os.path.join(input_directory, "combined_plots")
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Iterate through all input files and generate plots
for i in range(161):  # Assuming you have 161 input files numbered from 0 to 160
    input_file_path = os.path.join(input_directory, f"combined_data_frame_{i}.txt")
    
    # Load data from the input file
    combined_data = np.loadtxt(input_file_path)
    curvature = combined_data[:, 0]
    x = combined_data[:, 1]
    y = combined_data[:, 2]
    z = combined_data[:, 3]

    # Set limits on curvature, X, Y, and Z axes
    curvature_limit = 20  # Set your desired curvature limit
    x_limit = 0,100  # Set your desired X axis limits

    y_limit = 0,100  # Set your desired Y axis limits
    z_limit = 0, 100  # Set your desired Z axis limits
    
    # Create 3D scatter plot with curvature as colormap
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(x, y, z, c=curvature, cmap='viridis', s=5, vmin=0, vmax=curvature_limit)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim(x_limit)
    ax.set_ylim(y_limit)
    ax.set_zlim(z_limit)
    ax.set_title(f"3D Scatter Plot with Curvature (Frame {i})")
    plt.colorbar(scatter, ax=ax, label='Curvature')
    
    # Save the plot as an image
    plot_path = os.path.join(output_directory, f"plot_frame_{i}.png")
    plt.savefig(plot_path)
    plt.close()

    print(f"Plot for frame {i} generated: {plot_path}")

print("Plots generated for all frames.")




#     print(f"Frame {i} generated: {frame_path}")

# # Convert from indices to physical positions
#     stringData2 = np.copy(data[:,0:3])
#     stringData2[:,0] += 0.5*(nx-1)*dx
#     stringData2[:,1] += 0.5*(ny-1)*dy
#     stringData2[:,2] += 0.5*(nz-1)*dz 
    
#     eps = 0.5 # To stop tree construction issues related to input data > periodic boundaries
#     tree = scipy.spatial.cKDTree(stringData2[:,0:3],boxsize=[nx*dx+eps,ny*dy+eps,nz*dz+eps])
           
#     #neighbours = tree.query_ball_point(stringData[:,0:3],np.sqrt(dx**2+dy**2+dz**2))
#     neighbours = tree.query(stringData2[:,0:3],k=[2,3])
#     lengthCutOff = 0
#     # Curvature calculations:
#     # Already have the lengths of two sides of the triangle formed between query point and 2 neighbours:
#     a = neighbours[0][:,0]
#     b = neighbours[0][:,1]
           
#     # Calculate the total length of string in the simulation.
#     # Sum all neighbour distances for each point and then divide by two at the end to account for double counting.
#     cutoffLogic = (a<lengthCutOff) & (b<lengthCutOff)
#     # stringTotalLength[i] = 0.5*(np.sum(a[a<lengthCutOff]) +np.sum(b[b<lengthCutOff]))
           
#     # Need to calculate the distance between the two neighbours though
#     c = np.zeros(len(stringData2[:,0]))
           
#     # Logic to determine whether to account for periodicity or not
#     xLogic = abs(stringData2[neighbours[1][:,0],0]-stringData2[neighbours[1][:,1],0]) >= 0.5*nx*dx
#     yLogic = abs(stringData2[neighbours[1][:,0],1]-stringData2[neighbours[1][:,1],1]) >= 0.5*ny*dy
#     zLogic = abs(stringData2[neighbours[1][:,0],2]-stringData2[neighbours[1][:,1],2]) >= 0.5*nz*dz
           
#     c[~xLogic] += (stringData2[neighbours[1][~xLogic,0],0] - stringData2[neighbours[1][~xLogic,1],0])**2
#     c[xLogic] += (nx*dx - abs(stringData2[neighbours[1][xLogic,0],0] - stringData2[neighbours[1][xLogic,1],0]) )**2
           
#     c[~yLogic] += (stringData2[neighbours[1][~yLogic,0],1] - stringData2[neighbours[1][~yLogic,1],1])**2
#     c[yLogic] += (ny*dy - abs(stringData2[neighbours[1][yLogic,0],1] - stringData2[neighbours[1][yLogic,1],1]) )**2
           
#     c[~zLogic] += (stringData2[neighbours[1][~zLogic,0],2] - stringData2[neighbours[1][~zLogic,1],2])**2
#     c[zLogic] += (nz*dz - abs(stringData2[neighbours[1][zLogic,0],2] - stringData2[neighbours[1][zLogic,1],2]) )**2
           
#     c = np.sqrt(c) # Convert to the actual distance
           
#     # Calculate the curvature by fitting to a circle and taking inverse of the radius.
#     curv = np.sqrt( (a+b+c)*(-a+b+c)*(a-b+c)*(a+b-c) )/(a*b*c);
#     curv[np.isnan(curv)] = 0
#     curv[np.isinf(curv)] = 0
#     combined_data = np.column_stack((curv, modified_data[:, 0:3]))
    
#     # Save combined data for this frame into a text file
#     combined_data_file_path = os.path.join(output_directory, f"combined_data_frame_{i}.txt")
#     np.savetxt(combined_data_file_path, combined_data)
#     print(f"Combined data for frame {i} saved: {combined_data_file_path}")

#     # ... (your existing code)
    
# print("Combined curvature data and string positions saved for all frames.")