import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# Path to the input file
input_file_path = "/Users/pranavbharadwajgangrekalvemanoj/Desktop/Axions_Project/evolution_run_1/gifData_178.txt"

# Path to the directory where plots will be saved
output_directory = "/Users/pranavbharadwajgangrekalvemanoj/Desktop/Axions_Project/evolution_run_1/plots"

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Read data from the input file
with open(input_file_path, 'r') as file:
    data = file.read()
    

y = []
with open(input_file_path, 'r') as file:
    for line in file:
        line = line.strip()
        try:
            value = float(line)
            y.append(value)
        except ValueError:
            print(f"Skipping invalid value: {line}")
print(type(y[0]))
x = np.arange(len(y))
print(len(x))
plt.plot(x,y)
plt.xlabel("Index")
plt.ylabel("Value")
plt.show()
plot_file_path = os.path.join(output_directory, 'plot_of_1d_indices_frame_178.png')
plt.savefig(plot_file_path)
plt.close()

print(f"Plot saved at: {plot_file_path}")