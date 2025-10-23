import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# Path to the input file
input_file_path = "/Users/pranavbharadwajgangrekalvemanoj/Desktop/Axions_Project/evolution_run_1/gifData_0.txt"

# Path to the directory where plots will be saved
output_directory = "/Users/pranavbharadwajgangrekalvemanoj/Desktop/Axions_Project/evolution_run_1/plots"

# Ensure the output directory exists, create it if necessary
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Read data from the input file
with open(input_file_path, 'r') as file:
    data = file.read()

class Array:
    def __init__(self, A, B, C):
        self.A = A
        self.B = B
        self.C = C
        self.array = [0] * (A * B * C)

    def get_1d_index(self, a, b, c):
        return a * (self.B * self.C) + b * self.C + c

    @classmethod
    def get_3d_indices_from_1d(cls, index, B, C):
        a = index // (B * C)
        index %= (B * C)
        b = index // C
        c = index % C
        return a, b, c
#function original of the form array[C*(B*a + b) + c]
# Example usage
A = 4
B = 3
C = 2

# Create an instance of the Array class
array = Array(A, B, C)

# Convert 3D indices to 1D index
a, b, c = 2, 1, 0
one_d_index = array.get_1d_index(a, b, c)
print("1D Index:", one_d_index)

# Convert 1D index back to 3D indices using class method
a_new, b_new, c_new = Array.get_3d_indices_from_1d(one_d_index, B, C)
print("Converted 3D Indices:", a_new, b_new, c_new)

with open('test_gif.txt', 'r') as file:
    one_d_array = np.loadtxt(file)

# Assuming B and C dimensions (adjust these based on your array dimensions)
B, C = 10, 10

# Convert the 1D array to 3D arrays using the get_3d_indices_from_1d method
a, b, c = Array.get_3d_indices_from_1d(np.arange(len(one_d_array)), B, C)
array1 = one_d_array

# Create 3D coordinates for the plot
x = np.arange(len(array1))
y = np.arange(B)
z = np.arange(C)
x, y, z = np.meshgrid(x, y, z)

# Create a 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the 3D array with a colormap
sc = ax.scatter(x, y, z, c=array1.flatten(), cmap='viridis')
plt.colorbar(sc)

# Set labels for each axis
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# Show the plot
plt.show()
plot_file_path = os.path.join(output_directory, 'plot.png')
plt.savefig(plot_file_path)
plt.close()

print(f"Plot saved at: {plot_file_path}")


