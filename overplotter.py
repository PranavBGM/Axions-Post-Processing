
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

input_directory_101 = "/Users/pranavbharadwajgangrekalvemanoj/Desktop/Axions_Project/to_scp_060324_101/npy_files_full_positive_boost_ev_1/"
input_directory_101_eps_03 = "/Users/pranavbharadwajgangrekalvemanoj/Desktop/Axions_Project/to_scp_230424_101_eps_0.3/npy_files_full_positive_boost_ev_1/"
input_directory_101_eps_07 = "/Users/pranavbharadwajgangrekalvemanoj/Desktop/Axions_Project/to_scp_230424_101_eps_0.7/npy_files_full_positive_boost_ev_1/"
input_directory_201 = "/Users/pranavbharadwajgangrekalvemanoj/Desktop/Axions_Project/to_scp_060324_201/npy_files_full_positive_boost_ev_1/"
input_directory_201_no_res = "/Users/pranavbharadwajgangrekalvemanoj/Desktop/Axions_Project/to_scp_120324_201_all/npy_files_full_positive_boost_ev_1/"
input_directory_401 = "/Users/pranavbharadwajgangrekalvemanoj/Desktop/Axions_Project/to_scp_090424_401_all/npy_files_full_positive_boost_ev_1/"
input_directory_401_no_res = "/Users/pranavbharadwajgangrekalvemanoj/Desktop/Axions_Project/to_scp_140324_401_all/npy_files_full_positive_boost_ev_1/"

input_directory_201_eps_03 = "/Users/pranavbharadwajgangrekalvemanoj/Desktop/Axions_Project/to_scp_250424_201_eps_0.3/npy_files_full_positive_boost_ev_1/"
input_directory_201_eps_07 = "/Users/pranavbharadwajgangrekalvemanoj/Desktop/Axions_Project/to_scp_250424_201_eps_0.7/npy_files_full_positive_boost_ev_1/"
input_directory_201_res_eps_03 = "/Users/pranavbharadwajgangrekalvemanoj/Desktop/Axions_Project/to_scp_250424_201_res_eps_0.3/npy_files_full_positive_boost_ev_1/"
input_directory_201_res_eps_07 = "/Users/pranavbharadwajgangrekalvemanoj/Desktop/Axions_Project/to_scp_250424_201_res_eps_0.7/npy_files_full_positive_boost_ev_1/"



radii_101 = np.arange(0, 49, 1)
radii_201 = np.arange(0, 99, 1)
radii_401 = np.arange(0, 199, 1)

data_alpha_slice_101_0 = np.loadtxt(f"{input_directory_101}101_data_alpha_slice_0.txt")
# data_alpha_slice_101_eps_03_0 = np.loadtxt(f"{input_directory_101_eps_03}101_data_alpha_slice_0.txt")
# data_alpha_slice_101_eps_07_0 = np.loadtxt(f"{input_directory_101_eps_07}101_data_alpha_slice_0.txt")

data_alpha_slice_201_0 = np.loadtxt(f"{input_directory_201}201_data_alpha_slice_0.txt")
data_not_res_alpha_slice_201_0 = np.loadtxt(f"{input_directory_201_no_res}201_data_alpha_slice_0.txt")
data_alpha_slice_201_eps_03_0 = np.loadtxt(f"{input_directory_201_eps_03}201_data_alpha_slice_0.txt")
data_alpha_slice_201_eps_07_0 = np.loadtxt(f"{input_directory_201_eps_07}201_data_alpha_slice_0.txt")
data_alpha_slice_201_res_eps_03_0 = np.loadtxt(f"{input_directory_201_res_eps_03}201_data_alpha_slice_0.txt")
data_alpha_slice_201_res_eps_07_0 = np.loadtxt(f"{input_directory_201_res_eps_07}201_data_alpha_slice_0.txt")

data_alpha_slice_201_1 = np.loadtxt(f"{input_directory_201}201_data_alpha_slice_1.txt")
data_not_res_alpha_slice_201_1 = np.loadtxt(f"{input_directory_201_no_res}201_data_alpha_slice_1.txt")
data_alpha_slice_201_eps_03_1 = np.loadtxt(f"{input_directory_201_eps_03}201_data_alpha_slice_1.txt")
data_alpha_slice_201_eps_07_1 = np.loadtxt(f"{input_directory_201_eps_07}201_data_alpha_slice_1.txt")
data_alpha_slice_201_res_eps_03_1 = np.loadtxt(f"{input_directory_201_res_eps_03}201_data_alpha_slice_1.txt")
data_alpha_slice_201_res_eps_07_1 = np.loadtxt(f"{input_directory_201_res_eps_07}201_data_alpha_slice_1.txt")

data_alpha_slice_201_2 = np.loadtxt(f"{input_directory_201}201_data_alpha_slice_2.txt")
data_not_res_alpha_slice_201_2 = np.loadtxt(f"{input_directory_201_no_res}201_data_alpha_slice_2.txt")
data_alpha_slice_201_eps_03_2 = np.loadtxt(f"{input_directory_201_eps_03}201_data_alpha_slice_2.txt")
data_alpha_slice_201_eps_07_2 = np.loadtxt(f"{input_directory_201_eps_07}201_data_alpha_slice_2.txt")
data_alpha_slice_201_res_eps_03_2 = np.loadtxt(f"{input_directory_201_res_eps_03}201_data_alpha_slice_2.txt")
data_alpha_slice_201_res_eps_07_2 = np.loadtxt(f"{input_directory_201_res_eps_07}201_data_alpha_slice_2.txt")



data_alpha_slice_201_0 = np.loadtxt(f"{input_directory_201}201_data_alpha_slice_0.txt")
data_alpha_slice_401_0 = np.loadtxt(f"{input_directory_401}401_data_alpha_slice_0.txt")
data_not_res_alpha_slice_201_0 = np.loadtxt(f"{input_directory_201_no_res}201_data_alpha_slice_0.txt")
data_not_res_alpha_slice_401_0 = np.loadtxt(f"{input_directory_401_no_res}401_data_alpha_slice_0.txt")

data_alpha_slice_101_1 = np.loadtxt(f"{input_directory_101}101_data_alpha_slice_1.txt")
# data_alpha_slice_101_eps_03_1 = np.loadtxt(f"{input_directory_101_eps_03}101_data_alpha_slice_1.txt")
# data_alpha_slice_101_eps_07_1 = np.loadtxt(f"{input_directory_101_eps_07}101_data_alpha_slice_1.txt")
data_alpha_slice_201_1 = np.loadtxt(f"{input_directory_201}201_data_alpha_slice_1.txt")
data_alpha_slice_401_1 = np.loadtxt(f"{input_directory_401}401_data_alpha_slice_1.txt")
data_not_res_alpha_slice_201_1 = np.loadtxt(f"{input_directory_201_no_res}201_data_alpha_slice_1.txt")
data_not_res_alpha_slice_401_1 = np.loadtxt(f"{input_directory_401_no_res}401_data_alpha_slice_1.txt")


data_alpha_slice_101_2 = np.loadtxt(f"{input_directory_101}101_data_alpha_slice_2.txt")
# data_alpha_slice_101_eps_03_2 = np.loadtxt(f"{input_directory_101_eps_03}101_data_alpha_slice_2.txt")
# data_alpha_slice_101_eps_07_2 = np.loadtxt(f"{input_directory_101_eps_07}101_data_alpha_slice_2.txt")
data_alpha_slice_201_2 = np.loadtxt(f"{input_directory_201}201_data_alpha_slice_2.txt")
data_alpha_slice_401_2 = np.loadtxt(f"{input_directory_401}401_data_alpha_slice_2.txt")
data_not_res_alpha_slice_201_2 = np.loadtxt(f"{input_directory_201_no_res}201_data_alpha_slice_2.txt")
data_not_res_alpha_slice_401_2 = np.loadtxt(f"{input_directory_401_no_res}401_data_alpha_slice_2.txt")

data_del_alpha_101_0 = np.loadtxt(f"{input_directory_101}101_data_del_alpha_0.txt")
# data_del_alpha_101_eps_03_0 = np.loadtxt(f"{input_directory_101_eps_03}101_data_del_alpha_0.txt")
# data_del_alpha_101_eps_07_0 = np.loadtxt(f"{input_directory_101_eps_07}101_data_del_alpha_0.txt")

data_del_alpha_201_0 = np.loadtxt(f"{input_directory_201}201_data_del_alpha_0.txt")
data_not_res_del_alpha_201_0 = np.loadtxt(f"{input_directory_201_no_res}201_data_del_alpha_0.txt")
data_del_alpha_201_eps_03_0 = np.loadtxt(f"{input_directory_201_eps_03}201_data_del_alpha_0.txt")
data_del_alpha_201_eps_07_0 = np.loadtxt(f"{input_directory_201_eps_07}201_data_del_alpha_0.txt")
data_del_alpha_201_res_eps_03_0 = np.loadtxt(f"{input_directory_201_res_eps_03}201_data_del_alpha_0.txt")
data_del_alpha_201_res_eps_07_0 = np.loadtxt(f"{input_directory_201_res_eps_07}201_data_del_alpha_0.txt")

data_del_alpha_201_1 = np.loadtxt(f"{input_directory_201}201_data_del_alpha_1.txt")
data_not_res_del_alpha_201_1 = np.loadtxt(f"{input_directory_201_no_res}201_data_del_alpha_1.txt")
data_del_alpha_201_eps_03_1 = np.loadtxt(f"{input_directory_201_eps_03}201_data_del_alpha_1.txt")
data_del_alpha_201_eps_07_1 = np.loadtxt(f"{input_directory_201_eps_07}201_data_del_alpha_1.txt")
data_del_alpha_201_res_eps_03_1 = np.loadtxt(f"{input_directory_201_res_eps_03}201_data_del_alpha_1.txt")
data_del_alpha_201_res_eps_07_1 = np.loadtxt(f"{input_directory_201_res_eps_07}201_data_del_alpha_1.txt")

data_del_alpha_201_2 = np.loadtxt(f"{input_directory_201}201_data_del_alpha_2.txt")
data_not_res_del_alpha_201_2 = np.loadtxt(f"{input_directory_201_no_res}201_data_del_alpha_2.txt")
data_del_alpha_201_eps_03_2 = np.loadtxt(f"{input_directory_201_eps_03}201_data_del_alpha_2.txt")
data_del_alpha_201_eps_07_2 = np.loadtxt(f"{input_directory_201_eps_07}201_data_del_alpha_2.txt")
data_del_alpha_201_res_eps_03_2 = np.loadtxt(f"{input_directory_201_res_eps_03}201_data_del_alpha_2.txt")
data_del_alpha_201_res_eps_07_2 = np.loadtxt(f"{input_directory_201_res_eps_07}201_data_del_alpha_2.txt")



data_del_alpha_201_0 = np.loadtxt(f"{input_directory_201}201_data_del_alpha_0.txt")
data_del_alpha_401_0 = np.loadtxt(f"{input_directory_401}401_data_del_alpha_0.txt")
data_not_res_del_alpha_201_0 = np.loadtxt(f"{input_directory_201_no_res}201_data_del_alpha_0.txt")
data_not_res_del_alpha_401_0 = np.loadtxt(f"{input_directory_401_no_res}401_data_del_alpha_0.txt")

data_del_alpha_101_1 = np.loadtxt(f"{input_directory_101}101_data_del_alpha_1.txt")
# data_del_alpha_101_eps_03_1 = np.loadtxt(f"{input_directory_101_eps_03}101_data_del_alpha_1.txt")
# data_del_alpha_101_eps_07_1 = np.loadtxt(f"{input_directory_101_eps_07}101_data_del_alpha_1.txt")
data_del_alpha_201_1 = np.loadtxt(f"{input_directory_201}201_data_del_alpha_1.txt")
data_del_alpha_401_1 = np.loadtxt(f"{input_directory_401}401_data_del_alpha_1.txt")
data_not_res_del_alpha_201_1 = np.loadtxt(f"{input_directory_201_no_res}201_data_del_alpha_1.txt")
data_not_res_del_alpha_401_1 = np.loadtxt(f"{input_directory_401_no_res}401_data_del_alpha_1.txt")

data_del_alpha_101_2 = np.loadtxt(f"{input_directory_101}101_data_del_alpha_2.txt")
# data_del_alpha_101_eps_03_2 = np.loadtxt(f"{input_directory_101_eps_03}101_data_del_alpha_2.txt")
# data_del_alpha_101_eps_07_2 = np.loadtxt(f"{input_directory_101_eps_07}101_data_del_alpha_2.txt")
data_del_alpha_201_2 = np.loadtxt(f"{input_directory_201}201_data_del_alpha_2.txt")
data_del_alpha_401_2 = np.loadtxt(f"{input_directory_401}401_data_del_alpha_2.txt")
data_not_res_del_alpha_201_2= np.loadtxt(f"{input_directory_201_no_res}201_data_del_alpha_2.txt")
data_not_res_del_alpha_401_2 = np.loadtxt(f"{input_directory_401_no_res}401_data_del_alpha_2.txt")





fig, axs = plt.subplots(2, 3, figsize=(15, 10))

# # Plot the data



# axs[0,0].plot(radii_201*0.7, data_alpha_slice_201_eps_03_0, label=r' $\phi \dot{\alpha}$ 201 $\Delta x = 0.7$, $\epsilon = 0.3$', color='lightsteelblue')
# axs[0,0].plot(radii_201*0.7, data_not_res_alpha_slice_201_0, label=r' $\phi \dot{\alpha}$ 201 $\Delta x = 0.7$, $\epsilon = 0.5$', color='royalblue')
# axs[0,0].plot(radii_201*0.7, data_alpha_slice_201_eps_07_0, label=r' $\phi \dot{\alpha}$ 201 $\Delta x = 0.7$, $\epsilon = 0.7$', color='mediumblue')

# axs[0,0].plot(radii_201*0.7, data_del_alpha_201_eps_03_0, label=r'$\phi \dot{\Delta \alpha}$ 201 $\Delta x = 0.7$, $\epsilon = 0.3$', color='lightcoral')
# axs[0,0].plot(radii_201*0.7, data_not_res_del_alpha_201_0, label=r'$\phi \dot{\Delta \alpha}$ 201 $\Delta x = 0.7$, $\epsilon = 0.5$', color='red')
# axs[0,0].plot(radii_201*0.7, data_del_alpha_201_eps_07_0, label=r'$\phi \dot{\Delta \alpha}$ 201 $\Delta x = 0.7$, $\epsilon = 0.7$', color='darkred')

# # axs[0,0].set_xlabel('Radius in physical spatial units (Number of grid points x spatial resolution)')
# # axs[0,0].set_ylabel('Power spectrum')
# axs[0,0].set_ylim(0,2e-5)


# # Add a legend
# axs[0,0].legend()
# # axs[0,0].set_title('First bin contributions')




# axs[0,1].plot(radii_201*0.7, data_alpha_slice_201_eps_03_1, label=r' $\phi \dot{\alpha}$ 201 $\Delta x = 0.7$, $\epsilon = 0.3$', color='lightsteelblue')
# axs[0,1].plot(radii_201*0.7, data_not_res_alpha_slice_201_1, label=r' $\phi \dot{\alpha}$ 201 $\Delta x = 0.7$, $\epsilon = 0.5$', color='royalblue')
# axs[0,1].plot(radii_201*0.7, data_alpha_slice_201_eps_07_1, label=r' $\phi \dot{\alpha}$ 201 $\Delta x = 0.7$, $\epsilon = 0.7$', color='mediumblue')

# axs[0,1].plot(radii_201*0.7, data_del_alpha_201_eps_03_1, label=r'$\phi \dot{\Delta \alpha}$ 201 $\Delta x = 0.7$, $\epsilon = 0.3$', color='lightcoral')
# axs[0,1].plot(radii_201*0.7, data_not_res_del_alpha_201_1, label=r'$\phi \dot{\Delta \alpha}$ 201 $\Delta x = 0.7$, $\epsilon = 0.5$', color='red')
# axs[0,1].plot(radii_201*0.7, data_del_alpha_201_eps_07_1, label=r'$\phi \dot{\Delta \alpha}$ 201 $\Delta x = 0.7$, $\epsilon = 0.7$', color='darkred')
# axs[0,1].set_ylim(0,2e-5)
# # axs[0,1].set_xlabel('Radius in physical spatial units (Number of grid points x spatial resolution)')
# # axs[0,1].set_ylabel('Power spectrum')
# axs[0,1].set_xticklabels([])
# axs[0,1].set_yticklabels([])
# # Add a legend
# # axs[0,1].legend()
# # axs[0,1].set_title('Second bin contributions')




# axs[0,2].plot(radii_201*0.7, data_alpha_slice_201_eps_03_2, label=r' $\phi \dot{\alpha}$ 201 $\Delta x = 0.7$, $\epsilon = 0.3$', color='lightsteelblue')
# axs[0,2].plot(radii_201*0.7, data_not_res_alpha_slice_201_2, label=r' $\phi \dot{\alpha}$ 201 $\Delta x = 0.7$, $\epsilon = 0.5$', color='royalblue')
# axs[0,2].plot(radii_201*0.7, data_alpha_slice_201_eps_07_2, label=r' $\phi \dot{\alpha}$ 201 $\Delta x = 0.7$, $\epsilon = 0.7$', color='mediumblue')

# axs[0,2].plot(radii_201*0.7, data_del_alpha_201_eps_03_2, label=r'$\phi \dot{\Delta \alpha}$ 201 $\Delta x = 0.7$, $\epsilon = 0.3$', color='lightcoral')
# axs[0,2].plot(radii_201*0.7, data_not_res_del_alpha_201_2, label=r'$\phi \dot{\Delta \alpha}$ 201 $\Delta x = 0.7$, $\epsilon = 0.5$', color='red')
# axs[0,2].plot(radii_201*0.7, data_del_alpha_201_eps_07_2, label=r'$\phi \dot{\Delta \alpha}$ 201 $\Delta x = 0.7$, $\epsilon = 0.7$', color='darkred')
# axs[0,2].set_ylim(0,2e-5)
# # axs[0,2].set_xlabel('Radius in physical spatial units (Number of grid points x spatial resolution)')
# # axs[0,2].set_ylabel('Power spectrum')
# axs[0,2].set_xticklabels([])
# axs[0,2].set_yticklabels([])
# # Add a legend
# # axs[0,2].legend()
# # axs[0,2].set_title('Third bin contributions')






# axs[1,0].plot(radii_201*0.35, data_alpha_slice_201_res_eps_03_0, label=r' $\phi \dot{\alpha}$ 201 $\Delta x = 0.35$, $\epsilon = 0.3$', color='lightsteelblue')
# axs[1,0].plot(radii_201*0.35, data_alpha_slice_201_0, label=r' $\phi \dot{\alpha}$ 201 $\Delta x = 0.35$, $\epsilon = 0.5$', color='royalblue')
# axs[1,0].plot(radii_201*0.35, data_alpha_slice_201_res_eps_07_0, label=r' $\phi \dot{\alpha}$ 201 $\Delta x = 0.35$, $\epsilon = 0.7$', color='mediumblue')

# axs[1,0].plot(radii_201*0.35, data_del_alpha_201_res_eps_03_0, label=r'$\phi \dot{\Delta \alpha}$ 201 $\Delta x = 0.35$, $\epsilon = 0.3$', color='lightcoral')
# axs[1,0].plot(radii_201*0.35, data_del_alpha_201_0, label=r'$\phi \dot{\Delta \alpha}$ 201 $\Delta x = 0.35$, $\epsilon = 0.5$', color='red')
# axs[1,0].plot(radii_201*0.35, data_del_alpha_201_res_eps_07_0, label=r'$\phi \dot{\Delta \alpha}$ 201 $\Delta x = 0.35$, $\epsilon = 0.7$', color='darkred')
# axs[1,0].set_ylim(0,9e-5)
# # Hide x and y grid values for the subplot

# # axs[1,0].set_xlabel('Radius in physical spatial units (Number of grid points x spatial resolution)')
# # axs[1,0].set_ylabel('Power spectrum')

# # Add a legend
# axs[1,0].legend()
# # axs[1,0].set_title('First bin contributions')



# axs[1,1].plot(radii_201*0.35, data_alpha_slice_201_res_eps_03_1, label=r' $\phi \dot{\alpha}$ 201 $\Delta x = 0.35$, $\epsilon = 0.3$', color='lightsteelblue')
# axs[1,1].plot(radii_201*0.35, data_alpha_slice_201_1, label=r' $\phi \dot{\alpha}$ 201 $\Delta x = 0.35$, $\epsilon = 0.5$', color='royalblue')
# axs[1,1].plot(radii_201*0.35, data_alpha_slice_201_res_eps_07_1, label=r' $\phi \dot{\alpha}$ 201 $\Delta x = 0.35$, $\epsilon = 0.7$', color='mediumblue')

# axs[1,1].plot(radii_201*0.35, data_del_alpha_201_res_eps_03_1, label=r'$\phi \dot{\Delta \alpha}$ 201 $\Delta x = 0.35$, $\epsilon = 0.3$', color='lightcoral')
# axs[1,1].plot(radii_201*0.35, data_del_alpha_201_1, label=r'$\phi \dot{\Delta \alpha}$ 201 $\Delta x = 0.35$, $\epsilon = 0.5$', color='red')
# axs[1,1].plot(radii_201*0.35, data_del_alpha_201_res_eps_07_1, label=r'$\phi \dot{\Delta \alpha}$ 201 $\Delta x = 0.35$, $\epsilon = 0.7$', color='darkred')
# axs[1,1].set_ylim(0,9e-5)
# # axs[0,1].set_xlabel('Radius in physical spatial units (Number of grid points x spatial resolution)')
# # axs[0,1].set_ylabel('Power spectrum')
# axs[1,1].set_xticklabels([])
# axs[1,1].set_yticklabels([])
# # Add a legend
# # axs[1,1].legend()
# # axs[1,1].set_title('Second bin contributions')




# axs[1,2].plot(radii_201*0.35, data_alpha_slice_201_res_eps_03_2, label=r' $\phi \dot{\alpha}$ 201 $\Delta x = 0.35$, $\epsilon = 0.3$', color='lightsteelblue')
# axs[1,2].plot(radii_201*0.35, data_alpha_slice_201_2, label=r' $\phi \dot{\alpha}$ 201 $\Delta x = 0.35$, $\epsilon = 0.5$', color='royalblue')
# axs[1,2].plot(radii_201*0.35, data_alpha_slice_201_res_eps_07_2, label=r' $\phi \dot{\alpha}$ 201 $\Delta x = 0.35$, $\epsilon = 0.7$', color='mediumblue')

# axs[1,2].plot(radii_201*0.35, data_del_alpha_201_res_eps_03_2, label=r'$\phi \dot{\Delta \alpha}$ 201 $\Delta x = 0.35$, $\epsilon = 0.3$', color='lightcoral')
# axs[1,2].plot(radii_201*0.35, data_del_alpha_201_2, label=r'$\phi \dot{\Delta \alpha}$ 201 $\Delta x = 0.35$, $\epsilon = 0.5$', color='red')
# axs[1,2].plot(radii_201*0.35, data_del_alpha_201_res_eps_07_2, label=r'$\phi \dot{\Delta \alpha}$ 201 $\Delta x = 0.35$, $\epsilon = 0.7$', color='darkred')

# # axs[0,2].set_xlabel('Radius in physical spatial units (Number of grid points x spatial resolution)')
# # axs[0,2].set_ylabel('Power spectrum')
# axs[1,2].set_ylim(0,9e-5)

axs[0,2].set_yticklabels([])
axs[1,2].set_yticklabels([])
# plt.show()



# Add a legend
# axs[1,2].legend()
# axs[1,2].set_title('Third bin contributions')

# axs[0].plot(radii_201*0.35, data_del_alpha_201_0, label=r' $\phi \dot{\Delta \alpha}$ 201 $\Delta x = 0.35$' , color='red', linestyle='--')
# axs[0].plot(radii_201*0.35, data_alpha_slice_201_0, label=r' $\phi \dot{\alpha}$ 201 $\Delta x = 0.35$', color='blue', linestyle='--')
# axs[0].plot(radii_401*0.175, data_alpha_slice_401_0, label=r' $\phi \dot{\alpha}$ 401 $\Delta x = 0.175$', color='blue', linestyle='-.')
# axs[0].plot(radii_401*0.175, data_del_alpha_401_0, label=r' $\phi \dot{\Delta \alpha}$ 401 $\Delta x = 0.175$' , color='red', linestyle='-.')
axs[0,0].plot(radii_101*0.7, data_del_alpha_101_0, label=r'$\phi \dot{\Delta \alpha}$, $N_x = 101$, $\Delta x = 0.7$', color='lime')
axs[0,0].plot(radii_201*0.7, data_not_res_del_alpha_201_0, label=r' $\phi \dot{\Delta \alpha}$, $N_x = 201$, $\Delta x = 0.7$', color='forestgreen')
axs[0,0].plot(radii_401*0.7, data_not_res_del_alpha_401_0, label=r' $\phi \dot{\Delta \alpha}$, $N_x = 401$, $\Delta x = 0.7$', color='darkolivegreen')

axs[0,0].plot(radii_101*0.7, data_alpha_slice_101_0, label=r'$\phi \dot{\alpha}$, $N_x = 101$, $\Delta x = 0.7$', color='#D87EFF')

axs[0,0].plot(radii_201*0.7, data_not_res_alpha_slice_201_0, label=r' $\phi \dot{\alpha}$, $N_x = 201$, $\Delta x = 0.7$', color='#B000FC')


axs[0,0].plot(radii_401*0.7, data_not_res_alpha_slice_401_0, label=r' $\phi \dot{\alpha}$, $N_x = 401$, $\Delta x = 0.7$',color='#7300A4')
axs[0,0].set_ylim(0,5e-5)
axs[0,1].set_ylim(0,5e-5)
axs[0,2].set_ylim(0,5e-5)
axs[1,0].set_xlabel('Radius of circular mask', fontsize=14)
axs[1,0].set_ylabel('Power spectrum', fontsize=14)

axs[0,0].set_xlim(-2, 37)
# axs[0,0].set_xticklabels([])
axs[0,1].set_yticklabels([])

axs[0,1].set_xlim(-2, 37)
# axs[1,0].set_xticklabels([])
axs[1,1].set_yticklabels([])
axs[0,2].set_xlim(-2, 37)
axs[1,0].set_ylim(0,8e-5)
axs[1,1].set_ylim(0,8e-5)
axs[1,2].set_ylim(0,8e-5)
# axs[1,0].set_xlim(0, 35)
# axs[1,1].set_xlim(0, 35)
# axs[1,2].set_xlim(0, 35)

# Add a legend
axs[0,0].legend(fontsize=14)
# axs[0,0].set_title('First bin contributions')



# axs[0,1].plot(radii_201*0.35, data_alpha_slice_201_1, label=r' $\phi \dot{\alpha}$ 201 $\Delta x = 0.35$', color='blue', linestyle='--')
# axs[0,1].plot(radii_201*0.35, data_del_alpha_201_1, label=r' $\phi \dot{\Delta \alpha}$ 201 $\Delta x = 0.35$', color='red', linestyle='--')

axs[0,1].plot(radii_101*0.7, data_del_alpha_101_1, label=r'$\phi \dot{\Delta \alpha}$, $N_x = 101$, $\Delta x = 0.7$', color='lime')
# axs[0,1].plot(radii_101*0.7, data_alpha_slice_201_eps_03_1, label=r' $\phi \dot{\alpha}$ 101 $\Delta x = 0.7$, $\epsilon = 0.3$', color='blue', linestyle='--')
# axs[0,1].plot(radii_101*0.7, data_alpha_slice_201_eps_07_1, label=r' $\phi \dot{\alpha}$ 101 $\Delta x = 0.7$, $\epsilon = 0.7$', color='blue', linestyle='-.')

# axs[0,1].plot(radii_101*0.7, data_del_alpha_201_eps_03_1, label=r'$\phi \dot{\Delta \alpha}$ 101 $\Delta x = 0.7$, $\epsilon = 0.3$', color='red', linestyle='--')
# axs[0,1].plot(radii_101*0.7, data_del_alpha_201_eps_07_1, label=r'$\phi \dot{\Delta \alpha}$ 101 $\Delta x = 0.7$, $\epsilon = 0.7$', color='red', linestyle='-.')
# axs[0,1].plot(radii_401*0.175, data_alpha_slice_401_1, label=r' $\phi \dot{\alpha}$ 401 $\Delta x = 0.175$', color='blue', linestyle='-.')
# axs[0,1].plot(radii_401*0.175, data_del_alpha_401_1, label=r' $\phi \dot{\Delta \alpha}$ 401 $\Delta x = 0.175$' , color='red', linestyle='-.')


axs[0,1].plot(radii_201*0.7, data_not_res_del_alpha_201_1, label=r' $\phi \dot{\Delta \alpha}$, $N_x = 201$, $\Delta x = 0.7$', color='forestgreen')

axs[0,1].plot(radii_401*0.7, data_not_res_del_alpha_401_1, label=r' $\phi \dot{\Delta \alpha}$, $N_x = 401$, $\Delta x = 0.7$', color='darkolivegreen')

axs[0,1].plot(radii_101*0.7, data_alpha_slice_101_1, label=r' $\phi \dot{\alpha}$, $N_x = 101$, $\Delta x = 0.7$', color='#D87EFF')
axs[0,1].plot(radii_201*0.7, data_not_res_alpha_slice_201_1, label=r' $\phi \dot{\alpha}$, $N_x = 201$, $\Delta x = 0.7$', color='#B000FC')


axs[0,1].plot(radii_401*0.7, data_not_res_alpha_slice_401_1, label=r' $\phi \dot{\alpha}$, $N_x = 401$, $\Delta x = 0.7$', color='#7300A4')

# Add a legend
# axs[1].legend()
# axs[1].set_title('Second bin contributions')



# axs[0,2].plot(radii_201*0.35, data_alpha_slice_201_2, label=r' $\phi \dot{\alpha}$ 201 $\Delta x = 0.35$', color='blue', linestyle='--')
# axs[0,2].plot(radii_201*0.35, data_del_alpha_201_2, label=r' $\phi \dot{\Delta \alpha}$ 201 $\Delta x = 0.35$', color='red', linestyle='--')

# axs[0,2].plot(radii_401*0.175, data_alpha_slice_401_2, label=r' $\phi \dot{\alpha}$ 401 $\Delta x = 0.175$', color='blue', linestyle='-.')
# axs[0,2].plot(radii_401*0.175, data_del_alpha_401_2, label=r' $\phi \dot{\Delta \alpha}$ 401 $\Delta x = 0.175$' , color='red', linestyle='-.')
# axs[2].plot(radii_101*0.7, data_alpha_slice_101_eps_03_2, label=r' $\phi \dot{\alpha}$ 101 $\Delta x = 0.7$, $\epsilon = 0.3$', color='blue', linestyle='--')
# axs[2].plot(radii_101*0.7, data_alpha_slice_101_eps_07_2, label=r' $\phi \dot{\alpha}$ 101 $\Delta x = 0.7$, $\epsilon = 0.7$', color='blue', linestyle='-.')

# axs[2].plot(radii_101*0.7, data_del_alpha_101_eps_03_2, label=r'$\phi \dot{\Delta \alpha}$ 101 $\Delta x = 0.7$, $\epsilon = 0.3$', color='red', linestyle='--')
# axs[2].plot(radii_101*0.7, data_del_alpha_101_eps_07_2, label=r'$\phi \dot{\Delta \alpha}$ 101 $\Delta x = 0.7$, $\epsilon = 0.7$', color='red', linestyle='-.')

axs[0,2].plot(radii_101*0.7, data_del_alpha_101_2, label=r'$\phi \dot{\Delta \alpha}$, $N_x = 101$, $\Delta x = 0.7$', color='lime')
axs[0,2].plot(radii_201*0.7, data_not_res_del_alpha_201_2, label=r' $\phi \dot{\Delta \alpha}$, $N_x = 201$, $\Delta x = 0.7$', color='forestgreen')
axs[0,2].plot(radii_401*0.7, data_not_res_del_alpha_401_2, label=r' $\phi \dot{\Delta \alpha}$, $N_x = 401$, $\Delta x = 0.7$', color='darkolivegreen')

axs[0,2].plot(radii_101*0.7, data_alpha_slice_101_2, label=r' $\phi \dot{\alpha}$, $N_x = 101$, $\Delta x = 0.7$', color='#D87EFF')
axs[0,2].plot(radii_201*0.7, data_not_res_alpha_slice_201_2, label=r' $\phi \dot{\alpha}$, $N_x = 201$, $\Delta x = 0.7$', color='#B000FC')
axs[0,2].plot(radii_401*0.7, data_not_res_alpha_slice_401_2, label=r' $\phi \dot{\alpha}$, $N_x = 401$, $\Delta x = 0.7$', color='#7300A4')
# Add a legend
# axs[2].legend()
# axs[2].set_title('Third bin contributions')




axs[1,0].plot(radii_101*0.7, data_del_alpha_101_0, label=r'$\phi \dot{\Delta \alpha}$, $N_x = 101$, $\Delta x = 0.7$', color='lime')
axs[1,0].plot(radii_201*0.35, data_del_alpha_201_0, label=r' $\phi \dot{\Delta \alpha}$, $N_x = 201$, $\Delta x = 0.35$' , color='forestgreen')
axs[1,0].plot(radii_401*0.175, data_del_alpha_401_0, label=r' $\phi \dot{\Delta \alpha}$, $N_x = 401$, $\Delta x = 0.175$' , color='darkolivegreen')

axs[1,0].plot(radii_101*0.7, data_alpha_slice_101_0, label=r' $\phi \dot{\alpha}$, $N_x = 101$, $\Delta x = 0.7$', color='#D87EFF')
axs[1,0].plot(radii_201*0.35, data_alpha_slice_201_0, label=r' $\phi \dot{\alpha}$, $N_x = 201$, $\Delta x = 0.35$', color='#B000FC')
axs[1,0].plot(radii_401*0.175, data_alpha_slice_401_0, label=r' $\phi \dot{\alpha}$, $N_x = 401$, $\Delta x = 0.175$', color='#7300A4')

axs[1,0].legend(fontsize =14)



axs[1,1].plot(radii_101*0.7, data_del_alpha_101_1, label=r'$\phi \dot{\Delta \alpha}$, $N_x = 101$, $\Delta x = 0.7$', color='lime')
axs[1,1].plot(radii_201*0.35, data_del_alpha_201_1, label=r' $\phi \dot{\Delta \alpha}$, $N_x = 201$, $\Delta x = 0.35$', color='forestgreen')
axs[1,1].plot(radii_401*0.175, data_del_alpha_401_1, label=r' $\phi \dot{\Delta \alpha}$, $N_x = 401$, $\Delta x = 0.175$' , color='darkolivegreen')

axs[1,1].plot(radii_101*0.7, data_alpha_slice_101_1, label=r' $\phi \dot{\alpha}$, $N_x = 101$, $\Delta x = 0.7$', color='#D87EFF')
axs[1,1].plot(radii_201*0.35, data_alpha_slice_201_1, label=r' $\phi \dot{\alpha}$, $N_x = 201$, $\Delta x = 0.35$', color='#B000FC')
axs[1,1].plot(radii_401*0.175, data_alpha_slice_401_1, label=r' $\phi \dot{\alpha}$, $N_x = 401$, $\Delta x = 0.175$', color='#7300A4')


# axs[1,1].legend()
axs[1,2].plot(radii_101*0.7, data_del_alpha_101_2, label=r'$\phi \dot{\Delta \alpha}$, $N_x = 101$, $\Delta x = 0.7$', color='lime')
axs[1,2].plot(radii_201*0.35, data_del_alpha_201_2, label=r' $\phi \dot{\Delta \alpha}$, $N_x = 201$, $\Delta x = 0.35$', color='forestgreen')
axs[1,2].plot(radii_401*0.175, data_del_alpha_401_2, label=r' $\phi \dot{\Delta \alpha}$, $N_x = 401$, $\Delta x = 0.175$' , color='darkolivegreen')

axs[1,2].plot(radii_101*0.7, data_alpha_slice_101_2, label=r' $\phi \dot{\alpha}$, $N_x = 101$, $\Delta x = 0.7$', color='#D87EFF')
axs[1,2].plot(radii_201*0.35, data_alpha_slice_201_2, label=r' $\phi \dot{\alpha}$, $N_x = 201$, $\Delta x = 0.35$', color='#B000FC')
axs[1,2].plot(radii_401*0.175, data_alpha_slice_401_2, label=r' $\phi \dot{\alpha}$, $N_x = 401$, $\Delta x = 0.175$', color='#7300A4')


axs[0, 0].tick_params(axis='both', which='major', labelsize=14)
axs[0, 1].tick_params(axis='both', which='major', labelsize=14)
axs[0, 2].tick_params(axis='both', which='major', labelsize=14)
axs[1, 0].tick_params(axis='both', which='major', labelsize=14)
axs[1, 1].tick_params(axis='both', which='major', labelsize=14)
axs[1, 2].tick_params(axis='both', which='major', labelsize=14)


# axs[2,2].legend()
plt.subplots_adjust(wspace=0, hspace=0.15)
plt.show()

# Show the plot


# import numpy as np
# import matplotlib.pyplot as plt

# def load_data(directory, filename):
#     return np.loadtxt(f"{directory}{filename}")

# def plot_data(ax, radii, data, label, color, linestyle, marker=None):
#     ax.plot(radii, data, label=label, color=color, linestyle=linestyle, marker=marker)

# def generate_label(prefix, index, resolution):
#     return f"{prefix} {index} Δx = {resolution}"

# input_directories = [
#     "/Users/pranavbharadwajgangrekalvemanoj/Desktop/Axions_Project/to_scp_060324_101/npy_files_full_positive_boost_ev_1/",
#     "/Users/pranavbharadwajgangrekalvemanoj/Desktop/Axions_Project/to_scp_060324_201/npy_files_full_positive_boost_ev_1/",
#     "/Users/pranavbharadwajgangrekalvemanoj/Desktop/Axions_Project/to_scp_120324_201_all/npy_files_full_positive_boost_ev_1/",
#     "/Users/pranavbharadwajgangrekalvemanoj/Desktop/Axions_Project/to_scp_090424_401_all/npy_files_full_positive_boost_ev_1/",
#     "/Users/pranavbharadwajgangrekalvemanoj/Desktop/Axions_Project/to_scp_140324_401_all/npy_files_full_positive_boost_ev_1/"
# ]

# radii = [np.arange(0, end, 1) for end in [49, 99, 199]]


# indices = [1, 2, 2, 4, 4]

# data_alpha_slice = [[load_data(dir, f"{indices[i]}01_data_alpha_slice_{j}.txt") for j in range(3)] for i, dir in enumerate(input_directories)]
# data_del_alpha = [[load_data(dir, f"{indices[i]}01_data_del_alpha_{j}.txt") for j in range(3)] for i, dir in enumerate(input_directories)]

# fig, axs = plt.subplots(2, 3, figsize=(15, 10))

# resolutions = [0.7, 0.7, 0.35, 0.7, 0.175]

# for i in range(3):
#     for j in range(2):
#         for k in range(len(input_directories)):
#             radii = np.arange(0, len(data_alpha_slice[k][i]), 1) * resolutions[k]
#             plot_data(axs[j, i], radii, data_alpha_slice[k][i], generate_label('φ α', indices[k]*100+1, resolutions[k]), 'blue', '-')
#             plot_data(axs[j, i], radii, data_del_alpha[k][i], generate_label('φ Δα', indices[k]*100+1, resolutions[k]), 'red', '-')
#         axs[j, i].legend()
#         axs[j, i].set_title(f'{i+1} bin contributions')

# plt.show()

# -------------------------------------------------------------------------------------------------------------

# fig, axs = plt.subplots(2, 3, figsize=(15, 10))

# # Plot the data



# axs[0,0].plot(radii_201*0.7, data_alpha_slice_201_eps_03_0, label=r' $\phi \dot{\alpha}$, $\epsilon = 0.3$', color='lightsteelblue')
# axs[0,0].plot(radii_201*0.7, data_not_res_alpha_slice_201_0, label=r' $\phi \dot{\alpha}$, $\epsilon = 0.5$', color='royalblue')
# axs[0,0].plot(radii_201*0.7, data_alpha_slice_201_eps_07_0, label=r' $\phi \dot{\alpha}$, $\epsilon = 0.7$', color='mediumblue')

# axs[0,0].plot(radii_201*0.7, data_del_alpha_201_eps_03_0, label=r'$\phi \dot{\Delta \alpha}$, $\epsilon = 0.3$', color='lightcoral')
# axs[0,0].plot(radii_201*0.7, data_not_res_del_alpha_201_0, label=r'$\phi \dot{\Delta \alpha}$, $\epsilon = 0.5$', color='red')
# axs[0,0].plot(radii_201*0.7, data_del_alpha_201_eps_07_0, label=r'$\phi \dot{\Delta \alpha}$, $\epsilon = 0.7$', color='darkred')

# # axs[0,0].set_xlabel('Radius in physical spatial units (Number of grid points x spatial resolution)')
# # axs[0,0].set_ylabel('Power spectrum')
# axs[0,0].set_ylim(0,2e-5)



# # axs[0,0].set_xticklabels([])
# # Add a legend
# axs[0,0].legend()
# # axs[0,0].set_title('First bin contributions')




# axs[0,1].plot(radii_201*0.7, data_alpha_slice_201_eps_03_1, label=r' $\phi \dot{\alpha}$, $\epsilon = 0.3$', color='lightsteelblue')
# axs[0,1].plot(radii_201*0.7, data_not_res_alpha_slice_201_1, label=r' $\phi \dot{\alpha}$, $\epsilon = 0.5$', color='royalblue')
# axs[0,1].plot(radii_201*0.7, data_alpha_slice_201_eps_07_1, label=r' $\phi \dot{\alpha}$, $\epsilon = 0.7$', color='mediumblue')

# axs[0,1].plot(radii_201*0.7, data_del_alpha_201_eps_03_1, label=r'$\phi \dot{\Delta \alpha}$, $\epsilon = 0.3$', color='lightcoral')
# axs[0,1].plot(radii_201*0.7, data_not_res_del_alpha_201_1, label=r'$\phi \dot{\Delta \alpha}$, $\epsilon = 0.5$', color='red')
# axs[0,1].plot(radii_201*0.7, data_del_alpha_201_eps_07_1, label=r'$\phi \dot{\Delta \alpha}$, $\epsilon = 0.7$', color='darkred')
# axs[0,1].set_ylim(0,2e-5)
# # axs[0,1].set_xlabel('Radius in physical spatial units (Number of grid points x spatial resolution)')
# # axs[0,1].set_ylabel('Power spectrum')
# # axs[0,1].set_xticklabels([])
# axs[0,1].set_yticklabels([])
# # Add a legend
# # axs[0,1].legend()
# # axs[0,1].set_title('Second bin contributions')




# axs[0,2].plot(radii_201*0.7, data_alpha_slice_201_eps_03_2, label=r' $\phi \dot{\alpha}$, $\epsilon = 0.3$', color='lightsteelblue')
# axs[0,2].plot(radii_201*0.7, data_not_res_alpha_slice_201_2, label=r' $\phi \dot{\alpha}$, $\epsilon = 0.5$', color='royalblue')
# axs[0,2].plot(radii_201*0.7, data_alpha_slice_201_eps_07_2, label=r' $\phi \dot{\alpha}$, $\epsilon = 0.7$', color='mediumblue')

# axs[0,2].plot(radii_201*0.7, data_del_alpha_201_eps_03_2, label=r'$\phi \dot{\Delta \alpha}$, $\epsilon = 0.3$', color='lightcoral')
# axs[0,2].plot(radii_201*0.7, data_not_res_del_alpha_201_2, label=r'$\phi \dot{\Delta \alpha}$, $\epsilon = 0.5$', color='red')
# axs[0,2].plot(radii_201*0.7, data_del_alpha_201_eps_07_2, label=r'$\phi \dot{\Delta \alpha}$, $\epsilon = 0.7$', color='darkred')
# axs[0,2].set_ylim(0,2e-5)
# # axs[0,2].set_xlabel('Radius in physical spatial units (Number of grid points x spatial resolution)')
# # axs[0,2].set_ylabel('Power spectrum')
# # axs[2,0].set_xticklabels([])
# axs[0,2].set_yticklabels([])
# # Add a legend
# # axs[0,2].legend()
# # axs[0,2].set_title('Third bin contributions')






# axs[1,0].plot(radii_201*0.35, data_alpha_slice_201_res_eps_03_0, label=r' $\phi \dot{\alpha}$, $\epsilon = 0.3$', color='lightsteelblue')
# axs[1,0].plot(radii_201*0.35, data_alpha_slice_201_0, label=r' $\phi \dot{\alpha}$, $\epsilon = 0.5$', color='royalblue')
# axs[1,0].plot(radii_201*0.35, data_alpha_slice_201_res_eps_07_0, label=r' $\phi \dot{\alpha}$, $\epsilon = 0.7$', color='mediumblue')

# axs[1,0].plot(radii_201*0.35, data_del_alpha_201_res_eps_03_0, label=r'$\phi \dot{\Delta \alpha}$, $\epsilon = 0.3$', color='lightcoral')
# axs[1,0].plot(radii_201*0.35, data_del_alpha_201_0, label=r'$\phi \dot{\Delta \alpha}$, $\epsilon = 0.5$', color='red')
# axs[1,0].plot(radii_201*0.35, data_del_alpha_201_res_eps_07_0, label=r'$\phi \dot{\Delta \alpha}$, $\epsilon = 0.7$', color='darkred')
# axs[1,0].set_ylim(0,9e-5)
# # Hie x and y grid values for the subplot
# # axs[1,0].set_xticklabels([])
# # axs[1,0].set_xlabel('Radius in physical spatial units (Number of grid points x spatial resolution)')
# # axs[1,0].set_ylabel('Power spectrum')

# # Add a legend
# axs[1,0].legend()
# # axs[1,0].set_title('First bin contributions')



# axs[1,1].plot(radii_201*0.35, data_alpha_slice_201_res_eps_03_1, label=r' $\phi \dot{\alpha}$, $\epsilon = 0.3$', color='lightsteelblue')
# axs[1,1].plot(radii_201*0.35, data_alpha_slice_201_1, label=r' $\phi \dot{\alpha}$, $\epsilon = 0.5$', color='royalblue')
# axs[1,1].plot(radii_201*0.35, data_alpha_slice_201_res_eps_07_1, label=r' $\phi \dot{\alpha}$, $\epsilon = 0.7$', color='mediumblue')

# axs[1,1].plot(radii_201*0.35, data_del_alpha_201_res_eps_03_1, label=r'$\phi \dot{\Delta \alpha}$, $\epsilon = 0.3$', color='lightcoral')
# axs[1,1].plot(radii_201*0.35, data_del_alpha_201_1, label=r'$\phi \dot{\Delta \alpha}$, $\epsilon = 0.5$', color='red')
# axs[1,1].plot(radii_201*0.35, data_del_alpha_201_res_eps_07_1, label=r'$\phi \dot{\Delta \alpha}$, $\epsilon = 0.7$', color='darkred')
# axs[1,1].set_ylim(0,9e-5)
# # axs[0,1].set_xlabel('Radius in physical spatial units (Number of grid points x spatial resolution)')
# # axs[0,1].set_ylabel('Power spectrum')
# # axs[1,1].set_xticklabels([])
# axs[1,1].set_yticklabels([])
# # Add a legend
# # axs[1,1].legend()
# # axs[1,1].set_title('Second bin contributions')




# axs[1,2].plot(radii_201*0.35, data_alpha_slice_201_res_eps_03_2, label=r' $\phi \dot{\alpha}$, $\epsilon = 0.3$', color='lightsteelblue')
# axs[1,2].plot(radii_201*0.35, data_alpha_slice_201_2, label=r' $\phi \dot{\alpha}$, $\epsilon = 0.5$', color='royalblue')
# axs[1,2].plot(radii_201*0.35, data_alpha_slice_201_res_eps_07_2, label=r' $\phi \dot{\alpha}$, $\epsilon = 0.7$', color='mediumblue')

# axs[1,2].plot(radii_201*0.35, data_del_alpha_201_res_eps_03_2, label=r'$\phi \dot{\Delta \alpha}$, $\epsilon = 0.3$', color='lightcoral')
# axs[1,2].plot(radii_201*0.35, data_del_alpha_201_2, label=r'$\phi \dot{\Delta \alpha}$, $\epsilon = 0.5$', color='red')
# axs[1,2].plot(radii_201*0.35, data_del_alpha_201_res_eps_07_2, label=r'$\phi \dot{\Delta \alpha}$, $\epsilon = 0.7$', color='darkred')

# # axs[0,2].set_xlabel('Radius in physical spatial units (Number of grid points x spatial resolution)')
# # axs[0,2].set_ylabel('Power spectrum')
# axs[1,2].set_ylim(0,9e-5)

# # axs[2,1].set_xticklabels([])
# axs[1,2].set_yticklabels([])

# axs[1,0].set_xlabel('Radius of circular mask', fontsize=14)
# axs[1,0].set_ylabel('Power spectrum', fontsize=14)

# axs[0, 0].text(0.30, 0.95, r'$\Delta x = 0.7$', fontsize=12, verticalalignment='top', horizontalalignment='right', transform=axs[0, 0].transAxes)
# axs[1, 0].text(0.30, 0.95, r'$\Delta x = 0.35$', fontsize=12, verticalalignment='top', horizontalalignment='right', transform=axs[1, 0].transAxes)

# axs[0, 0].tick_params(axis='both', which='major', labelsize=14)
# axs[0, 1].tick_params(axis='both', which='major', labelsize=14)
# axs[0, 2].tick_params(axis='both', which='major', labelsize=14)
# axs[1, 0].tick_params(axis='both', which='major', labelsize=14)
# axs[1, 1].tick_params(axis='both', which='major', labelsize=14)
# axs[1, 2].tick_params(axis='both', which='major', labelsize=14)
# axs[0,0].legend(fontsize=14)
# axs[1,0].legend(fontsize=14)
# plt.subplots_adjust(wspace=0, hspace=0.15)
# plt.show()

