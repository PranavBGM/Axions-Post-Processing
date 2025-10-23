#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 05:07:28 2023

@author: pranavbharadwajgangrekalvemanoj
"""

# Given dataset
# Given data in the specified format
dataset = """
Star number: 1 
Main sequence Star Time 0.0 Mass 100.000 
Hertzsprung Gap Time 3.5 Mass 69.551 
Naked Helium MS Time 3.5 Mass 29.837 
Naked Helium HG Time 4.0 Mass 12.157 
Black Hole Time 4.0 Mass 1.984 
Black Hole Time 17250.0 Mass 1.984 

Star number: 2 
Main sequence Star Time 0.0 Mass 35.000 
Hertzsprung Gap Time 5.3 Mass 31.455 
Core Helium Burning Time 5.3 Mass 31.364 
Naked Helium MS Time 5.7 Mass 12.015 
Naked Helium HG Time 5.9 Mass 9.678 
Black Hole Time 5.9 Mass 1.812 
Black Hole Time 17250.0 Mass 1.812 

Star number: 3 
Main sequence Star Time 0.0 Mass 18.000 
Hertzsprung Gap Time 10.0 Mass 17.397 
Core Helium Burning Time 10.0 Mass 17.386 
First AGB Time 11.1 Mass 10.541 
Neutron Star Time 11.2 Mass 1.555 
Neutron Star Time 17250.0 Mass 1.555 

Star number: 4 
Main sequence Star Time 0.0 Mass 6.500 
Hertzsprung Gap Time 57.3 Mass 6.500 
Giant Branch Time 57.5 Mass 6.500 
Core Helium Burning Time 57.6 Mass 6.498 
First AGB Time 64.9 Mass 6.378 
Second AGB Time 65.2 Mass 6.329 
Oxygen/Neon WD Time 65.8 Mass 1.211 
Oxygen/Neon WD Time 17250.0 Mass 1.211 

Star number: 5 
Main sequence Star Time 0.0 Mass 3.200 
Hertzsprung Gap Time 318.2 Mass 3.200 
Giant Branch Time 320.0 Mass 3.200 
Core Helium Burning Time 321.8 Mass 3.199 
First AGB Time 394.2 Mass 3.171 
Second AGB Time 396.9 Mass 3.141 
Carbon/Oxygen WD Time 397.8 Mass 0.772 
Carbon/Oxygen WD Time 17250.0 Mass 0.772 

Star number: 6 
Main sequence Star Time 0.0 Mass 2.100 
Hertzsprung Gap Time 1011.5 Mass 2.100 
Giant Branch Time 1019.3 Mass 2.100 
Core Helium Burning Time 1035.8 Mass 2.098 
First AGB Time 1299.1 Mass 2.076 
Second AGB Time 1303.8 Mass 2.048 
Carbon/Oxygen WD Time 1305.2 Mass 0.638 
Carbon/Oxygen WD Time 17250.0 Mass 0.638 

Star number: 7 
Main sequence Star Time 0.0 Mass 1.700 
Hertzsprung Gap Time 1873.6 Mass 1.700 
Giant Branch Time 1901.1 Mass 1.700 
Core Helium Burning Time 1977.1 Mass 1.668 
First AGB Time 2109.8 Mass 1.644 
Second AGB Time 2113.5 Mass 1.613 
Carbon/Oxygen WD Time 2115.0 Mass 0.601 
Carbon/Oxygen WD Time 17250.0 Mass 0.601 

Star number: 8 
Main sequence Star Time 0.0 Mass 1.300 
Hertzsprung Gap Time 4258.5 Mass 1.300 
Giant Branch Time 4482.8 Mass 1.300 
Core Helium Burning Time 4810.2 Mass 1.191 
First AGB Time 4941.7 Mass 1.159 
Second AGB Time 4945.7 Mass 1.115 
Carbon/Oxygen WD Time 4946.7 Mass 0.552 
Carbon/Oxygen WD Time 17250.0 Mass 0.552 

Star number: 9 
Main sequence Star Time 0.0 Mass 1.100 
Hertzsprung Gap Time 7684.5 Mass 1.100 
Giant Branch Time 8089.3 Mass 1.100 
Core Helium Burning Time 8720.0 Mass 0.901 
First AGB Time 8851.6 Mass 0.863 
Second AGB Time 8856.0 Mass 0.800 
Carbon/Oxygen WD Time 8856.3 Mass 0.525 
Carbon/Oxygen WD Time 17250.0 Mass 0.525 

Star number: 10 
Main sequence Star Time 0.0 Mass 1.000 
Hertzsprung Gap Time 11003.1 Mass 1.000 
Giant Branch Time 11582.8 Mass 0.999 
Core Helium Burning Time 12326.3 Mass 0.766 
First AGB Time 12458.7 Mass 0.728 
Second AGB Time 12463.3 Mass 0.593 
Carbon/Oxygen WD Time 12463.5 Mass 0.520 
Carbon/Oxygen WD Time 17250.0 Mass 0.520 

Star number: 11 
Main sequence Star Time 0.0 Mass 0.930 
Hertzsprung Gap Time 14594.7 Mass 0.930 
Giant Branch Time 15363.7 Mass 0.929 
Core Helium Burning Time 16203.7 Mass 0.664 
First AGB Time 16338.9 Mass 0.631 
Second AGB Time 16343.7 Mass 0.532 
Carbon/Oxygen WD Time 16343.9 Mass 0.518 
Carbon/Oxygen WD Time 17250.0 Mass 0.518 

Star number: 12 
Main sequence Star Time 0.0 Mass 0.780 
Main sequence Star Time 17250.0 Mass 0.780 

Star number: 13 
Low Mass MS Star Time 0.0 Mass 0.690 
Low Mass MS Star Time 17250.0 Mass 0.690 

Star number: 14 
Low Mass MS Star Time 0.0 Mass 0.600 
Low Mass MS Star Time 17250.0 Mass 0.600 

Star number: 15 
Low Mass MS Star Time 0.0 Mass 0.150 
Low Mass MS Star Time 17250.0 Mass 0.150 
"""


# Split the dataset into sections for each star
star_data = dataset.strip().split('\n\n')

# Initialize empty lists for each variable
star_numbers = []
stages = []
times = []
masses = []

# Iterate through each star's data and extract values into lists
for star_info in star_data:
    lines = star_info.split('\n')
    star_number = int(lines[0].split()[-1])
    for line in lines[1:]:
        parts = line.split()
        if len(parts) >= 6 and parts[1] in ['MS', 'HG', 'WD', 'Gap', 'Burning', 'Star', 'Naked', 'Helium', 'Black']:
            if 'Time' in parts:
                time_index = parts.index('Time')
                time = float(parts[time_index + 1])
                star_numbers.append(star_number)
                stages.append(parts[1])
                times.append(time)
                masses.append(float(parts[-1]))

# Print the lists for each variable
print("Star Numbers:", star_numbers)
print("Stages:", stages)
print("Times:", times)
print("Masses:", masses)

star_sums = []
current_star_number = star_numbers[0]
current_sum = 0

for star_number, time in zip(star_numbers, times):
    if star_number == current_star_number:
        current_sum += time
    else:
        star_sums.append(current_sum)
        current_star_number = star_number
        current_sum = time

# Append the sum for the last star
star_sums.append(current_sum)

# Print the sum of times for each star
for star_number, time_sum in zip(set(star_numbers), star_sums):
    print(f"Star {star_number} - Sum of Times: {time_sum}")
