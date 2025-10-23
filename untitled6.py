#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 10:47:14 2023

@author: connorsheehan
"""

# To begin, we import some libraries that we will need later.

# The numpy library will allow us to do FFTs
import numpy as np

# The pyplot module from matplotlib will allow us to plot things.
from  matplotlib import pyplot as plt

# pi is a useful value!
from math import pi

import os

# Use the numpy loadtxt routine to read data.

# this is an example file, change it to point at your data.
# filename=os.path.join(pathcrabtemplate, "example_data/20220522_051826_B0329+54.npz")
filename=("20230131_151540_B1933+16.npz")
obsdata = np.load(filename)
print(obsdata['header'])
print("")

data = obsdata['data']
print("Data array shape:",data.shape)

# Here we infer the number of channels and numnber of phase bins based on the last entry in the file.
nsub, nchan,nphase = data.shape

# Print this out for verification
print("Nsub = {} Nchan = {} Nphase= {}".format(nsub, nchan,nphase))
        

# Here is where we reshape the 1-d array into a 2-d data structure
iphase=np.arange(nphase)
ichan=np.arange(nchan)
isub = np.arange(nsub)

# We can integrate over all frequency channels by using the `sum` routine from numpy...
fully_averaged=np.mean(data,axis=(0,1))
time_averaged = np.mean(data,axis=0)
freq_averaged = np.mean(data,axis=1)

period = 0.437446
initial_dm = 170

freq_centre = 611
bandwidth = 10
freq_res = bandwidth / nchan
freq_min = freq_centre - 0.5*(bandwidth)


time = (iphase / nphase) * period
freq = (ichan * freq_res + freq_min)

peak_flux_time = []

for i in range(0,len(time_averaged)):
    max_index = np.argmax(time_averaged[i])
    max_bin = max_index + 1
    print(max_bin)
    peak_flux_time.append((max_bin / nphase) * period)
    
print(peak_flux_time)
selected_freq = freq[13:29] 
selected_time = peak_flux_time[13:29]
plt.figure(figsize=(16,6))
plt.subplot(121)
plt.plot(np.polyfit(selected_freq,selected_time,1))
plt.subplot(122)
plt.plot(1/(selected_freq**2), selected_time)

# Plot the 2-d data:
plt.figure(figsize=(16,6))
plt.subplot(131)
plt.imshow(time_averaged,aspect='auto',origin='lower')
plt.xlabel("Phase (iphase)")
plt.ylabel("Channels (ichan)")

plt.subplot(132)
plt.imshow(freq_averaged,aspect='auto',origin='lower')
plt.xlabel("Phase (iphase)")
plt.ylabel("Sub-integrations (isub)")

# Plot the integrated profile:
plt.subplot(133)
plt.plot(fully_averaged)
plt.xlabel("Phase (iphase)")
plt.ylabel("Intensity (Arbitrary)")
plt.title("Integrated profile")
plt.show()

##
#  This function will shift each row of a 2-d (3-d) array by the the number of columns specified in the "shifts" array.
#  data_in - the 2-d (3-d) array to work on
#  shifts - the shifts to apply
#  Returns: The shifted array
##
def shift_rows(data_in, shifts):
    shifted = np.zeros_like(data_in)
    if data_in.ndim == 3:
        for sub in range(nsub):
            shifted[sub] = shift_rows(data_in[sub],shifts)
    else:
        for chan in range(nchan):
            shifted[chan] = np.roll(data_in[chan],int(shifts[chan]))
    return shifted

# This example scaling is wrong - you need to determine the right values to scale!

period = 0.437446
initial_dm = 170

freq_centre = 611
bandwidth = 10
freq_res = bandwidth / nchan
freq_min = freq_centre - 0.5*(bandwidth)


time = (iphase / nphase) * period
freq = (ichan * freq_res + freq_min)

plt.figure(figsize=(16,6))
plt.subplot(131)
plt.imshow(time_averaged,aspect='auto',origin='lower')
plt.xlabel("Phase (iphase)")
plt.ylabel("Channels (ichan)")

plt.subplot(132)
plt.imshow(freq_averaged,aspect='auto',origin='lower')
plt.xlabel("Phase (iphase)")
plt.ylabel("Sub-integrations (isub)")

# Plot the integrated profile:
plt.subplot(133)
plt.plot(fully_averaged)
plt.xlabel("Phase (iphase)")
plt.ylabel("Intensity (Arbitrary)")
plt.title("Integrated profile")
plt.show()


# This array is going to conain the phase shifts in "bins".
# For this demo, we just shift by the frequency, which is clearly not correct!

#Dedispersion:
timedelay = -initial_dm / (2.4010e-4 * (freq**2))
bindelay=timedelay * nphase / period

# Here we call our row-shifting function
# Remeber that the shift is in phase bins, not time units!
dedispersed = shift_rows(time_averaged,bindelay)

# Again, sum along axis zero to integrate over frequency.
integrated=np.sum(dedispersed,axis=0)

# plot the data again... If you have de-dispersed correctly, you the S/N should be maximised.
plt.figure(figsize=(12,12))
plt.subplot(211)

# Here we show how to change the scaling of the axes so that they have physical units.
plt.imshow(dedispersed,aspect='auto',origin='lower',extent=(0,1,freq[0],freq[-1]))
plt.xlabel("Phase (units?)")
plt.ylabel("Frequency (units?)")


plt.subplot(212)
plt.plot(np.linspace(0,1,nphase),integrated)
plt.xlabel("Phase (iphase)")
plt.ylabel("Intensity (Arbitrary)")
plt.title("Integrated profile")
plt.show()


#bindelay = timedelay * 1024 / period of pulsar
#where time delay is from equation in script