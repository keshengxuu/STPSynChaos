# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 21:18:48 2024

@author: admin
"""

from __future__ import print_function

import matplotlib.pyplot as plt

import csv



def read_csv_file(filename):
    """Reads a CSV file and return it as a list of rows."""
    data = []
    for row in csv.reader(open(filename)):
# transfer the string type to the float type
        row=list((float(x) for x in row))
        data.append(row)
    return data


# Define the ranges for jee and jei
jee_values = [round(j * 0.1, 2) for j in range(21)]  # Values from 0.00 to 2.00 in steps of 0.05
jei_values = [round(j * 0.1, 2) for j in range(21)]  # Values from 0.00 to 2.00 in steps of 0.1

spikestrain = []

for jee_index in range(21):
    jee_value = jee_values[jee_index]
    for jei_index in range(21):
        jei_value = jei_values[jei_index]
        s = jee_index * 21 + jei_index  # Compute s index based on jee and jei values
        spikes = read_csv_file(f'STD_chaos_Spikes/Spikes_{s:04d}_jee_{jee_value:.2f}_jei_{jei_value:.2f}.txt')
        spikestrain.append(spikes)


dt = 0.01

flag = 0

for i in range(441):
    spikes = spikestrain[i]
    with open("sptimes_chaos/Sptraintimes_ex_%d.txt"%i,'w') as f:
        writer=csv.writer(f,'excel')
        flag = 0
        for sp in spikes:
            if flag>=0 and flag <1000:
                writer.writerow(['%.3f'%(x*dt) for x in sp])
            flag =flag+1
            