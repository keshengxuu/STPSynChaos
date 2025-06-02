#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 15:23:52 2021

@author: ksxuphy
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




spikestrain_Nonchaos = [read_csv_file('Spikes_%04d_jei_0001.txt'%s ) for s in range(1,4)]
spikestrain_chaos = [read_csv_file('Spikes_%04d_jei_0002.txt'%s ) for s in range(1,4) ]#STP CHAOS

dt = 0.01

flag = 0

def sptimes(spiketrain, filename):

    for i in range(3):
        spikes = spiketrain[i]
        with open("%s_%d.txt"%(filename,i),'w') as f:
            writer=csv.writer(f,'excel')
            flag = 0
            for sp in spikes:
                if flag <1000:
                    writer.writerow(['%.3f'%(x*dt) for x in sp])
                flag =flag+1
    return 

            
            
sptimes(spikestrain_Nonchaos,'sptimes_Nonchaos')      

sptimes(spikestrain_chaos,'sptimes_chaos')      

# for i in range(6):
#     spikes = spikestrain[i]
#     with open("EXSpikes_%d.txt"%i,'w') as f:
#         writer=csv.writer(f,'excel')
#         flag = 0
#         for sp in spikes:
#             if flag <1000:
#                 writer.writerow(['%.3f'%(x) for x in sp])
#             flag =flag+1
