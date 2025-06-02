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




#spikestrain_Nonchaos =[read_csv_file('spiketrainSTP_chaos06/Spikes_%04d_jei_030.txt'%s ) for s in range(1,11)]
spikestrain_chaos =  [read_csv_file('spiketrainSTP_chaos10/Spikes_%04d_jei_100.txt'%s ) for s in range(1,11) ]#STP NONCHAOS

dt = 0.01

flag = 0

def sptimes(spiketrain, filename):

    for i in range(10):
        spikes = spiketrain[i]
        with open("%s_%d.txt"%(filename,i),'w') as f:
            writer=csv.writer(f,'excel')
            flag = 0
            for sp in spikes:
                #if 1000 <= flag <1250:
                if flag <1000:
                    writer.writerow(['%.3f'%(x*dt) for x in sp])
                flag =flag+1
    return 

            
            
# sptimes(spikestrain_Nonchaos,'Sptraintimes_in')      

# sptimes(spikestrain_chaos,'SptraintimesSTP_in')      



#sptimes(spikestrain_Nonchaos,'Sptraintimes_ex')      

sptimes(spikestrain_chaos,'SptraintimesSTP_ex') 
#sptimes(spikestrain_chaos,'SptraintimesSTP_in')   

# for i in range(6):
#     spikes = spikestrain[i]
#     with open("EXSpikes_%d.txt"%i,'w') as f:
#         writer=csv.writer(f,'excel')
#         flag = 0
#         for sp in spikes:
#             if flag <1000:
#                 writer.writerow(['%.3f'%(x) for x in sp])
#             flag =flag+1
