#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 15:23:52 2021

@author: ksxuphy
"""
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import csv


def read_csv_file(filename):
    """Reads a CSV file and return it as a list of rows."""
    data = []
    for row in csv.reader(open(filename)):
# transfer the string type to the float type
        row=list((float(x) for x in row))
        data.append(row)
    return data



def adjust_spines(ax, spines):
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward',0))  # outward by 10 points
#            spine.set_smart_bounds(True)
        else:
            spine.set_color('none')  # don't draw spine

    # turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        # no yaxis ticks
        ax.yaxis.set_ticks([])

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
        # no xaxis ticks
        ax.xaxis.set_ticks([])
        

#Fonts!
# plt.rcParams['mathtext.sf'] = 'Arial'
# plt.rcParams['font.family'] = 'sans-serif'
# plt.rcParams['font.sans-serif'] = 'Arial'

#changing the xticks and  yticks fontsize for all sunplots
plt.rc('xtick', labelsize=11)
plt.rc('ytick', labelsize=11)
plt.rc('font',size=12)
plt.rc('font', weight='bold')
plt.rcParams['axes.labelweight'] = 'bold'

vart = np.loadtxt('varance/vart.txt')
vart_stf30 = np.loadtxt('varance/vart_stf30.txt')
vart_stf60 = np.loadtxt('varance/vart_stf60.txt')
vart_stf100 = np.loadtxt('varance/vart_stf100.txt')



gee = np.linspace(0, 1,11)



fig=plt.figure(1,figsize=(10,6))

plt.plot(gee[1:],vart,label = "nonstp03")
plt.plot(gee[1:],vart_stf30,label = "nonstp030")
plt.plot(gee[1:],vart_stf60,label = "nonstp060")
plt.plot(gee[1:],vart_stf100,label = "nonstp100")


plt.legend()




plt.savefig('VarancePLot.png',dpi = 300)
