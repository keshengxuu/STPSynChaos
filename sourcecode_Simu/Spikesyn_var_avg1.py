
"""
Created on Mon Apr  8 20:48:38 2024

@author: admin
"""

import numpy as np
import contextlib
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import rc, cm
import matplotlib as mlp
from numpy.ma import masked_array
from matplotlib.colors import Normalize
import matplotlib.gridspec as gridspec
from matplotlib import rc, cm
import csv
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pyspike as spk


plt.rcParams['mathtext.sf'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'


dt = 0.01

def read_csv_file(filename):
    """Reads a CSV file and return it as a list of rows."""
    data = []
    for row in csv.reader(open(filename)):

        row=list((float(x) for x in row))
        data.append(row)
    return data

spike_trains_Nonchaos = [spk.load_spike_trains_from_txt("sptimes_chaos/Sptraintimes_ex_%d.txt"%fl,separator=',',
                                               edges=(0, 1000)) for fl in range(441)]

N=1250
Ne=1000

output_filename = "average_nonchaos.txt"
output_filename1 = "varance_nonchaos.txt"


listdata = []
listdata0_varance = []


for ind0 in range(441):
    
    spike_sync = spk.spike_sync_matrix(spike_trains_Nonchaos[ind0], interval=(0, 1000))
    listdata.append(np.mean(spike_sync))
    listdata0_varance.append(np.var(spike_sync))
 

with open(output_filename, 'w') as f:
    for value in listdata:
        f.write(f"{value}\n")

  
with open(output_filename1, 'w') as f:
    for value in listdata0_varance:
        f.write(f"{value}\n")





    
    
    
    
    
    
    







