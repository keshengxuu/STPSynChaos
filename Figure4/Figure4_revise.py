
"""
Created on Mon Apr  8 20:48:38 2024
keshengxu@gmail.com
@author: keshengxu
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

import bct 				#Import Brain Connectivity Toolbox


plt.rcParams['mathtext.sf'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
#plt.rcParams['font.sans-serif'] = 'DejaVu Sans'




plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
plt.rc('font',size=8)
plt.rc('font', weight='bold')
plt.rcParams['axes.labelweight'] = 'bold'




# function to sort each row of the matrix
def sortByRow(mat):
	return np.sort(mat, axis=1)

# function to sort the matrix row-wise and column-wise
def sortMatRowAndColWise(mat):
	# sort rows of mat[][]
	mat = sortByRow(mat)

	# get transpose of mat[][]
	mat = mat.transpose()

	# again sort rows of mat[][]
	mat = sortByRow(mat)

	# again get transpose of mat[][]
	mat = mat.transpose()

	return mat

def sort2(mat):
    
    return mat[:,np.argmax(mat, axis=1)]

cmap00=plt.get_cmap('viridis_r')# cmap00=plt.get_cmap('jet')
#cmap00=plt.get_cmap('jet')# cmap00=plt.get_cmap('jet')
dt = 0.01


ax11 = []
def read_csv_file(filename):
    """Reads a CSV file and return it as a list of rows."""
    data = []
    for row in csv.reader(open(filename)):

        row=list((float(x) for x in row))
        data.append(row)
    return data

def adjust_spines(ax, spines):
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward',0)) 

        else:
            spine.set_color('none')  # don't draw spine

  
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


gs = gridspec.GridSpec(nrows=2, ncols=3, left=0.05, right=0.98,top=0.97,bottom=0.08,
                        wspace=0.4,hspace=0.15)


spikes1 = [read_csv_file('Spikes_%04d_jei_0001.txt'%s ) for s in range(1,4)]#NONSTP CHAOS
 
spikes2 = [read_csv_file('Spikes_%04d_jei_0002.txt'%s ) for s in range(1,4) ]#STP CHAOS

spike_trains_Nonchaos = [spk.load_spike_trains_from_txt("sptimes_Nonchaos_%d.txt"%fl,separator=',',
                                               edges=(0, 1000)) for fl in range(3)]


spike_trains_chaos = [spk.load_spike_trains_from_txt("sptimes_chaos_%d.txt"%fl,separator=',',
                                               edges=(0, 1000)) for fl in range(3)]


fig=plt.figure(1,figsize=(12.5,6))

plt.clf()


N=1250
Ne=1000

# The first line graph






sli_para = 1


cmap = colors.ListedColormap(['navy', 'limegreen','lightpink'])
bounds=[0,0.85,0.95,1]
norm = colors.BoundaryNorm(bounds, cmap.N)

ax1 = plt.subplot(gs[0,0])
for i in range(N):
    if i<Ne:
        ax1.plot(dt*np.array(spikes1[sli_para][i]),i*np.ones_like(spikes1[sli_para][i]),'.',color ='orange',markersize=1,rasterized=True)
#     else:
#         ax1.plot(dt*np.array(spikes1[sli_para][i]),i*np.ones_like(spikes1[sli_para][i]),'g.',markersize=1,rasterized=True)
# plt.title('JEE =0.4')


plt.ylabel('Neuron index',labelpad =-5)
#plt.xlabel('Times',labelpad=0.5)


# ax1.set_ylabel('Neuron index')
# ax1.set_xlabel('Times')
ax1.set_xlim([200,590])

ax12 = ax1.twinx()

f0 = spk.spike_sync_profile(spike_trains_Nonchaos[1])
x0, y0 = f0.get_plottable_data(averaging_window_size=5)

ax12.plot(x0, y0, '-',color= 'red',  label="averaged SPIKE-Sync profile")

ax12.set_ylabel('Averaged SPIKE-Sync profile', fontsize=8,color='red',labelpad=1)
ax12.set_ylim([0,1])
ax12.set_xlim([200,600])
# ax12.set_yticks([0.00,0.05,0.1],['0.00','0.05','0.10'])
ax12.tick_params(axis='y', labelcolor='red')





ax2 = plt.subplot(gs[0,1])
#isi_distance = spk.isi_distance_matrix(spike_trains_Nonchaos[1])
#spike_distance = spk.spike_distance_matrix(spike_trains_Nonchaos[1], interval=(0, 1200))
spike_sync = spk.spike_sync_matrix(spike_trains_Nonchaos[1], interval=(0, 1000))
#im2= ax2.imshow(spike_sync, vmin =0.7, vmax = 1.0, interpolation='spline16',cmap=plt.cm.get_cmap('terrain'))
im2= ax2.imshow(spike_sync, vmin =0.8, vmax = 1.0,interpolation='spline16',cmap=plt.colormaps.get_cmap('gist_ncar'),origin='lower')

#im2= ax2.imshow(spike_sync, vmin =0.7, vmax = 1.0, interpolation='spline16',cmap=plt.cm.get_cmap('brg'))

#plt.xlabel('Neuron index',labelpad =-0.5)
plt.ylabel('Neuron index',labelpad =-2.5)

# ax2.set_xlabel('Neuron index')
# ax2.set_ylabel('Neuron index')

ax2.set_xticks([199,399,599,799,999],['200','400','600','800','1000'])
ax2.set_yticks([199,399,599,799,999],['200','400','600','800','1000'])

cbar1 = fig.colorbar(im2, ax=ax2)

cbar1.set_ticks(np.linspace(0.8,1.0,5))

cbar1.set_label(r'Spike-Sync',fontsize=8,labelpad=1)
#cbar1.set_ticks(np.linspace(0,1,3))
##change the appearance of ticks anf tick labbel
cbar1.ax.tick_params(labelsize=8)




ax3 = plt.subplot(gs[0,2])
spike_sync0 = spk.spike_sync_matrix(spike_trains_Nonchaos[1], interval=(0, 1000))

ci,q=bct.modularity_louvain_und(spike_sync0,gamma=1, hierarchy=False, seed=None)


fit_data=spike_sync0[np.argsort(ci)]
fit_data=fit_data[:,np.argsort(ci)]



im3=ax3.imshow(fit_data, interpolation='spline16',cmap=cmap, norm=norm,origin='lower')







#im3= ax3.imshow(spike_sync10,interpolation='spline16',cmap=plt.colormaps.get_cmap('gist_ncar'), norm=norm)

#plt.xlabel('Neuron index',labelpad =-0.5)
plt.ylabel('Neuron index',labelpad =-2.5)

# ax3.set_xlabel('Neuron index')
# ax3.set_ylabel('Neuron index')
ax3.set_xticks([199,399,599,799,999],['200','400','600','800','1000'])
ax3.set_yticks([199,399,599,799,999],['200','400','600','800','1000'])

cbar3 = fig.colorbar(im3, ax=ax3,cmap=cmap, norm=norm, boundaries=bounds,ticks=[0, 0.85,0.95,1.0])


cbar3.set_label(r'Spike-Sync',fontsize=8,labelpad=5)
#cbar1.set_ticks(np.linspace(0,1,3))
##change the appearance of ticks anf tick labbel
cbar3.ax.tick_params(labelsize=8)






#The second line graph
ax4 = plt.subplot(gs[1,0])
for i in range(N):
    if i<Ne:
        ax4.plot(dt*np.array(spikes2[1][i]),i*np.ones_like(spikes2[1][i]),'.',color ='orange',markersize=1,rasterized=True)
    # else:
    #     ax4.plot(dt*np.array(spikes2[sli_para][i]),i*np.ones_like(spikes2[sli_para][i]),'g.',markersize=1,rasterized=True)
    
    
plt.ylabel('Neuron index',labelpad =-5)
plt.xlabel('Times',labelpad=0.5)    
# ax4.set_xlabel('Times')
# ax4.set_ylabel('Neuron index')
ax4.set_xlim([0,1000])
ax4.set_ylim([0,1050])



ax42 = ax4.twinx()

f1 = spk.spike_sync_profile(spike_trains_chaos[1])
x1, y1 = f1.get_plottable_data(averaging_window_size=5)

ax42.plot(x1, y1, '-',color= 'red',  label="averaged SPIKE-Sync profile")

ax42.set_ylabel('Averaged SPIKE-Sync profile',fontsize=8, color='red',labelpad=1)
ax42.set_ylim([0,1.03])
ax42.set_xlim([0,1000])
# ax12.set_yticks([0.00,0.05,0.1],['0.00','0.05','0.10'])
ax42.tick_params(axis='y', labelcolor='red')




ax5 = plt.subplot(gs[1,1])

#spike_sync = spk.spike_sync_matrix(spike_trains_chaos[1], interval=(0, 2000))
#isi_distance = spk.isi_distance_matrix(spike_trains_chaos[1])
#spike_distance = spk.spike_distance_matrix(spike_trains_chaos[1], interval=(0, 1200))

spike_sync = spk.spike_sync_matrix(spike_trains_chaos[1], interval=(0, 1000))
#im5= ax5.imshow(spike_sync, vmin =0.7, vmax = 1.0, interpolation='spline16',cmap=plt.cm.get_cmap('terrain')) #the_older_format
im5= ax5.imshow(spike_sync, vmin =0.8, vmax = 1.0, interpolation='spline16',cmap=plt.colormaps.get_cmap('gist_ncar'),origin='lower')
#im5= ax5.imshow(spike_sync, vmin =0.7, vmax = 1.0, interpolation='spline16',cmap=plt.cm.get_cmap('brg'))

plt.xlabel('Neuron index',labelpad =-0.5)
plt.ylabel('Neuron index',labelpad =-2.5)

# ax5.set_xlabel('Neuron index')
# ax5.set_ylabel('Neuron index')
ax5.set_xticks([199,399,599,799,999],['200','400','600','800','1000'])
ax5.set_yticks([199,399,599,799,999],['200','400','600','800','1000'])
#ax5.set_xticks([199,399,599,799,999],['200','400','600','800','1000'])
# ax5.set_xticks([199,399,599,799,999],['200','400','600','800','1000'])              
# ax6.set_xticks([199,399,599,799,999],['200','400','600','800','1000'])

cbar2 = fig.colorbar(im5, ax=ax5)

cbar2.set_ticks(np.linspace(0.8,1.0,5))

cbar2.set_label(r'Spike-Sync',fontsize=8,labelpad=5)
#cbar1.set_ticks(np.linspace(0,1,3))
##change the appearance of ticks anf tick labbel
cbar2.ax.tick_params(labelsize=8)







ax6 = plt.subplot(gs[1,2])
spike_sync0 = spk.spike_sync_matrix(spike_trains_chaos[1], interval=(0, 1000))
#spike_sync = sortMatRowAndColWise(spike_sync0)


ci,q=bct.modularity_louvain_und(spike_sync0,gamma=3, hierarchy=False, seed=None)


fit_data=spike_sync0[np.argsort(ci)]
fit_data=fit_data[:,np.argsort(ci)]



im6=ax6.imshow(fit_data, interpolation='spline16',cmap=cmap, norm=norm,origin='lower')





#im6= ax6.imshow(spike_sync10,interpolation='spline16',cmap=plt.colormaps.get_cmap('gist_ncar'))

plt.xlabel('Neuron index',labelpad =-0.5)
plt.ylabel('Neuron index')

# ax3.set_xlabel('Neuron index')
# ax3.set_ylabel('Neuron index')
ax6.set_xticks([199,399,599,799,999],['200','400','600','800','1000'])
ax6.set_yticks([199,399,599,799,999],['200','400','600','800','1000'])

cbar6 = fig.colorbar(im6, ax=ax6,cmap=cmap, norm=norm, boundaries=bounds,ticks=[0, 0.85,0.95,1.0])

cbar6.set_label(r'Spike-Sync',fontsize=8,labelpad =1)
#cbar1.set_ticks(np.linspace(0,1,3))
##change the appearance of ticks anf tick labbel
cbar6.ax.tick_params(labelsize=8)






# plt.figtext(0.009,0.97,'(A)',fontsize = 16)
# plt.figtext(0.009,0.49,'(B)',fontsize = 16)

ax1.text(210,950,'(a)',bbox=dict(facecolor='white', alpha=1, edgecolor='white',pad=0),fontdict=dict(fontsize=8, fontweight='bold',color= 'black'),fontsize =18)
ax2.text(50,910,'(b)',bbox=dict(facecolor='white', alpha=0.8, edgecolor='white',pad=0),fontdict=dict(fontsize=8, fontweight='bold',color= 'black'),fontsize =18)
ax3.text(50,910,'(c)',bbox=dict(facecolor='white', alpha=0.8, edgecolor='white',pad=0),fontdict=dict(fontsize=8, fontweight='bold',color= 'black'),fontsize =18)
ax4.text(70,950,'(d)',bbox=dict(facecolor='white', alpha=1, edgecolor='white',pad=0),fontdict=dict(fontsize=8, fontweight='bold',color= 'black'),fontsize =18)
ax5.text(50,910,'(e)',bbox=dict(facecolor='white', alpha=0.8, edgecolor='white',pad=0),fontdict=dict(fontsize=8, fontweight='bold',color= 'black'),fontsize =18)
ax6.text(50,910,'(f)',bbox=dict(facecolor='white', alpha=0.8, edgecolor='white',pad=0),fontdict=dict(fontsize=8, fontweight='bold',color= 'black'),fontsize =18)

# plt.figtext(0.355,0.97,r'$\mathsf{(C_1)}$',fontsize = 'x-large')
# plt.figtext(0.355,0.48,r'$\mathsf{(D_1)}$',fontsize = 'x-large')

# plt.figtext(0.705,0.97,r'$\mathsf{(C_2)}$',fontsize = 'x-large')
# plt.figtext(0.705,0.48,r'$\mathsf{(D_2)}$',fontsize = 'x-large')


plt.draw()


# axins1.text(1090,588,'JEI=0.3',color='black',rotation=0,fontsize='large')
# axins4.text(1390,588,'JEI=0.3',color='black',rotation=0,fontsize='large')   

# axins1.text(1090,660,'NONSTP',color='black',rotation=0,fontsize='small')
# axins4.text(1390,660,'STP',color='black',rotation=0,fontsize='small')   




#plt.subplots_adjust(bottom=0.1,left=0.49,wspace = 0.11,hspace = 0.2,right=0.5, top=0.7)


plt.savefig('Figure4.png',dpi = 300)





