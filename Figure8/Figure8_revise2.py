
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
import bct 	


plt.rcParams['mathtext.sf'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
#plt.rcParams['font.sans-serif'] = 'DejaVu Sans'




plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.rc('font',size=12)
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



cmap00=plt.get_cmap('viridis_r')# cmap00=plt.get_cmap('jet')
#cmap00=plt.get_cmap('jet')# cmap00=plt.get_cmap('jet')
dt = 0.01

cmap = colors.ListedColormap(['navy', 'limegreen','lightpink'])
bounds=[0,0.85,0.95,1]
norm = colors.BoundaryNorm(bounds, cmap.N)

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


gs = gridspec.GridSpec(nrows=4, ncols=4, left=0.07, right=0.925,top=0.968,bottom=0.06,
                        wspace=0.3,hspace=0.23)




spike_trains_chaos = [spk.load_spike_trains_from_txt("sptimes_chaos/Sptraintimes_ex_%d.txt"%fl,separator=',',
                                               edges=(0, 1000)) for fl in range(10)]


spike_trains_STPchaos30 = [spk.load_spike_trains_from_txt("sptimes030_STPchaos/SptraintimesSTP_ex_%d.txt"%fl,separator=',',
                                               edges=(0, 1000)) for fl in range(10)]

spike_trains_STPchaos60 = [spk.load_spike_trains_from_txt("sptimes060_STPchaos/SptraintimesSTP_ex_%d.txt"%fl,separator=',',
                                               edges=(0, 1000)) for fl in range(10)]

spike_trains_STPchaos100 = [spk.load_spike_trains_from_txt("sptimes100_STPchaos/SptraintimesSTP_ex_%d.txt"%fl,separator=',',
                                               edges=(0, 1000)) for fl in range(10)]



fig=plt.figure(1,figsize=(12,11))

plt.clf()


N=1250
Ne=1000



ax = [plt.subplot(gs[i,j]) for i in range(4) for j in range(4)]


# ax1 = plt.subplot(gs[0,0])

index0 = [1,3,6,9]

for i, ind0 in zip(range(4),index0):
    
    #isi_distance = spk.isi_distance_matrix(spike_trains_Nonchaos[1])
    #spike_distance = spk.spike_distance_matrix(spike_trains_Nonchaos[1], interval=(0, 1200))
    spike_sync0  = spk.spike_sync_matrix(spike_trains_chaos[ind0], interval=(0, 1000))
    
    ci,q=bct.modularity_louvain_und(spike_sync0,gamma=1.0, hierarchy=False, seed=None)
    fit_data=spike_sync0[np.argsort(ci)]
    fit_data=fit_data[:,np.argsort(ci)]
    
    #im2= ax2.imshow(spike_sync, vmin =0.7, vmax = 1.0, interpolation='spline16',cmap=plt.cm.get_cmap('terrain'))
    if i == 0:
        im0= ax[i].imshow(fit_data, interpolation='spline16',cmap=cmap, norm=norm,origin='lower') #vmin =0.8, vmax = 1.0,interpolation='spline16',cmap=plt.colormaps.get_cmap('gist_ncar'))
    ax[i].imshow(fit_data, interpolation='spline16',cmap=cmap, norm=norm,origin='lower')# vmin =0.8, vmax = 1.0,interpolation='spline16',cmap=plt.colormaps.get_cmap('gist_ncar'))
    ax[i].set_xticks([249,499,749,999],['250','500','750','1000'])
    ax[i].set_yticks([249,499,749,999],['250','500','750','1000'])
    
  


for i, ind0 in zip(range(4,8),index0):
    #isi_distance = spk.isi_distance_matrix(spike_trains_Nonchaos[1])
    #spike_distance = spk.spike_distance_matrix(spike_trains_Nonchaos[1], interval=(0, 1200))
    spike_sync0 = spk.spike_sync_matrix(spike_trains_STPchaos30[ind0], interval=(0, 1000))
    
    ci,q=bct.modularity_louvain_und(spike_sync0,gamma=1.0, hierarchy=False, seed=None)
    fit_data=spike_sync0[np.argsort(ci)]
    fit_data=fit_data[:,np.argsort(ci)]
    #im2= ax2.imshow(spike_sync, vmin =0.7, vmax = 1.0, interpolation='spline16',cmap=plt.cm.get_cmap('terrain'))
    if i == 4:
        im1= ax[i].imshow(fit_data, interpolation='spline16',cmap=cmap, norm=norm,origin='lower')#, vmin =0.8, vmax = 1.0,interpolation='spline16',cmap=plt.colormaps.get_cmap('gist_ncar'))
    ax[i].imshow(fit_data, interpolation='spline16',cmap=cmap, norm=norm,origin='lower')#vmin =0.8, vmax = 1.0,interpolation='spline16',cmap=plt.colormaps.get_cmap('gist_ncar'))
    ax[i].set_xticks([249,499,749,999],['250','500','750','1000'])
    ax[i].set_yticks([249,499,749,999],['250','500','750','1000'])
    


for i, ind0 in zip(range(8,12),index0):
    #isi_distance = spk.isi_distance_matrix(spike_trains_Nonchaos[1])
    #spike_distance = spk.spike_distance_matrix(spike_trains_Nonchaos[1], interval=(0, 1200))
    spike_sync0 = spk.spike_sync_matrix(spike_trains_STPchaos60[ind0], interval=(0, 1000))
    
    ci,q=bct.modularity_louvain_und(spike_sync0,gamma=1.0, hierarchy=False, seed=None)
    fit_data=spike_sync0[np.argsort(ci)]
    fit_data=fit_data[:,np.argsort(ci)]
    #im2= ax2.imshow(spike_sync, vmin =0.7, vmax = 1.0, interpolation='spline16',cmap=plt.cm.get_cmap('terrain'))
    if i == 8:
        im2= ax[i].imshow(fit_data, interpolation='spline16',cmap=cmap, norm=norm,origin='lower')#, vmin =0.8, vmax = 1.0,interpolation='spline16',cmap=plt.colormaps.get_cmap('gist_ncar'))
    ax[i].imshow(fit_data, interpolation='spline16',cmap=cmap, norm=norm,origin='lower')#, vmin =0.8, vmax = 1.0,interpolation='spline16',cmap=plt.colormaps.get_cmap('gist_ncar'))
    ax[i].set_xticks([249,499,749,999],['250','500','750','1000'])
    ax[i].set_yticks([249,499,749,999],['250','500','750','1000'])
  
 

for i, ind0 in zip(range(12,16),index0):
    #isi_distance = spk.isi_distance_matrix(spike_trains_Nonchaos[1])
    #spike_distance = spk.spike_distance_matrix(spike_trains_Nonchaos[1], interval=(0, 1200))
    spike_sync0 = spk.spike_sync_matrix(spike_trains_STPchaos100[ind0], interval=(0, 1000))
    
    ci,q=bct.modularity_louvain_und(spike_sync0,gamma=1.0, hierarchy=False, seed=None)
    fit_data=spike_sync0[np.argsort(ci)]
    fit_data=fit_data[:,np.argsort(ci)]
    #im2= ax2.imshow(spike_sync, vmin =0.7, vmax = 1.0, interpolation='spline16',cmap=plt.cm.get_cmap('terrain'))
    if i == 12:
        im3= ax[i].imshow(fit_data, interpolation='spline16',cmap=cmap, norm=norm,origin='lower')#, vmin =0.8, vmax = 1.0,interpolation='spline16',cmap=plt.colormaps.get_cmap('gist_ncar'))
    ax[i].imshow(fit_data, interpolation='spline16',cmap=cmap, norm=norm,origin='lower')#, vmin =0.8, vmax = 1.0,interpolation='spline16',cmap=plt.colormaps.get_cmap('gist_ncar'))
    ax[i].set_xticks([249,499,749,999],['250','500','750','1000'])
    ax[i].set_yticks([249,499,749,999],['250','500','750','1000']) 
    
    
    
ay_set = [ax[0],ax[4],ax[8],ax[12]]    


for ay0  in ay_set:
    ay0.set_ylabel('Neuron index')

ax[12].set_xlabel('Neuron index') 
ax[13].set_xlabel('Neuron index')   
ax[14].set_xlabel('Neuron index')    
ax[15].set_xlabel('Neuron index')     
    
    
    
    
    
    
    

axx= [ax[3]]
im_Set= [im0]



cax1 = fig.add_axes([0.93, 0.779, 0.014, 0.19]) #the parameter setting for colorbar position
cbar1=fig.colorbar(im0, cax=cax1,cmap=cmap, norm=norm, boundaries=bounds,ticks=[0, 0.85,0.95,1.0]) # extend='min')
#cbar1=fig.colorbar(im1, cax=cax1, orientation="horizontal")
#cbar1.set_label(r'$\mathsf{\chi}$',fontsize='x-large',labelpad=-13, x=-0.07, rotation=0,color='red')
cbar1.set_label(r'Spike-Sync',fontsize=10,labelpad=2)
#cbar1.set_ticks(np.linspace(0.8,1.0,5))
##change the appearance of ticks anf tick labbel
cbar1.ax.tick_params(labelsize=10)
cax1.xaxis.set_ticks_position("top")


cax2 = fig.add_axes([0.93, 0.539, 0.015, 0.19]) #the parameter setting for colorbar position
cbar2=fig.colorbar(im1, cax=cax2,cmap=cmap, norm=norm, boundaries=bounds,ticks=[0, 0.85,0.95,1.0]) # extend='min')
#cbar1=fig.colorbar(im1, cax=cax1, orientation="horizontal")
#cbar1.set_label(r'$\mathsf{\chi}$',fontsize='x-large',labelpad=-13, x=-0.07, rotation=0,color='red')
cbar2.set_label(r'Spike-Sync',fontsize=10,labelpad=2)
#cbar2.set_ticks(np.linspace(0.8,1.0,5))
##change the appearance of ticks anf tick labbel
cbar2.ax.tick_params(labelsize=10)
cax2.xaxis.set_ticks_position("top")



cax3 = fig.add_axes([0.93, 0.30, 0.015, 0.19]) #the parameter setting for colorbar position
cbar3=fig.colorbar(im2, cax=cax3,cmap=cmap, norm=norm, boundaries=bounds,ticks=[0, 0.85,0.95,1.0]) # extend='min')
#cbar1=fig.colorbar(im1, cax=cax1, orientation="horizontal")
#cbar1.set_label(r'$\mathsf{\chi}$',fontsize='x-large',labelpad=-13, x=-0.07, rotation=0,color='red')
cbar3.set_label(r'Spike-Sync',fontsize=10,labelpad=2)
#cbar3.set_ticks(np.linspace(0.8,1.0,5))
##change the appearance of ticks anf tick labbel
cbar3.ax.tick_params(labelsize=10)
cax3.xaxis.set_ticks_position("top")



cax4 = fig.add_axes([0.93, 0.064, 0.015, 0.19]) #the parameter setting for colorbar position
cbar4=fig.colorbar(im3, cax=cax4,cmap=cmap, norm=norm, boundaries=bounds,ticks=[0, 0.85,0.95,1.0]) # extend='min')
#cbar1=fig.colorbar(im1, cax=cax1, orientation="horizontal")
#cbar1.set_label(r'$\mathsf{\chi}$',fontsize='x-large',labelpad=-13, x=-0.07, rotation=0,color='red')
cbar4.set_label(r'Spike-Sync',fontsize=10,labelpad=2)
#cbar4.set_ticks(np.linspace(0.8,1.0,5))
##change the appearance of ticks anf tick labbel
cbar4.ax.tick_params(labelsize=10)
cax4.xaxis.set_ticks_position("top")





lets=[r'$J_{EE}$'+'--> 0.20','0.4','0.70','1.00']
letss = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)','(j)','(k)','(l)','(m)','(n)','(o)','(p)']
for i,let in zip(range(4),lets):
    ax[i].set_title('%s'%let,fontsize =16)

for i,let in zip(range(16),letss):
     ax[i].text(50,900,'%s'%let,bbox=dict(facecolor='white', alpha=0.8, edgecolor='white',pad=0),fontdict=dict(fontsize=8, fontweight='bold',color= 'black'),fontsize =18)


# plt.figtext(0.02,0.98,'(A)',fontsize = 20) 
# plt.figtext(0.02,0.745,'(B)',fontsize = 20) 
# plt.figtext(0.02,0.515,'(C)',fontsize = 20) 
# plt.figtext(0.02,0.27,'(D)',fontsize = 20) 



plt.savefig('Figure7_revise2.png',dpi = 300)





