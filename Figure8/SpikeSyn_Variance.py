from __future__ import print_function
import pyspike as spk
from pyspike import SpikeTrain
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec



plt.rcParams['mathtext.sf'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'

#changing the xticks and  yticks fontsize for all sunplots
plt.rc('xtick', labelsize=7)
plt.rc('ytick', labelsize=7)
plt.rc('font',size=7)

num =10
spike_trains = [spk.load_spike_trains_from_txt("sptimes_chaos/Sptraintimes_ex_%d.txt"%fl,separator=',',
                                               edges=(0, 2000)) for fl in range(num)]
spike_trains30 = [spk.load_spike_trains_from_txt("sptimes030_STPchaos/SptraintimesSTP_ex_%d.txt"%fl,separator=',',
                                               edges=(0, 2000)) for fl in range(num)]


spike_trains60 = [spk.load_spike_trains_from_txt("sptimes060_STPchaos/SptraintimesSTP_ex_%d.txt"%fl,separator=',',
                                               edges=(0, 2000)) for fl in range(num)]


spike_trains100 = [spk.load_spike_trains_from_txt("sptimes100_STPchaos/SptraintimesSTP_ex_%d.txt"%fl,separator=',',
                                               edges=(0, 2000)) for fl in range(num)]


var03=[]
vart_stf30=[]
vart_stf60 = []
vart_stf100 = []
binss =20
for i in range(num):
    spike_sync = spk.spike_sync_matrix(spike_trains[i], interval=(1000, 2000))
    sp_sync= spike_sync[np.triu_indices(np.size(spike_sync,0))]
    Var = np.var(sp_sync)
    var03.append(Var)




for i in range(num):
    spike_sync = spk.spike_sync_matrix(spike_trains30[i], interval=(1000, 2000))
    sp_sync= spike_sync[np.triu_indices(np.size(spike_sync,0))]
    # Multistability var(spike_sync)
    Var = np.var(sp_sync)
    vart_stf30.append(Var)
    

for i in range(num):
    spike_sync = spk.spike_sync_matrix(spike_trains60[i], interval=(1000, 2000))
    sp_sync= spike_sync[np.triu_indices(np.size(spike_sync,0))]
    # Multistability var(spike_sync)
    Var = np.var(sp_sync)
    vart_stf60.append(Var)


for i in range(num):
    spike_sync = spk.spike_sync_matrix(spike_trains100[i], interval=(1000, 2000))
    sp_sync= spike_sync[np.triu_indices(np.size(spike_sync,0))]
    # Multistability var(spike_sync)
    Var = np.var(sp_sync)
    vart_stf100.append(Var)


np.savetxt('varance/vart.txt',np.array(var03)*1000)
np.savetxt('varance/vart_stf30.txt',np.array(vart_stf30)*1000)
np.savetxt('varance/vart_stf60.txt',np.array(vart_stf60)*1000)
np.savetxt('varance/vart_stf100.txt',np.array(vart_stf100)*1000)





