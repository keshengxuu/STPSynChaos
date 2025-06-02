# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 19:08:34 2025

@author: admin
"""

import matplotlib
matplotlib.use('Agg')
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import Wavelets
from NetworkarchitectureB import Networkmodel
from mpi4py import MPI
import csv

rank = MPI.COMM_WORLD.rank
threads = MPI.COMM_WORLD.size

def expnorm(tau1, tau2):
    if tau1 > tau2:
        t2 = tau2; t1 = tau1
    else:
        t2 = tau1; t1 = tau2
    tpeak = t1 * t2 / (t1 - t2) * np.log(t1 / t2)
    return (np.exp(-tpeak / t1) - np.exp(-tpeak / t2)) / (1 / t2 - 1 / t1)

def ISI_Phase(Num_Neurs, TIME, spikes):
    Phase = np.zeros((TIME.size, Num_Neurs), dtype=np.float32)
    for ci in range(Num_Neurs):  # Num_Neurs is total number of neurons
        mth = 0  # the flag of mth spikes of neuron i at time t
        nt = 0
        for t in TIME:
            if np.sum(spikes[ci]) < 2:
                Phase[nt, ci] = 0
            elif (mth + 1) < np.array(spikes[ci]).size:
                Phase[nt, ci] = 2.0 * np.pi * (t - spikes[ci][mth]) / (spikes[ci][mth + 1] - spikes[ci][mth])
                nt = nt + 1
                if t > spikes[ci][mth + 1]:
                    mth = mth + 1
    return Phase

def HR_network(X, i, ux_ex):
    global firing
    x, y, z, sex, sey, six, siy, sexe, seye = X  # agregué variable 's'
    
    ISyn = (sey + seye) * (x - VsynE) + siy * (x - VsynI)
        
    firingExt = np.random.binomial(1, iRate * dt, size=N)

    if any(i > delay_dt):
        firing = (V_t[i - delay_dt, range(N)] > theta) * (V_t[i - delay_dt - 1, range(N)] < theta)

    return np.array([y - a0 * x**3 + b0 * x**2 + i_ext0 - z - ISyn,
                     c0 - d0 * x**2 - y,
                     r0 * (s0 * (x - k0) - z),
                     -sex * (1 / tau1E + 1 / tau2E) - sey / (tau1E * tau2E) + np.dot(CMeMatrix, ux_ex) + np.dot(CMieMatrix, firing[0:Ne]),
                     sex,
                     -six * (1 / tau1I + 1 / tau2I) - siy / (tau1I * tau2I) + np.dot(CMiMatrix, firing[Ne:]) + np.dot(CMeiMatrix, firing[Ne:]),
                     six,
                     -sexe * (1 / tau1E + 1 / tau2E) - seye / (tau1E * tau2E) + firingExt * GsynExt,
                     sexe])  

equil = 400  # 400
Trun = 2000  # 2000    
Total = Trun + equil  # ms

dt = 0.01  # ms

a0 = 1.0
b0 = 3.0
c0 = 1.0
d0 = 5.0
s0 = 4.0
r0 = 0.006
k0 = -1.56
i_ext0 = 1.5  # chaos

# Synaptic parameters
mGsynE = 1.0; mGsynI = 40; mGsynExt = 0.6  # mean
sGsynE = 0.2; sGsynI = 2; sGsynExt = 0.2
VsynE = 0; VsynI = -1.0  # reversal potential
tau1E = 3; tau2E = 1
tau1I = 4; tau2I = 1
Pe = 0.3; Pi = 0.3
iRate = 6  # 10*Pi
P_SW = 0.01

factE = 1000 * dt * expnorm(tau1E, tau2E)
factI = 1000 * dt * expnorm(tau1I, tau2I)
W_gap = 0  # 0.001
W_gapi = 0.001  # 0.05

mdelay = 1.5; sdelay = 0.1  # ms

theta = 0

Ne = 1000 
Ni = 250 
N = Ne + Ni

GsynE = np.random.normal(mGsynE, sGsynE, size=(N, Ne)) / factE
GsynExt = np.random.normal(mGsynExt, sGsynExt, size=N)
GsynExt = GsynExt * (GsynExt > 0) / factE
GsynI = np.random.normal(mGsynI, sGsynI, size=(N, Ni)) / factI
delay = np.random.normal(mdelay, sdelay, size=N)

np.random.seed(10)

firing = np.zeros(N) 

delay_dt = (delay / dt).astype(int)
equil_dt = int(equil / dt)
Time = np.arange(0, Total, dt)
nsteps = len(Time)
Time2 = Time[equil_dt:]

# Fixed parameters
jee = 3.0
jei = 0.2
jie = 0.02
jii = 0.03

# Varying parameters
tau_rec_values = np.arange(50, 2050, 100)
tau_facil_values = np.arange(50, 2050, 100)
U = 0.1

Vals = [(v1, v2) for v1 in tau_rec_values for v2 in tau_facil_values]
isim = 0

for val in Vals:
    tau_rec, tau_facil = val
    
    lastSpike_ex = np.zeros(Ne)
    interval_ex = np.zeros(Ne)
    tmp_ex = np.zeros(Ne)
    ux_ex = np.zeros(Ne)
    x_ex = np.zeros(Ne)
    u_ex = np.zeros(Ne)
    
    minvtaur = -1. / tau_rec 
    minvtauf = -1. / tau_facil
    
    dtype = np.float32
    record_x = np.zeros((Ne, nsteps), dtype=dtype)
    record_u = np.zeros((Ne, nsteps), dtype=dtype)
    record_ux = np.zeros((Ne, nsteps), dtype=dtype)
    record_y = np.zeros((Ne, nsteps), dtype=dtype)
    
    vex_before = np.zeros((Ne))
    vih_before = np.zeros((Ni))
    
    vex = np.zeros((Ne))
    vih = np.zeros((Ni))

    EX_record = np.zeros((nsteps, Ne), dtype=float)
    INH_record = np.zeros((nsteps, Ni), dtype=float)

    if isim % threads == rank:
        net = Networkmodel(Ne1=Ne,   # The numbers of excitatroy ensemble
                            Ni1=Ni,  # The numbers of inhibitory ensemble
                            cp=P_SW,  # the probabilty to add new links
                            W_gap=0,  # the weight of gap junction for excitatoty population.
                            W_Chem1=jee,  # the weight of itself chemical synapse in excitatoty population.
                            W_ei1=jei,  # the weight of inhibitory synapse currents from inhibitory to excitatory population.
                            W_ii1=jii,  # the weight of itself inhibitory currents in inhibitory population
                            W_ie1=jie)  # the weight of excitatory currents in inhibitory population from excitatory population

        wEE_gap, wEE_Chem, wEI_chem, wII_chem, wIE_chem = net.SmallWorld(deg=10, opt=2)
    
        CMeMatrix = np.concatenate((wEE_Chem, np.zeros((Ni, Ne))), axis=0) * GsynE
        CMeiMatrix = np.concatenate((wEI_chem, np.zeros((Ni, Ni))), axis=0) 
        CMiMatrix = np.concatenate((np.zeros((Ne, Ni)), wII_chem), axis=0) * GsynI
        CMieMatrix = np.concatenate((np.zeros((Ne, Ne)), wIE_chem), axis=0) 
        CM0 = np.concatenate((wEE_gap, np.zeros((Ni, Ne))), axis=0) 
        CMgapMat = np.concatenate((CM0, np.zeros((N, Ni))), axis=1)
    
        ISyn_t = np.zeros((nsteps, N))
        V_t = np.zeros((nsteps, N))
   
        x_init = np.random.uniform(-2, 1, size=N)
        y = np.random.uniform(-2, 1, size=N)
        z = np.random.uniform(-3, -2, size=N)
        sex = np.zeros_like(x_init)
        sey = np.zeros_like(x_init)
        six = np.zeros_like(x_init)
        siy = np.zeros_like(x_init)
        sexe = np.zeros_like(x_init)
        seye = np.zeros_like(x_init)
     
        X = (x_init, y, z, sex, sey, six, siy, sexe, seye)
        
        for i in range(nsteps):
            t = i * dt
            interval_ex = t - lastSpike_ex
            
            u_ex = U + (u_ex - U) * np.exp(interval_ex * minvtauf)
            tmp_ex = 1.0 - u_ex
            x_ex = 1.0 + (x_ex - 1) * np.exp(interval_ex * minvtaur)
            
            if i >= 1:
                for ci in range(Ne):
                    if V_t[i, ci] >= -0.1 and V_t[i - 1, ci] < -0.1:
                        x_ex[ci] *= tmp_ex[ci]
                        u_ex[ci] += U * tmp_ex[ci]
                        lastSpike_ex[ci] = t
            ux_ex = u_ex * x_ex 
           
            X += dt * HR_network(X, i, ux_ex)
            V_t[i] = X[0] 
           
        V_t = V_t[equil_dt:, :]
        Time_simu = Time2
                    
        spikes = [(np.diff(1 * (V_t[:, i] > theta)) == 1).nonzero()[0] for i in range(N)]
        
        with open("Spikes_tau_CHAOS40res1_I30STD15/Spikes_%04d_tau_rec_%d_tau_facil_%d.txt" % (isim, tau_rec, tau_facil), 'w') as f:
            writer = csv.writer(f, 'excel')
            for sp in spikes:
                writer.writerow(['%.3f' % x for x in sp])       
                            
    isim += 1
