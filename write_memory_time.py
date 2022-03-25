import numpy as np
from matplotlib import pyplot as plt
import os

N = 10000
B = 1500

#%%

# af = 0.7

nnd_cycles = np.load('data/2B/[0.7]_af/nnd_memories/nnd_S#1.npy')[:,0]
nnd_memory = np.load('data/2B/[0.7]_af/nnd_memories/nnd_S#1.npy')[:,1]**2 * N * np.pi / (4 * B**2)

der_cycles = np.load('data/2B/[0.7]_af/2nd_derivative_memories/der_S#1.npy')[:,0]
der_memory = np.load('data/2B/[0.7]_af/2nd_derivative_memories/der_S#1.npy')[:,1]


#---------------------------- Individual Memory Plot -------------------------#

plt.figure(dpi=400,figsize=(8,8))

plt.plot(nnd_cycles,nnd_memory,c='#2c7bb6',linewidth=2) 
plt.axis([0,150,0,1])

plt.plot(der_cycles,der_memory,c='#d7191c',linewidth=2) 
plt.axis([0,150,0,1])

plt.legend(['nnd','2nd derivative'],fontsize=18)
plt.xlabel('Cycle Number', fontsize=18)
plt.ylabel('Memory Read', fontsize=18)

#%%

nnd_files = os.listdir('data/2B/[0.7]_af/nnd_memories')

nnd_converging_cycle_array =  []
nnd_y_sum = np.zeros(300)

plt.figure(dpi=400,figsize=(8,8))
for fname in nnd_files:
    nnd_memory_array = np.load('data/2B/[0.7]_af/nnd_memories/' + fname)
    nnd_memory_array_x = nnd_memory_array[:,0]
    nnd_memory_array_y = nnd_memory_array[:,1]**2 * N * np.pi / (4 * B**2)
    nnd_index = np.where(nnd_memory_array_y < 0.698)
    #print(nnd_index[0][-1])
    nnd_converging_cycle_array.append(nnd_index[0][-1])
    plt.plot(nnd_memory_array_x,nnd_memory_array_y,c='#2c7bb6',linewidth=2,alpha=0.7) 
    
    nnd_y_sum = nnd_y_sum + nnd_memory_array_y

nnd_avg_converging_cycle = np.mean(nnd_converging_cycle_array)
nnd_std_converging_cycle = np.std(nnd_converging_cycle_array)

plt.axis([0,150,0,1])
plt.xlabel('Cycle Number',fontsize=18)
plt.ylabel('Memory Read',fontsize=18)


plt.figure(dpi=400,figsize=(8,8))
plt.plot(nnd_memory_array_x,nnd_y_sum/10,c='#2c7bb6',linewidth=2) 
plt.fill_betweenx([0,1],nnd_avg_converging_cycle - nnd_std_converging_cycle, nnd_avg_converging_cycle+ nnd_std_converging_cycle, alpha=0.2,color='#2c7bb6')  
plt.plot([nnd_avg_converging_cycle,nnd_avg_converging_cycle],[0,1],c='#2cb668',linewidth=2)


plt.axis([0,150,0,1])
plt.xlabel('Cycle Number',fontsize=18)
plt.ylabel('Memory Read',fontsize=18)


#%%
der_files = os.listdir('data/2B/[0.7]_af/2nd_derivative_memories')

der_converging_cycle_array =  []
der_y_sum = np.zeros(300)
plt.figure(dpi=400,figsize=(8,8))
for fname in der_files:
    der_memory_array = np.load('data/2B/[0.7]_af/2nd_derivative_memories/' + fname)
    der_memory_array_x = der_memory_array[:,0]
    der_memory_array_y = der_memory_array[:,1]
    der_index = np.where(der_memory_array_y < 0.699)
    der_index_int = der_index[0][-1]
    
    der_converging_cycle_array.append(der_memory_array_x[der_index_int])
    plt.plot(der_memory_array_x,der_memory_array_y,c='#d7191c',linewidth=2,alpha=0.5) 
    
    der_memory_array_y = np.nan_to_num(der_memory_array_y)
    der_y_sum = der_y_sum + der_memory_array_y


der_avg_converging_cycle = np.nanmean(der_converging_cycle_array)
der_std_converging_cycle = np.nanstd(der_converging_cycle_array)

plt.axis([0,150,0,1])
plt.xlabel('cycle #',fontsize=18)
plt.ylabel('read memory',fontsize=18)


plt.figure(dpi=400,figsize=(8,8))
plt.plot(der_memory_array_x,der_y_sum/10,c='#d7191c',linewidth=2) 
plt.fill_betweenx([0,1],der_avg_converging_cycle - der_std_converging_cycle, der_avg_converging_cycle+ der_std_converging_cycle, alpha=0.2,color='#d7191c')  
plt.plot([der_avg_converging_cycle,der_avg_converging_cycle],[0,1],c='#d7197b',linewidth=2)


plt.axis([0,150,0,1])
plt.xlabel('Cycle Number',fontsize=18)
plt.ylabel('Memory Read',fontsize=18)



#%%

# reading time plot for af = 0.7

N_array = np.load('data/2B/[0.7]_af/reading_time/N_array.npy')
nat_time = np.load('data/2B/[0.7]_af/reading_time/nat_time.npy')
nnd_time = np.load('data/2B/[0.7]_af/reading_time/nnd_time.npy')

plt.figure(dpi=400,figsize=(8,8))

plt.plot(N_array,nnd_time,c='#2c7bb6',linewidth=2)
plt.plot(N_array,nat_time,c='#d7191c',linewidth=2)
plt.xlabel('number of particles', fontsize=18)
plt.ylabel('time [seconds]', fontsize=18)
plt.legend(['nnd','2nd derivative'],fontsize=18)




