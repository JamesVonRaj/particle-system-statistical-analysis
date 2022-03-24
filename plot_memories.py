import numpy as np
from matplotlib import pyplot as plt


#%%

#plots for area fraction = [0.7]

second_derivative_memory = np.load('data/2A/[0.7]_af/2nd_memory.npy')
first_derivative_memory = np.load('data/2A/[0.7]_af/1st_memory.npy')
nnd_memory = np.load('data/2A/[0.7]_af/nnd_memory.npy')

plt.figure(dpi=400,figsize=(8,8))
plt.plot(nnd_memory[:,0],nnd_memory[:,1],c='#2c7bb6',linewidth=2)
plt.axis([0,1,0,max(nnd_memory[:,1])*1.1])
plt.xlabel('area fraction',fontsize=18)
plt.ylabel("count",fontsize=18)

plt.figure(dpi=400,figsize=(8,8))
plt.plot(np.arange(0,1,0.01),second_derivative_memory,c='#fdae61',linewidth=2)
plt.xlabel('area fraction',fontsize=18)
plt.ylabel("f''",fontsize=18)

plt.figure(dpi=400,figsize=(8,8))
plt.plot(np.arange(0,1,0.01),first_derivative_memory,c='#d7191c',linewidth=2)
plt.xlabel('area fraction',fontsize=18)
plt.ylabel("f'",fontsize=18)

#%%

#plots for area fraction = [0.5,0.6]

second_derivative_memory = np.load('data/2A/[0.5,0.6]_af/2nd_memory.npy')
first_derivative_memory = np.load('data/2A/[0.5,0.6]_af/1st_memory.npy')
nnd_memory = np.load('data/2A/[0.5,0.6]_af/nnd_memory.npy')

plt.figure(dpi=400,figsize=(8,8))
plt.plot(nnd_memory[:,0],nnd_memory[:,1],c='#2c7bb6',linewidth=2)
plt.axis([0,1,0,max(nnd_memory[:,1])*1.1])
plt.xlabel('area fraction',fontsize=18)
plt.ylabel("count",fontsize=18)

plt.figure(dpi=400,figsize=(8,8))
plt.plot(np.arange(0,1,0.01),second_derivative_memory,c='#fdae61',linewidth=2)
plt.xlabel('area fraction',fontsize=18)
plt.ylabel("f''",fontsize=18)

plt.figure(dpi=400,figsize=(8,8))
plt.plot(np.arange(0,1,0.01),first_derivative_memory,c='#d7191c',linewidth=2)
plt.xlabel('area fraction',fontsize=18)
plt.ylabel("f'",fontsize=18)

#%%

#plots for area fraction = [0.55,0.6]

second_derivative_memory = np.load('C:/Users/james/OneDrive/Computing/SwellPy/March_Meeting/data/2A/[0.55,0.6]_af/2nd_memory.npy')
first_derivative_memory = np.load('C:/Users/james/OneDrive/Computing/SwellPy/March_Meeting/data/2A/[0.55,0.6]_af/1st_memory.npy')
nnd_memory = np.load('C:/Users/james/OneDrive/Computing/SwellPy/March_Meeting/data/2A/[0.55,0.6]_af/nnd_memory.npy')

plt.figure(dpi=400,figsize=(8,8))
plt.plot(nnd_memory[:,0],nnd_memory[:,1],c='#2c7bb6',linewidth=2)
plt.axis([0,1,0,max(nnd_memory[:,1])*1.1])
plt.xlabel('area fraction',fontsize=18)
plt.ylabel("count",fontsize=18)

plt.figure(dpi=400,figsize=(8,8))
plt.plot(np.arange(0,1,0.01),second_derivative_memory,c='#fdae61',linewidth=2)
plt.xlabel('area fraction',fontsize=18)
plt.ylabel("f''",fontsize=18)

plt.figure(dpi=400,figsize=(8,8))
plt.plot(np.arange(0,1,0.01),first_derivative_memory,c='#d7191c',linewidth=2)
plt.xlabel('area fraction',fontsize=18)
plt.ylabel("f'",fontsize=18)
#%%

#plots for area fraction = [0.4,0.7]

second_derivative_memory = np.load('C:/Users/james/OneDrive/Computing/SwellPy/March_Meeting/data/2A/[0.4,0.7]_af/2nd_memory.npy')
first_derivative_memory = np.load('C:/Users/james/OneDrive/Computing/SwellPy/March_Meeting/data/2A/[0.4,0.7]_af/1st_memory.npy')
nnd_memory = np.load('C:/Users/james/OneDrive/Computing/SwellPy/March_Meeting/data/2A/[0.4,0.7]_af/nnd_memory.npy')

plt.figure(dpi=400,figsize=(8,8))
plt.plot(nnd_memory[:,0],nnd_memory[:,1],c='#2c7bb6',linewidth=2)
plt.axis([0,1,0,max(nnd_memory[:,1])*1.1])
plt.xlabel('area fraction',fontsize=18)
plt.ylabel("count",fontsize=18)

plt.figure(dpi=400,figsize=(8,8))
plt.plot(np.arange(0,1,0.01),second_derivative_memory,c='#fdae61',linewidth=2)
plt.xlabel('area fraction',fontsize=18)
plt.ylabel("f''",fontsize=18)

plt.figure(dpi=400,figsize=(8,8))
plt.plot(np.arange(0,1,0.01),first_derivative_memory,c='#d7191c',linewidth=2)
plt.xlabel('area fraction',fontsize=18)
plt.ylabel("f'",fontsize=18)