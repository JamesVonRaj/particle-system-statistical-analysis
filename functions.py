import numpy as np
import math
import scipy.spatial.ckdtree
from scipy.signal import peak_widths, find_peaks
from sklearn.cluster import DBSCAN



# gives nnd plot (x,y) data
# Parameters: particle data (X) : array_like, shape (n,m)
#             boxsize (B) : scalar
def make_NND(X,B): 
    
    tree = scipy.spatial.cKDTree(X,boxsize=B)
    dist,point = tree.query(X,k=[2],distance_upper_bound=30)
        
    b1 = np.linspace(0,30,601) #resolution of probability distribution
    hist, edges = np.histogram(dist, bins=b1)
    edges_pruned = edges[0:len(edges)-1]
    hist_data = np.vstack((edges_pruned,hist)).T
    x = hist_data[:,0]
    y = hist_data[:,1]
    
    return (x,y)



# gives 1D array of nnd distances
# Parameters: particle data (X) : array_like, shape (n,m)
#             boxsize (B) : scalar
def make_NND_dist(X,B):

    tree = scipy.spatial.cKDTree(X,boxsize=B)
    dist,point = tree.query(X,k=[2],distance_upper_bound=30)
    
    return dist



# gives the weighted average and standard deviation
# Parameters: values, weights : Numpy ndarrays with the same shape
def weighted_avg_and_std(values, weights):

    average = np.average(values, weights=weights)
    variance = np.average((values-average)**2, weights=weights)
    
    return (average, math.sqrt(variance))



# gives (m,3) arrays of cluster particles and stray particles along with their 
# cluster labels
# Parameters: epsilon (eps) : scalar
#             minimum_sample_number (min_samples) : scalar
def compute_dbscan(epsilon,minimum_sample_number,X):
    
    db = DBSCAN(eps=epsilon, min_samples=minimum_sample_number).fit(X)
    
    labels = np.array(db.labels_)
    labels = np.reshape(labels, (labels.shape[0], 1)) 
    data = np.hstack((X,labels))
    
    clusters = data[np.logical_not(np.logical_and(data[:,2] == -1, data[:,2] == -1))]
    strays = data[np.logical_and(data[:,2] == -1, data[:,2] == -1)]
    
    return (clusters,strays)



# gives 1D arrays of cluster size for each cluster in the system.
# Also gives 1D array of each particle's size value. 
# Parameters: cluster labels (cluster_labels) : 1D array
def cluster_data(cluster_labels):

    cluster_values = []
    cluster_value_arrays = []
    max_cluster_label = int(np.max(cluster_labels)) #shows how many clusters there are
    
    for i in range(max_cluster_label + 1): #iterates through every cluster label
        clust_val = np.where(cluster_labels == i)
        cluster_value_arrays.append(clust_val)
        
        clust_val = np.size(clust_val)
        cluster_values.append(clust_val)
    
    
    cluster_size_array = []
    for i in cluster_labels:
        cluster_size_array.append(cluster_values[int(i)]) #Used for colorbar for
        #cluster plot. gives array that assigns cluster size for each particle. 
        #earch particle then gets a size parameter.
    
    return (cluster_values, cluster_size_array)



# gives 1D arrays of the mean and standard deviation for clusters.
# Also gives 1D array of the size of each cluster. 
# Parameters: cluster array from cluster_data function (cluster) : (m,3) array
#             boxsize (B) : scalar
def nnd_mean_and_std(cluster,B):
    
    mean_nnd_clusters = []
    std_nnd_clusters = []
    size_nnd_clusters = []
    
    int_max_csa = int(np.max(cluster[:,2]) + 1) # max cluster size integer
    
    for i in range(int_max_csa):
    
        single_cluster_loc = np.where(cluster[:,2] == i) # gives array of 
        # cluster particles location in cluster array.
        
        single_cluster_array = []
        for j in single_cluster_loc: # iterates over each particle location 
        # for a single cluster.
            
            single_cluster_array.append(cluster[j,0:2]) # gives array of (x,y)
            # location for each particle in a specific cluster.
        
        single_cluster_array = np.array(single_cluster_array)
        single_cluster_array = single_cluster_array[0][:,:]
        
    
        single_cluster_array = np.array(single_cluster_array)
        dist = make_NND_dist(single_cluster_array,B)
        
        dist = np.unique(dist) # removes duplicate distances in a cluster.
        if np.size(dist) != 1:
            mean_nnd_clusters.append(np.mean(dist))
            
            std_nnd_clusters.append(np.std(dist))
            size_nnd_clusters.append(len(single_cluster_array[:,0]))
    
    return (mean_nnd_clusters,std_nnd_clusters,size_nnd_clusters)



# gives 1D arrays of the particle number, full width half max,
# and nnd peak location for clusters.
# Parameters: cluster array from cluster_data function (cluster) : (m,3) array
#             boxsize (B) : scalar
def nnd_pkh_loc_and_fwhm(cluster,B):
    
    particle_number = []
    fwhm = []
    pkh_loc_array = []
    
    int_max_csa = int(np.max(cluster[:,2]) + 1)
    
    for i in range(int_max_csa):
    
        single_cluster_loc = np.where(cluster[:,2] == i)
        single_cluster_array = []
        
        for j in single_cluster_loc:
            single_cluster_array.append(cluster[j,0:2])
        
        single_cluster_array = np.array(single_cluster_array)
        single_cluster_array = single_cluster_array[0][:,:]
    
        single_cluster_array = np.array(single_cluster_array)
        radius_x,radius_y = make_NND(single_cluster_array,B)
        
        pkh_loc = radius_x[np.where(radius_y == max(radius_y))[0][0]]
        
        pkh_loc_index = [np.where(radius_y == max(radius_y))[0][0]]
        
        results_half = peak_widths(radius_y, pkh_loc_index, rel_height=0.5)[0][0] * 0.1
    
        fwhm.append(results_half)
        pkh_loc_array.append(pkh_loc)
        particle_number.append(len(single_cluster_array[:,0]))
    
    return (particle_number,fwhm,pkh_loc_array)





# gives 1D arrays of the particle number, full width half max,
# and nnd peak location for clusters with two memories written into the system 
# Parameters: cluster array from cluster_data function (cluster) : (m,3) array
#             boxsize (B) : scalar
def two_memory_cluster_data(cluster,B):
    particle_number_1 = []
    fwhm_1 = []
    pkh_loc_array_1 = []
    
    particle_number_2 = []
    fwhm_2 = []
    pkh_loc_array_2 = []
    int_max_csa = int(np.max(cluster[:,2]) + 1)
    
    for i in range(int_max_csa):
    
        single_cluster_loc = np.where(cluster[:,2] == i)
        single_cluster_array = []
        
        for j in single_cluster_loc:
            single_cluster_array.append(cluster[j,0:2])
        
        single_cluster_array = np.array(single_cluster_array)
        single_cluster_array = single_cluster_array[0][:,:]
    
        single_cluster_array = np.array(single_cluster_array)
        radius_x,radius_y = make_NND(single_cluster_array, single_cluster_array,B)
        
        pkh_loc = radius_x[np.where(radius_y == max(radius_y))[0][0]]
        
        pkh_loc_index = [np.where(radius_y == max(radius_y))[0][0]]
        
        results_half = peak_widths(radius_y, pkh_loc_index, rel_height=0.5)[0][0] * 0.1
        
        fwhm_1.append(results_half)
        pkh_loc_array_1.append(pkh_loc)
        particle_number_1.append(len(single_cluster_array[:,0]))
        
        if len(single_cluster_array[:,0]) > 8:
        
            peaks, _  = find_peaks(radius_y)
            pkh_2 = np.sort(radius_y[peaks])[-2]
            pkh_loc_2 = radius_x[np.where(radius_y == pkh_2)[0][0]]
            
            pkh_loc_index_2 = [np.where(radius_y == pkh_2)[0][0]]
            
            results_half_2 = peak_widths(radius_y, pkh_loc_index_2, rel_height=0.5)[0][0] * 0.1
            
            fwhm_2.append(results_half_2)
            pkh_loc_array_2.append(pkh_loc_2)
            particle_number_2.append(len(single_cluster_array[:,0]))
    
    return (particle_number_1,fwhm_1,pkh_loc_array_1,particle_number_2,fwhm_2,pkh_loc_array_2)


def cluster_count_hist(counts,bins):
    fake = np.array([])
    for i in range(len(counts)):
        a, b = bins[i], bins[i+1]
        sample = a + (b-a)*np.random.rand(counts[i])
        fake = np.append(fake, sample)
    
    return (fake, bins)
