

#Colab Version

import numpy as np
import pandas as pd
import scipy as sc
import math
import re
from Bio import SeqIO
import time
from datetime import datetime
from sklearn.model_selection import train_test_split
from  sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import sklearn
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import silhouette_score
# from sompy.sompy import SOMFactory




genes = pd.read_csv("SC_gene_expression.csv")
gfeatures = genes.iloc[0:,1:]
gfeatures = gfeatures.iloc[:,:].fillna(0)#fill all missing value(NaN) replaced by zero
gfeatures = np.array(gfeatures)
gf = np.array(gfeatures)
gname = genes.iloc[:,0]
gn = np.array(gname)
m, n= gf.shape
print(m,n)

#fill out missing value(NaN) in dataset with average
for i in range(n):
    s = np.sum(gf[:,i])
    avg = s/m
    gf[:,i] = np.where(gf[:,i]==0, avg, gf[:,i])




#Descretization
gf = np.where(gf<-0.1,-1 , gf)
gf = np.where(gf > 0.1, 1, gf)
gf = np.where((gf >= -0.1) & (gf <= 0.1), 0, gf)





#Kmeans
comp = np.zeros([10,1])
lbl = np.zeros([m,10])


for i in range(10):
    km = KMeans(n_clusters=4)#, random_state=0
    km.fit(gf)
    # kmcenters = km.cluster_centers_
    # print(km.labels_)
    # print(np.shape(km.labels_))

    # print(kmcenters[1,:])
    kmlbl = km.labels_
    # print("kmlabels size is:", np.shape(kmlabels))

    silkm = silhouette_score(gf, kmlbl)

    comp[i,0] = silkm
    lbl[:,i] = kmlbl
kmlabels = lbl[:,ind]
ind = np.argmax(comp)
dbikm = davies_bouldin_score(gf, kmlabels)
print("DBI for kmeans is:", dbikm)
print("silhouette score for kmeans is:", max(comp))

kmcenters = km.cluster_centers_
print("size of center  is :", np.shape(kmcenters))

kmcenters = np.where(kmcenters < -0.1, -1, kmcenters)
kmcenters = np.where(kmcenters > 0.1, 1, kmcenters)
kmcenters = np.where((kmcenters >= -0.1) & (kmcenters <= 0.1), 0, kmcenters)

#Mutual Information
nmikm = 0
km0 = 0
km1 = 0
km2 = 0
km3 = 0
kmname0 = []
kmname1 = []
kmname2 = []
kmname3 = []

kmnames0 = pd.DataFrame(columns=["Kmeans", "cluster"])
kmnames1 = pd.DataFrame(columns=["Kmeans", "cluster"])
kmnames2 = pd.DataFrame(columns=["Kmeans", "cluster"])
kmnames3 = pd.DataFrame(columns=["Kmeans", "cluster"])



for i in range(m):
    if kmlabels[i] == 0:
        nmikm = nmikm + normalized_mutual_info_score(kmcenters[0,:],gf[i,:])
        km0 += 1
        kmname0.append(gname[i])
        kmnames0 = kmnames0.append({'Kmeans':gname[i] , 'cluster': 0}, ignore_index=True)
    elif kmlabels[i] ==1:
        nmikm = nmikm + normalized_mutual_info_score(kmcenters[1, :], gf[i, :])
        km1 += 1
        kmname1.append(gname[i])
        kmnames1 = kmnames1.append({'Kmeans':gname[i] , 'cluster': 1}, ignore_index=True)
    elif kmlabels[i] == 2:
        nmikm = nmikm + normalized_mutual_info_score(kmcenters[2, :], gf[i, :])
        km2 += 1
        kmname2.append(gname[i])
        kmnames2 = kmnames2.append({'Kmeans':gname[i] , 'cluster': 2}, ignore_index=True)
    elif kmlabels[i] == 3:
        nmikm = nmikm + normalized_mutual_info_score(kmcenters[3, :], gf[i, :])
        km3 += 1
        kmname3.append(gname[i])
        kmnames3 = kmnames3.append({'Kmeans':gname[i] , 'cluster': 3}, ignore_index=True)



print("Normalized mutual information for Kmeans is:", nmikm)
print("the number of cluster0 in kmeans is:", km0)
print("the number of cluster1 in kmeans is:", km1)
print("the number of cluster2 in kmeans is:", km2)
print("the number of cluster3 in kmeans is:", km3)


kmnames0.to_csv("KmeansClusters0.csv", index=False)
kmnames1.to_csv("KmeansClusters1.csv", index=False)
kmnames2.to_csv("KmeansClusters2.csv", index=False)
kmnames3.to_csv("KmeansClusters3.csv", index=False)







#GMM
comp2 = np.zeros([10,1])
lbl2 = np.zeros([m,10])
for i in range(10):
    gmm = GaussianMixture(n_components=4, covariance_type='full')#, random_state=0
    gmm.fit(gf)
    # print("prediction by GMM is:")
    # print(gmm.predict(gf))
    #Find GMM centers
    gmmlbl = gmm.predict(gf)

    silgmm = silhouette_score(gf, gmmlbl)
    # print("silhouette score for GMM is:", silgmm)



    gmmcenter = np.empty(shape=(gmm.n_components, gf.shape[1]))
    for i in range(gmm.n_components):
        density = sc.stats.multivariate_normal(cov=gmm.covariances_[i], mean=gmm.means_[i]).logpdf(gf)
        gmmcenter[i, :] = gf[np.argmax(density)]
    # print(gmmcenter[3,:])
    # print("size of gmmcenter  is :", np.shape(gmmcenter))
    comp2[i, 0] = silgmm
    lbl2[:, i] = gmmlbl

ind2 = np.argmax(comp2)
gmmlabels = lbl2[:,ind2]
dbigmm = davies_bouldin_score(gf, gmmlabels)
print("DBI for GMM is:", dbigmm)
print("silhouette score for GMM is:", max(comp2))

nmigmm = 0
gmm0 = 0
gmm1 = 0
gmm2 = 0
gmm3 = 0
gmmname0 = []
gmmname1 = []
gmmname2 = []
gmmname3 = []

gmmnames0 = pd.DataFrame(columns=["GMM", "cluster"])
gmmnames1 = pd.DataFrame(columns=["GMM", "cluster"])
gmmnames2 = pd.DataFrame(columns=["GMM", "cluster"])
gmmnames3 = pd.DataFrame(columns=["GMM", "cluster"])
for i in range(m):
    if gmmlabels[i] == 0:
        nmigmm = nmigmm + normalized_mutual_info_score(gmmcenter[0,:],gf[i,:])
        gmm0 += 1
        gmmname0.append(gname[i])
        gmmnames0 = gmmnames0.append({'GMM': gname[i], 'cluster': 0}, ignore_index=True)
    elif gmmlabels[i] ==1:
        nmigmm = nmigmm + normalized_mutual_info_score(gmmcenter[1, :], gf[i, :])
        gmm1 += 1
        gmmname1.append(gname[i])
        gmmnames1 = gmmnames1.append({'GMM': gname[i], 'cluster': 1}, ignore_index=True)
    elif gmmlabels[i] == 2:
        nmigmm = nmigmm + normalized_mutual_info_score(gmmcenter[2, :], gf[i, :])
        gmm2 += 1
        gmmname2.append(gname[i])
        gmmnames2 = gmmnames2.append({'GMM': gname[i], 'cluster': 2}, ignore_index=True)
    elif gmmlabels[i] == 3:
        nmigmm = nmigmm + normalized_mutual_info_score(gmmcenter[3, :], gf[i, :])
        gmm3 += 1
        gmmname3.append(gname[i])
        gmmnames3 = gmmnames3.append({'GMM': gname[i], 'cluster': 3}, ignore_index=True)





print("Normalized mutual information for GMM is:", nmigmm)

print("the number of cluster0 in gmm is:", gmm0)
print("the number of cluster1 in gmm is:", gmm1)
print("the number of cluster2 in gmm is:", gmm2)
print("the number of cluster3 in gmm is:", gmm3)


        

gmmnames0.to_csv("gmmClusters0.csv", index=False)
gmmnames1.to_csv("gmmClusters1.csv", index=False)
gmmnames2.to_csv("gmmClusters2.csv", index=False)
gmmnames3.to_csv("gmmClusters3.csv", index=False)
















# print(gmmcenter[0,:])

# print("this shape is:", np.shape(np.empty(shape=(gmm.n_components,gf.shape[1]))))



#SOM
# som =sm.SOM(gf)
# from  minisom import MiniSom
# w = 10
# h = 10
# print(len(gf[0]))
# som = MiniSom(h, w, len(gf[0]), learning_rate=0.5,
#               sigma=1, neighborhood_function='triangle', random_seed=10)
#
# som.train_random(gf, 100, verbose=True)
# print(som.winner(gf[0:10,:]))
# win_map = som.win_map(gf)
# print(som.activation_response(gf))



#Metric parameter for clustering















#fill out missing value(NaN) in dataset
# k = np.sum(gf[:,0])
# print(k)
# cnt = 0
# for j in range(n):
#     for i in range(m):
# gfeatures.iloc[:,0].fillna(0)
# for i in range(m):
#      if np.isnan(gf[i,0]):
#          cnt += 1
#
#
#
# print(cnt)
# cnt2 = 0
# gfeatures.iloc[:,0].fillna(0)
# for i in range(m):
#     for j in range(n):
#         if gf[i,j] == 0:
#             cnt2 += 1
#
# print(cnt2)
#
#
#
# print(gf[:,0])


# code to replace all negative value with 0
# result = np.where(ini_array1<0, 0, ini_array1)



