# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 19:37:55 2020

@author: Chetan
"""
                                            """
                                          K-means 
                                            """
import pandas as pd
import matplotlib.pylab as plt
from sklearn.cluster import	KMeans
from scipy.spatial.distance import cdist 
import numpy as np

df = pd.read_excel("C:\\Users\\ADMIN\\Desktop\\chetan\\assignment\\6.clustering\\EastWestAirlines.xlsx","data")
print(df)
def n (i):
     x = (i-i.min())	/	(i.max()	-	i.min())
     return (x)

df_norm = n(df.iloc[:,1:])

k = list(range(1,10))
k
TWSS = [] # variable for storing total within sum of squares for each kmeans 
for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    WSS = [] # variable for storing within sum of squares for each cluster 
    for j in range(i):
        WSS.append(sum(cdist(df_norm.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,df_norm.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))

plt.plot(k,TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS");plt.xticks(k)

model=KMeans(n_clusters=5) 
model.fit(df_norm)

model.labels_
df['clust']=pd.Series(model.labels_)   
df.head(10) 
df.columns
df = df.iloc[:,[0,12,1,2,3,4,5,6,7,8,9,10,11]]
df.iloc[:,1:7].groupby(df.clust).mean()
df.groupby(df.clust).mean()
df.to_csv("C:\\Users\\ADMIN\\Desktop\\chetan\\assignment\\6.clusteringflight.csv",index=False)



                                            """
                                          H-cluster
                                            """

df1 = pd.read_excel("C:\\Users\\ADMIN\\Desktop\\chetan\\assignment\\6.clustering\\EastWestAirlines.xlsx","data")

def n (i):
     x = (i-i.min())	/	(i.max()	-	i.min())
     return (x)

df_norm = n(df1.iloc[:,1:])

from scipy.cluster.hierarchy import linkage 
import scipy.cluster.hierarchy as sch 
z = linkage(df_norm,method='complete',metric='euclidean')
plt.figure(figsize=(20, 5));
plt.title('Hierarchical Clustering Dendrogram');
plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(
    z,
    leaf_rotation=0.,  
    leaf_font_size=8.,  
)
plt.show()

from sklearn.cluster import	AgglomerativeClustering 
clusters = AgglomerativeClustering(n_clusters=5,linkage='complete',affinity = "euclidean")
clusters.fit(df_norm)
clusters.labels_

clusters_labels = pd.Series(clusters.labels_)
df1['clust'] = clusters_labels
df1.columns
df1 = df.iloc[:,[0,12,1,2,3,4,5,6,7,8,9,10,11]]
df1.columns
df1.clust
df1.groupby(df1.clust).mean()

df1.to_csv("flight2.csv",index=False)
