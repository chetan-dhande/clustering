# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 17:55:12 2020

@author: Chetan
"""

import pandas as pd
import matplotlib.pyplot as plt 
df = pd.read_csv("C:\\Users\\ADMIN\\Desktop\\chetan\\assignment\\6.clustering\\crime_data.csv")
df.columns
df1= df.iloc[:,1:]

from scipy.cluster.hierarchy import linkage 
import scipy.cluster.hierarchy as sch 

z = linkage(df1,method='complete',metric='euclidean')
plt.figure(figsize=(20, 5));
plt.title('Hierarchical Clustering Dendrogram');
plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(
    z,
    leaf_rotation=0.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()

from sklearn.cluster import	AgglomerativeClustering 
clusters = AgglomerativeClustering(n_clusters=3,linkage='complete',affinity = "euclidean")
clusters.fit(df1)
clusters.labels_

clusters_labels = pd.Series(clusters.labels_)
df['clust'] = clusters_labels
df.columns
df = df.iloc[:,[5,0,1,2,3,4]]
df.columns
df.clust
df.groupby(df.clust).mean()

df.to_csv("crime.csv",index=False)
