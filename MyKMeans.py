import numpy as np


import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


class MyKMeans:
	mX =[] #dataset rows contains position of clustered elements
	labels=[]
	
	max_iter=300 #Maximum number of iterations of the k-means algorithm for a single run.
	
	n=8 #number of clusters
	n_init=10 #Number of times the k-means algorithm is run with different centroid seeds.
	cluster_means=[]
	
	
	def __init__(self, n_clusters = 8, n_init = 10, max_iter=300):
		self.n = n_clusters
		self.n_init = n_init
		self.max_iter = max_iter
	
	
	def fit_predict(self ,X):
		self.mX=X
		
		self.predict_once()
		
		best_cluster_means = np.copy(self.cluster_means)
		best_total_deviation= self.total_deviation()
		
		for i in range(self.n_init-1):
			self.predict_once()
			if (self.total_deviation() > best_total_deviation):
				best_cluster_means =  np.copy(self.cluster_means)
				best_total_deviation= self.total_deviation()
		self.cluster_means = best_cluster_means
		self.actualise_labels()
		
	def total_deviation(self):
		tot_dev = 0.0
		for i in range(self.n):
			tot_dev+=np.sum(np.std(self.mX[self.labels == i], axis=1))
			
		return tot_dev

	def predict_once(self): #run k-mean nagoritm with random init once
		self.set_random_centroids()
		
		self.actualise_labels()
		
		means_changed = True
		
		iterations_count = 0
		while (means_changed and iterations_count < self.max_iter):
			iterations_count += 1
			old_means = np.copy(self.cluster_means)
			self.actualise_means()
			self.actualise_labels()
			means_changed = not((old_means == self.cluster_means).all())
		
	def set_random_centroids(self):
		min_x_vector = np.amin(self.mX,axis=0)#initialise min vector (all coordinates minimal)
		max_x_vector = np.amax(self.mX,axis=0)#initialise max vector (all coordinates maximal)
		
		range_x_vector = max_x_vector - min_x_vector #range of values
		
		self.cluster_means = np.random.rand(self.n,len(self.mX[0])) 
		for point in self.cluster_means:
			point *= range_x_vector
			point += min_x_vector
		
		
		
	def distances_to_means(self, point): ##calculate points distance to means
		distances = np.empty(self.n)
		for i in range(self.n):
			distances[i] = np.linalg.norm(self.cluster_means[i] - point)
		return distances
	
	def actualise_labels(self):
		self.labels = np.empty(len(self.mX), dtype=int)
		for i in range(len(self.mX)):
			distances = self.distances_to_means(self.mX[i])
			self.labels[i] = np.argmin(distances)
			
	def actualise_means(self):
		for i in range (self.n):
			if(len(self.mX[self.labels==i])):
				self.cluster_means[i]= np.mean(self.mX[self.labels==i],axis=0)
		
"""
km=MyKMeans(n_clusters = 3)

X, y = make_blobs(
   n_samples=150, n_features=2,
   centers=3, cluster_std=0.5,
   shuffle=True, random_state=0
)

  
km.fit_predict(X)

plt.scatter(
    X[km.labels == 0, 0], X[km.labels == 0, 1],
    s=50, c='lightgreen',
    marker='s', edgecolor='black',
    label='cluster 1'
)

plt.scatter(
    X[km.labels == 1, 0], X[km.labels == 1, 1],
    s=50, c='orange',
    marker='o', edgecolor='black',
    label='cluster 2'
)

plt.scatter(
    X[km.labels == 2, 0], X[km.labels == 2, 1],
    s=50, c='lightblue',
    marker='v', edgecolor='black',
    label='cluster 3'
)


# plot the centroids
plt.scatter(
    km.cluster_means[:, 0], km.cluster_means[:, 1],
    s=250, marker='+',
    c='red',
    label='centroids'
)
plt.legend(scatterpoints=1)
plt.grid()
plt.show()
"""
