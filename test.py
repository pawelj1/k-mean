import unittest
from MyKMeans import MyKMeans
from sklearn.cluster import KMeans
import numpy as np
from sklearn.datasets import make_blobs


import unittest

class TestStringMethods(unittest.TestCase):

	def test_MyKMeans(self):
			
				rng = np.random.RandomState(0)
				X, y = make_blobs(
					n_samples=150, n_features=2,
					centers=3, cluster_std=0.5,
					shuffle=True, random_state=rng
					)
				km1=MyKMeans(n_clusters = 3, n_init=50)
				km2=KMeans(n_clusters = 3, n_init=50)
				km1.fit_predict(X)
				km2.fit_predict(X)
				self.assertEqual(km1.labels.all(), km2.labels_.all(),msg=("Clustering effects are not equal result of sklearn.cluster.KMeans"))
			
			

if __name__ == '__main__':
    unittest.main()
