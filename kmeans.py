import numpy as np
import random
import time


class KMeans:
	def __init__(self, data_points: np.ndarray, n_clusters: int=2) -> None:
		self._data_points = np.array([np.append(data_point, 0) for data_point in data_points])
		self._n_clusters = n_clusters
		self._dimension = np.shape(self._data_points[0])[0]
		self._centroids = np.zeros((n_clusters, self._dimension - 1))

		# Set initial centroids.
		self._set_initial_seeds()

	def _set_initial_seeds(self) -> None:
		# The cluster numbers are from 0 to n_clusters - 1 (both inclusive).
		for i in range(self._n_clusters):
			random_index = random.randint(0, len(self._data_points) - 1)
			while any(np.equal(self._data_points[random_index][:self._dimension - 1], centroid).all() for centroid in self._centroids):
				random_index = random.randint(0, len(self._data_points) - 1)
			self._centroids[i] = np.array(self._data_points[random_index][:self._dimension - 1])

	def _set_centroids(self, _round: int) -> None:
		# The cluster numbers are from 0 to n_clusters - 1 (both inclusive).
		for i in range(self._n_clusters):
			_sum = np.zeros((self._dimension - 1, ))
			count = 0
			for data in self._data_points:
				if data[self._dimension - 1] == i:
					_sum += np.array(data[:self._dimension - 1])
					count += 1
			_mean = _sum *  (1 / count)
			self._centroids[i] = np.around(_mean, decimals=_round)

	def _assign_clusters(self) -> None:
		for i, data in enumerate(self._data_points):
			distances = [np.linalg.norm(np.subtract(data[:self._dimension - 1], centroid)) for centroid in self._centroids]
			min_distance = min(distances)
			min_distance_cluster = distances.index(min_distance)
			self._data_points[i][self._dimension - 1] = min_distance_cluster

	def cluster_data(self, _round: int=2):
		convergence = False
		while not convergence:
			old_centroids = self._centroids.copy()
			self._assign_clusters()
			self._set_centroids(_round)
			if np.equal(self._centroids, old_centroids).all():
				convergence = True


if __name__ == "__main__":
	sample_data = np.array([[1, 1], [1, 2], [3, 1], [3, 0.5], [4, 5],
							[4, 2], [4, 3], [4, 4], [2.5, 2.5]])
	# sample_data = np.random.random((10000, 20))
	start_time = time.time()
	kmeans = KMeans(sample_data, n_clusters=2)
	kmeans.cluster_data()
	print(f"{time.time() - start_time} s")
