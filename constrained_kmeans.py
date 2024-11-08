import numpy as np
import random
import time

class ConstrainedKMeans:
	def __init__(self, labeled_data: np.ndarray, unlabeled_data: np.ndarray, n_clusters: int=2,
				 _round: int=2) -> None:
		self._labeled_data = labeled_data
		self._unlabeled_data = np.array([np.append(data_point, 0) for data_point in unlabeled_data])
		self._n_clusters = n_clusters
		self._dimension = np.shape(self._unlabeled_data[0])[0]
		self._centroids = np.zeros((n_clusters, self._dimension - 1))

		# Set initial seeds.
		self._set_initial_seeds(_round=2)

	def _set_initial_seeds(self, _round) -> None:
		# The cluster numbers are from 0 to n_clusters - 1 (both inclusive).
		for i in range(self._n_clusters):
			mask = np.in1d(self._labeled_data[:, 1], np.asarray([i]))
			labeled_cluster_data = self._labeled_data[mask]
			if labeled_cluster_data.size > 0:
				_sum = np.zeros((self._dimension - 1,))
				for label in labeled_cluster_data:
					_sum += np.array(label[:self._dimension - 1])
				_mean = _sum *  (1 / labeled_cluster_data.size)
				self._centroids[i] = np.around(_mean, decimals=_round)
			else:
				random_index = random.randint(0, len(self._unlabeled_data) - 1)
				while any(np.equal(self._unlabeled_data[random_index][:self._dimension - 1], centroid).all() for centroid in
						self._centroids):
					random_index = random.randint(0, len(self._unlabeled_data) - 1)
				self._centroids[i] = np.array(self._unlabeled_data[random_index][:self._dimension - 1])

	def _set_centroids(self, _round: int) -> None:
		# The cluster numbers are from 0 to n_clusters - 1 (both inclusive).
		for i in range(self._n_clusters):
			_sum = np.zeros((self._dimension - 1, ))
			count = 0
			for data in np.concatenate((self._unlabeled_data, self._labeled_data)):
				if data[self._dimension - 1] == i:
					_sum += np.array(data[:self._dimension - 1])
					count += 1
			_mean = _sum *  (1 / count)
			self._centroids[i] = np.around(_mean, decimals=_round)

	def _assign_clusters(self) -> None:
		for i, data in enumerate(self._unlabeled_data):
			distances = [np.linalg.norm(np.subtract(data[:self._dimension - 1], centroid)) for centroid in
						 self._centroids]
			min_distance = min(distances)
			min_distance_cluster = distances.index(min_distance)
			self._unlabeled_data[i][self._dimension - 1] = min_distance_cluster

	def cluster_data(self, _round: int=2):
		convergence = False
		while not convergence:
			old_centroids = self._centroids.copy()
			self._assign_clusters()
			self._set_centroids(_round)
			print(self._centroids)
			print()
			if np.equal(self._centroids, old_centroids).all():
				convergence = True

			print("Labeled Data")
			print(self._labeled_data)
			print("Unlabeled Data")
			print(self._unlabeled_data)
			print()


if __name__ == "__main__":
	labeled_data = np.array([[1, 1, 0], [1, 2, 0], [4, 5, 1], [4, 4, 1]])
	unlabeled_data = np.array([[3, 1], [3, 0.5], [4, 2], [4, 3], [2.5, 2.5]])
	# sample_data = np.random.random((10000, 20))
	start_time = time.time()
	kmeans = ConstrainedKMeans(labeled_data, unlabeled_data, n_clusters=2)
	kmeans.cluster_data()
	print(f"{time.time() - start_time} s")