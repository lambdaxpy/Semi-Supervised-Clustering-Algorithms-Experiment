import numpy as np
import random
import time


class COPKMeans:
	def __init__(self, data_points: np.ndarray, ml: np.ndarray[tuple[int]], cl: np.ndarray[tuple[int]],
				 n_clusters: int=2):
		self._data_points = np.array([np.append(data_point, -1) for data_point in data_points])
		self._ml = ml
		self._cl = cl
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

	def _assign_linked_ml_to_cluster(self, index: int, cluster_number: int):
		for ml_tuple in self._ml:
			if ml_tuple[0] == index:
				self._data_points[ml_tuple[1]][self._dimension - 1] = cluster_number
			elif ml_tuple[1] == index:
				self._data_points[ml_tuple[0]][self._dimension - 1] = cluster_number

	def _set_centroids(self, _round: int) -> None:
		# The cluster numbers are from 0 to n_clusters - 1 (both inclusive).
		for i in range(self._n_clusters):
			_sum = np.zeros((self._dimension - 1, ))
			count = 0
			for data in self._data_points:
				if data[self._dimension - 1] == i:
					_sum += np.array(data[:self._dimension - 1])
					count += 1
			_mean = _sum
			if count > 0:
				_mean = _sum *  (1 / count)
			self._centroids[i] = np.around(_mean, decimals=_round)

	def _violate_constraints(self, data_point: np.ndarray, cluster_number: int) -> bool:
		for ml_tuple in self._ml:
			if np.equal(self._data_points[ml_tuple[0]], data_point).all():
				ml_cluster_number = self._data_points[ml_tuple[1]][self._dimension - 1]
				if ml_cluster_number != cluster_number and ml_cluster_number != -1:
					# print("YES")
					return True
			if np.equal(self._data_points[ml_tuple[1]], data_point).all():
				ml_cluster_number = self._data_points[ml_tuple[0]][self._dimension - 1]
				if ml_cluster_number != cluster_number and ml_cluster_number != -1:
					# print("YES")
					return True

		for cl_tuple in self._cl:
			if np.equal(self._data_points[cl_tuple[0]], data_point).all():
				cl_cluster_number = self._data_points[cl_tuple[1]][self._dimension - 1]
				if cl_cluster_number == cluster_number:
					# print("YES")
					return True
			if np.equal(self._data_points[cl_tuple[1]], data_point).all():
				cl_cluster_number = self._data_points[cl_tuple[0]][self._dimension - 1]
				if cl_cluster_number == cluster_number:
					# print("YES")
					return True
		return False

	def _assign_clusters(self) -> bool:
		for i, data in enumerate(self._data_points):
			clusters_violating = []
			distances = [np.linalg.norm(np.subtract(data[:self._dimension - 1], centroid)) for centroid in self._centroids]
			min_distance = min(distances)
			min_distance_cluster = distances.index(min_distance)
			# print("Data point:", data)
			# print("Min Distance Cluster:", min_distance_cluster)
			# print("Centroids:", self._centroids)
			while self._violate_constraints(data, min_distance_cluster):
				clusters_violating.append(min_distance_cluster)
				if len(clusters_violating) == len(distances):
					print("Not worked for:", data, "with index:", i)
					return False
				min_distance = min(list(filter(lambda x: distances.index(x) not in clusters_violating, distances)))
				min_distance_cluster = distances.index(min_distance)
				# print("Min Distance Cluster:", min_distance_cluster)
			self._data_points[i][self._dimension - 1] = min_distance_cluster
		return True

	def cluster_data(self, _round: int=2):
		convergence = False
		while not convergence:
			old_centroids = self._centroids.copy()
			if not self._assign_clusters():
				print("Assignment not worked.")
				break
			self._set_centroids(_round)
			# print(self._centroids)
			# print()
			if np.equal(self._centroids, old_centroids).all():
				convergence = True

			# print(self._data_points)
			# print()

		print(self._data_points)

if __name__ == "__main__":
	sample_data = np.array([[1, 1], [1, 2], [3, 1], [3, 0.5], [4, 5],
							[4, 2], [4, 3], [4, 4], [2.5, 2.5]])

	ml = np.array([(0, 1)])
	cl = np.array([(0, 8), (1, 8)])
	# sample_data = np.random.random((10000, 20))
	start_time = time.time()
	kmeans = COPKMeans(sample_data, ml, cl, n_clusters=2)
	kmeans.cluster_data()
	print(f"{time.time() - start_time} s")