import pandas as pd
import numpy as np
from kmeans import KMeans
from seeded_kmeans import SeededKMeans
import time
import statistics
import matplotlib.pyplot as plt


COLUMNS = ["Sex", "Marital status", "Age", "Income", "Occupation", "Settlement size"]

df = pd.read_csv("segmentation data.csv")
df = df[COLUMNS]
data_points = df.to_numpy()

kmeans_times = []
seeded_kmeans_times = []
median_kmeans_times = []
median_seeded_kmeans_times = []
seeded_kmeans_label_num = []

kmeans_iterations = []
seeded_kmeans_iterations = []
median_kmeans_iterations = []
median_seeded_kmeans_iterations = []


def plot_kmeans_clustering(data_points):
	colors = ["red", "green", "blue", "yellow", "orange"]
	for data_point in data_points:
		plt.scatter(data_point[0], data_point[3], color=colors[data_point[-1]])
	plt.show()


def save_kmeans_clustering(data_points):
	np.savetxt("labeled_customer.csv", data_points, delimiter=",", fmt='%f')
	print("Save finished.")


for i in range(100):
	for j in range(10):
		start_time = time.time()
		kmeans = KMeans(data_points, n_clusters=5)
		kmeans.cluster_data(_round=2)
		end_time = time.time() - start_time
		# print(f"{j}: Time passed (KMeans): {end_time}")
		print(f"{j}: Iteration Amount (KMeans): {kmeans._iteration}")
		kmeans_times.append(end_time)
		kmeans_iterations.append(kmeans._iteration)

		# save_kmeans_clustering(kmeans._data_points)

		labeled_data = np.array(kmeans._data_points[:j])
		unlabeled_data = np.array(np.delete(kmeans._data_points, np.s_[-1:], axis=1)[j:])

		start_time = time.time()
		seeded_kmeans = SeededKMeans(labeled_data, unlabeled_data, n_clusters=5, _round=2)
		seeded_kmeans.cluster_data(_round=2)
		end_time = time.time() - start_time
		seeded_kmeans_times.append(end_time)
		seeded_kmeans_iterations.append(seeded_kmeans._iteration)
		# print(seeded_kmeans._unlabeled_data)
		print(f"{j}: Iteration Amount (Seeded-KMeans): {seeded_kmeans._iteration}")
		# print(f"{j}: Time passed (Seeded-KMeans): {time.time() - start_time}")

	# median_kmeans_time = statistics.median(kmeans_times)
	# median_seeded_kmeans_time = statistics.median(seeded_kmeans_times)

	# median_kmeans_times.append(median_kmeans_time)
	# median_seeded_kmeans_times.append(median_seeded_kmeans_time)

	median_kmeans_iteration = statistics.median(kmeans_iterations)
	median_seeded_kmeans_iteration = statistics.median(seeded_kmeans_iterations)

	median_kmeans_iterations.append(median_kmeans_iteration)
	median_seeded_kmeans_iterations.append(median_seeded_kmeans_iteration)

	# print(f"{i}: Median time (KMeans): {median_kmeans_time}")
	# print(f"{i}: Median time (Seeded-KMeans): {median_seeded_kmeans_time}")

	print(f"{i}: Median iterations (KMeans): {median_kmeans_iteration}")
	print(f"{i}: Median iterations (Seeded-KMeans): {median_seeded_kmeans_iteration}")

plt.plot(range(100), median_seeded_kmeans_iterations, label="Seeded-KMeans")
plt.plot(range(100), median_kmeans_iterations, label="KMeans")

plt.xlabel("Number of Labels")
plt.ylabel("Number of Iterations")

plt.show()