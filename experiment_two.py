import pandas as pd
import numpy as np
from kmeans import KMeans
from cop_kmeans import COPKMeans
import time
import statistics
import matplotlib.pyplot as plt


COLUMNS = ["Sex", "Marital status", "Age", "Income", "Occupation", "Settlement size"]
COLUMNS_LABEL = ["Sex", "Marital status", "Age", "Income", "Occupation", "Settlement size", "Label"]

df = pd.read_csv("segmentation data.csv")
df = df[COLUMNS]
data_points = df.to_numpy()

labeled_df = pd.read_csv("labeled_customer.csv")
labeled_df = labeled_df[COLUMNS_LABEL]
labeled_data_points = df.to_numpy()


def calculate_rand_index(clustered_points):
    n = len(clustered_points)
    p = 0
    for i in range(n):
        for j in range(i + 1, n):
            data_point_one = labeled_data_points[i]
            data_point_two = labeled_data_points[j]
            clustered_point_one = clustered_points[i]
            clustered_point_two = clustered_points[j]

            if clustered_point_one[-1] == clustered_point_two[-1] and \
                    data_point_one[-1] == data_point_two[-1]:
                p += 1

            if clustered_point_one[-1] != clustered_point_two[-1] and \
                    data_point_one[-1] != data_point_two[-1]:
                p += 1

    rand_index = 2 * p / (n * (n - 1))
    return rand_index


def get_constraints():
    ml = []
    cl = []
    for i in range(100):
        if labeled_data_points[i][-1] == labeled_data_points[i + 100][-1]:
            ml.append((i, i + 100))
        else:
            cl.append((i, i + 100))

    # transitive_closure(ml, cl, len(labeled_data_points))
    return ml, cl


kmeans = KMeans(data_points, n_clusters=5)
kmeans.cluster_data(_round=2)
print("Accuracy (KMeans):", calculate_rand_index(kmeans._data_points))

ml, cl = get_constraints()
print(ml)
print(cl)
cop_kmeans = COPKMeans(data_points, np.array(ml), np.array(cl), n_clusters=5)
cop_kmeans.cluster_data(_round=2)
print("Accuracy (COP-KMeans):", calculate_rand_index(cop_kmeans._data_points))

