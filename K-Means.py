# Kevin Clark    
# K-Means Algorithm for CSE 494 Assignment 

from copy import deepcopy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Calculates Euclidean Distance
def euclidean_distance(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)

# Set Up Data Points
df = pd.DataFrame([
    [2,8], [3,3], [1,2], [5,8], [7,3], [6,4], [8,4], [4,7]],
    columns = ['x', 'y'])
x_coordinates = df['x'].values
y_coordinates = df['y'].values
coordinates = np.array(list(zip(x_coordinates, y_coordinates)))

#Set Up Centroids
centroid_df = pd.DataFrame([
    [2,8], [1,2], [6,4]],
    columns = ['x', 'y'])
centroid_x_coordinates = centroid_df['x'].values
centroid_y_coordinates = centroid_df['y'].values
centroid = np.array(list(zip(centroid_x_coordinates, centroid_y_coordinates)), dtype=np.float32)
previous_centroid = np.zeros(centroid.shape)
# Set K-Value (# of Clusters)
k =  3

# Initialize Clusters
clusters = np.zeros(len(coordinates))

# Calculates distance between old and new centroids
Error = euclidean_distance(centroid, previous_centroid, None)

# K-Means Calculation
while Error != 0:
    # Sets Points to the correct cluster
    for i in range(len(coordinates)):
        distances = euclidean_distance(coordinates[i], centroid)
        cluster = np.argmin(distances)
        clusters[i] = cluster
    previous_centroid = deepcopy(centroid)
    # Calculates new centroid
    for i in range(k):
        points = [coordinates[j] for j in range(len(coordinates)) if clusters[j] == i]
        centroid[i] = np.mean(points, axis=0)
    Error = euclidean_distance(centroid, previous_centroid, None)

# Displays Optimized K-Means
colors = ['R', 'G', 'B']
fig, ax = plt.subplots()
for i in range(k):
        points = np.array([coordinates[j] for j in range(len(coordinates)) if clusters[j] == i])
        ax.scatter(points[:, 0], points[:, 1], c=colors[i])
ax.scatter(centroid[:, 0], centroid[:, 1], c = 'black')

plt.show()