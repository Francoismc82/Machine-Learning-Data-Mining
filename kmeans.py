# K-Means clustering implementation
from sklearn.cluster import KMeans
import random
import numpy as np
import csv
from math import sqrt
import matplotlib.pyplot as plt

import pandas as pd
import math
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
#from random import *

# Some hints on how to start have been added to this file.
# You will have to add more code than just the hints provided here for the full implementation.
# ====
# Define a function that computes the distance between two data points


def distance(pointA, pointB):
    distance = math.sqrt(
        math.pow((pointA[0] - pointB[0]), 2) +
        math.pow((pointA[0] - pointB[0]), 2)
    )
    return distance


print(distance([1, 2], [14, 18]))

# ====
# Define a function that reads data in from the csv files  HINT: http://docs.python.org/2/library/csv.html

dataset = pd.read_csv('data2008.csv')

x = dataset[['BirthRate(Per1000 - 2008)',
             'LifeExpectancy(2008)']].values.reshape(-1, 2)
k = 3


# ====
# Write the initialisation procedure

clusters = {
    0: [],
    1: [],
    2: []
}

predictions = []


def scatter_kmeans(x, k, r=123):
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=r)
    y_pred = kmeans.fit_predict(x)
    predictions = y_pred

    print(y_pred)
    colours = 'rbgcmy'
    for c in range(k):
        plt.scatter(x[y_pred == c, 0], x[y_pred == c, 1],
                    c=colours[c], label='Cluster{}'.format(c))
        plt.scatter(kmeans.cluster_centers_[c, 0], kmeans.cluster_centers_[
                    c, 1], marker='x', c='black')

    plt.xlabel('BirthRate')
    plt.ylabel('Life Expectancy')
    plt.legend()
    plt.show()

    countries = dataset.iloc[:, 0]
    countries = np.array(countries)
    print(countries)

    for i in range(len(predictions)):
        cluster_no = predictions[i]
        clusters[cluster_no].append(countries[i])


# Print out the results
    print(clusters)


scatter_kmeans(x, k)
