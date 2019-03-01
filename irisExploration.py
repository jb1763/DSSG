#Common Libraries
import matplotlib.pyplot as plt
from sklearn import datasets
import pandas as pd

#load sepal data

iris = datasets.load_iris()
sepalx = iris.data[:, :2]
y = iris.target

xmin, xmax = sepalx[:,0].min() - .5, sepalx[:,0].max() + .5
ymin, ymax = sepalx[:,1].min() - .5, sepalx[:,1].max() + .5

plt.figure(2, figsize=(8,6))
plt.clf()

#Plot the training points
plt.scatter(sepalx[:,0], sepalx[:,1], c=y, cmap='magma', edgecolor='k')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.xlim(xmin,xmax)
plt.ylim(ymin,ymax)

#load sepal data
iris = datasets.load_iris()
petalx = iris.data[:, 2:4]
y = iris.target

xmin, xmax = petalx[:,0].min() - .5, petalx[:,0].max() + .5
ymin, ymax = petalx[:,1].min() - .5, petalx[:,1].max() + .5

plt.figure(2, figsize=(8,6))
plt.clf()

#Plot the training points
plt.scatter(petalx[:,0], petalx[:,1], c=y, cmap='magma', edgecolor='k')
plt.xlabel('Petal length')
plt.ylabel('Petal width')

plt.xlim(xmin,xmax)
plt.ylim(ymin,ymax)
