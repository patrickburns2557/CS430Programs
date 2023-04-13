import scipy.io
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

#Load the X table from the .mat file
mat = scipy.io.loadmat("K-means-data.mat")
data = mat["X"]



#Run KMeans based on the passed in initType and return the number of iterations performed
def KMeansIterations(initType):
    kmeans = KMeans(n_clusters=3, n_init="auto", init=initType).fit(data)
    return kmeans.n_iter_


#Run KMeans based on the pased in initType and graph the clusters
def KMeansAndPlot(initType):
    plt.figure("K-Means using " + str(initType) + " initialization")
    
    #Run kmeans on the data and use 3 clusters
    kmeans = KMeans(n_clusters=3, n_init="auto", init=initType).fit(data)

    #Create numpy arrays of width 2 to store each datapoint in depending on it's cluster
    label0 = np.zeros(shape=(1,2))
    label1 = np.zeros(shape=(1,2))
    label2 = np.zeros(shape=(1,2))
    
    #Split up the datapoints based on their cluster assigned from KMeaans
    for i in range(len(kmeans.labels_)):
        if kmeans.labels_[i] == 0:
            label0 = np.vstack([label0, data[i]])
        elif kmeans.labels_[i] == 1:
            label1 = np.vstack([label1, data[i]])
        elif kmeans.labels_[i] == 2:
            label2 = np.vstack([label2, data[i]])
    
    #Must remove the 0 row from the beginning of the numpy array
    label0 = np.delete(label0, (0), axis=0)
    label1 = np.delete(label1, (0), axis=0)
    label2 = np.delete(label2, (0), axis=0)

    #Print the centroid locations using the specified initialization type
    print("Centroid locations using " + str(initType) + " initialization:")
    for centroid in kmeans.cluster_centers_:
        print("(" + str(centroid[0]) + ", " + str(centroid[1]) + ")")
    print()

    #Plot
    plt.scatter(label0[:, 0], label0[:, 1], marker=".", linewidths=0.5, color="green", label="cluster 1")
    plt.scatter(label1[:, 0], label1[:, 1], marker=".", linewidths=0.5, color="blue", label="cluster 2")
    plt.scatter(label2[:, 0], label2[:, 1], marker=".", linewidths=0.5, color="red", label="cluster 3")
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker="x", linewidths=3, color="black", label="centroids")
    plt.title("K-Means Clustering Using " + str(initType) + " Initialization\nNumber of iterations: " + str(kmeans.n_iter_))
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.legend(loc='best')




#Calculate K-means using both methods of initialization and open window plots for each one
#Note, the colors of each cluster may not match between the windowsbecause a seed was not specified for Scikit-learn's K-Means algorithm 
KMeansAndPlot("random")
KMeansAndPlot("k-means++")


#Run K-Means 20 times each for random initialization and k-means++ initialization
# and record the number of iterations each run took 
randomIters = []
kmeansplusplus = []
for i in range(20):
    randomIters.append(KMeansIterations("random"))
    kmeansplusplus.append(KMeansIterations("k-means++"))
randomAvg = sum(randomIters) / len(randomIters)
kmeansplusplusAvg = sum(kmeansplusplus) / len(kmeansplusplus)

print("Average iterations from 20 runs using random initialization: " + str(randomAvg))
print("Average iterations from 20 runs using k-means++ initialization: " + str(kmeansplusplusAvg))

if kmeansplusplusAvg < randomAvg:
    print("K-means++ initialization of centroids needed less iterations than random initialization.")
else:
    print("Random initialization of centroids needed less iterations than k-means++ initialization. (Extremely rare)")

plt.show()