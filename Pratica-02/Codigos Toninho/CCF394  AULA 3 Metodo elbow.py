from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt


# Carrega o conjunto de dados iris
iris = load_iris()
X = iris.data

# Cria um objeto k-means com inicialização k-means++
# calculate distortion for a range of number of cluster
distortions = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++')
    kmeans.fit(X)
    distortions.append(kmeans.inertia_)

# plot
plt.plot(range(1, 11), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()



#labels= kmeans.labels_
# Treina o modelo com o conjunto de dados
kmeans.fit(X)

# Imprime os centros dos clusters encontrados
print(kmeans.cluster_centers_)
#print(labels)
