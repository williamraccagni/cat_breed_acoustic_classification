import random
import sklearn
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def elbow_method(title : str, filename : str, samples_dataset : np.array, max_k : int, seed : int):

    sse = []
    K = range(1, (max_k + 1))
    for k in K:
        kmeanModel = sklearn.cluster.KMeans(n_clusters=k, random_state=seed)
        kmeanModel.fit(samples_dataset)
        sse.append(kmeanModel.inertia_)

    # plt.figure(figsize=(16, 8))
    plt.plot(K, sse, 'bx-')
    plt.xticks(K)
    plt.xlabel('k')
    plt.ylabel('Sum of Squared Errors')
    plt.title(title + ': Elbow Method')
    # plt.show()
    plt.savefig('./k_means/' + filename + '_elbow_method.png')
    plt.clf() # clear plot for next method



def silhoutte_method(title : str, filename : str, samples_dataset : np.array, max_k : int, seed : int) -> int:

    silhouette_coefficients = []
    K = range(2, (max_k + 1))
    for k in K:
        kmeans = sklearn.cluster.KMeans(n_clusters=k, random_state=seed)
        kmeans.fit(samples_dataset)
        score = sklearn.metrics.silhouette_score(samples_dataset, kmeans.labels_)
        silhouette_coefficients.append(score)
        print("For n_clusters={}, The Silhouette Coefficient is {}".format(k, score))

    plt.plot(K, silhouette_coefficients)
    plt.xticks(K)
    plt.xlabel('k')
    plt.ylabel('Silhouette Coefficient')
    plt.title(title + ': Silhouette Method')
    # plt.show()
    plt.savefig('./k_means/' + filename + '_silhouette_method.png')
    plt.clf()  # clear plot for next method

    max_value = max(silhouette_coefficients)
    best_k = silhouette_coefficients.index(max_value) + 2
    print('Silhouette Method best k is: ' + str(best_k))

    return best_k

def calculate_k_means(title : str, filename : str, samples_dataset : np.array, k_value: int, seed : int):

    kmeans = sklearn.cluster.KMeans(n_clusters=k_value, random_state=seed, n_init=30, max_iter=500)
    labels = kmeans.fit_predict(samples_dataset)
    u_labels = np.unique(labels) # make the labels unique

    # PCA for dimensionality reduction
    pca = PCA(2)
    # prepare data in 2 dimensions
    df = pca.fit_transform(samples_dataset)
    print(df.shape)

    # random colors for legend
    colors = []
    for x in range(1, (k_value + 1)):
        c = "#{:06x}".format(random.randint(0, 0xFFFFFF))
        colors.append(c)

    # plot the results
    for i in u_labels:
        # plt.scatter(samples_dataset[labels == i, 0], samples_dataset[labels == i, 1], label=i, c=colors[i])
        plt.scatter(df[labels == i, 0], df[labels == i, 1], label=i, c=colors[i])
    plt.legend()
    plt.title(title + ': K-Means Clusters')
    plt.savefig('./k_means/' + filename + '_k_means.png')
    plt.clf()  # clear plot for next method

def two_k_prediction(path : str, title : str, filename : str, X, y, labels_map : list, seed : int):

    kmeans = sklearn.cluster.KMeans(n_clusters=2, random_state=seed, n_init=30, max_iter=500)

    # clear

    y_pred = kmeans.fit_predict(X)

    print('K-Means Model, scores:')

    print('Confusion Matrix:')
    print(sklearn.metrics.confusion_matrix(y_true=y, y_pred=y_pred, labels=np.unique(y)))
    print("Accuracy:", sklearn.metrics.accuracy_score(y_true=y, y_pred=y_pred))
    print('Classification results:')
    print(sklearn.metrics.classification_report(y_true=y, y_pred=y_pred))

    # Confusion Matrix Plot
    matrix = sklearn.metrics.confusion_matrix(y_true=y, y_pred=y_pred, normalize='true')
    print(matrix)

    sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels_map).plot()

    plt.title(title + ':\nK-Means Classifier Confusion Matrix')
    plt.savefig(path + '/' + filename + '_kmeans_confusion_matrix.png')
    plt.clf()  # clear plot for next method





    # inverted

    y_pred = [int(1 - x) for x in y_pred]

    print('K-Means Model, scores (inverted predictions):')

    print('Confusion Matrix:')
    print(sklearn.metrics.confusion_matrix(y_true=y, y_pred=y_pred, labels=np.unique(y)))
    print("Accuracy:", sklearn.metrics.accuracy_score(y_true=y, y_pred=y_pred))
    print('Classification results:')
    print(sklearn.metrics.classification_report(y_true=y, y_pred=y_pred))

    # Confusion Matrix Plot
    matrix = sklearn.metrics.confusion_matrix(y_true=y, y_pred=y_pred, normalize='true')
    print(matrix)

    sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels_map).plot()

    plt.title(title + ':\nK-Means Classifier Confusion Matrix')
    plt.savefig(path + '/' + filename + '_kmeans_confusion_matrix_in.png')
    plt.clf()  # clear plot for next method

