import json
import random
import sklearn
import sklearn.cluster
import numpy as np
import matplotlib.pyplot as plt

import k_means

def elbow_method(samples_dataset : np.array, max_k : int, seed : int):

    sse = []
    K = range(1, (max_k + 1))
    for k in K:
        kmeanModel = sklearn.cluster.KMeans(n_clusters=k, random_state=seed)
        kmeanModel.fit(samples_dataset)
        sse.append(kmeanModel.inertia_)

    return sse

def silhoutte_method(samples_dataset : np.array, max_k : int, seed : int):

    silhouette_coefficients = []
    K = range(2, (max_k + 1))
    for k in K:
        kmeans = sklearn.cluster.KMeans(n_clusters=k, random_state=seed)
        kmeans.fit(samples_dataset)
        score = sklearn.metrics.silhouette_score(samples_dataset, kmeans.labels_)
        silhouette_coefficients.append(score)

    return silhouette_coefficients

if __name__ == '__main__':

    # ----Variables----
    dataset_path = "./cats_dataset"
    seed = 0

    #ELBOW VISUALIZATION

    # - Time Domain

    # Opening JSON file
    with open('./features/normalized_time_domain_features.json') as json_file:
        tdf_dataset = json.load(json_file)

    np_tdf = np.array([tdf_dataset[str(index)] for index in range(len(tdf_dataset.keys()))])

    td_values = elbow_method(np_tdf, 14, seed)


    # Frequency Domain

    # Opening JSON file
    with open('./features/normalized_frequency_domain_features.json') as json_file:
        fdf_dataset = json.load(json_file)

    np_fdf = np.array([fdf_dataset[str(index)] for index in range(len(fdf_dataset.keys()))])

    fd_values = elbow_method(np_fdf, 14, seed)

    # Time and frequency

    # Opening JSON file
    with open('./features/normalized_timeplusfrequency_domain_features.json') as json_file:
        tpfdf_dataset = json.load(json_file)

    np_tpfdf = np.array([tpfdf_dataset[str(index)] for index in range(len(tpfdf_dataset.keys()))])

    tfd_values = elbow_method(np_tpfdf, 14, seed)



    # VISUALIZATION ELBOW

    max_k = 14
    K = range(1, (max_k + 1))

    # time plot
    plt.plot(K, td_values, label = "Time Domain")
    # frequency plot
    plt.plot(K, fd_values, label="Frequency Domain")
    # time/frequency plot
    plt.plot(K, tfd_values, label="Time and Frequency Domain")

    plt.legend()

    plt.xticks(K)
    plt.xlabel('k')
    plt.ylabel('Sum of Squared Errors')
    plt.title('Elbow Method')
    plt.show()
    # plt.savefig('./k_means/' + filename + '_elbow_method.png')
    plt.clf()  # clear plot for next method






    # silhouette method

    # time Domain

    # Opening JSON file
    with open('./features/normalized_time_domain_features.json') as json_file:
        tdf_dataset = json.load(json_file)

    np_tdf = np.array([tdf_dataset[str(index)] for index in range(len(tdf_dataset.keys()))])

    sil_td_co = silhoutte_method(np_tdf, 14, seed)

    # frequency Domain

    # Opening JSON file
    with open('./features/normalized_frequency_domain_features.json') as json_file:
        fdf_dataset = json.load(json_file)

    np_fdf = np.array([fdf_dataset[str(index)] for index in range(len(fdf_dataset.keys()))])

    sil_fd_co = silhoutte_method(np_fdf, 14, seed)


    # time and frequency domain

    # Opening JSON file
    with open('./features/normalized_timeplusfrequency_domain_features.json') as json_file:
        tpfdf_dataset = json.load(json_file)

    np_tpfdf = np.array([tpfdf_dataset[str(index)] for index in range(len(tpfdf_dataset.keys()))])

    sil_tfd_co = silhoutte_method(np_tpfdf, 14, seed)


    #silhoutte_visualization

    max_k = 14
    K = range(2, (max_k + 1))

    # time plot
    plt.plot(K, sil_td_co, label="Time Domain")
    # frequency plot
    plt.plot(K, sil_fd_co, label="Frequency Domain")
    # time/frequency plot
    plt.plot(K, sil_tfd_co, label="Time and Frequency Domain")

    plt.legend()


    plt.xticks(K)
    plt.xlabel('k')
    plt.ylabel('Silhouette Coefficient')
    plt.title('Silhouette Method')
    plt.show()
    # plt.savefig('./k_means/' + filename + '_silhouette_method.png')
    plt.clf()  # clear plot for next method