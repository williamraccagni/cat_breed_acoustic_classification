import csv
from os import walk
import pandas as pd
import numpy as np
import json
import sklearn
from sklearn.preprocessing import StandardScaler

import utils
import time_domain_features_extraction as tdfe
import frequency_domain_features_extraction as fdfe
import k_means
import knn_model
import mlp_model

if __name__ == '__main__':

    # ----Variables----
    dataset_path = "./cats_dataset"
    seed = 0

    to_do_steps = [4] # <---------- INSERT THE STEPS TO DO HERE
    # -----------------


    # ---- Paths and Labels Extraction ----
    if 1 in to_do_steps:
        # ----Files list from Dataset Folder
        _, _, dataset_files = next(walk(dataset_path))
        dataset_files.sort()
        print('Number of Files from Dataset: ' + str(len(dataset_files)))

        # ----Get labels from Files list
        # [filepath, cat_id, emission context, breed, sex]
        names_and_labels = []
        for x in dataset_files:
            splitted_string = x.split('_')
            names_and_labels.append([ x, splitted_string[1], splitted_string[0], splitted_string[2], splitted_string[3] ])
        pd.DataFrame(names_and_labels).to_csv('./features/names_and_labels.csv', header=False) # index=False, header=False)
        print('names_and_labels.csv saved!')



    # ---- TIME DOMAIN ----



    # ---- Time Domain Features Extraction ----
    if 2 in to_do_steps:

        # Read filenames
        with open('./features/names_and_labels.csv', newline='') as f:
            reader = csv.reader(f)
            fn_dataset = list(reader)

        # remove useless information
        for i in range(len(fn_dataset)):
            fn_dataset[i] = [ int(fn_dataset[i][0]), fn_dataset[i][1] ] # [id, path]

        tdf_dataset = tdfe.time_domain_dataset(dataset_path, fn_dataset, frame_length=1024, hop_length=512)



        # time domain features json dump

        # listing for json
        for index in tdf_dataset.keys():
            tdf_dataset[index] = tdf_dataset[index].tolist()

        print('Time Domain Features:')

        for index in tdf_dataset.keys():
            print(str(index) + ': ' + str(tdf_dataset[index]) )

        with open('./features/time_domain_features.json', 'w') as json_file:
            json.dump(tdf_dataset, json_file)

        print('time_domain_features.json saved!')


    # ---- Time Domain Features Normalization ----
    if 3 in to_do_steps:

        # Opening JSON file
        with open('./features/time_domain_features.json') as json_file:
            tdf_dataset = json.load(json_file)

        # standardize data
        standardscaler = sklearn.preprocessing.StandardScaler()

        X = [tdf_dataset[str(index)] for index in range(len(tdf_dataset))]
        normalized_X = standardscaler.fit_transform(X)

        # making normalized dictionary
        normalized_tdf_dataset = {}

        for index in range(len(X)):
            print('Sample ' + str(index) + ':')
            print(X[index])

            print('Sample ' + str(index) + ' normalized:')
            print(normalized_X[index].tolist())
            normalized_tdf_dataset[index] = normalized_X[index].tolist()

        with open('./features/normalized_time_domain_features.json', 'w') as json_file:
            json.dump(normalized_tdf_dataset, json_file)

        print('normalized_time_domain_features.json saved!')


    # ---- Time Domain K-Means ----
    if 4 in to_do_steps:

        # Opening JSON file
        with open('./features/normalized_time_domain_features.json') as json_file:
            tdf_dataset = json.load(json_file)

        np_tdf = np.array([tdf_dataset[str(index)] for index in range(len(tdf_dataset.keys())) ])

        k_means.elbow_method( 'Time Domain' ,'time_domain', np_tdf, 14, seed)
        best_k = k_means.silhoutte_method('Time Domain', 'time_domain', np_tdf, 14, seed)
        k_means.calculate_k_means('Time Domain', 'time_domain', np_tdf, best_k, seed)



    # ---- Time Domain Cat Breed Prediction ----
    if 5 in to_do_steps:

        # Loading Xs
        with open('./features/normalized_time_domain_features.json') as json_file:
            tdf_dataset = json.load(json_file)

        X = [tdf_dataset[str(index)] for index in range(len(tdf_dataset.keys())) ]

        # loading ys
        with open('./features/names_and_labels.csv', newline='') as f:
            reader = csv.reader(f)
            fn_dataset = list(reader)

        # ->converting in numbers
        y = []
        for index in range(len(fn_dataset)):
            if fn_dataset[index][4] == 'MC':
                y.append(0)
            if fn_dataset[index][4] == 'EU':
                y.append(1)

        labels_map = ['MC', 'EU']

        # plot ys balance
        utils.plot_labels_balancing('results', 'Cat Breed', 'cat_breed',
                                    y, labels_map, [0,1])

        # K Means prediction
        k_means.two_k_prediction('results/time_domain/kmeans', 'Time Domain, Cat Breed', 'time_domain_cat_breed',
                              X, y, labels_map, seed=seed)

        # KNN
        knn_model.execute_knn('results/time_domain/knn', 'Time Domain, Cat Breed', 'time_domain_cat_breed',
                              X, y, labels_map, k_value=3, test_size=0.3, seed=seed)

        # MLP
        mlp_model.execute_mlp('results/time_domain/mlp', 'Time Domain, Cat Breed', 'time_domain_cat_breed',
                              X, y, model_type=0, batch_size=100, epochs=1500, labels_map=labels_map, test_size=0.3, seed=seed)



    # ---- FREQUENCY DOMAIN -----



    # ---- Frequency Domain Features Extraction ----
    if 6 in to_do_steps:

        # Read filenames
        with open('./features/names_and_labels.csv', newline='') as f:
            reader = csv.reader(f)
            fn_dataset = list(reader)

        # remove useless information
        for i in range(len(fn_dataset)):
            fn_dataset[i] = [int(fn_dataset[i][0]), fn_dataset[i][1]]  # [id, path]

        fdf_dataset = fdfe.frequency_domain_dataset(dataset_path, fn_dataset, fft_length=1024, hop_length=512)



        # frequency domain features json dump

        # listing for json
        for index in fdf_dataset.keys():
            fdf_dataset[index] = fdf_dataset[index].tolist()

        print('Frequency Domain Features:')

        for index in fdf_dataset.keys():
            print(str(index) + ': ' + str(fdf_dataset[index]))

        with open('./features/frequency_domain_features.json', 'w') as json_file:
            json.dump(fdf_dataset, json_file)

        print('frequency_domain_features.json saved!')



    # ---- Frequency Domain Features Normalization ----
    if 7 in to_do_steps:

        # Opening JSON file
        with open('./features/frequency_domain_features.json') as json_file:
            fdf_dataset = json.load(json_file)

        # standardize data
        standardscaler = sklearn.preprocessing.StandardScaler()

        X = [fdf_dataset[str(index)] for index in range(len(fdf_dataset))]
        normalized_X = standardscaler.fit_transform(X)

        # making normalized dictionary
        normalized_fdf_dataset = {}

        for index in range(len(X)):
            print('Sample ' + str(index) + ':')
            print(X[index])

            print('Sample ' + str(index) + ' normalized:')
            print(normalized_X[index].tolist())
            normalized_fdf_dataset[index] = normalized_X[index].tolist()

        with open('./features/normalized_frequency_domain_features.json', 'w') as json_file:
            json.dump(normalized_fdf_dataset, json_file)

        print('normalized_frequency_domain_features.json saved!')



    # ---- Frequency Domain K-Means ----
    if 8 in to_do_steps:

        # Opening JSON file
        with open('./features/normalized_frequency_domain_features.json') as json_file:
            fdf_dataset = json.load(json_file)

        np_fdf = np.array([fdf_dataset[str(index)] for index in range(len(fdf_dataset.keys())) ])

        k_means.elbow_method('Frequency Domain' ,'frequency_domain', np_fdf, 14, seed)
        best_k = k_means.silhoutte_method('Frequency Domain' ,'frequency_domain', np_fdf, 14, seed)
        k_means.calculate_k_means('Frequency Domain' ,'frequency_domain', np_fdf, best_k, seed)


    # ---- Frequency Domain Cat Breed Prediction ----
    if 9 in to_do_steps:

        # Loading Xs
        with open('./features/normalized_frequency_domain_features.json') as json_file:
            fdf_dataset = json.load(json_file)

        X = [fdf_dataset[str(index)] for index in range(len(fdf_dataset.keys())) ]

        # loading ys
        with open('./features/names_and_labels.csv', newline='') as f:
            reader = csv.reader(f)
            fn_dataset = list(reader)

        # ->converting in numbers
        y = []
        for index in range(len(fn_dataset)):
            if fn_dataset[index][4] == 'MC':
                y.append(0)
            if fn_dataset[index][4] == 'EU':
                y.append(1)

        labels_map = ['MC', 'EU']

        # K Means prediction
        k_means.two_k_prediction('results/frequency_domain/kmeans', 'Frequency Domain, Cat Breed', 'frequency_domain_cat_breed',
                              X, y, labels_map, seed=seed)

        # KNN
        knn_model.execute_knn('results/frequency_domain/knn', 'Frequency Domain, Cat Breed', 'frequency_domain_cat_breed',
                              X, y, labels_map, k_value=3, test_size=0.3, seed=seed)

        # MLP
        mlp_model.execute_mlp('results/frequency_domain/mlp', 'Frequency Domain, Cat Breed', 'frequency_domain_cat_breed',
                              X, y, model_type=0, batch_size=100, epochs=1500, labels_map=labels_map, test_size=0.3, seed=seed)



    # ---- TIME PLUS FREQUENCY DOMAIN ----



    # ---- Time plus Frequency Domain Features save----
    if 10 in to_do_steps:

        # Opening normalized time JSON file
        with open('./features/normalized_time_domain_features.json') as json_file:
            tdf_dataset = json.load(json_file)

        # Opening normalized frequency JSON file
        with open('./features/normalized_frequency_domain_features.json') as json_file:
            fdf_dataset = json.load(json_file)

        print(tdf_dataset['0'])
        print(fdf_dataset['0'])

        X = [ (tdf_dataset[str(index)] + fdf_dataset[str(index)]) for index in range(len(tdf_dataset))]

        print(X[0])
        print(len(X[0]))

        # making normalized dictionary
        tpfdf_dataset = {}

        for index in range(len(X)):
            tpfdf_dataset[index] = X[index]

        with open('./features/normalized_timeplusfrequency_domain_features.json', 'w') as json_file:
            json.dump(tpfdf_dataset, json_file)

        print('normalized_timeplusfrequrncy_domain_features.json saved!')



    # ---- Time plus Frequency Domain K-Means ----
    if 11 in to_do_steps:

        # Opening JSON file
        with open('./features/normalized_timeplusfrequency_domain_features.json') as json_file:
            tpfdf_dataset = json.load(json_file)

        np_tpfdf = np.array([tpfdf_dataset[str(index)] for index in range(len(tpfdf_dataset.keys())) ])

        k_means.elbow_method('Time and Frequency Domain' ,'timeplusfrequency_domain', np_tpfdf, 14, seed)
        best_k = k_means.silhoutte_method('Time and Frequency Domain' ,'timeplusfrequency_domain', np_tpfdf, 14, seed)
        k_means.calculate_k_means('Time and Frequency Domain' ,'timeplusfrequency_domain', np_tpfdf, best_k, seed)



    # ---- Time plus Frequency Domain Cat Breed Prediction ----
    if 12 in to_do_steps:

        # Loading Xs
        with open('./features/normalized_timeplusfrequency_domain_features.json') as json_file:
            tpfdf_dataset = json.load(json_file)

        X = [tpfdf_dataset[str(index)] for index in range(len(tpfdf_dataset.keys())) ]

        # loading ys
        with open('./features/names_and_labels.csv', newline='') as f:
            reader = csv.reader(f)
            fn_dataset = list(reader)

        # ->converting in numbers
        y = []
        for index in range(len(fn_dataset)):
            if fn_dataset[index][4] == 'MC':
                y.append(0)
            if fn_dataset[index][4] == 'EU':
                y.append(1)

        labels_map = ['MC', 'EU']

        # K Means prediction
        k_means.two_k_prediction('results/timeplusfrequency_domain/kmeans', 'Time and Frequency Domain, Cat Breed', 'timeplusfrequency_domain_cat_breed',
                              X, y, labels_map, seed=seed)

        # KNN
        knn_model.execute_knn('results/timeplusfrequency_domain/knn', 'Time and Frequency Domain, Cat Breed', 'timeplusfrequency_domain_cat_breed',
                              X, y, labels_map, k_value=3, test_size=0.3, seed=seed)

        # MLP
        mlp_model.execute_mlp('results/timeplusfrequency_domain/mlp', 'Time and Frequency Domain, Cat Breed', 'timeplusfrequency_domain_cat_breed',
                              X, y, model_type=0, batch_size=100, epochs=1500, labels_map=labels_map, test_size=0.3, seed=seed)
