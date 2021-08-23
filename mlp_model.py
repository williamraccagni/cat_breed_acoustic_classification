import numpy as np
import sklearn
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import copy

def mlp_model(input_len : int):

    # Create Model
    mlp = keras.Sequential([
        # input layer and first dense layer
        keras.layers.Dense(int(input_len/2), input_shape=(input_len,), activation='relu'),
        # second dense layer
        keras.layers.Dense(int(input_len/4), activation='relu'),
        # output layer
        keras.layers.Dense(1, activation='sigmoid')
    ])

    # Compile Model
    mlp.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=[tf.keras.metrics.Precision()])

    return mlp



def execute_mlp(path : str, title : str, filename : str, X, y, model_type : int,
                batch_size : int, epochs : int, labels_map : list, test_size, seed : int):

    # For reproducibility
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # ----Holdout Method----
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=test_size, random_state=seed)
    # get subtrain and validation sets
    X_subtrain, X_validation, y_subtrain, y_validation = sklearn.model_selection.train_test_split(X_train, y_train,
                                                                                                  test_size=test_size,
                                                                                                  random_state=seed)

    # load model
    if model_type == 0:
        mlp = mlp_model(len(X[0]))

    # summary of the network
    mlp.summary()

    # clear keras session
    tf.keras.backend.clear_session()

    # it trains the model for a fixed number of epochs
    mlp.fit(x=X_subtrain, y=y_subtrain, validation_data=(X_validation, y_validation), batch_size=batch_size, epochs=epochs)




    # predictions and labels
    train_predictions = [int(round(x[0])) for x in mlp.predict(X_train)]
    predictions = [int(round(x[0])) for x in mlp.predict(X_test)]

    print('MLP Model, scores (Holdout Method):')

    print('Confusion Matrix (test set):')
    print(sklearn.metrics.confusion_matrix(y_true=y_test, y_pred=predictions, labels=np.unique(y_test)))

    print("Accuracy (train set):",
          sklearn.metrics.accuracy_score(y_true=y_train, y_pred=train_predictions))
    print("Accuracy (test set):",
          sklearn.metrics.accuracy_score(y_true=y_test, y_pred=predictions))

    print('Classification results (test set):')
    print(sklearn.metrics.classification_report(y_true=y_test, y_pred=predictions))

    # Confusion Matrix Plot
    matrix = sklearn.metrics.confusion_matrix(y_true=y_test, y_pred=predictions, normalize='true')
    print(matrix)

    sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels_map).plot()

    plt.title(title + ':\nMLP Classifier Confusion Matrix\nHoldout Method, Test Set')
    plt.savefig(path + '/' + filename + '_mlp_confusion_matrix_hm.png')
    plt.clf()  # clear plot for next method




    # ----K-Fold Method----

    # load model
    if model_type == 0:
        mlp = mlp_model(len(X[0]))

    # summary of the network
    mlp.summary()

    # At first get train and set with Holdout Method using next value after seed
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=test_size,
                                                                                random_state=(seed + 1))
    # KFold Method
    kf = sklearn.model_selection.KFold(n_splits=10, random_state=seed, shuffle=True)
    kf.get_n_splits(np.array(X_train))

    # KFold Cross Validation
    best_accuracy = 0.0
    for subtrain_index, validation_index in kf.split(np.array(X_train)):
        X_subtrain_fold, X_validation_fold = np.array(X_train)[subtrain_index], np.array(X_train)[validation_index]
        y_subtrain_fold, y_validation_fold = np.array(y_train)[subtrain_index], np.array(y_train)[validation_index]

        # clear keras session
        tf.keras.backend.clear_session()

        if model_type == 0:
            mlp = mlp_model(len(X[0]))
        mlp.fit(x=X_subtrain_fold, y=y_subtrain_fold, validation_data=(X_validation_fold, y_validation_fold),
                batch_size=batch_size, epochs=epochs)

        y_pred = [int(round(x[0])) for x in mlp.predict(X_validation_fold)]
        actual_score = sklearn.metrics.accuracy_score(y_validation_fold, y_pred)

        if actual_score > best_accuracy:
            best_accuracy = actual_score
            best_mlp = mlp


    # predictions and labels
    train_predictions = [int(round(x[0])) for x in best_mlp.predict(X_train)]
    predictions = [int(round(x[0])) for x in best_mlp.predict(X_test)]

    print('MLP Model, scores (K-Fold Method):')

    print('Confusion Matrix (test set):')
    print(sklearn.metrics.confusion_matrix(y_true=y_test, y_pred=predictions, labels=np.unique(y_test)))

    print("Accuracy (train set):",
          sklearn.metrics.accuracy_score(y_true=y_train, y_pred=train_predictions))
    print("Accuracy (test set):",
          sklearn.metrics.accuracy_score(y_true=y_test, y_pred=predictions))

    print('Classification results (test set):')
    print(sklearn.metrics.classification_report(y_true=y_test, y_pred=predictions))

    # Confusion Matrix Plot
    matrix = sklearn.metrics.confusion_matrix(y_true=y_test, y_pred=predictions, normalize='true')
    print(matrix)

    sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels_map).plot()

    plt.title(title + ':\nMLP Classifier Confusion Matrix\nK-Fold Best Classifier, Test Set')
    plt.savefig(path + '/' + filename + '_mlp_confusion_matrix_kf.png')
    plt.clf()  # clear plot for next method