import sklearn
import matplotlib.pyplot as plt
import numpy as np
import copy

def execute_knn(path : str, title : str, filename : str, X, y, labels_map : list, k_value : int, test_size, seed : int):

    # KNN model ---- Holdout Method
    knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=k_value)

    # Holdout Method split
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=test_size, random_state=seed)

    # train knn
    knn.fit(X=X_train, y=y_train)

    # predictions and labels
    train_predictions = knn.predict(X=X_train)
    predictions = knn.predict(X=X_test)

    print('KNN Model, scores (Holdout Method):')

    print('Confusion Matrix (test set):')
    print(sklearn.metrics.confusion_matrix(y_true=y_test, y_pred=predictions, labels=np.unique(y_test)) )
    print("Accuracy (train set):", sklearn.metrics.accuracy_score(y_true=y_train, y_pred=train_predictions))
    print("Accuracy (test set):", sklearn.metrics.accuracy_score(y_true=y_test, y_pred=predictions))
    print('Classification results (test set):')
    print(sklearn.metrics.classification_report(y_true=y_test, y_pred=predictions) )



    # Confusion Matrix Plot Test Accuracy
    sklearn.metrics.plot_confusion_matrix(estimator=knn, X=X_test, y_true=y_test, normalize='true', display_labels=labels_map)

    plt.title(title + ':\nKNN Classifier Confusion Matrix\nHoldout Method, Test Set')

    plt.savefig(path + '/' + filename + '_knn_confusion_matrix_hm.png')
    plt.clf()  # clear plot for next method








    # ----KFold Cross Validation----

    # KNN model
    knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=k_value)

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

        knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=k_value)
        knn.fit(X_subtrain_fold, y_subtrain_fold)

        y_pred = knn.predict(X_validation_fold)
        actual_score = sklearn.metrics.accuracy_score(y_validation_fold, y_pred)

        if actual_score > best_accuracy:
            best_accuracy = actual_score
            best_knn = copy.deepcopy(knn)



    # predictions and labels
    train_predictions = best_knn.predict(X_train)
    predictions = best_knn.predict(X_test)

    print('KNN Model, scores (K-Fold Method):')

    print('Confusion Matrix (test set):')
    print(sklearn.metrics.confusion_matrix(y_true=y_test, y_pred=predictions, labels=np.unique(y_test)))

    print("Best K-Fold Classifier Accuracy (train set):", sklearn.metrics.accuracy_score(y_true=y_train, y_pred=train_predictions))
    print("Best K-Fold Classifier Accuracy (test set):", sklearn.metrics.accuracy_score(y_true=y_test, y_pred=predictions))

    print('Classification results (test set):')
    print(sklearn.metrics.classification_report(y_true=y_test, y_pred=predictions))



    # Confusion Matrix plot Test Accuracy
    sklearn.metrics.plot_confusion_matrix(estimator=best_knn, X=X_test, y_true=y_test, normalize='true', display_labels=labels_map)

    plt.title(
        title + ':\nKNN Classifier Confusion Matrix\nK-Fold Best Classifier, Test Set')

    plt.savefig(path + '/' + filename + '_knn_confusion_matrix_kf.png')
    plt.clf()  # clear plot for next method