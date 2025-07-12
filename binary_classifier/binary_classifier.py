import os
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    seed = 0
    data_directory_path = '../batches'
    test_size = 0.2

    data = [os.path.join(data_directory_path, f) for f in os.listdir(data_directory_path) if f.endswith('.npy')]

    X = []
    y = []

    for batch in data:
        print(batch)
        data = np.load(batch, allow_pickle=True).item()
        for X_batch, y_batch in zip(data['X'], data['y']):
            X.append(X_batch)
            y.append(y_batch)

    X = np.array(X)
    y = np.array(y)

    X = X.reshape(X.shape[0], -1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

    print("Training model")

    clf = RandomForestClassifier(random_state=seed, n_estimators=200)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_train)

    with open('best_model.pkl', 'wb') as model_file:
        pickle.dump(clf, model_file)

    y_pred_train = clf.predict(X_train)

    print(
        f"Train set:\n"
        f"Accuracy: {accuracy_score(y_train, y_pred_train)}\n"
        f"Precision: {precision_score(y_train, y_pred_train, average='weighted', zero_division=1)}\n"
        f"Recall: {recall_score(y_train, y_pred_train, average='weighted', zero_division=1)}\n"
        f"F1: {f1_score(y_train, y_pred_train, average='weighted', zero_division=1)}")

    y_pred_test = clf.predict(X_test)
    print(
        f"Test set:\n"
        f"Accuracy: {accuracy_score(y_test, y_pred_test)}\n"
        f"Precision: {precision_score(y_test, y_pred_test, average='weighted', zero_division=1)}\n"
        f"Recall: {recall_score(y_test, y_pred_test, average='weighted', zero_division=1)}\n"
        f"F1: {f1_score(y_test, y_pred_test, average='weighted', zero_division=1)}\n"
    )
