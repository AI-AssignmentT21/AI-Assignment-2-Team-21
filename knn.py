import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report



def display_kNN():
    data = pd.read_csv('cleaned_data.csv')
    x_tmp = data['comserd']
    y_tmp = data['state']
    points = []

    for i in range(len(x_tmp)):
        points.append([x_tmp[i], y_tmp[i]])

    x = np.array(points)
    y = data['jaild']

    plt.xlabel('comserd')
    plt.ylabel('state')

    colors = ['green' if val == 1 else 'red' for val in y]

    plt.scatter(x[:, 0], x[:, 1], c=colors)
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Mandatory prison sentence',
                              markerfacecolor='green', markersize=5),
                       Line2D([0], [0], marker='o', color='w', label='No mandatory prison sentence',
                              markerfacecolor='red', markersize=5)]
    plt.legend(handles=legend_elements, title='Legend', loc='upper right', markerscale=2)

    plt.show()


def kNN():
    data = pd.read_csv('cleaned_data.csv') 
    targets = data['jaild']
    clustering_dataset = data[['state', 'comserd']]

    training_proportion = 0.7
    X_train, X_test, y_train, y_test = train_test_split(clustering_dataset, targets,
                                                        test_size=1.0 - training_proportion,
                                                        random_state=42)

    neigh = KNeighborsClassifier(n_neighbors=1, metric='euclidean')

    neigh.fit(X_train, y_train)

    y_prediction = neigh.predict(X_test)
    print(classification_report(y_test, y_prediction, zero_division=0))



kNN()
display_kNN()
