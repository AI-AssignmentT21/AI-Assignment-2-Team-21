import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LinearRegression



def display_all_linear_regression():
    data = pd.read_csv('cleaned_data.csv')
    list_pairs1 = ['state', 'year', 'mrall', 'beertax', 'mlda', 'jaild', 'comserd', 'vmiles', 'unrate', 'perinc']
    list_pairs2 = ['state', 'year', 'mrall', 'beertax', 'mlda', 'jaild', 'comserd', 'vmiles', 'unrate', 'perinc']

    rows = len(list_pairs1)
    cols = len(list_pairs2)

    fig, axes = plt.subplots(rows, cols, figsize=(50,50))

    for i, elt1 in enumerate(list_pairs1):
        for j, elt2 in enumerate(list_pairs2):
            if elt1 != elt2:
                x = data[elt1]
                y = data[elt2]
                axes[i, j].scatter(x, y)
                axes[i, j].set_xlabel(elt1)
                axes[i, j].set_ylabel(elt2)
            
    plt.tight_layout()
    plt.savefig('graph.png')


def display_linear_regression():
    elt1 = 'vmiles'
    elt2 = 'mrall'
    data = pd.read_csv('cleaned_data.csv')
    x = data[elt1]
    y = data[elt2]

    plt.xlabel(elt1)
    plt.ylabel(elt2)

    plt.scatter(x, y)
    plt.show()


def linear_regression():
    data = pd.read_csv('cleaned_data.csv')
    X = data[['vmiles']]
    y = data['mrall']

    training_proportion = 0.7
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=1.0 - training_proportion,
                                                        random_state=42)
    
    regressor = LinearRegression()

    regressor.fit(X_train, y_train)

    print("coef: " + str(regressor.coef_[0]))
    print("intercept: " + str(regressor.intercept_))
    print()

    y_prediction = regressor.predict(X_test)

    print("R² score: ", metrics.r2_score(y_test, y_prediction))
    print("MAE: ", metrics.mean_absolute_error(y_test, y_prediction))
    print("MSE: ", metrics.mean_squared_error(y_test, y_prediction))
    print("RMSE: ", np.sqrt(metrics.mean_squared_error(y_test, y_prediction)))

    ordonne = np.linspace(4, 11, 100)
    plt.scatter(X, y)
    plt.xlabel('vmiles')
    plt.ylabel('mrall')
    plt.plot(ordonne, regressor.coef_[0]*ordonne+regressor.intercept_, color='r')
    plt.show()



linear_regression()
