#!/usr/bin/env python3

import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.svm import SVC
import keras
from keras.models import Sequential
from keras.layers import Dense

plotData = True
n = 2
Ntrain = 1000
Ntest = 10000
alpha = [0.33, 0.34, 0.33]
meanVectors = np.transpose([[-18, 0, 18], [-8, 0, 8]])
covEvalues = np.transpose([[3.2**2, 0], [0, 0.6**2]])
covEvectors = [
    np.array([[1, -1], [1, 1]]) / math.sqrt(2),
    np.array([[1, 0], [0, 1]]),
    np.array([[1, -1], [1, 1]]) / math.sqrt(2)
]

K = 10

def generate_data(N):
    data_labels = np.random.choice(3, N, replace=True, p=alpha)
    ind0 = np.array((data_labels==0).nonzero())
    ind1 = np.array((data_labels==1).nonzero())
    ind2 = np.array((data_labels==2).nonzero())
    N0 = np.shape(ind0)[1]
    N1 = np.shape(ind1)[1]
    N2 = np.shape(ind2)[1]
    Xtrain = np.zeros((N, n))
    x0 = np.transpose(np.matmul(np.matmul(covEvectors[0], covEvalues**(1/2)), np.random.standard_normal((n, N0))) + meanVectors[0].reshape(2,1))
    x1 = np.transpose(np.matmul(np.matmul(covEvectors[1], covEvalues**(1/2)), np.random.standard_normal((n, N1))) + meanVectors[1].reshape(2,1))
    x2 = np.transpose(np.matmul(np.matmul(covEvectors[2], covEvalues**(1/2)), np.random.standard_normal((n, N2))) + meanVectors[2].reshape(2,1))
    np.put_along_axis(Xtrain, np.transpose(ind0), x0, axis=0)
    np.put_along_axis(Xtrain, np.transpose(ind1), x1, axis=0)
    np.put_along_axis(Xtrain, np.transpose(ind2), x2, axis=0)
    return Xtrain

# Uses K-Fold cross validation to find the best hyperparameters for an MLP model, and plots the results
def train_MLP_hyperparams(TrainingData_labels, TrainingData_features):
    num_perceptron_candidates = list(range(1, 21))
    hyperparam_candidates = np.array([[['sigmoid', n] for n in num_perceptron_candidates], [['softplus', n] for n in num_perceptron_candidates]])
    hyperparam_performance = np.zeros((np.shape(hyperparam_candidates)[0] * np.shape(hyperparam_candidates)[1]))
    for (i, hyperparams) in enumerate(np.reshape(hyperparam_candidates, (-1, 2))):

        skf = KFold(n_splits=K, shuffle=False)

        total_loss = 0

        for(k, (train, test)) in enumerate(skf.split(TrainingData_features, TrainingData_labels)):
            loss = min(map(lambda _: MLP_loss(hyperparams, TrainingData_features[train], TrainingData_labels[train], TrainingData_features[test], TrainingData_labels[test])[1], range(4)))
            total_loss += loss

        loss = total_loss / K
        hyperparam_performance[i] = loss

        print(i, loss)

    plt.style.use('seaborn-white')

    max_perf_index = np.argmin(hyperparam_performance)
    max_perf_x1 = max_perf_index % 20
    max_perf_x2 = max_perf_index // 20
    best_nonlinearity = hyperparam_candidates[max_perf_x2][max_perf_x1][0]
    best_num_perceptrons = hyperparam_candidates[max_perf_x2][max_perf_x1][1]

    plt.plot(list(range(1,21)), hyperparam_performance[0:20], 'b.')
    plt.plot(list(range(1,21)), hyperparam_performance[20:40], 'r.')
    plt.title("MLP K-Fold Hyperparameter Validation Performance")
    plt.xlabel("Number of perceptrons in hidden layer")
    plt.ylabel("MLP MSE loss")
    plt.legend(["Sigmoid activation", "Softplus activation"])
    plt.ylim([0,5])
    plt.plot(max_perf_x1 + 1, hyperparam_performance[max_perf_index], 'gx')
    print("The best MLP loss was " + str(hyperparam_performance[max_perf_index]) + ".")
    plt.show()
    return (best_nonlinearity, best_num_perceptrons)

# Trains an MLP with the given hyperparams on the train data, then validates its performance on the given test data.
# Returns the trained model and respective validation loss.
def MLP_loss(hyperparams, train_features, train_labels, test_features, test_labels):
    (nonlinearity, num_perceptrons) = hyperparams
    sgd = keras.optimizers.SGD(lr=0.002)
    model = Sequential()
    model.add(Dense(int(num_perceptrons), activation=str(nonlinearity), input_dim=1))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer=sgd)
    model.fit(train_features, train_labels, epochs=1000, batch_size=100, verbose=0)
    loss = model.evaluate(test_features, test_labels)
    return (model, loss)


Xtrain = generate_data(Ntrain)
Xtest = generate_data(Ntest)

if plotData:
    plt.subplot(1,2,1)
    plt.plot(Xtrain[:,0], Xtrain[:,1], '.')
    plt.title('Training Data')
    plt.subplot(1,2,2)
    plt.plot(Xtest[:,0], Xtest[:,1], '.')
    plt.title('Testing Data')
    plt.show()

hyperparams = train_MLP_hyperparams(Xtrain[:,1], Xtrain[:,0])
(best_nonlinearity, best_num_perceptrons) = hyperparams

print("The best performance was achieved with " + str(best_num_perceptrons) + " perceptrons using " + str(best_nonlinearity) + " as an activation function.")
(model, loss) = min(map(lambda _: MLP_loss(hyperparams, Xtrain[:,0], Xtrain[:,1], Xtest[:,0], Xtest[:,1]), range(5)), key=lambda r: r[1])
print("The test dataset was fit with a loss of " + str(loss) + ".")

predictions = model.predict(Xtest[:,0])
plt.plot(Xtest[:,0], Xtest[:,1], 'b.')
plt.plot(Xtest[:,0], predictions, 'r.')
plt.title('Trained model performance on test dataset')
plt.legend(['Test data', 'Predicted data'])
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()
