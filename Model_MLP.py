import keras
from keras import layers
import numpy as np
from sklearn.neural_network import MLPClassifier
from Evaluation import evaluation


def Model_MLP(Train_Data, Train_Target, Test_Data, Test_Target):
    # Create model object
    clf = MLPClassifier(hidden_layer_sizes=(6, 5),
                        random_state=5,
                        verbose=True,
                        learning_rate_init=0.01)

    # Fit data onto the model
    clf.fit(Train_Data, Train_Target)

    # Make prediction on test dataset
    ypred = clf.predict(Test_Data)

    Eval = evaluation(ypred, Test_Target)

    return Eval, ypred


