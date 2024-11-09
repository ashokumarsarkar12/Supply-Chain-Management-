import keras
from keras import layers
import cv2 as cv
import numpy as np
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.neural_network import MLPClassifier
from Evaluation import evaluation


def Model_AE_LSTM_MLP(Train_Data, Train_Target, Test_Data, Test_Target):
    # This is the size of our encoded representations
    encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

    # This is our input image
    input_img = keras.Input(shape=(Train_Data.shape[1],))
    # "encoded" is the encoded representation of the input
    encoded = layers.Dense(encoding_dim, activation='relu')(input_img)
    # "decoded" is the lossy reconstruction of the input
    decoded = layers.Dense(Train_Target.shape[1], activation='sigmoid')(encoded)

    # This model maps an input to its reconstruction
    autoencoder_lstm_mlp = keras.Model(input_img, decoded)
    # This model maps an input to its encoded representation
    encoder = keras.Model(input_img, encoded)
    # This is our encoded (32-dimensional) input
    encoded_input = keras.Input(shape=(encoding_dim,))
    # Retrieve the last layer of the autoencoder model
    decoder_layer = autoencoder_lstm_mlp.layers[-1]
    # Create the decoder model
    decoder = keras.Model(encoded_input, decoder_layer(encoded_input))
    autoencoder_lstm_mlp.compile(optimizer='adam', loss='binary_crossentropy')
    autoencoder_lstm_mlp.add(LSTM(50), input_shape=(1, 200))
    autoencoder_lstm_mlp.add(Dense(Train_Target.shape[1]))
    autoencoder_lstm_mlp.add(Dense(Train_Target.shape[1]))
    autoencoder_lstm_mlp.add(MLPClassifier(hidden_layer_sizes=(6, 5),random_state=5,verbose=True,learning_rate_init=0.01))
    autoencoder_lstm_mlp.compile(loss='mean_squared_error', optimizer='adam')

    # (x_train, y_train), (x_test, y_test) = mnist.load_data()
    Train_Data = Train_Data.astype('float32') / 255.
    Test_Data = Test_Data.astype('float32') / 255.
    Train_Data = Train_Data.reshape((len(Train_Data), np.prod(Train_Data.shape[1:])))
    Test_Data = Test_Data.reshape((len(Test_Data), np.prod(Test_Data.shape[1:])))
    autoencoder_lstm_mlp.fit(Train_Data, Train_Target,
                    epochs= 5,
                    batch_size=256,
                    shuffle=True,
                    validation_data=(Test_Data, Test_Target))
    # Encode and decode some digits
    # Note that we take them from the *test* set
    encoded_imgs = encoder.predict(Test_Data)
    pred = decoder.predict(encoded_imgs)

    Eval = evaluation(pred, Test_Target)

    return Eval, pred
