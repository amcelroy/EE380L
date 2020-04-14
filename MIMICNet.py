import keras
from keras.models import Model
from keras.layers import Input, Dense

class MIMICNet:
    def __init__(self):
        self.__model = keras.Model

    def create(self, columns=28):
        l1 = Input(shape=(28,))
        l2 = Dense(1024, activation='relu', )
        return self.__model