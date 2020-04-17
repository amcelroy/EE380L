import keras
from keras import regularizers
from keras.models import Model
from keras.layers import Input, Dense, Dropout, ReLU


class MIMICNet:
    def __init__(self):
        self.__model = keras.Model

    def layer(self, input, neurons=64):
        d = Dense(neurons,
                  activation='relu',
                  #activity_regularizer=regularizers.l2(0.01)
                  )(input)
        # drop = Dropout(.2)(act)
        return d

    def create(self, columns=28):
        l1 = Input(shape=(columns,))
        l2 = self.layer(l1, neurons=128)
        l3 = self.layer(l2, neurons=32)
        out = Dense(2, activation='softmax')(l3)
        self.__model = Model(inputs=l1, outputs=out)

        return self.__model

    def compile(self):
        opt = keras.optimizers.SGD(lr=.001, momentum=.01, nesterov=True)
        self.__model.compile(
            optimizer=opt,
            loss='binary_crossentropy',
            metrics=[keras.metrics.accuracy]
        )
