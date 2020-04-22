
import keras
from keras import regularizers
from keras.callbacks import Callback
from keras.models import Model
from keras.layers import Input, Dense, Dropout, ReLU


class MIMICNet(Callback):
    def __init__(self):
        self.__model = keras.Model
        self.__train_acc = []
        self.__test_acc = []

    def layer(self, input, neurons=64):
        d = Dense(neurons,
                  activation='relu',
                  kernel_regularizer=regularizers.l2(0.00001)
                  )(input)
        drop = Dropout(.5)(d)
        return drop

    def create(self, columns=28):
        l1 = Input(shape=(columns,))
        l = self.layer(l1, neurons=1024)
        l = self.layer(l, neurons=512)
        out = Dense(2, activation='softmax')(l)
        self.__model = Model(inputs=l1, outputs=out)

        return self.__model

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        self.__train_acc.append(logs['accuracy'])
        self.__test_acc.append(logs['val_accuracy'])

    def compile(self):
        opt = keras.optimizers.Adam(lr=.0001)
        self.__model.compile(
            optimizer=opt,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
