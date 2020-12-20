import tensorflow as tf


class CrimeModel(tf.keras.Model):

    def __init__(self, units=32):

        super(CrimeModel, self).__init__()

        self.lstm = tf.keras.layers.LSTM(32)
        self.dense = tf.keras.layers.Dense(16, activation='relu')
        self.final = tf.keras.layers.Dense(1, activation='sigmoid')

    def __call__(self, x, training):

        x = self.lstm(x)
        x = self.dense(x)
        x = self.final(x)

        return x