import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import confusion_matrix, accuracy_score

(x_train, y_train), (x_test, y_test)= imdb.load_data(num_words=20000)
# num_words: it'll take most frequent 20000 words to make processing easier

# apply padding to make all reviews of fixed length
x_train= pad_sequences(x_train, maxlen=100)
x_test=pad_sequences(x_test, maxlen=100)
# if review<100 0 added and if>100 truncated

# x_train.shape=(25000, 100), x_test.shape=(25000, 100)
model=tf.keras.models.Sequential()
# Adding embedding layer: (Embedding layer gives the machine a way to
# understand the basic meaning of each word and creates vectors)
model.add(tf.keras.layers.Embedding(input_dim=20000, output_dim=128,
                                    input_shape=(100,)))
# Embedding layer can only be the first layer of an rnn model
# input_dim= num of unique words, output_dim= num of vector dimensions

# Adding LSTM layer: (It helps the machine remember what happened earlier in
# story and uses gates to control information to forget from past
# new information to learn from current and what to remember for next sentence.

model.add(tf.keras.layers.LSTM(units=128, activation='tanh'))
model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
# for rnn models, rmsprop works better than adam

history= model.fit(x_train, y_train, batch_size=128, epochs=5,
          validation_data=(x_test, y_test))

y_pred=(model.predict(x_test)>0.5).astype('int32')
print(y_pred[0], y_test[0])
print(y_pred[10], y_test[10])
print(y_pred[-4], y_test[-4])
# 0: negative   1: positive
cm=confusion_matrix(y_test, y_pred)
print(cm)
acc_cm= accuracy_score(y_test, y_pred)
print(acc_cm)

epoch_range=range(1,6)
plt.plot(epoch_range, history.history['accuracy'])
plt.plot(epoch_range, history.history['val_accuracy'])
plt.title("Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(['Train', 'val'], loc='upper left')
plt.show()

plt.plot(epoch_range, history.history['loss'])
plt.plot(epoch_range, history.history['val_loss'])
plt.title("Model Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(['Train', 'val'], loc='upper left')
plt.show()