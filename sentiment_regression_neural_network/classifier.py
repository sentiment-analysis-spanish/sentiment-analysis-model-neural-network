#tar -cJf filename.tar.xz /path/to/folder_or_file ...
#https://blog.mimacom.com/text-classification/
#https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/
from keras.datasets import imdb
import nltk
from nltk.corpus import stopwords
from sklearn.preprocessing import MultiLabelBinarizer
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, GRU, Flatten, GlobalMaxPool1D, Dropout, Conv1D
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences

import pandas as pd
import glob
import re
import numpy as np
import pickle



class SentimentAnalysisClassifier:
    def __init__(self):
        with open('../data/neural_network_config/tokenizer.pickle', 'rb') as handle:
            self.tokenizer = pickle.load(handle)
        self.multilabel_binarizer = MultiLabelBinarizer()
        self.model = None
        self.maxlen = 500


    def create_train_and_test_data(self, sentences, y):
        print("separating data into test data and train data")
        sentences_train, sentences_test, y_train, y_test = train_test_split(
        sentences, y, test_size=0.25, random_state=1000)

        X_train = self.tokenizer.texts_to_sequences(sentences_train)
        X_test = self.tokenizer.texts_to_sequences(sentences_test)

        X_train = pad_sequences(X_train, padding='post', maxlen=self.maxlen)
        X_test = pad_sequences(X_test, padding='post', maxlen=self.maxlen)
        return X_train, X_test, y_train, y_test

    def create_model(self, vocab_size):
        print("creating model")
        filter_length = 300

        #model = Sequential()
        #model.add(Embedding(vocab_size, 20, input_length=maxlen))
        #model.add(Dropout(0.15))
        #model.add(GlobalMaxPool1D())
        #model.add(Dense(output_size, activation='sigmoid'))

        self.model = Sequential()
        self.model.add(Embedding(vocab_size, 20, input_length=self.maxlen))
        self.model.add(Dropout(0.1))
        self.model.add(Conv1D(filter_length, 3, padding='valid', activation='relu', strides=1))
        self.model.add(GlobalMaxPool1D())
        self.model.add(Dense(1, activation='sigmoid'))

        #self.model = Sequential()
        #self.model.add(Embedding(vocab_size, 20, input_length=self.maxlen))
        #self.model.add(GRU(128, return_sequences=True))
        #self.model.add(GRU(128))
        #self.model.add(Dense(1, activation='sigmoid'))
        
        self.model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
              
    def save_model(self, model):
        print("saving model")
        # serialize model to JSON
        model_json = model.to_json()
        with open("../data/neural_network_config/model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights("../data/neural_network_config/model.h5")
        print("Saved model to disk")

    def create_and_train_model(self):
        filename = "../data/json_bundle_reviews/large-bundle-clean.json"
        df = pd.read_json(filename)
        #df = self.clean_news(df)

        y = df.sentiment.values
        sentences = df['content'].values

        X_train, X_test, y_train, y_test = self.create_train_and_test_data(sentences, y)

        vocab_size = len(self.tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index

        self.create_model(vocab_size)
        
        print("training model")
        callbacks = [
        ModelCheckpoint(filepath='../data/neural_network_config/temp-model.h5', save_best_only=True)]

        history = self.model.fit(X_train, y_train,
                            epochs=40,
                            batch_size=100,
                            validation_data=(X_test, y_test),
                            callbacks=callbacks)

        #loss, accuracy = self.model.evaluate(X_train, y_train, verbose=False)
        #print("Training Accuracy: {:.4f}".format(accuracy))

        #loss, accuracy = self.model.evaluate(X_test, y_test, verbose=False)
        #print("Testing Accuracy:  {:.4f}".format(accuracy))


        self.save_model(self.model)


if __name__== "__main__":
    classifier = SentimentAnalysisClassifier()
    classifier.create_and_train_model()