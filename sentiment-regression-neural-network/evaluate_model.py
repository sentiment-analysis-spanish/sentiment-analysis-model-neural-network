from keras.models import model_from_json
from keras.models import model_from_json
import pickle
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.optimizers import Adam

maxlen = 100
def main():
    # loading
    with open('../data/neural_network_config/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # load json and create model
    json_file = open('../data/neural_network_config/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights("../data/neural_network_config/model.h5")
    print("Loaded model from disk")


    # evaluate loaded model on test data
    loaded_model.compile(optimizer=Adam(0.015), loss='mean_squared_error', metrics=['mse', 'mae', 'mape', 'cosine'])
        


    sentence_test = ["me gusta mucho, todo está bien", "que alegría que alboroto otro perrito piloto. muy correcto", "lo odio, fatal, que horror", "malo malísimo feo caca"]
    xnew = tokenizer.texts_to_sequences(sentence_test)
    xnew = pad_sequences(xnew, padding='post', maxlen=maxlen)
    print(xnew)

    ynew = loaded_model.predict(xnew)
    print(ynew)
    for proba in ynew:
        print("----")
        print(proba)


    #print(ynew)
    #print(multilabel_binarizer.inverse_transform(np.array([ynew])))

if __name__== "__main__":
  main()