from keras.models import model_from_json
from keras.models import model_from_json
import pickle
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.optimizers import Adam
from classifier import SentimentAnalysisClassifier

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
    loaded_model.compile(optimizer='adam', loss='mean_squared_error',  metrics=['mae','accuracy'])

    clean = SentimentAnalysisClassifier().clean_text
    sentence_test = ["IMPORTANTE: El restaurante cierra los días de partido o eventos.  El confort y la elegancia son uno de los dúos más apreciados por los sibaritas de todo el mundo. Pero Chamartín ha sabido convertirlo en un muy distinguido trío, puesto que suma la gastronomía asiática a esta lujosa ecuación.  Y es que Zen Market es mucho más que un restaurante, puesto que su calidad raya la perfección. No es casualidad que Ignacio García de Vinuesa y su equipo de profesionales le hayan dado forma a este rincón que también es deportista: se encuentra en una zona exclusiva del estadio Santiago Bernabéu.  Si pensabas que habías probado un pato lacado auténtico, desde eltenedor.es te pedimos que reserves una mesa y disfrutes de un auténtico pato lacado en caviar o quizás un bogavante salteado con sal y pimienta Sechuan. Si ya los probaste, reserva en cualquier caso y para cualquier ocasión. Zen Market es toda una experiencia. ¡No te lo puedes perder!",
    "me gusta mucho, todo está bien", 
    "Los platos bien.. Comida divertida. El servicio regular, tardaron en servir el café 20 minutos",
    "decepcionante, la verdad que no me ha gustado demasiado, poco recomendable", 
    "Elegimos el hotel por su buena situación, en el centro, en la calle Ancha. Apenas a unos metros de la Catedral y el barrio Húmedo. En ese sentido, perfecto. El problema es que esa localización precisamente es lo que genera que sea imposible dormir. Nos tocó una habitación en el primer piso de la fachada que da a la calle Cervantes, repleta de bares. Hasta las 3 de la mañana con ruidos constantes, golpes, gritos. A partir de esa hora, borrachos gritando. No pudimos pegar ojo, algo que también fue culpa de la estrecha cama de 1,35 cm. Una auténtica pesadilla.  El hotel es un tres estrellas anclado en los finales de los ochenta. El baño y el mobiliario así lo demuestran. Las habitaciones están dispersas por varios edificios, pasillos y más pasillos, escaleras, etc...Caótico.  Otro aspecto negativo fue la televisión. Sólo tenía sintonizadas dos cadenas y no había posibilidad de resintonizar el aparato, ya que estaba bloqueado. Además, estaba contectada la descripción para ciegos, imposible de quitar, por lo que ver una película (lo único que se podía ver, la verdad), era toda una epopeya.  Al menos pudimos disfrutar de un circuito de spa gratuito. Se trata de un spa humilde, pero que no nos vino mal. SI hubiéramos sabido que íbamos a dormir así, nos quedamos en la piscina.  El desayuno no es buffet, es un café con zumo natural y una tostada.  Habíamos reservado aparcamiento, pero viendo las opiniones al respecto acerca de la estrechez e incomodidad, decidimos aparcar en uno público que está a 50 metros...al menos no tengo que volverme loco para entrar en una calle peatonal, maniobrar de locura y arriesgarme.  https://www.tripadvisor.es/ShowUserReviews-g187492-d8670166-r384189174-Hotel_Paris-Leon_Province_of_Leon_Castile_and_Leon.html#"
    "La comida está correcta, buena, pero un poco escasa. El precio adecuado si vas con promoción de El Tenedor. Aunque los camareros son agradables, el servicio es un poco caótico, con intervalos irregulares a la hora de traer la comida. Además, pedimos dos veces la carta para pedir más raciones, pero nunca la trajeron, y la cuenta venía con un error en el descuento aplicado",
    "Meh. Iba con unas expectativas medias y salí decepcionado. Es verdad que la carta pinta bien y que la están renovando (cambio de cocinero), pero no hay nada que sorprenda o deje un buen recuerdo. Sabores que intentan ser originales y acaban con el producto principal. Presentaciones muy similares, abusando de almendras, lechugas... Lo único destacable fueron los postres, especialmente un trampantojo de huevo. ",
    "que alegría que alboroto otro perrito piloto. muy correcto", "lo odio, fatal, que horror", "malo malísimo feo caca, horrible"]
    
    sentence_test = [clean(item) for item in sentence_test]
    
    xnew = tokenizer.texts_to_sequences(sentence_test)
    xnew = pad_sequences(xnew, padding='post', maxlen=maxlen)

    ynew = loaded_model.predict(xnew)
    print(ynew)

    #print(ynew)
    #print(multilabel_binarizer.inverse_transform(np.array([ynew])))

    for i in range(0, len(ynew)):
        print("----")
        print(sentence_test[i])
        print(ynew[i] * 100)

if __name__== "__main__":
  main()