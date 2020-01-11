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
    "La entrada en el 2020 ha comportado cambios en la edad de jubilación y en el cálculo de los años cotizados que se tienen en cuenta para determinar la prestación. Las medidas son de carácter automático, ya que forman parte de la reforma de 2011 que hace que la edad para jubilarse se retrase paulatinamente hasta llegar a los 67 años.  ADVERTISING  Todo esto se da mientras resta pendiente saber cuál será la subida de las prestaciones en el 2020, ya que aunque el Gobierno en funciones ha prometido que se subirán el 0,9% y no perderán poder adquisitivo, la medida no se tomará hasta que esté formado un Ejecutivo. En diciembre de 2019 en España se contabilizaban 6.089.294 pensiones de jubilación, con una prestación media de 1.143,55 euros mensuales.   Pensiones en 2020 Los cambios en la edad de jubilación  Respecto a la edad de jubilación, cada año se va retrasando en virtud del régimen establecido en la reforma de 2011 aprobada durante el mandato de José Luis Rodríguez Zapatero. De esta forma, en 2020 la edad legal ordinaria será de 65 años y 10 meses. Esta edad se aplicará a aquellos que han cotizado menos de 37 años.  Si una persona llega a los 65 años en 2020 y ha cotizado 37 años o más, ya podrá jubilarse con 65 años.  En el caso de la jubilación parcial, en la que se combina trabajo y prestación, el mínimo será de 61 años y 10 meses con 35 años o más cotizados; o de 62 años y 8 meses con 33 años cotizados.  Con cada año que pasa es necesaria más edad para acceder a la jubilación, tanto si se ha cotizado por encima o por debajo de los periodos de referencia  Con cada año que pasa es necesaria más edad para acceder a la jubilación, tanto si se ha cotizado por encima o por debajo de los periodos de referencia Pensiones en 2020 Los cambios en el cálculo de la pensión  Por lo que respecta al cálculo de la pensión que se cobrará la momento de jubilarse, en 2020 se tendrán en cuenta los últimos 23 años cotizados. Estos años cotizados conforman la base reguladora, que es la suma de las bases de cotización en dicho periodo. Hay que tener en cuenta que cuantos más años se tengan en cuenta es posible que se recorte más la pensión, ya que en los últimos años de vida laboral es cuando mejores salarios se suelen cobrar.   Esta es otra de las reformas introducidas con los cambios en las pensiones de la década anterior, momento hasta el que se tenían en cuenta los últimos 15 años trabajados. La idea es que para 2022 ya se tengan en cuenta los últimos 25 años cotizados. De esta manera, en 2021 se computarán los últimos 24 años trabajados y en 2022 los últimos 25 años cotizados.  La base reguladora de la pensión se obtiene de dividir los meses de los años exigidos por el divisor correspondiente La base reguladora de la pensión se obtiene de dividir los meses de los años exigidos por el divisor correspondiente (LV) En 2023 El recorte de las pensiones que viene  Otra de las medidas que tendrán un fuerte calado en el sistema es la llegada del factor de sostenibilidad, que se aplicará a partir de 2023 e irá recortando las nuevas pensiones, teniendo en cuenta que los pensionistas vivirán más. Dicha medida en un principio debía aplicarse en 2019.  El conjunto de medidas se puede consultar al detalle en la guía para la jubilación del Ministerio de Trabajo, Migraciones y Seguridad Social.",
    "El Juzgado de Violencia de Género de Elche (Alicante) investiga a un hombre detenido el pasado miércoles por una supuesta agresión a su hija, de 19 años. Tras decretar su libertad provisional, se ha abierto una causa por los supuestos delitos de malos tratos y abuso sexual. La investigación trata de determinar también si, fruto de tal relación incestuosa, la hija de la víctima, de 15 meses, lo es también del investigado.  MÁS INFORMACIÓN El Defensor del Pueblo pide a Interior un plan de seguimiento a menores víctimas de violencia machista Un hombre mata a su expareja y a su hija de 3 años en Esplugues de Llobregat  Según han confirmado a EL PAÍS fuentes del Tribunal Superior de Justicia de la Comunidad Valenciana (TSJCV), tras la detención el juez decretó la libertad provisional del acusado así como una orden de alejamiento con respecto a su hija. Ni la Fiscalía ni la víctima habían solicitado prisión preventiva para el acusado. La causa, que de momento lleva el juzgado de Violencia de Género, se ha iniciado con una investigación previa y el magistrado ha encargado una prueba pericial psicológica de la víctima.  Los   indicios apuntan a que padre e hija mantenían una relación  , afirman las mismas fuentes. Si se confirma, el caso continuará en el juzgado de Violencia de Género mientras que de lo contrario se trasladará a un juzgado de instrucción ordinario al considerarse maltrato de un padre hacia una hija.  Según ha adelantado el periódico Información, la denuncia partió de una vecina de la víctima, que la encontró junto a su casa llorando y con un ojo amoratado. Tras la detención del presunto agresor y su paso a disposición judicial, padre e hija declararon que la bebé es la hija en común de ambos y que mantuvieron relaciones desde que la víctima tenía 16 años. ADVERTISING  Ads by Teads   Larga condena Según el mismo diario, la muchacha estuvo acogida en una vivienda tutelada de la Generalitat valenciana después de que su padre ingresara en prisión y cumpliera una larga condena. Al quedar en libertad, el arrestado contactó de nuevo con ella, la entonces menor abandonó el piso tutelado y se fue a vivir con su padre. Poco después se quedó embarazada y dio a luz cuando todavía era menor de edad. Las mismas fuentes aseguran que ambos declararon en sede judicial que son los padres del bebé.  También apuntan las mismas fuentes que el informe psicológico solicitado por la Fiscalía pretende establecer el estado psicológico de la muchacha, que se ha ido a vivir a casa de una tía, hermana del detenido, por carecer de medios económicos y no poder volver al domicilio familiar.",
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