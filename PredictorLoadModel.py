#from tkinter import filedialog
#from tkinter import *
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
from tensorflow.python.keras.models import load_model

longitud, altura = 256, 256
modelo = './modelo/modelo.h5'
pesos_modelo = './modelo/pesos.h5'
cnn = load_model(modelo)
cnn.load_weights(pesos_modelo)

predicciones = ['Adolfo', 'Aldo', 'Daniel','Quillo','Sergio','Vania']

def predict(file):
  x = load_img(file, target_size=(longitud, altura))
  x = img_to_array(x)
  x = np.expand_dims(x, axis=0)
  print("Prediction..")
  array = cnn.predict(x) ## [1,0,0]
  result = array[0]
  answer = np.argmax(result)
  print(predicciones[answer])
  return answer

predict("./test/adolfo1.png")
predict("./test/adolfo2.png")
predict("./test/image_test.png")
predict("./test/aldo1.png")
predict("./test/aldo2.png")
predict("./test/daniel.png")
predict("./test/daniel1.png")
predict("./test/quillo1.png")
predict("./test/quillo2.png")
predict("./test/sergio1.png")
predict("./test/sergio2.png")
predict("./test/vania1.png")
predict("./test/vania2.png")
predict("./test/vania3.png")
predict("./test/vania4.png")
