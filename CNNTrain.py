# -*- coding: utf-8 -*-
import sys
import os  # Manejo de archivos de sistema
import tensorflow as tf
# Para realizar el procesamiento de imagenes
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
# Para entrenar el algoritmo
from tensorflow.python.keras import optimizers
# Librería para hacer redes neuronales secuenciales
from tensorflow.python.keras.models import Sequential
# Tipos de capas
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D
# Para cerrar cualquier sesión de Keras en bkgrd
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.client import device_lib
from keras.utils.vis_utils import plot_model

# Mata cualquier session de keras
K.clear_session()

# Early stopping after 2 epochs if does not improve the train proccess
early_stopping_monitor = EarlyStopping(patience=2)

# Folder de imagenes de entrenamiento
data_entrenamiento = './data/entrenamiento'
data_validacion = './data/validacion'  # Imagenes para validar

num_clases = 6

"""
Parameters
"""
epocas = 20
# Tamaño de la imagen a procesar (Normalizada)
longitud, altura = 256, 256
batchsize = 64
filtrosConv1 = 32
# Filtros por cada convolucion (Mostrar filtros)
filtrosConv2 = 64
tamano_filtro1 = (3, 3)
tamano_filtro2 = (2, 2)             # Tamaño de pixel por cada convolución
tamano_pool = (2, 2)                # Tamaño del filtro maxpooling
clases = num_clases                 # Las clases a categorizar
lr = 0.0001                          # Learning rate entre más pequeño mejora la

# Preprocesamiento de las imagenes
# Preparamos nuestras imagenes Se hace un generador
entrenamiento_datagen = ImageDataGenerator(
    rescale=1. / 255,      # Normalizamos los valores de los pixeles 0-255...
    shear_range=0.2,        # Permite inclinar la imagen
    zoom_range=0.2,         # Las imagenes pueden tener zoom o por secciones
    horizontal_flip=True    # toma la imagen y la invierte
)

# Para validar solo se hace la re-escalación
test_datagen = ImageDataGenerator(rescale=1. / 255)

entrenamiento_generator = entrenamiento_datagen.flow_from_directory(
    data_entrenamiento,             # Nuestro set de entrenamiento
    target_size=(altura, longitud),  # Resolución en pixeles
    batch_size=batchsize,
    class_mode='categorical'      # Modo de categorización
)

print(entrenamiento_generator.class_indices)

validacion_generator = test_datagen.flow_from_directory(
    data_validacion,            # Nuestro generador de datos de validacion
    target_size=(altura, longitud),
    batch_size=batchsize,
    class_mode='categorical'      # Modo de categorización
)

cnn = Sequential()  # La red neuronal convolucional sera secuencial
# Agrega capa 1 de convolucion 2D, 150x150 y activación relu
cnn.add(Convolution2D(filtrosConv1, tamano_filtro1, padding="same",
                      input_shape=(longitud, altura, 3), activation='relu'))

# Agrega capa 2 un Maxpooling
cnn.add(MaxPooling2D(pool_size=tamano_pool))

# Agrega capa 3 otra capa convolucional (Input_shape solo se usa en la primera capa)
cnn.add(Convolution2D(filtrosConv2, tamano_filtro2, padding="same"))

# Agrega capa 4 Maxpooling
cnn.add(MaxPooling2D(pool_size=tamano_pool))

# Agrega capa 5 Aplana la imagen a una sola dimesion
cnn.add(Flatten())

# Agrega capa 6 agrega 256 neuronas con activacion relu
cnn.add(Dense(256, activation="relu"))

# Agrega capa 7 y activa aleatoriamente solo la mitad de las neuronas
cnn.add(Dropout(0.5))

# Agrega capa 8 con solo 3 neuronas para clasificar y su activacion es softmax
cnn.add(Dense(clases, activation="softmax"))

# Compila la cnn con la finalidad de mejorar la accuracy
cnn.compile(loss='categorical_crossentropy',
            optimizer=optimizers.Adam(lr=lr),
            metrics=['accuracy'])

plot_model(cnn, to_file='model_plot.png',
           show_shapes=True, show_layer_names=True)


# Entrena a la cnn
cnn.fit_generator(
    entrenamiento_generator,                # Con que imagenes se entrena
    steps_per_epoch=5784/batchsize,                  # Pasos por epoca
    epochs=epocas,                          # cuantas epocas
    validation_data=validacion_generator,   # Imagenes para validar validadas
    validation_steps=2892/batchsize,       # cuantos pasos por validadcion
    use_multiprocessing=True,
    callbacks=[early_stopping_monitor]
)
print("Entrenamiento terminado")

target_dir = './modelo/'
if not os.path.exists(target_dir):
    os.mkdir(target_dir)  # crealo si no existe
cnn.save('./modelo/modelo.h5')
cnn.save_weights('./modelo/pesos.h5')

print("Modelos guardado")
