# CNN Face Recognition

Face recognition involves identifying or verifying a person from a digital image or video frame and is still one of the most challenging tasks in computer vision today. The conventional face recognition pipeline consists of face detection, face alignment, feature extraction, and classification. 

This software uses a python's server based on REST to connect with a mobile application. The mobile app, sends the face to get recognized in a large dataset of images using the CNN. The it posts the result with the label representations the class of the received image.

# Model of the CNN:

![Alt text](https://github.com/adolfos94/CNN-Face-Recognition/blob/master/model_plot.png?raw=true "CNN MODEL")

## Functions

- Load Model
```python
modelo = './modelo/modelo.h5'
pesos_modelo = './modelo/pesos.h5'
cnn = load_model(modelo)
cnn.load_weights(pesos_modelo)
```

- Predictor

```python
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
```
