# -*- coding: utf-8 -*-

from flask import Flask, request, Response
import json
import jsonpickle
import numpy as np
import cv2
import base64

app = Flask(__name__)

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
global graph
graph = tf.get_default_graph()

predicciones = ['Adolfo', 'Aldo', 'Daniel','Quillo','Sergio','Vania']

def predict(file):
    with graph.as_default():
        x = load_img(file, target_size=(longitud, altura))
        x = img_to_array(x)
        x = np.expand_dims(x, axis=0)
        array = cnn.predict(x) ## [1,0,0]
        result = array[0]
        print(result)
        answer = np.argmax(result)
        
        print(predicciones[answer])
        
        return answer

@app.route("/")
def hello():
    return "API FACE RECOGNITION!"

# route http posts to this method
@app.route('/api/test', methods=['POST'])
def test():
    # convert string of image data to uint8
    nparr = np.fromstring(base64.b64decode(request.form['img']), np.uint8)
    # decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.resize(img,(256,256))
    cv2.imwrite('./test/image_test.png',img)

    answer = predict("./test/image_test.png")
    
    # build a response dict to send back to client
    response = {'message': 'Hola: ' + predicciones[answer]}
    # encode response using jsonpickle
    response_pickled = jsonpickle.encode(response)

    return Response(response=response_pickled, status=200, mimetype="application/json")

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True, port=5000)