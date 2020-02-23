# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 09:54:07 2019

@author: jawad_zf1uaw5
"""

import flask
import numpy as np
import tensorflow as tf
#from tensorflow import keras
import keras
from keras.models import load_model
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
def init():
    print('entry in to initiation')
    global model,graph
    print('exit')
    # load the pre-trained Keras model
    #model = load_model('C:/Users/jawad_zf1uaw5/.spyder-py3/my_model.h5')
    with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
        model = load_model('C:/Users/jawad_zf1uaw5/.spyder-py3/my_model.h5')
    #model = keras.models.load_model('C:/Users/jawad_zf1uaw5/Desktop/pers/DL/my_model.h5')
    #model = keras.models.load_model('C:/Users/jawad_zf1uaw5/.spyder-py3/my_model.h5')
    print('model loaded ....')
    graph = tf.get_default_graph()
    return
def getParameters():
    parameters = []
    parameters.append(flask.request.args.get('Cylinders'))
    parameters.append(flask.request.args.get('Displacement'))
    parameters.append(flask.request.args.get('Horsepower'))
    parameters.append(flask.request.args.get('Weight'))
    parameters.append(flask.request.args.get('Acceleration'))
    parameters.append(flask.request.args.get('Model Year'))
    parameters.append(flask.request.args.get('USA'))
    parameters.append(flask.request.args.get('Europe'))
    parameters.append(flask.request.args.get('Japan'))
    return parameters
def sendResponse(responseObj):
    response = flask.jsonify(responseObj)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Methods', 'GET')
    response.headers.add('Access-Control-Allow-Headers', 'accept,content-type,Origin,X-Requested-With,Content-Type,access_token,Accept,Authorization,source')
    response.headers.add('Access-Control-Allow-Credentials', True)
    return response
# API for prediction
@app.route("/predict", methods=["GET"])
def predict():
    iD = flask.request.args.get('id')
    print('getting parameters / features')
    parameters = getParameters()
    inputFeature = np.asarray(parameters).reshape(1, 9)
    with graph.as_default():
        print('prediction in progress')
        raw_prediction = str(model.predict(inputFeature)[0][0])
        print('prediction completed')
    return sendResponse({iD: raw_prediction})
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server...""please wait until server has fully started"))
    init()
    app.run(threaded=True)