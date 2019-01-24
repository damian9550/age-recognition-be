from flask import Flask
from flask import request
from flask import jsonify
from flask_cors import CORS
from PIL import Image
from io import BytesIO
import io, base64,os
import face_recognition
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import numpy as np
import keras
import cv2

app = Flask(__name__)
CORS(app)
model = load_model("model_1.h5")
model._make_predict_function()
datagen = ImageDataGenerator(rescale=1./255)


def get_face_encoding_from_base64(strImg): 
    image = face_recognition.load_image_file(io.BytesIO(base64.b64decode(strImg)))
    return image

def decode_age(y, age_range=(10,80)):
    return (age_range[1]-age_range[0]) * y + 10	
	
def predict_age(data):
    data = np.asarray(data, dtype="float32")
    data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
    # reshape to be [samples][width][height][pixels]
    X_test = data.reshape(1, data.shape[0], data.shape[1], 3)
    # convert from int to float
    X_test = X_test.astype('float32')
    p = model.predict_generator(datagen.flow(X_test), verbose=0, steps=1)
    result = decode_age(p)

    return np.round(result)

@app.route('/api/dummy')
def hello_flask():    
    return '<html><body>Hello, <strong>Flask</strong>!</body></html>'
	
@app.route('/api/uploadFile', methods=['POST'])
def uploadFile():
    file = request.values['file']
    header = file.find(",") + 1
    file = file[header:]
    image = get_face_encoding_from_base64(file)
	
    facesList = face_recognition.face_locations(image)
    print(facesList)
    results = []
    for top, right, bottom, left in facesList:
	
        image_tmp = image[top:bottom, left:right]
        image_tmp = Image.fromarray(image_tmp)
        image_tmp.save('out.bmp')
		
        age = predict_age(image_tmp)
        print(age)
        result = {"top": top, "right" : right, "bottom": bottom, "left": left, "age": age.item()}
        results.append(result)
		
    return jsonify(results)
