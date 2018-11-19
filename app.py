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

app = Flask(__name__)
CORS(app)
model = load_model("model_1.h5")
datagen = ImageDataGenerator(rescale=1./255)


def get_face_encoding_from_base64(strImg): 
    image = face_recognition.load_image_file(io.BytesIO(base64.b64decode(strImg)))
    return image
	
	
def predict_age(data):
    data = np.asarray(data, dtype="float32")

    # reshape to be [samples][width][height][pixels]
    X_test = data.reshape(1, data.shape[0], data.shape[1], 3)
    # convert from int to float
    X_test = X_test.astype('float32')

    p = model.predict_generator(datagen.flow(X_test), verbose=0)

    return np.argmax(p) + 10

@app.route('/')
def hello_flask():    
    return '<html><body>Hello, <strong>Flask</strong>!</body></html>'
	
@app.route('/uploadFile', methods=['POST'])
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
        #age = predict_age(image_tmp) #not working yet
        age = 20 #MOCK!
		
        result = {"top": top, "right" : right, "bottom": bottom, "left": left, "age": age}
        results.append(result)
		
    return jsonify(results)
