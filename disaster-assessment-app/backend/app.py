from urllib import response
from flask import Flask, json, jsonify, request
from flask_cors import CORS

import flask
import numpy as np
import cv2

from predict_earthquake import pred as pred_earthquake
from predict_forest_fire import pred as pred_forest_fire

app = Flask(__name__)
CORS(app)   

@app.route('/test', methods = ["POST"])
def test_image():

    f = request.files['myFile']
    f.save(f.filename)
    
    # Resize accd to model requirements
    image = cv2.imread(f.filename)
    print(image.shape)
    image = cv2.resize(image,(1024,1024))
    cv2.imwrite(f.filename, image)

    demo = request.form['demoMode']

    if not demo:
        model = request.form['selectedModel']
        if model == 'fire':
            # Call Fire detection model's predict!
            pred_forest_fire(f.filename)
        else:
            pred_earthquake(f.filename)  #earthquake

    # % Damage Calculation
    filename = f.filename.split('/')[-1]
    exact_name, extension = filename.split('.')

    image = cv2.imread('outputs/'+exact_name+'_mask.'+extension)
    number_of_white_pix = 0
    try:
        mask = cv2.inRange(image, (0, 0, 50), (50, 50,255))
        number_of_white_pix = np.sum(mask == 255)
    except:
        print('Code mein bug hai...but code rukna nahi cahiye')
    try:
        mask = cv2.inRange(image, (50, 0, 0), (255, 50,50))
        number_of_white_pix += np.sum(mask == 255)
    except:
        print('Code mein bug hai...but code rukna nahi cahiye')

    try:
        area_damage = round((number_of_white_pix/mask.size)*100,2)
    except:
        area_damage = 0
    print("Area Damaged: ", area_damage, ' %')
   
    return flask.jsonify(status = 'success', url=f"http://localhost:5000/files?name={'outputs/'+f.filename}", areaDamage = area_damage), 200

@app.route('/files',methods=['GET'])
def send_image():
    return flask.send_file(request.args.get("name"))

if __name__ == "__main__":
    app.run(host = 'localhost',port=5000, debug=True)
