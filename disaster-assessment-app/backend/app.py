from urllib import response
from flask import Flask, json, jsonify, request
from flask_cors import CORS

import flask
import numpy as np
import cv2

from predict import pred

app = Flask(__name__)
CORS(app)   

@app.route('/test', methods = ["POST"])
def test_image():

    f = request.files['myFile']
    # f.save(f.filename)
    
    # Resize accd to model requirements
    image = cv2.imread(f.filename)
    print(image.shape)
    image = cv2.resize(image,(1024,1024))
    cv2.imwrite(f.filename, image)

    
    model = request.form['selectedModel']

    if model == 'fire':
        # Call Fire detection model's predict!
        pass
    else:
        pred(f.filename)  #earthquake

    # % Damage Calculation
    
    image = cv2.imread('outputs/'+f.filename)
    mask = cv2.inRange(image, (0, 0, 50), (50, 50,255))
    number_of_white_pix = np.sum(mask == 255)
    area_damage = round((number_of_white_pix/mask.size)*100,2)
    print("Area Damaged: ", area_damage, ' %')
   
    return flask.jsonify(status = 'success', url=f"http://localhost:5000/files?name={'outputs/'+f.filename}", areaDamage = area_damage), 200

@app.route('/files',methods=['GET'])
def send_image():
    return flask.send_file(request.args.get("name"))

if __name__ == "__main__":
    app.run(host = 'localhost',port=5000, debug=True)
