from flask import Flask, json, jsonify, request
from flask_cors import CORS
from predict_earthquake import pred
# from dbms.dict_db.model import Model

app = Flask(__name__)
CORS(app)   
# model = Model()

@app.route('/test', methods = ["POST"])
def test_image():
    print(request.data)
    print("Working!")
    f = request.files['myFile']
    f.save(f.filename)
    pred(f.filename)
    return {"YAY":"YAYY"}, 200
if __name__ == "__main__":
    app.run(host = 'localhost',port=5000, debug=True)


# to-d0
# Resize --> 1024*1024
#Directly pass to model!