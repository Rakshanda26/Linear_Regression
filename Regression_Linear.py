import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    # return 'Hello World'
    return render_template('home.html')



@app.route('/linear_predict', methods=['POST'])
def linear_reg():
    data = request.get_json()
    print("data is = ",data)
    new_data = pd.DataFrame(data)
    print("new_data is = " , new_data)
    output = model.predict(new_data)
    print(output)
    predicted_value = output[0][0]
    print("predicted_value is = ", predicted_value)
    return jsonify({"prediction": predicted_value})


if __name__ == "__main__":
    app.run(debug=True, port=5002)
