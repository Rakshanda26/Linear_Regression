import pickle
from flask import Flask, request, render_template

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(request.form['TV'])]  # Corrected field name to match your form input
    final_features = [data]
    print(final_features)
    # Replace this with your model prediction logic
    output = model.predict(final_features)[0][0]
    print(output)

    return render_template('home.html', prediction_text="Prediction for sales is {}".format(output))

if __name__ == '__main__':
    app.run(debug=True, port=5004)
