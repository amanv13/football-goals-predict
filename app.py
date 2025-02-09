from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Load the trained model
model_path = 'model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract data from form
        int_features = [float(x) for x in request.form.values()]
        final_features = [np.array(int_features)]
        
        # Make prediction
        prediction = model.predict(final_features)
        output = prediction[0]

        return render_template('index.html', prediction_text=f'Prediction: {output}')
    except Exception as e:
        return jsonify({'error': str(e)}), 400  # Return error response if any issue occurs

if __name__ == "__main__":
    app.run(debug=True)
