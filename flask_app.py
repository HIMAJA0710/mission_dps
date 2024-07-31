from flask import Flask, request, jsonify
import pickle
from datetime import datetime

app = Flask(__name__)

# Load the pre-trained model (update the path as needed)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    year = data.get('year')
    month = data.get('month')

    if year is None or month is None:
        return jsonify({'error': 'Invalid input'}), 400

    try:
        date = datetime(year, month, 1)
    except ValueError:
        return jsonify({'error': 'Invalid date'}), 400

    month_numeric = date.toordinal()
    prediction = model.predict([[month_numeric]])[0]

    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
