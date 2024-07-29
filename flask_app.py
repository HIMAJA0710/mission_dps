from flask import Flask, request, jsonify
import pickle
from datetime import datetime

app = Flask(__name__)

# Load the pre-trained model (update the path as needed)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/predict', methods=['POST'])
def predict():
    # Get the JSON data from the request
    data = request.get_json()

    # Extract year and month from the JSON data
    year = data.get('year')
    month = data.get('month')

    if year is None or month is None:
        return jsonify({'error': 'Invalid input'}), 400

    # Convert the year and month to a datetime object
    try:
        date = datetime(year, month, 1)
    except ValueError:
        return jsonify({'error': 'Invalid date'}), 400

    # Convert the date to ordinal for prediction
    month_numeric = date.toordinal()

    # Make the prediction
    prediction = model.predict([[month_numeric]])[0]

    # Return the prediction as a JSON response
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
