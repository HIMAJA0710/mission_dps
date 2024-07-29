# DPD Challenge Mission 1

This repository contains a Streamlit application for analyzing and forecasting traffic accident data. The application reads a CSV file, preprocesses the data, performs linear regression to forecast future values, and displays the results using interactive visualizations.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Model](#model)
- [Endpoints](#endpoints)
- [Visualizations](#visualizations)
- [Contributing](#contributing)
- [License](#license)

## Installation

To set up and run this project locally, follow these steps:

1. Clone the repository:
    ```sh
    git clone https://github.com/HIMAJA0710/mission_dps.git
    cd dpd-challenge-mission1
    ```

2. Create a virtual environment:
    ```sh
    python -m venv venv
    ```

3. Activate the virtual environment:

    - On Windows:
      ```sh
      venv\Scripts\activate
      ```

    - On macOS and Linux:
      ```sh
      source venv/bin/activate
      ```

4. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### Streamlit Application

1. Ensure you have your data file located at `C:\Users\amide\OneDrive\Documents\GitHub\mission_dps\monatszahlen2405_verkehrsunfaelle_export_31_05_24_r.csv`.

2. Run the Streamlit application:
    ```sh
    streamlit run app.py
    ```

3. Open your browser and navigate to `http://localhost:8501` to view the application.

### Flask API for Predictions

1. Start the Flask API:
    ```sh
    python app.py
    ```

2. The API will be available at `http://127.0.0.1:5000`. You can test the prediction endpoint by sending a POST request to `http://127.0.0.1:5000/predict` with a JSON body:
    ```json
    {
        "year": 2020,
        "month": 10
    }
    ```

3. The response will be in the following format:
    ```json
    {
        "prediction": <float>
    }
    ```

## Data

The data used in this project is traffic accident data, stored in a CSV file. The file is preprocessed to include only relevant columns and records up to the year 2020.

### Preprocessing Steps:
- Load the initial data.
- Drop records after the year 2020.
- Select relevant columns.
- Convert 'MONAT' column to datetime.
- Handle missing values and sort the data by date.

## Model

A linear regression model is used to forecast the number of traffic accidents. The model is trained on the preprocessed data, specifically for 'Alkoholunfälle' (alcohol-related accidents).

### Training Steps:
- Filter the data for 'Alkoholunfälle' and 'insgesamt' (total).
- Convert dates to ordinal values.
- Split the data into training and testing sets.
- Train the linear regression model and calculate the mean squared error.

## Endpoints

### `/predict` (POST)

Predict the number of traffic accidents for a given year and month.

#### Request
- JSON body:
    ```json
    {
        "year": <int>,
        "month": <int>
    }
    ```

#### Response
- JSON body:
    ```json
    {
        "prediction": <float>
    }
    ```

## Visualizations

The application includes several visualizations to display the data and model predictions:

- Historical number of accidents per category.
- Processed data.
- Model predictions compared with actual values.

### Example Visualizations:

[Historical Data](historial no.of accidents.png)
[Model Prediction](model prediction.png)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.
