from flask import Flask, render_template, request
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)

# Load model and data
model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))
car = pd.read_csv('Cleaned_Car_data.csv')

@app.route('/', methods=['GET'])
def index():
    companies = sorted(car['company'].unique())
    years = sorted(car['year'].unique(), reverse=True)
    fuel_types = sorted(car['fuel_type'].unique())

    companies.insert(0, 'Select Company')

    # Company → Models mapping
    company_model_dict = {}
    for company in car['company'].unique():
        company_model_dict[company] = sorted(
            car[car['company'] == company]['name'].unique()
        )

    return render_template(
        'index.html',
        companies=companies,
        years=years,
        fuel_types=fuel_types,
        company_model_dict=company_model_dict
    )

@app.route('/predict', methods=['POST'])
def predict():
    company = request.form.get('company')
    car_model = request.form.get('car_models')
    year = int(request.form.get('year'))
    fuel_type = request.form.get('fuel_type')
    kms_driven = int(request.form.get('kilo_driven'))

    input_df = pd.DataFrame(
        [[car_model, company, year, kms_driven, fuel_type]],
        columns=['name', 'company', 'year', 'kms_driven', 'fuel_type']
    )

    prediction = model.predict(input_df)

    return str(round(prediction[0], 2))

if __name__ == '__main__':
    app.run(debug=True)
