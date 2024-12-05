# app.py
from flask import Flask, render_template, request #for taking i/p data 
import pickle #This module is used to load the machine learning model
import pandas as pd

app = Flask(__name__)

# Load the trained model
model_path = 'model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)  

# Define the route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define the route for the prediction page
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Retrieve the form data
        age = int(request.form['age'])
        restbp = float(request.form['restbp'])
        chol = float(request.form['chol'])
        maxhr = float(request.form['maxhr'])
        oldpeak = float(request.form['oldpeak'])
        slope = int(request.form['slope'])
        ca = int(request.form['ca'])
        thal = request.form['thal']

        # Create a DataFrame for model prediction
        input_data = pd.DataFrame([[age, restbp, chol, maxhr, oldpeak, slope, ca, thal]],
                                  columns=['Age', 'RestBP', 'Chol', 'MaxHR', 'Oldpeak', 'Slope', 'Ca', 'Thal'])

        # Encode the input data if necessary
        input_data = pd.get_dummies(input_data, columns=['Thal', 'Slope'], drop_first=True)

        # Align columns to match the model's input
        model_columns = list(model.feature_names_in_)
        input_data = input_data.reindex(columns=model_columns, fill_value=0)

        # Make prediction using the trained model
        prediction_probability = model.predict_proba(input_data)[0][1] * 100  # Probability of heart disease
        if prediction_probability < 40:
            result = 'Heart is Healthy!'
            remedies = None
        else:
            result = f'{prediction_probability:.2f}% likelihood of Heart Disease.'
            remedies = get_remedies(prediction_probability)  # Get remedies based on probability

        return render_template('predict.html', prediction_text=f'Result: {result}', remedies=remedies)

    return render_template('predict.html')
    #  return jsonify({
    #         'prediction_text': result,
    #         'remedies': remedies
    #     })

# Function to get remedies based on prediction percentage
def get_remedies(probability):
    if 40 <= probability < 50:
        return ["Regular exercise", "Healthy diet", "Routine check-ups"]
    elif 50 <= probability < 60:
        return ["Consult a cardiologist", "Include heart-healthy foods", "Monitor blood pressure"]
    elif 60 <= probability < 70:
        return ["Medication as prescribed", "Increase physical activity", "Limit salt intake"]
    elif 70 <= probability < 80:
        return ["Strict dietary control", "Daily monitoring", "Stress management"]
    elif 80 <= probability < 90:
        return ["Specialist care", "Cardiac rehabilitation", "Potential surgical intervention"]
    else:  # 90-100
        return ["Immediate medical attention", "Advanced treatment", "Lifestyle changes under supervision"]
if __name__ == '__main__':
    app.run(debug=True)
