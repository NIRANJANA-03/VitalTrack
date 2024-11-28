from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load your trained model and scaler
#model = joblib.load('models/random_forest_model.joblib')
model = joblib.load('models/knn_model.joblib')
scaler = joblib.load('models/scaler.joblib')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract data from the form
        data = request.form.to_dict()

        # Create an array with zeros for all one-hot encoded features
        features = np.zeros(21)  # Adjust the size if necessary

        # Handle 'MTRANS' one-hot encoding
        if data.get('MTRANS') == 'Bike':
            features[17] = 1
        elif data.get('MTRANS') == 'Motorbike':
            features[18] = 1
        elif data.get('MTRANS') == 'Public_Transportation':
            features[19] = 1
        elif data.get('MTRANS') == 'Walking':
            features[20] = 1  # Index should be corrected based on actual size

        # Handle 'CALC' one-hot encoding
        if data.get('CAEC') == 'Frequently':
            features[14] = 1
        elif data.get('CAEC') == 'Sometimes':
            features[15] = 1
        elif data.get('CAEC') == 'no':
            features[16] = 1

        # Handle 'Gender' one-hot encoding
        if data.get('Gender') == 'Male':
            features[13] = 1
        elif data.get('Gender') == 'Female':
            features[13] = 0
        # Note: Assuming that 'Female' is the default case and doesn't need explicit encoding

        # Convert other features
        features[0] = int(data.get('Age', 0))
        features[1] = float(data.get('Height', 0)) / 100 
        features[2] = int(data.get('Weight', 0))
        features[3] = 1 if data.get('family_history_with_overweight') == 'yes' else 0
        features[4] = 1 if data.get('FAVC') == 'yes' else 0
        features[5] = int(data.get('FCVC', 0))
        features[6] = int(data.get('NCP', 0))
        features[7] = 1 if data.get('SMOKE') == 'yes' else 0
        features[8] = float(data.get('CH2O', 0.0))
        features[9] = 1 if data.get('SCC') == 'yes' else 0
        features[10] = int(data.get('FAF', 0))
        features[11] = int(data.get('TUE', 0))
        
        if data.get('CALC') == 'Frequently':
            features[12] = 2
        elif data.get('CALC') == 'Sometimes':
            features[12] = 1
        elif data.get('CALC') == 'no':
            features[12] = 0

        # features_array = np.array(features).reshape(1, -1)

        # Scale the features
        features_array = np.array(features).reshape(1, -1)

        # Scale the features
        scaled_features = scaler.transform(features_array)

        # Make prediction
        prediction = model.predict(scaled_features)
        class_labels = {
            0: "Insufficient_Weight",
            1: "Normal_Weight",
            2: "Overweight_Level_I",
            3:"Overweight_Level_II",
            4:  "Obesity_Type_I",
            5: "Obesity_Type_II",
            6: "Obesity_Type_III"
        }

        advice = {
            "Normal_Weight": ("Keep up the good work! Maintain a balanced diet and regular exercise.", "You’re doing great! Keep focusing on a healthy lifestyle."),
            "Overweight_Level_I": ("Consider starting a regular exercise routine and consulting with a nutritionist.", "Small steps can lead to big changes. You’ve got this!"),
            "Overweight_Level_II": ("It's important to consult with a healthcare provider for personalized advice.", "Believe in yourself. Positive changes are possible with determination."),
            "Obesity_Type_I": ("Seek advice from a healthcare professional and consider a structured weight management plan.", "You have the power to make positive changes. Start with small, achievable goals."),
            "Insufficient_Weight": ("Consult with a doctor to ensure there are no underlying health issues and consider a balanced diet plan.", "Your health is important. Focus on gaining weight in a healthy way."),
            "Obesity_Type_II": ("Consult with a healthcare provider to develop a comprehensive weight management strategy.", "Remember, every effort counts. You can achieve your goals with perseverance."),
            "Obesity_Type_III": ("Immediate consultation with a healthcare provider is crucial. Develop a comprehensive health plan.", "Believe in yourself. With the right support and effort, positive change is achievable.")
        }

        # Get the predicted class label
        predicted_class = int(prediction[0])
        predicted_label = class_labels.get(predicted_class, "Unknown")
        advice_text, positive_message = advice.get(predicted_label, ("No advice available.", "Keep striving for improvement."))

        # Render result.html with the prediction, advice, and positive message
        return render_template('result.html', 
                               predicted_class=predicted_label,
                               advice=advice_text,
                               positive_message=positive_message)
        
        
    except Exception as e:
        return jsonify({'error': str(e)})




if __name__ == '__main__':
    app.run(debug=True)
