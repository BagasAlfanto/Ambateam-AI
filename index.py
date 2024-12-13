from flask import Flask, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from scipy.stats import randint

# Read the dataset from a xlsx file
df = pd.read_excel('data_output.xlsx')

# Split the dataset into features and target
X = df.drop(columns=['understood'])
y = df['understood']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with a StandardScaler and a RandomForestRegressor
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestRegressor())
])

# Define the hyperparameter grid for the RandomizedSearchCV
param_dist = {
    'model__n_estimators': randint(50, 200),
    'model__max_depth': randint(10, 50)
}

# Create a RandomizedSearchCV object
search = RandomizedSearchCV(pipeline, param_dist, n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)

# Fit the RandomizedSearchCV object to the training data
search.fit(X_train, y_train)

# Get the best model from the RandomizedSearchCV
best_model = search.best_estimator_

# Make predictions on the testing set
y_pred = best_model.predict(X_test)

# Calculate the mean squared error of the model
mse = mean_squared_error(y_test, y_pred)

# Print the mean squared error of the model
print(f'Mean Squared Error: {mse}')

# Create a Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the AI Model API!"

@app.route('/predict', methods=['POST'])
def predict():
    # Get the JSON data from the request
    data = request.get_json()

    # Parse the JSON data into a DataFrame with allowed columns
    allowed = ['duration', 'total_quiz', 'total_subject']
    data = {k: data[k] for k in allowed}

    # Check if any data is missing
    for key, value in data.items():
        if value is None:
            return jsonify({'error': f'Missing value for {key}'}), 400

    # Create a DataFrame from the JSON data
    df = pd.DataFrame(data, index=[0])

    # Make a prediction using the best model
    prediction = best_model.predict(df)

    # Create feedback based on the prediction
    feedback = ''

    # calculate the feedback each value 0,1 0,3 0,5 0,7 and above 0,9
    if prediction[0] < 0.3:
        feedback = "Keep pushing! Every step forward is progress. You've got this!"
    elif prediction[0] < 0.5:
        feedback = "You're making progress! Just a little more focus and effort, and you'll see amazing results!"
    elif prediction[0] < 0.7:
        feedback = "You're doing amazing! Keep building on what you've learned, and you'll hit your goals!"
    elif prediction[0] < 0.9:
        feedback = "You're so close to perfection! Keep going—you're almost there!"
    else:
        feedback = "Outstanding! You've nailed it—time to set even bigger goals!"

    # Create a response JSON
    response = {
        'prediction': prediction[0],
        'feedback': feedback
    }

    # Return the response JSON
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
