# app.py
from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64
import warnings
from groq import Groq
from sklearn.metrics import classification_report, accuracy_score
import markdown
warnings.filterwarnings("ignore", category=UserWarning)
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


app = Flask(__name__)
GROQ_CLIENT = Groq(api_key=os.getenv("GROQ_KEY"))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, '.', 'data', 'cleaned_sleep_data.csv')
MODEL_PATH = os.path.join(BASE_DIR, '.', 'models', 'lgbm_sleep_model.pkl')



df = pd.read_csv(CSV_PATH)
model = joblib.load(MODEL_PATH)
#df = pd.read_csv('data/cleaned_sleep_data.csv')

#model = joblib.load('models/xgboost_sleep_model.pkl')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET'])
def show_prediction_form():
    return render_template('predict.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        form = request.form

        user_input = [
            int(form['gender']),
            int(form['age']),
            int(form['occupation']),
            float(form['sleep_duration']),
            int(form['physical_activity']),
            int(form['stress_level']),
            int(form['bmi_category']),
            int(form['heart_rate']),
            int(form['daily_steps']),
            int(form['sleep_disorder']),
            int(form['systolic_bp']),
            int(form['diastolic_bp'])
        ]

        prediction = model.predict([user_input])[0] + 4  # score from 4–9

        occupation_map = {
            0: "Accountant", 1: "Doctor", 2: "Engineer", 3: "Lawyer",
            4: "Manager", 5: "Nurse", 6: "Sales Representative", 7: "Salesperson",
            8: "Scientist", 9: "Software Engineer", 10: "Teacher"
        }

        bmi_map = {
            0: "Normal", 1: "Underweight", 2: "Obese", 3: "Overweight"
        }

        disorder_map = {
            0: "Insomnia", 1: "Sleep Apnea", 2: "None"
        }

        user_input_dict = {
            "Age": int(form['age']),
            "Gender": int(form['gender']),
            "Occupation": occupation_map.get(int(form['occupation']), "Unknown"),
            "Sleep Duration": float(form['sleep_duration']),
            "Physical Activity Level": int(form['physical_activity']),
            "Stress Level": int(form['stress_level']),
            "BMI Category": bmi_map.get(int(form['bmi_category']), "Unknown"),
            "Heart Rate": int(form['heart_rate']),
            "Daily Steps": int(form['daily_steps']),
            "Sleep Disorder": disorder_map.get(int(form['sleep_disorder']), "Unknown"),
            "Blood Pressure": f"{form['systolic_bp']}/{form['diastolic_bp']}"
        }

        ai_feedback = sleep_doc_groq_feedback(user_input_dict, prediction)
        if not ai_feedback:
            ai_feedback = get_default_feedback(prediction)

        return render_template('result.html', prediction=prediction, ai_feedback=ai_feedback)

    except Exception as e:
        return f"Error: {e}"


@app.route('/dashboard')
def dashboard():
    # Correlation heatmap
    corr = df.corr(numeric_only=True)
    fig1 = plt.figure(figsize=(8,6))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    heatmap_img = plot_to_img(fig1)
    plt.close(fig1)

    # Stress vs Sleep Quality
    fig2 = plt.figure()
    sns.regplot(x='Stress Level', y='Quality of Sleep', data=df)
    stress_img = plot_to_img(fig2)
    plt.close(fig2)

    # Heart Rate vs Sleep Quality
    fig3 = plt.figure()
    sns.regplot(x='Heart Rate', y='Quality of Sleep', data=df)
    heart_img = plot_to_img(fig3)
    plt.close(fig3)

    return render_template('dashboard.html',
                           heatmap_img=heatmap_img,
                           stress_img=stress_img,
                           heart_img=heart_img)


@app.route('/filters', methods=['GET', 'POST'])
def filters():
    occupation_map = {
        0: "Accountant",
        1: "Doctor",
        2: "Engineer",
        3: "Lawyer",
        4: "Manager",
        5: "Nurse",
        6: "Sales Representative",
        7: "Salesperson",
        8: "Scientist",
        9: "Software Engineer",
        10: "Teacher"
    }

    filtered_df = df.copy()
    occupations = sorted([(key, val) for key, val in occupation_map.items()], key=lambda x: x[1])
    stress_levels = sorted(df['Stress Level'].unique())

    selected_occupation = request.form.get('occupation')
    selected_stress = request.form.get('stress')
    selected_sleep_min = request.form.get('sleep_min')
    selected_sleep_max = request.form.get('sleep_max')

    # Apply filters
    if request.method == 'POST':
        if selected_occupation and selected_occupation != 'All':
            filtered_df = filtered_df[filtered_df['Occupation'] == int(selected_occupation)]

        if selected_stress and selected_stress != 'All':
            filtered_df = filtered_df[filtered_df['Stress Level'] == int(selected_stress)]

        if selected_sleep_min and selected_sleep_max:
            filtered_df = filtered_df[
                (filtered_df['Sleep Duration'] >= float(selected_sleep_min)) &
                (filtered_df['Sleep Duration'] <= float(selected_sleep_max))
                ]

    # Metrics
    avg_quality = round(filtered_df['Quality of Sleep'].mean(), 2) if not filtered_df.empty else 'N/A'
    avg_hr = round(filtered_df['Heart Rate'].mean(), 2) if not filtered_df.empty else 'N/A'
    avg_steps = round(filtered_df['Daily Steps'].mean(), 2) if not filtered_df.empty else 'N/A'

    # Visual
    fig = plt.figure()
    sns.histplot(filtered_df['Quality of Sleep'], bins=6, kde=True)
    plt.title("Sleep Quality Distribution")
    plot_img = plot_to_img(fig)
    plt.close(fig)

    return render_template(
        'filters.html',
        occupations=occupations,
        stress_levels=stress_levels,
        plot_img=plot_img,
        avg_quality=avg_quality,
        avg_hr=avg_hr,
        avg_steps=avg_steps
    )


def plot_to_img(fig):
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png')
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return encoded


def sleep_doc_groq_feedback(user_input, sleep_quality_score):
    prompt = f"""
    A user has received a sleep quality score of {sleep_quality_score:.1f}/10.

    Their info:
    - Age: {user_input['Age']}
    - Gender: {"Male" if user_input['Gender'] == 1 else "Female"}
    - Occupation: {user_input['Occupation']}
    - Sleep Duration: {user_input['Sleep Duration']} hours
    - Physical Activity Level: {user_input['Physical Activity Level']} min/day
    - Stress Level: {user_input['Stress Level']}/10
    - BMI Category: {user_input['BMI Category']}
    - Blood Pressure: {user_input['Blood Pressure']}
    - Heart Rate: {user_input['Heart Rate']} bpm
    - Daily Steps: {user_input['Daily Steps']}
    - Sleep Disorder: {user_input['Sleep Disorder']}

    Give the user a personalized and friendly message based on these inputs.
    Suggest 2–3 simple and relevant changes they can try, and offer support or encouragement.
    The tone should be friendly and not alarming, but still based on modern sleep medicine.
    The message should be brief and not overly wordy. The message should not include a signature handle of any kind. 
    """

    messages = [
        {
            "role": "system",
            "content": "You are a compassionate AI born sleep coach and clinician called SleepDocGroq who is helping users optimize their sleep habits with encouragement and medically accurate insights."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    client = GROQ_CLIENT

    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=messages,
            temperature=1.0,
            max_tokens=1000
        )
        return markdown.markdown(response.choices[0].message.content)
    except Exception as e:
        return None  # fallback logic will kick in

def get_default_feedback(score):
    if score >= 8:
        print("Awesome! Your sleep quality looks excellent. Keep maintaining your healthy habits!")
        return "Awesome! Your sleep quality looks excellent. Keep maintaining your healthy habits!"
    elif score >= 6:
        print("Not bad! There's room for improvement. Try being consistent with your bedtime and avoid screens before bed.")
        return "Not bad! There's room for improvement. Try being consistent with your bedtime and avoid screens before bed."
    elif score >= 4:
        print("Your sleep quality could use a boost. Watch out for stress, and make time to unwind before bed.")
        return "Your sleep quality could use a boost. Watch out for stress, and make time to unwind before bed."
    else:
        print("Your sleep score is low. It might help to reduce stimulants, improve your sleep routine, or consult a sleep specialist.")
        return "Your sleep score is low. It might help to reduce stimulants, improve your sleep routine, or consult a sleep specialist."



##run this in cli to get accuracy scores
#python -c "from app import evaluate_model; evaluate_model()"
def evaluate_model():

    # Define features and label
    features = [
        'Gender', 'Age', 'Occupation', 'Sleep Duration', 'Physical Activity Level',
        'Stress Level', 'BMI Category', 'Heart Rate', 'Daily Steps',
        'Sleep Disorder', 'Systolic BP', 'Diastolic BP'
    ]
    label = 'Quality of Sleep'

    x = df[features]
    y = df[label] - 4  # reverse shift if needed (model was trained on 0–5)


    # Predict
    y_pred = model.predict(x)

    # Report
    print("Accuracy:", accuracy_score(y, y_pred))
    print(classification_report(y, y_pred))


if __name__ == '__main__':
    #print("WARNING: This is a development server. Do not use it in a production deployment."
    #    " Use a production WSGI server instead.Running on http://127.0.0.1:5000"
    #    " 33mPress CTRL+C to quit. * Debugger PIN: 101-256-703")
    app.run(debug=True)
    #app.run()
