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
warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)

df = pd.read_csv('data/cleaned_sleep_data.csv')

model = joblib.load('models/xgboost_sleep_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET'])
def show_prediction_form():
    return render_template('predict.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        user_input = [
            int(request.form['gender']),
            int(request.form['age']),
            int(request.form['occupation']),
            float(request.form['sleep_duration']),
            int(request.form['physical_activity']),
            int(request.form['stress_level']),
            int(request.form['bmi_category']),
            int(request.form['heart_rate']),
            int(request.form['daily_steps']),
            int(request.form['sleep_disorder']),
            int(request.form['systolic_bp']),
            int(request.form['diastolic_bp'])
        ]

        prediction = model.predict([user_input])[0] + 4  # shift from 0–5 back to 4–9 range
        return render_template('result.html', prediction=prediction)
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
    filtered_df = df.copy()
    occupations = sorted(df['Occupation'].unique())
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


if __name__ == '__main__':
    app.run(debug=True)
