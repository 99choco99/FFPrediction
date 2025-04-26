import tensorflow as tf
from tensorflow import keras
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template
from flask_bootstrap import Bootstrap5
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired

np.random.seed(42)

# Flask 앱 초기화
app = Flask(__name__)
app.config['SECRET_KEY'] = 'hard to guess string'
bootstrap = Bootstrap5(app)

# 모델과 파이프라인 변수를 글로벌로 선언만
model = None
pipeline = None

# WTForms로 입력 폼 생성
class LabForm(FlaskForm):
    longitude = StringField('longitude(1-7)', validators=[DataRequired()])
    latitude = StringField('latitude(1-7)', validators=[DataRequired()])
    month = StringField('month(01-Jan ~ Dec-12)', validators=[DataRequired()])
    day = StringField('day(00-sun ~ 06-sat, 07-hol)', validators=[DataRequired()])
    avg_temp = StringField('avg_temp', validators=[DataRequired()])
    max_temp = StringField('max_temp', validators=[DataRequired()])
    max_wind_speed = StringField('max_wind_speed', validators=[DataRequired()])
    avg_wind = StringField('avg_wind', validators=[DataRequired()])
    submit = SubmitField('Submit')

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/prediction', methods=['GET', 'POST'])
def lab():
    global model, pipeline

    # 요청이 올 때 처음 한 번만 모델과 파이프라인 로드
    if model is None:
        model = keras.models.load_model("fires_model.keras")
    if pipeline is None:
        pipeline = joblib.load("pipeline.pkl")

    form = LabForm()

    if form.validate_on_submit():
        longitude = float(form.longitude.data)
        latitude = float(form.latitude.data)
        month = form.month.data
        day = form.day.data
        avg_temp = float(form.avg_temp.data)
        max_temp = float(form.max_temp.data)
        max_wind_speed = float(form.max_wind_speed.data)
        avg_wind = float(form.avg_wind.data)

        data = pd.DataFrame([{
            'longitude': longitude,
            'latitude': latitude,
            'month': month,
            'day': day,
            'avg_temp': avg_temp,
            'max_temp': max_temp,
            'max_wind_speed': max_wind_speed,
            'avg_wind': avg_wind
        }])

        X_prepared = pipeline.transform(data)
        pred_log = model.predict(X_prepared)[0][0]
        burned_area = np.exp(pred_log) - 1

        return render_template('result.html', result=round(burned_area, 2))

    return render_template('prediction.html', form=form)

if __name__ == '__main__':
    app.run(debug=True)
