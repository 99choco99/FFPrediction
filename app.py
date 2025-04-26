import tensorflow as tf
from tensorflow import keras
import joblib

print("TensorFlow version:", tf.__version__)
print("Keras version:", keras.__version__)

import numpy as np
import pandas as pd
from flask import Flask, render_template

from flask_bootstrap import Bootstrap5
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
STRING_FIELD = StringField('max_wind_speed', validators=[DataRequired()])

np.random.seed(42)

# Flask 앱 초기화
app = Flask(__name__)
app.config['SECRET_KEY'] = 'hard to guess string'
bootstrap = Bootstrap5(app)

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

# 모델과 전처리 파이프라인 로드
model = keras.models.load_model("model.h5")
pipeline = joblib.load("pipeline.pkl")

# 홈(index) 페이지
@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

# /prediction 페이지
@app.route('/prediction', methods=['GET', 'POST'])
def lab():
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