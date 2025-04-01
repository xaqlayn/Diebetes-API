from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import json

from starlette.middleware.cors import CORSMiddleware

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],

)

# model input
class raw_data(BaseModel):
    Pregnancies: int
    Glucose: int
    BloodPressure: int
    SkinThickness: int
    Insulin: int
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

# load model
with open('diabetes_model.sav', 'rb') as file:
    model = pickle.load(file)

@app.post('/Diabetes predict')
async def predict(data: raw_data):
    input_data = data.json()
    input_dict = json.loads(input_data)
    prediction = model.predict([[input_dict['Pregnancies'], input_dict['Glucose'], input_dict['BloodPressure'], input_dict['SkinThickness'], input_dict['Insulin'], input_dict['BMI'], input_dict['DiabetesPedigreeFunction'], input_dict['Age']]])
    if prediction[0] == 1:
        prediction = 'Diabetic'
    else:
        prediction = 'Not Diabetic'

    return {
        'prediction': prediction
    }

