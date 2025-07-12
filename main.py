from fastapi import FastAPI
from pydantic import BaseModel
import pickle

from starlette.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model input schema
class RawData(BaseModel):
    Pregnancies: int
    Glucose: int
    BloodPressure: int
    SkinThickness: int
    Insulin: int
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

# Load the ML model
with open('diabetes_model.sav', 'rb') as file:
    model = pickle.load(file)

@app.get("/")
def read_root():
    return {"message": "FastAPI is running!"}

@app.post("/diabetes-predict")
async def predict(data: RawData):
    input_list = [
        data.Pregnancies,
        data.Glucose,
        data.BloodPressure,
        data.SkinThickness,
        data.Insulin,
        data.BMI,
        data.DiabetesPedigreeFunction,
        data.Age,
    ]
    prediction = model.predict([input_list])
    result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"
    return {"prediction": result}
