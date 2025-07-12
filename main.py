from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pickle
import logging
from starlette.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

app = FastAPI(
    title="Diabetes Prediction API",
    description="Predict diabetes likelihood using medical data.",
    version="1.1.0"
)

def web_form():
    # Interactive HTML form with JS and modern CSS
    return """
    <html>
    <head>
        <title>Diabetes Prediction</title>
        <style>
            body {
                font-family: 'Segoe UI', Arial, sans-serif;
                background: #f7fafc;
                color: #222;
                margin: 0;
                padding: 0;
            }
            .container {
                max-width: 400px;
                margin: 40px auto;
                background: #fff;
                border-radius: 12px;
                box-shadow: 0 2px 16px rgba(0,0,0,0.08);
                padding: 32px 24px 24px 24px;
            }
            h2 {
                text-align: center;
                margin-bottom: 24px;
                color: #337ab7;
                font-weight: 600;
            }
            label {
                display: block;
                margin-bottom: 6px;
                font-size: 15px;
                color: #555;
            }
            input[type="number"], input[type="text"] {
                width: 100%;
                padding: 8px 10px;
                border-radius: 5px;
                border: 1px solid #d1d5db;
                margin-bottom: 16px;
                font-size: 16px;
            }
            input[type="submit"] {
                background: #337ab7;
                color: #fff;
                border: none;
                border-radius: 6px;
                padding: 10px 0;
                font-size: 18px;
                cursor: pointer;
                width: 100%;
                margin-top: 10px;
                transition: background 0.2s;
            }
            input[type="submit"]:hover {
                background: #23527c;
            }
            #result {
                text-align: center;
                margin-top: 18px;
                font-size: 18px;
                font-weight: 500;
                color: #444;
            }
        </style>
        <script>
        async function predictDiabetes(event) {
            event.preventDefault();
            const form = event.target;
            const data = {
                Pregnancies: parseInt(form.Pregnancies.value),
                Glucose: parseInt(form.Glucose.value),
                BloodPressure: parseInt(form.BloodPressure.value),
                SkinThickness: parseInt(form.SkinThickness.value),
                Insulin: parseInt(form.Insulin.value),
                BMI: parseFloat(form.BMI.value),
                DiabetesPedigreeFunction: parseFloat(form.DiabetesPedigreeFunction.value),
                Age: parseInt(form.Age.value)
            };
            document.getElementById('result').innerHTML = 'Predicting...';
            const response = await fetch('/diabetes-predict', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(data)
            });
            const result = await response.json();
            document.getElementById('result').innerHTML =
                'Prediction: <b>' + result.prediction + '</b>' +
                (result.probability !== undefined ? '<br>Probability: <b>' + result.probability + '</b>' : '');
        }
        </script>
    </head>
    <body>
        <div class="container">
            <h2>Diabetes Prediction Form</h2>
            <form onsubmit="predictDiabetes(event)">
                <label for="Pregnancies">Pregnancies</label>
                <input type="number" name="Pregnancies" min="0" required>
                <label for="Glucose">Glucose</label>
                <input type="number" name="Glucose" min="0" max="300" required>
                <label for="BloodPressure">Blood Pressure</label>
                <input type="number" name="BloodPressure" min="0" max="200" required>
                <label for="SkinThickness">Skin Thickness</label>
                <input type="number" name="SkinThickness" min="0" max="99" required>
                <label for="Insulin">Insulin</label>
                <input type="number" name="Insulin" min="0" max="900" required>
                <label for="BMI">BMI</label>
                <input type="number" step="any" name="BMI" min="10" max="80" required>
                <label for="DiabetesPedigreeFunction">Diabetes Pedigree Function</label>
                <input type="number" step="any" name="DiabetesPedigreeFunction" min="0" max="2.5" required>
                <label for="Age">Age</label>
                <input type="number" name="Age" min="1" max="120" required>
                <input type="submit" value="Predict">
            </form>
            <div id="result"></div>
        </div>
    </body>
    </html>
    """

@app.get("/", response_class=HTMLResponse)
@app.get("/web", response_class=HTMLResponse)
def home():
    return web_form()

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Improved input schema with validation
class RawData(BaseModel):
    Pregnancies: int = Field(..., ge=0, description="Number of times pregnant")
    Glucose: int = Field(..., ge=0, le=300, description="Glucose concentration")
    BloodPressure: int = Field(..., ge=0, le=200, description="Blood pressure value")
    SkinThickness: int = Field(..., ge=0, le=99, description="Skin thickness in mm")
    Insulin: int = Field(..., ge=0, le=900, description="Insulin level")
    BMI: float = Field(..., ge=10, le=80, description="Body Mass Index")
    DiabetesPedigreeFunction: float = Field(..., ge=0, le=2.5, description="DPF value")
    Age: int = Field(..., ge=1, le=120, description="Age in years")

# Setup logging
logging.basicConfig(level=logging.INFO)

# Load the ML model with error handling
try:
    with open('diabetes_model.sav', 'rb') as file:
        model = pickle.load(file)
except Exception as e:
    logging.error("Failed to load model: %s", e)
    raise RuntimeError("Model loading failed")

@app.post("/diabetes-predict", tags=["Prediction"])
async def predict(data: RawData):
    """
    Predict diabetes risk.
    Returns label and probability.
    """
    try:
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
        # Try to get probability (if model supports it)
        if hasattr(model, "predict_proba"):
            prob = float(model.predict_proba([input_list])[0][1])
            logging.info(f"Prediction: {result}, Probability: {prob:.2f}")
            return {"prediction": result, "probability": round(prob, 2)}
        else:
            logging.info(f"Prediction: {result}")
            return {"prediction": result}
    except Exception as e:
        logging.error("Prediction error: %s", e)
        raise HTTPException(status_code=500, detail="Prediction failed.")
