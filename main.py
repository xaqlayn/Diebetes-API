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

@app.get("/web", response_class=HTMLResponse)
def web_form():
    # Interactive HTML form with JS to show result on page
    return """
    <html>
    <head>
        <title>Diabetes Prediction</title>
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
            document.getElementById('result').innerHTML = 'Loading...';
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
        <h2>Diabetes Prediction Form</h2>
        <form onsubmit="predictDiabetes(event)">
            Pregnancies: <input type="number" name="Pregnancies" min="0" required><br>
            Glucose: <input type="number" name="Glucose" min="0" max="300" required><br>
            BloodPressure: <input type="number" name="BloodPressure" min="0" max="200" required><br>
            SkinThickness: <input type="number" name="SkinThickness" min="0" max="99" required><br>
            Insulin: <input type="number" name="Insulin" min="0" max="900" required><br>
            BMI: <input type="number" step="any" name="BMI" min="10" max="80" required><br>
            DiabetesPedigreeFunction: <input type="number" step="any" name="DiabetesPedigreeFunction" min="0" max="2.5" required><br>
            Age: <input type="number" name="Age" min="1" max="120" required><br>
            <input type="submit" value="Predict">
        </form>
        <div id="result" style="margin-top:20px;"></div>
    </body>
    </html>
    """

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

@app.get("/", tags=["Health"])
def read_root():
    """Health check endpoint."""
    return {"message": "FastAPI is running!"}

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
