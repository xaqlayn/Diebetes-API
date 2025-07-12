from pydantic import BaseModel, Field

class RawData(BaseModel):
    Pregnancies: int = Field(..., ge=0, description="Number of times pregnant")
    Glucose: int = Field(..., ge=0, le=300, description="Glucose concentration")
    BloodPressure: int = Field(..., ge=0, le=200, description="Blood pressure value")
    SkinThickness: int = Field(..., ge=0, le=99, description="Skin thickness in mm")
    Insulin: int = Field(..., ge=0, le=900, description="Insulin level")
    BMI: float = Field(..., ge=10, le=80, description="Body Mass Index")
    DiabetesPedigreeFunction: float = Field(..., ge=0, le=2.5, description="DPF value")
    Age: int = Field(..., ge=1, le=120, description="Age in years")
