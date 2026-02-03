# Diabetes Prediction API

A FastAPI-based machine learning API to predict diabetes risk from clinical features.

Note: The repository was originally named "Diebetes-API" (typo). This project is the Diabetes Prediction API — consider renaming the repo for clarity. This branch fixes docs and modernizes the API surface while keeping the original endpoint for compatibility.

Contents
- Features
- Tech stack
- Quickstart
- API (modernized spec + compatibility)
- Web UI
- Development & testing
- Deployment notes
- Migration & compatibility
- Contributing & License

Features
- Versioned, modern API under `/api/v1/*` with clear request/response schemas.
- Backwards-compatible legacy endpoint `/diabetes-predict` (PascalCase input).
- Health & metadata endpoint: `/api/v1/health`.
- Interactive modern demo UI served at `/` and `/web`.
- OpenAPI docs automatically at `/docs` and `/redoc`.

Tech stack
- Python 3.9 (runtime.txt specifies python-3.9.6)
- FastAPI
- pydantic
- scikit-learn (model file saved as `diabetes_model.sav`)
- Uvicorn for ASGI server

Quickstart (local)
1. Clone
   ```bash
   git clone https://github.com/xaqlayn/Diebetes-API.git
   cd Diebetes-API
   ```

2. Create a virtual environment and install dependencies
   ```bash
   python3.9 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
   If you don't have an up-to-date requirements.txt, install:
   ```bash
   pip install fastapi uvicorn scikit-learn==1.2.2 pydantic
   ```

3. Place your trained model
   - Put your trained scikit-learn model at the repository root named `diabetes_model.sav`, or set the env var `MODEL_PATH` to the model location.
   - Note: serverless platforms (like Vercel) have file-size limits — keep the model <50MB or host it externally.

4. Run the app
   ```bash
   chmod +x start.sh
   ./start.sh
   # or
   uvicorn main:app --host 0.0.0.0 --port 10000 --reload
   ```

Open:
- Demo UI: http://localhost:10000/
- Swagger UI: http://localhost:10000/docs
- ReDoc: http://localhost:10000/redoc

Modernized API (recommended)
- Base path: `/api/v1`
- Preferred JSON keys are snake_case (conventional for JSON APIs)
- Consistent response envelope and use of HTTP status codes

POST /api/v1/predict
- Request (application/json)
```json
{
  "pregnancies": 1,
  "glucose": 130,
  "blood_pressure": 82,
  "skin_thickness": 20,
  "insulin": 88,
  "bmi": 32.5,
  "diabetes_pedigree_function": 0.5,
  "age": 50
}
```

- Response 200
```json
{
  "model_version": "v1.0.0",
  "prediction": "diabetic",
  "probability": 0.78,
  "threshold": 0.5,
  "raw_prediction": 1
}
```

GET /api/v1/health
- Response 200
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_version": "v1.0.0"
}
```

Compatibility
- Legacy endpoint `/diabetes-predict` is still available and accepts PascalCase keys (Pregnancies, Glucose, ...). Migrate clients to `/api/v1/predict` and snake_case keys; keep legacy endpoint for the migration period.

Examples
- cURL (modern endpoint)
```bash
curl -X POST http://localhost:10000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"pregnancies":1,"glucose":130,"blood_pressure":82,"skin_thickness":20,"insulin":88,"bmi":32.5,"diabetes_pedigree_function":0.5,"age":50}'
```

- Legacy cURL
```bash
curl -X POST http://localhost:10000/diabetes-predict \
  -H "Content-Type: application/json" \
  -d '{"Pregnancies":1,"Glucose":130,"BloodPressure":82,"SkinThickness":20,"Insulin":88,"BMI":32.5,"DiabetesPedigreeFunction":0.5,"Age":50}'
```

Deployment notes
- Vercel serverless limits model file sizes — consider external model hosting (S3, GCS) if >50MB.
- Use environment variables (`MODEL_PATH`, `MODEL_VERSION`, `PORT`, `LOG_LEVEL`) for configuration in production.
- For production traffic, run multiple workers (Gunicorn + Uvicorn workers) or containerize with Docker.

Migration & deprecation plan (suggested)
1. Add /api/v1/predict client support and deploy.
2. Keep `/diabetes-predict` for 1–2 releases while tracking usage.
3. Announce deprecation and remove legacy endpoint after migration window.

Contributing
- Fork → branch → PR.
- Include tests for new functionality.
- Update docs when you change endpoints.



Contact
- Maintainer: xaqlayn
- Open an issue or PR in this repository for questions or changes.
