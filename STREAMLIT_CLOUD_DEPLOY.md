# Deploy Streamlit UI + API Call Verification

## 1. Push Project to GitHub

1. Create a GitHub repo and push this project.
2. Confirm these files are present:

- `app.py`
- `requirements.txt`
- `models/fruit_mobilenetv2.keras`
- `models/labels.json`

## 2. Deploy Streamlit App (Frontend)

1. Open Streamlit Community Cloud.
2. Click **New app**.
3. Select your repo and branch.
4. Set main file path to `app.py`.
5. Deploy.

## 3. Deploy FastAPI Backend (for API mode)

Important: `http://127.0.0.1:8000` only works on your local PC.
For Streamlit Cloud, your API must be public (Render/Railway/Fly.io/your VPS).

Suggested backend start command:

```bash
uvicorn api_service:app --host 0.0.0.0 --port 8000
```

After backend deploy, copy your public API URL, for example:

```text
https://fruit-api-example.onrender.com
```

## 4. Connect Streamlit to API

Use one of these methods:

### Method A: Streamlit Secrets (recommended)

In Streamlit Cloud app settings, add:

```toml
API_URL = "https://fruit-api-example.onrender.com"
```

The app reads this automatically as default API URL.

### Method B: Manual URL in UI

1. In the Streamlit app sidebar, enable **Use API backend**.
2. Paste the API URL in **API URL**.

## 5. Verify API Calls End-to-End

1. In sidebar, enable **Use API backend**.
2. Click **Test API connection**.
3. Expect a success message containing:

- `status=ok`
- `model_loaded=True`
- class list

4. Upload a fruit image and run prediction.
5. Confirm prediction result + segmented overlay are returned.

## 6. Troubleshooting

- API health check fails:
  - Verify API URL is reachable from browser.
  - Confirm `/health` endpoint works.
- CORS errors in browser:
  - Keep CORS enabled in `api_service.py`.
- Slow first prediction:
  - Model may be loading on first request.
- Streamlit deploy fails due model size:
  - Store model in repo LFS or a model download step in startup.

## 7. Local Commands (quick test)

Frontend:

```bash
python -m streamlit run app.py --server.port 8501
```

Backend:

```bash
python api_service.py
```

Health check:

```bash
python -c "import requests; print(requests.get('http://127.0.0.1:8000/health', timeout=20).json())"
```
