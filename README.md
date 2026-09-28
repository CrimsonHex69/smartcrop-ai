# SmartCrop AI

A hackathon MVP for the problem statement: **AI-Based Crop Recommendation System for Farmers**.

## What it does
- Takes N, P, K, temperature, humidity, pH and rainfall.
- Trains a Random Forest classifier on the public Crop Recommendation benchmark dataset.
- Returns a crop recommendation and top 3 model outputs.
- Optionally uses Gemini for a simple natural-language explanation.

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app automatically attempts to download `Crop_recommendation.csv` from the public GitHub copy used for this MVP. If your environment blocks outbound requests, download the CSV manually and put it beside `app.py`.

## Gemini (optional)

The core crop predictor does not require an API key. To enable Gemini explanations, set `GEMINI_API_KEY` as an environment variable, or add this to Streamlit Community Cloud secrets:

```toml
GEMINI_API_KEY = "your_key_here"
```

## Deployment

Push `app.py`, `requirements.txt` and (optionally) `Crop_recommendation.csv` to a GitHub repository. Deploy it from Streamlit Community Cloud.

## Demo flow

1. Click **Load demo values**.
2. Click **Recommend Crop**.
3. Show the recommended crop and top-3 outputs.
4. Open **AI explanation**.
5. Open **Model information** to show dataset size, crop classes and feature importance.

## Important

This is a prototype decision-support system. Do not claim that it guarantees yield, profit, or the best crop for a real farm. The benchmark dataset does not replace local soil testing, current weather data, agronomic expertise, or market considerations.
