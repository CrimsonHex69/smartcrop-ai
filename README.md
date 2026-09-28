# SmartCrop AI — Team Alpha Z

AI/ML crop decision-support prototype for IDEAFORGE 2.0.

## MVP flow
1. Enter soil and climate conditions.
2. Get a Random Forest crop recommendation.
3. See top recommendations and explainable context.
4. Compare leading crops.
5. Run a what-if simulation.
6. Browse the wider crop library.
7. Download a recommendation report.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

The app downloads the benchmark crop dataset on first run if it is not present locally.
