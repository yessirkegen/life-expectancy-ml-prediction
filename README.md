# Life Expectancy – GitHub Pages Demo

Static demo of a country-level ML model (WHO + World Bank).  
User picks a country & lifestyle scenario → we show **model-only** prediction (no backend).

## How it works
- We trained a RandomForest pipeline in Python (`models/best_rf.joblib`).
- `export_frontend_data.py` runs offline inference for **2019** for all countries and **all** scenario combinations (smoking, alcohol, activity, pollution) and saves:
  - `site/data/preds_2019.json` – predictions
  - `site/data/countries.json` – ISO3 → country name

The static site (`site/index.html`) loads those JSONs and renders the UI.

## Build / Run
```bash
# 1) generate JSONs from your trained model
python export_frontend_data.py

# 2) commit and push; enable GitHub Pages:
#    Settings → Pages → Source: Deploy from branch
#    Branch: main, Folder: /site
