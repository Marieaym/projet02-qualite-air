# Projet 02 — Analyse Climatique & Qualité de l'Air

**Étudiante :** Marie Yahaya Abdou  
**Module :** Analyse de Données Avancée avec Python — ADU 2025–2026

## Description
Analyse des données de qualité de l'air de Milan (2004–2005) à partir du dataset UCI Air Quality, enrichi des données ERA5 (Open-Meteo). Le projet couvre l'EDA, la détection des sources de pollution, un système d'alerte préventive et une prédiction ML à 6h.

## Résultats clés
- Trafic routier identifié comme source dominante (r CO/NOx = 0.79)
- 87 jours/an dépassant le seuil OMS
- Random Forest R² = 0.403 à 6h (vs LR R² = 0.027)
- Système d'alerte : rappel 74% à 3h d'avance

## Lancer le dashboard
```bash
pip install -r requirements.txt
cd dashboard
streamlit run app_v7.py
```

## Structure
- `data/raw/` : données brutes UCI
- `notebooks/` : notebook d'analyse complet
- `dashboard/` : application Streamlit interactive
- `rapport/` : rapport écrit et présentation