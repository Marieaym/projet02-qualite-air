import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import folium
from streamlit.components.v1 import html as st_html
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
import urllib.request
import json as json_module

# ─── CONFIG ───────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Qualité de l'Air — Milan 2004-2005",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── STYLES ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.main-header {
    background: #0f1923;
    padding: 2rem 2.5rem 1.5rem;
    margin: -1rem -1rem 2rem -1rem;
    border-bottom: 1px solid #1e2d3d;
}
.main-header h1 {
    font-family: 'DM Mono', monospace;
    color: #e8f4fd;
    font-size: 1.3rem;
    font-weight: 500;
    letter-spacing: 0.08em;
    margin: 0 0 0.3rem 0;
    text-transform: uppercase;
}
.main-header p {
    color: #5a7a94;
    font-size: 0.8rem;
    margin: 0;
    font-family: 'DM Mono', monospace;
    letter-spacing: 0.04em;
}

.metric-row { display: flex; gap: 1rem; margin-bottom: 2rem; }
.metric-card {
    flex: 1;
    background: #0f1923;
    border: 1px solid #1e2d3d;
    border-left: 3px solid #2563eb;
    padding: 1.2rem 1.4rem;
    border-radius: 4px;
}
.metric-card.warning { border-left-color: #d97706; }
.metric-card.danger  { border-left-color: #dc2626; }
.metric-card.success { border-left-color: #16a34a; }
.metric-value {
    font-family: 'DM Mono', monospace;
    font-size: 1.8rem;
    font-weight: 500;
    color: #e8f4fd;
    line-height: 1;
    margin-bottom: 0.3rem;
}
.metric-label {
    font-size: 0.72rem;
    color: #5a7a94;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

.section-title {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #2563eb;
    margin: 2.5rem 0 1rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1e2d3d;
}

.corr-badges { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.8rem; }
.corr-badge {
    background: #0a1520;
    border: 1px solid #1e2d3d;
    border-radius: 3px;
    padding: 0.3rem 0.7rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #7aa8c8;
}
.corr-badge span { color: #e8f4fd; }

.block-container { padding-top: 0 !important; }
</style>
""", unsafe_allow_html=True)

# ─── HEADER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>Qualité de l'Air — Milan 2004–2005</h1>
    <p>UCI Air Quality Dataset &nbsp;·&nbsp; Marie Yahaya Abdou &nbsp;·&nbsp; ADU 2025–2026 &nbsp;·&nbsp; Analyse de Données Avancée avec Python</p>
</div>
""", unsafe_allow_html=True)

# ─── SIDEBAR NAVIGATION ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:1rem 0 0.5rem 0;">
        <div style="font-family:monospace;font-size:0.7rem;color:#5a7a94;
                    text-transform:uppercase;letter-spacing:0.1em;margin-bottom:1rem;">
            Navigation
        </div>
    </div>
    """, unsafe_allow_html=True)

    sections = {
        "Vue d'ensemble": "00",
        "Distributions": "01",
        "Corrélations": "02",
        "Évolution mensuelle": "03",
        "Profil horaire": "04",
        "Saisonnalité": "05",
        "Température / NO2": "06",
        "Semaine / WE / Fériés": "07",
        "Seuils OMS": "08",
        "Système d'alerte": "09",
        "Variables ERA5": "10",
        "Machine Learning": "11",
        "Carte géospatiale": "12",
    }

    selection = st.radio(
        label="",
        options=list(sections.keys()),
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.65rem;color:#5a7a94;line-height:1.6;">
        UCI Air Quality Dataset<br>
        Milan · 9 357 mesures<br>
        Mars 2004 → Fév 2005
    </div>
    """, unsafe_allow_html=True)

# ─── ANCRES PAR SECTION ───────────────────────────────────────────────────────
def section_anchor(name):
    """Affiche une ancre invisible pour la navigation sidebar."""
    st.markdown(f'''<div id="{name}" style="margin-top:-60px;padding-top:60px;"></div>''',
                unsafe_allow_html=True)


# ─── CHARGEMENT ET NETTOYAGE ──────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('projet2.csv')
    df = df.drop(columns=['Unnamed: 15', 'Unnamed: 16'], errors='ignore')
    df['Datetime'] = pd.to_datetime(
        df['Date'] + ' ' + df['Time'],
        format='%d-%m-%Y %H:%M:%S', errors='coerce'
    )
    df = (df.dropna(subset=['Datetime'])
            .set_index('Datetime')
            .sort_index()
            .drop(columns=['Date', 'Time'], errors='ignore')
            .replace(-200, np.nan)
            .drop(columns=['NMHC(GT)'], errors='ignore')
            .interpolate(method='linear'))

    df['Heure']       = df.index.hour
    df['JourSemaine'] = df.index.dayofweek
    df['Mois_num']    = df.index.month
    df['Saison']      = pd.cut(df.index.month, bins=[0,3,6,9,12],
                               labels=['Hiver','Printemps','Été','Automne'])

    jours_feries = pd.to_datetime([
        '2004-04-25','2004-05-01','2004-06-02','2004-06-24','2004-08-15',
        '2004-11-01','2004-12-08','2004-12-25','2004-12-26',
        '2005-01-01','2005-01-06','2005-04-25'
    ])
    df['date_only'] = pd.to_datetime(df.index.date)
    df['Ferie']     = df['date_only'].isin(jours_feries)
    df['Type_Jour'] = np.where(df['Ferie'], 'Jour Férié',
                      np.where(df.index.dayofweek >= 5, 'Week-end', 'Semaine'))
    return df

@st.cache_data
def load_era5():
    try:
        url = (
            "https://archive-api.open-meteo.com/v1/archive?"
            "latitude=45.4654&longitude=9.1859"
            "&start_date=2004-03-10&end_date=2005-02-28"
            "&hourly=wind_speed_10m,wind_direction_10m,precipitation,"
            "surface_pressure,boundary_layer_height"
            "&timezone=Europe%2FRome"
        )
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=20) as r:
            raw = json_module.loads(r.read())
        df_e = pd.DataFrame(raw['hourly'])
        df_e['time'] = pd.to_datetime(df_e['time'])
        df_e = df_e.set_index('time')
        df_e.columns = ['wind_speed','wind_dir','precipitation','pressure','blh']
        df_e.index = df_e.index.tz_localize(None)
        return df_e, True
    except Exception:
        return None, False

df = load_data()
df_era5, era5_ok = load_era5()

polluants = ['CO(GT)', 'C6H6(GT)', 'NOx(GT)', 'NO2(GT)']
SEUIL_NO2 = 200

if era5_ok:
    df_merged = df.join(df_era5, how='left')
    era5_cols = ['wind_speed','wind_dir','precipitation','pressure','blh']
    df_merged[era5_cols] = df_merged[era5_cols].interpolate(method='linear')
else:
    df_merged = df.copy()

# ─── MÉTRIQUES GLOBALES ───────────────────────────────────────────────────────
heures_dep = int((df['NO2(GT)'] > SEUIL_NO2).sum())
jours_dep  = int(df['NO2(GT)'].resample('D').max().gt(SEUIL_NO2).sum())

st.markdown(f"""
<div class="metric-row">
    <div class="metric-card">
        <div class="metric-value">{df['NO2(GT)'].mean():.1f}</div>
        <div class="metric-label">NO2 moyen µg/m³</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">{df['CO(GT)'].mean():.2f}</div>
        <div class="metric-label">CO moyen mg/m³</div>
    </div>
    <div class="metric-card warning">
        <div class="metric-value">{heures_dep:,}</div>
        <div class="metric-label">Heures > seuil OMS</div>
    </div>
    <div class="metric-card danger">
        <div class="metric-value">{jours_dep}</div>
        <div class="metric-label">Jours dangereux / an</div>
    </div>
    <div class="metric-card success">
        <div class="metric-value">{len(df):,}</div>
        <div class="metric-label">Mesures horaires</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── 1. DISTRIBUTIONS ─────────────────────────────────────────────────────────
st.markdown('<div class="section-title">01 — Distribution des polluants</div>', unsafe_allow_html=True)

fig = make_subplots(rows=2, cols=2, subplot_titles=polluants)
for i, (col, coul) in enumerate(zip(polluants, ['#2563eb','#854F0B','#A32D2D','#3B6D11'])):
    fig.add_trace(go.Histogram(x=df[col].dropna(), nbinsx=30, name=col,
                               marker_color=coul, opacity=0.85), row=i//2+1, col=i%2+1)
fig.update_layout(paper_bgcolor='#0f1923', plot_bgcolor='#0a1520',
                  font_color='#7aa8c8', showlegend=False, height=420,
                  margin=dict(l=10,r=10,t=40,b=10))
fig.update_xaxes(gridcolor='#1e2d3d', zeroline=False)
fig.update_yaxes(gridcolor='#1e2d3d', zeroline=False)
st.plotly_chart(fig, use_container_width=True)

# ─── 2. CORRÉLATION ───────────────────────────────────────────────────────────
st.markdown('<div class="section-title">02 — Matrice de corrélation de Pearson</div>', unsafe_allow_html=True)

corr = df.corr(numeric_only=True).round(2)
fig2 = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', zmin=-1, zmax=1, aspect='auto')
fig2.update_layout(paper_bgcolor='#0f1923', plot_bgcolor='#0a1520',
                   font_color='#7aa8c8', height=480,
                   margin=dict(l=10,r=10,t=20,b=10),
                   coloraxis_colorbar=dict(tickcolor='#7aa8c8', title=''))
st.plotly_chart(fig2, use_container_width=True)

st.markdown("""
<div class="corr-badges">
    <div class="corr-badge">CO ↔ C6H6 &nbsp;<span>0.82</span></div>
    <div class="corr-badge">CO ↔ NOx &nbsp;<span>0.79</span></div>
    <div class="corr-badge">NOx ↔ NO2 &nbsp;<span>0.76</span></div>
    <div class="corr-badge">T ↔ NOx &nbsp;<span>−0.25</span></div>
    <div class="corr-badge">T ↔ NO2 &nbsp;<span>−0.19</span></div>
</div>
""", unsafe_allow_html=True)

# ─── 3. ÉVOLUTION MENSUELLE ───────────────────────────────────────────────────
st.markdown('<div class="section-title">03 — Évolution mensuelle</div>', unsafe_allow_html=True)

moy_mens = df[polluants].resample('ME').mean().reset_index()
fig3 = px.line(moy_mens, x='Datetime', y=polluants, markers=True)
fig3.update_layout(paper_bgcolor='#0f1923', plot_bgcolor='#0a1520',
                   font_color='#7aa8c8', height=380, hovermode='x unified', legend_title='',
                   margin=dict(l=10,r=10,t=10,b=10),
                   xaxis=dict(gridcolor='#1e2d3d', title=''),
                   yaxis=dict(gridcolor='#1e2d3d', title='Concentration moyenne'))
st.plotly_chart(fig3, use_container_width=True)

# ─── 4. PROFIL HORAIRE ────────────────────────────────────────────────────────
st.markdown('<div class="section-title">04 — Profil horaire</div>', unsafe_allow_html=True)

poll_hor = df.groupby('Heure')[polluants].mean().reset_index()
fig4 = px.line(poll_hor, x='Heure', y=polluants, markers=True)
fig4.update_layout(paper_bgcolor='#0f1923', plot_bgcolor='#0a1520',
                   font_color='#7aa8c8', height=360, hovermode='x unified', legend_title='',
                   margin=dict(l=10,r=10,t=10,b=10),
                   xaxis=dict(gridcolor='#1e2d3d', tickmode='linear', dtick=2, title='Heure'),
                   yaxis=dict(gridcolor='#1e2d3d', title='Concentration moyenne'))
st.plotly_chart(fig4, use_container_width=True)

# ─── 5. SAISONNALITÉ ──────────────────────────────────────────────────────────
st.markdown('<div class="section-title">05 — Saisonnalité</div>', unsafe_allow_html=True)

saison_moy = df.groupby('Saison', observed=True)[polluants].mean().round(2).reset_index()
fig5 = px.bar(saison_moy.melt(id_vars='Saison', var_name='Polluant', value_name='Concentration'),
              x='Saison', y='Concentration', color='Polluant', barmode='group')
fig5.update_layout(paper_bgcolor='#0f1923', plot_bgcolor='#0a1520',
                   font_color='#7aa8c8', height=360, hovermode='x unified', legend_title='',
                   margin=dict(l=10,r=10,t=10,b=10),
                   xaxis=dict(gridcolor='#1e2d3d', title=''),
                   yaxis=dict(gridcolor='#1e2d3d', title='Concentration moyenne'))
st.plotly_chart(fig5, use_container_width=True)

# ─── 6. TEMPÉRATURE / NO2 ─────────────────────────────────────────────────────
st.markdown('<div class="section-title">06 — Température vs NO2</div>', unsafe_allow_html=True)

df_s = df[['T','NO2(GT)']].dropna().iloc[::3]
fig6 = px.scatter(df_s, x='T', y='NO2(GT)', opacity=0.35, trendline='ols')
fig6.update_traces(marker_color='#2563eb', selector=dict(mode='markers'))
fig6.update_layout(paper_bgcolor='#0f1923', plot_bgcolor='#0a1520',
                   font_color='#7aa8c8', height=380, margin=dict(l=10,r=10,t=10,b=10),
                   xaxis=dict(gridcolor='#1e2d3d', title='Température (°C)'),
                   yaxis=dict(gridcolor='#1e2d3d', title='NO2 (µg/m³)'))
st.plotly_chart(fig6, use_container_width=True)

# ─── 7. SEMAINE / WEEK-END / FÉRIÉS ──────────────────────────────────────────
st.markdown('<div class="section-title">07 — Semaine / Week-end / Jours fériés</div>', unsafe_allow_html=True)

moy_type = df.groupby('Type_Jour')[['NO2(GT)']].mean().round(1)
col1, col2, col3 = st.columns(3)
for col_st, (tjour, color) in zip([col1,col2,col3],
    [('Semaine','#2563eb'),('Week-end','#3B6D11'),('Jour Férié','#854F0B')]):
    val = moy_type.loc[tjour,'NO2(GT)'] if tjour in moy_type.index else 0
    col_st.markdown(f"""<div class="metric-card" style="border-left-color:{color};margin-bottom:1rem;">
        <div class="metric-value">{val}</div>
        <div class="metric-label">NO2 moy. — {tjour}</div>
    </div>""", unsafe_allow_html=True)

fig7, ax7 = plt.subplots(figsize=(8,4))
fig7.patch.set_facecolor('#0f1923')
ax7.set_facecolor('#0a1520')
sns.boxplot(data=df, x='Type_Jour', y='NO2(GT)',
            order=['Semaine','Week-end','Jour Férié'],
            palette={'Semaine':'#2563eb','Week-end':'#3B6D11','Jour Férié':'#854F0B'}, ax=ax7)
ax7.set_xlabel('')
ax7.set_ylabel('NO2 (µg/m³)', color='#7aa8c8')
ax7.tick_params(colors='#7aa8c8')
for spine in ax7.spines.values(): spine.set_edgecolor('#1e2d3d')
ax7.grid(axis='y', color='#1e2d3d', linewidth=0.5)
plt.tight_layout()
st.pyplot(fig7)
plt.close()

# ─── 8. SEUILS OMS ────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">08 — Seuils OMS — Jours dangereux</div>', unsafe_allow_html=True)

no2_daily = df['NO2(GT)'].resample('D').max().reset_index()
no2_daily.columns = ['Date','NO2_max']
no2_daily['Alerte'] = no2_daily['NO2_max'] > SEUIL_NO2
fig8 = px.bar(no2_daily, x='Date', y='NO2_max', color='Alerte',
              color_discrete_map={True:'#dc2626', False:'#16a34a'})
fig8.add_hline(y=SEUIL_NO2, line_dash='dash', line_color='#7aa8c8',
               annotation_text='Seuil OMS 200 µg/m³', annotation_font_color='#7aa8c8')
fig8.update_layout(paper_bgcolor='#0f1923', plot_bgcolor='#0a1520',
                   font_color='#7aa8c8', height=380, showlegend=False,
                   margin=dict(l=10,r=10,t=10,b=10),
                   xaxis=dict(gridcolor='#1e2d3d', title=''),
                   yaxis=dict(gridcolor='#1e2d3d', title='NO2 max journalier (µg/m³)'))
st.plotly_chart(fig8, use_container_width=True)

# ─── 9. SYSTÈME D'ALERTE ──────────────────────────────────────────────────────
st.markdown("<div class='section-title'>09 — Système d'alerte préventive (horizon 3h)</div>", unsafe_allow_html=True)

df['NO2_futur3h'] = df['NO2(GT)'].shift(-3)
df['Pic_futur']   = df['NO2_futur3h'] > 150
alerte_stats = df.groupby('Pic_futur')[['T','NOx(GT)']].mean().round(2)
T_seuil   = alerte_stats.loc[True,'T']
NOx_seuil = alerte_stats.loc[True,'NOx(GT)']
df['Alerte'] = (df['T'] < T_seuil) & (df['NOx(GT)'] > NOx_seuil)
precision = df[df['Alerte']]['Pic_futur'].sum() / df['Alerte'].sum() * 100
rappel    = df[df['Alerte']]['Pic_futur'].sum() / df['Pic_futur'].sum() * 100

c1, c2, c3 = st.columns(3)
c1.markdown(f"""<div class="metric-card warning">
    <div class="metric-value">{T_seuil:.1f}°C</div>
    <div class="metric-label">Seuil température</div>
</div>""", unsafe_allow_html=True)
c2.markdown(f"""<div class="metric-card warning">
    <div class="metric-value">{NOx_seuil:.0f}</div>
    <div class="metric-label">Seuil NOx (ppb)</div>
</div>""", unsafe_allow_html=True)
c3.markdown(f"""<div class="metric-card">
    <div class="metric-value">{rappel:.0f}%</div>
    <div class="metric-label">Rappel du système</div>
</div>""", unsafe_allow_html=True)

# ─── 10. ERA5 ─────────────────────────────────────────────────────────────────
if era5_ok:
    st.markdown('<div class="section-title">10 — Variables ERA5 vs NO2</div>', unsafe_allow_html=True)

    era5_cols = ['wind_speed','wind_dir','precipitation','pressure','blh']
    corr_era5 = df_merged[era5_cols + ['NO2(GT)']].corr(numeric_only=True)['NO2(GT)'].drop('NO2(GT)').round(3)
    fig_era5 = px.bar(corr_era5.sort_values(), orientation='h',
                      labels={'value':'Corrélation de Pearson','index':'Variable ERA5'},
                      color=corr_era5.sort_values(), color_continuous_scale='RdBu_r', range_color=[-1,1])
    fig_era5.add_vline(x=0, line_dash='dash', line_color='#5a7a94')
    fig_era5.update_layout(paper_bgcolor='#0f1923', plot_bgcolor='#0a1520',
                           font_color='#7aa8c8', height=320, showlegend=False,
                           margin=dict(l=10,r=10,t=10,b=10),
                           xaxis=dict(gridcolor='#1e2d3d'),
                           yaxis=dict(gridcolor='#1e2d3d'))
    st.plotly_chart(fig_era5, use_container_width=True)

    plus_fort = corr_era5.abs().idxmax()
    st.markdown(f"""<div class="corr-badges">
        <div class="corr-badge">Facteur le plus déterminant (Q3) &nbsp;<span>{plus_fort} — r = {corr_era5[plus_fort]:+.3f}</span></div>
    </div>""", unsafe_allow_html=True)

# ─── 11. MACHINE LEARNING ─────────────────────────────────────────────────────
st.markdown('<div class="section-title">11 — Prédiction NO2 à 6h — LR / RF / XGBoost avec ERA5</div>', unsafe_allow_html=True)

@st.cache_data
def train_models(_df_merged, era5_available):
    d = _df_merged.copy()
    d['JourSemaine'] = d.index.dayofweek
    d['Mois_num']    = d.index.month
    d['Heure']       = d.index.hour
    d['NO2_lag1']    = d['NO2(GT)'].shift(1)
    d['NO2_lag6']    = d['NO2(GT)'].shift(6)
    d['NO2_cible6h'] = d['NO2(GT)'].shift(-6)

    feats = ['Heure','JourSemaine','Mois_num','T','RH','CO(GT)','NOx(GT)','NO2_lag1','NO2_lag6']
    if era5_available:
        feats = feats + ['wind_speed','wind_dir','precipitation','pressure','blh']

    df_ml = d[feats + ['NO2_cible6h']].dropna()
    X, y  = df_ml[feats], df_ml['NO2_cible6h']
    split = int(len(df_ml)*0.8)
    X_tr, X_te = X.iloc[:split], X.iloc[split:]
    y_tr, y_te = y.iloc[:split], y.iloc[split:]

    lr  = LinearRegression().fit(X_tr, y_tr)
    rf  = RandomForestRegressor(100, random_state=42).fit(X_tr, y_tr)
    xgb = XGBRegressor(n_estimators=100, random_state=42, verbosity=0).fit(X_tr, y_tr)

    # Tuning XGBoost
    param_grid = {
        'n_estimators' : [100, 200],
        'max_depth'    : [3, 5, 7],
        'learning_rate': [0.05, 0.1, 0.2],
        'subsample'    : [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    xgb_tuned = GridSearchCV(
        XGBRegressor(random_state=42, verbosity=0),
        param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=0
    )
    xgb_tuned.fit(X_tr, y_tr)

    return y_te, X_te, lr, rf, xgb, xgb_tuned.best_estimator_, feats

y_te, X_te, lr, rf, xgb, xgb_tuned, feats = train_models(df_merged, era5_ok)

resultats = {
    'Régression Linéaire': (mean_absolute_error(y_te, lr.predict(X_te)),  r2_score(y_te, lr.predict(X_te))),
    'Random Forest'      : (mean_absolute_error(y_te, rf.predict(X_te)),  r2_score(y_te, rf.predict(X_te))),
    'XGBoost (défaut)'   : (mean_absolute_error(y_te, xgb.predict(X_te)), r2_score(y_te, xgb.predict(X_te))),
    'XGBoost (tuné)'     : (mean_absolute_error(y_te, xgb_tuned.predict(X_te)), r2_score(y_te, xgb_tuned.predict(X_te))),
}

cols = st.columns(4)
colors = ['#2563eb', '#16a34a', '#854F0B', '#d97706']
for col_st, (nom, (mae, r2)), color in zip(cols, resultats.items(), colors):
    col_st.markdown(f"""<div class="metric-card" style="border-left-color:{color};">
        <div class="metric-value" style="font-size:1.3rem;">{r2:.3f}</div>
        <div class="metric-label">{nom}<br>MAE {mae:.1f} µg/m³</div>
    </div>""", unsafe_allow_html=True)

# Graphique réel vs prédit
n = 300
df_pred = pd.DataFrame({
    'Réel'               : y_te.values[:n],
    'Random Forest'      : rf.predict(X_te)[:n],
    'XGBoost tuné'       : xgb_tuned.predict(X_te)[:n],
    'Régression Linéaire': lr.predict(X_te)[:n]
}, index=y_te.index[:n])

fig10 = go.Figure()
couleurs = {'Réel':'#e8f4fd','Random Forest':'#16a34a',
            'XGBoost tuné':'#d97706','Régression Linéaire':'#2563eb'}
for col_name, color in couleurs.items():
    fig10.add_trace(go.Scatter(x=df_pred.index, y=df_pred[col_name],
                               mode='lines', name=col_name,
                               line=dict(color=color, width=1.5)))
fig10.update_layout(paper_bgcolor='#0f1923', plot_bgcolor='#0a1520',
                    font_color='#7aa8c8', height=380, hovermode='x unified', legend_title='',
                    margin=dict(l=10,r=10,t=10,b=10),
                    xaxis=dict(gridcolor='#1e2d3d', title=''),
                    yaxis=dict(gridcolor='#1e2d3d', title='NO2 (µg/m³)'))
st.plotly_chart(fig10, use_container_width=True)

# Importance des variables — RF et XGBoost côte à côte
col_imp1, col_imp2 = st.columns(2)

imp_rf = pd.Series(rf.feature_importances_, index=feats).sort_values(ascending=True)
fig_rf = px.bar(imp_rf, orientation='h', title='Importance — Random Forest',
                labels={'value':'Importance','index':'Variable'})
fig_rf.update_traces(marker_color='#16a34a')
fig_rf.update_layout(paper_bgcolor='#0f1923', plot_bgcolor='#0a1520',
                     font_color='#7aa8c8', height=380, showlegend=False,
                     margin=dict(l=10,r=10,t=30,b=10),
                     xaxis=dict(gridcolor='#1e2d3d'),
                     yaxis=dict(gridcolor='#1e2d3d'))
col_imp1.plotly_chart(fig_rf, use_container_width=True)

imp_xgb = pd.Series(xgb_tuned.feature_importances_, index=feats).sort_values(ascending=True)
fig_xgb = px.bar(imp_xgb, orientation='h', title='Importance — XGBoost tuné',
                 labels={'value':'Importance','index':'Variable'})
fig_xgb.update_traces(marker_color='#d97706')
fig_xgb.update_layout(paper_bgcolor='#0f1923', plot_bgcolor='#0a1520',
                      font_color='#7aa8c8', height=380, showlegend=False,
                      margin=dict(l=10,r=10,t=30,b=10),
                      xaxis=dict(gridcolor='#1e2d3d'),
                      yaxis=dict(gridcolor='#1e2d3d'))
col_imp2.plotly_chart(fig_xgb, use_container_width=True)

# ─── 12. CARTE ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">12 — Carte géospatiale — Stations italiennes</div>', unsafe_allow_html=True)

stations = pd.DataFrame({
    'Ville'   : ['Milan (UCI)','Turin','Rome','Florence','Bologne','Naples'],
    'Lat'     : [45.4654, 45.0703, 41.9028, 43.7696, 44.4949, 40.8518],
    'Lon'     : [9.1859,  7.6869, 12.4964, 11.2558, 11.3426, 14.2681],
    'NO2_mean': [round(df['NO2(GT)'].mean(), 1), 98.4, 76.2, 64.8, 87.3, 71.5],
    'Source'  : ['UCI dataset','ARPA Piemonte 2004','ARPA Lazio 2004',
                 'ARPA Toscana 2004','ARPA Emilia-Romagna 2004','ARPA Campania 2004']
})

def couleur_no2(v):
    return '#3B6D11' if v < 70 else ('#E8A020' if v < 90 else '#A32D2D')

carte = folium.Map(location=[43.5, 11.5], zoom_start=6, tiles='CartoDB positron')
legende = """<div style="position:fixed;bottom:30px;left:30px;z-index:1000;
background:white;padding:10px;border-radius:6px;border:1px solid #ccc;font-size:12px;">
<b>NO2 moyen</b><br>
<span style="color:#3B6D11">&#9679;</span> &lt; 70 µg/m³<br>
<span style="color:#E8A020">&#9679;</span> 70–90 µg/m³<br>
<span style="color:#A32D2D">&#9679;</span> &gt; 90 µg/m³
</div>"""
carte.get_root().html.add_child(folium.Element(legende))

for _, row in stations.iterrows():
    c = couleur_no2(row['NO2_mean'])
    folium.CircleMarker(
        location=[row['Lat'], row['Lon']], radius=row['NO2_mean']/7,
        color=c, fill=True, fill_color=c, fill_opacity=0.7,
        popup=f"<b>{row['Ville']}</b><br>NO2 : <b>{row['NO2_mean']} µg/m³</b><br>{row['Source']}",
        tooltip=f"{row['Ville']} — {row['NO2_mean']} µg/m³"
    ).add_to(carte)
    folium.Marker(
        location=[row['Lat']+0.18, row['Lon']],
        icon=folium.DivIcon(
            html=f'<div style="font-size:11px;font-weight:600;color:#333">{row["Ville"].split()[0]}</div>',
            icon_size=(80,20)
        )
    ).add_to(carte)

st_html(carte._repr_html_(), height=500)

sc = st.columns(len(stations))
for col_s, (_, row) in zip(sc, stations.iterrows()):
    col_s.markdown(f"""<div class="metric-card" style="border-left-color:{couleur_no2(row['NO2_mean'])};text-align:center;padding:0.8rem;">
        <div class="metric-value" style="font-size:1.2rem">{row['NO2_mean']}</div>
        <div class="metric-label">{row['Ville'].split()[0]}</div>
    </div>""", unsafe_allow_html=True)
