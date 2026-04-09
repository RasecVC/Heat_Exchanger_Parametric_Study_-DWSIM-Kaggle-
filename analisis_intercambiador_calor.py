# =============================================================================
# PROYECTO FINAL — Python para Analista de Datos en la Industria Petrolera
# Sistema de Analisis y Reporte de Datos
# Dataset: Heat Exchanger Parametric Study (DWSIM / Kaggle)
# Aplicacion: Monitoreo de intercambiador de calor en preheat train de crudo
# Autor: [Tu Nombre]
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import warnings
import os
import base64
from io import BytesIO
from datetime import date

warnings.filterwarnings('ignore')

figuras = []       # Almacena todas las figuras para exportacion
titulos_fig = []   # Titulos descriptivos en el mismo orden

# =============================================================================
# CONFIGURACION GENERAL
# =============================================================================

FILE_PATH  = 'heat_exchanger_dataset.csv'   # <- Ajusta la ruta si es necesario
OUTPUT_DIR = os.getcwd()

# Columnas originales (senales limpias de simulacion)
COLS_ORIG = [
    'hot_inlet_temperature_k',
    'cold_inlet_mass_flow_kg_s',
    'hot_outlet_temperature_k',
    'cold_outlet_temperature_k',
    'hx_1_heat_load_kw',
    'hot_outlet_pressure_pa',
    'cold_outlet_pressure_pa',
    'hot_outlet_mass_flow_kg_s',
    'cold_outlet_mass_flow_kg_s',
    'hx_1_logarithmic_mean_temperature_difference_lmtd_k',
]

# Columnas con ruido gaussiano (simulan lectura SCADA en campo)
COLS_NOISY = [c + '_noisy' for c in COLS_ORIG]

# Variable objetivo para el modelo predictivo
TARGET = 'hx_1_heat_load_kw'

# Etiquetas cortas para graficas
LABELS_CORTOS = {
    'hot_inlet_temperature_k'                                  : 'T_hot_in (K)',
    'cold_inlet_mass_flow_kg_s'                                : 'F_cold_in (kg/s)',
    'hot_outlet_temperature_k'                                 : 'T_hot_out (K)',
    'cold_outlet_temperature_k'                                : 'T_cold_out (K)',
    'hx_1_heat_load_kw'                                        : 'Heat Load (kW)',
    'hot_outlet_pressure_pa'                                   : 'P_hot_out (Pa)',
    'cold_outlet_pressure_pa'                                  : 'P_cold_out (Pa)',
    'hot_outlet_mass_flow_kg_s'                                : 'F_hot_out (kg/s)',
    'cold_outlet_mass_flow_kg_s'                               : 'F_cold_out (kg/s)',
    'hx_1_logarithmic_mean_temperature_difference_lmtd_k'      : 'LMTD (K)',
}

def registrar(fig, titulo):
    """Registra una figura cerrada en el listado global."""
    figuras.append(fig)
    titulos_fig.append(titulo)


# =============================================================================
# NIVEL 1 — CARGA, LIMPIEZA Y ESTADISTICA DESCRIPTIVA
# =============================================================================
print("\n" + "="*65)
print("  NIVEL 1: CARGA, LIMPIEZA Y ESTADISTICA DESCRIPTIVA")
print("="*65)

df = pd.read_csv(FILE_PATH)

print(f"\n[INFO] Dataset cargado: {df.shape[0]} filas x {df.shape[1]} columnas")
print(f"[INFO] Nulos totales   : {df.isnull().sum().sum()}")
print(f"[INFO] Filas duplicadas: {df.duplicated().sum()}")

# Estadistica descriptiva de variables originales
print("\n--- Estadistica Descriptiva (variables originales) ---")
desc = df[COLS_ORIG].describe().round(3)
print(desc.to_string())

# --- Grafica 1: Histogramas de las 10 variables originales ---
fig1, axes = plt.subplots(2, 5, figsize=(18, 7))
fig1.suptitle(
    "Distribucion de Variables del Proceso — Intercambiador de Calor\n"
    "Aplicacion: Preheat Train de Crudo | YPFB Refinacion",
    fontsize=13, fontweight='bold'
)
axes = axes.flatten()
colores = plt.cm.tab10(np.linspace(0, 1, 10))

for i, col in enumerate(COLS_ORIG):
    ax = axes[i]
    ax.hist(df[col], bins=40, color=colores[i], edgecolor='white', alpha=0.85)
    ax.set_title(LABELS_CORTOS[col], fontsize=9, fontweight='bold')
    ax.set_ylabel('Frecuencia', fontsize=8)
    ax.axvline(df[col].mean(),  color='red',    linestyle='--', linewidth=1.2, label='Media')
    ax.axvline(df[col].median(), color='orange', linestyle=':',  linewidth=1.2, label='Mediana')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
registrar(fig1, "Distribucion de Variables del Proceso (Histogramas)")
plt.close()
print("[OK] Grafica 1 generada: Histogramas")


# =============================================================================
# NIVEL 2 — ANALISIS DE CORRELACIONES Y KPIs OPERACIONALES
# =============================================================================
print("\n" + "="*65)
print("  NIVEL 2: CORRELACIONES, DELTA-P Y KPIs OPERACIONALES")
print("="*65)

# --- Grafica 2: Mapa de calor de correlaciones ---
corr = df[COLS_ORIG].corr()
labels_cortos = [LABELS_CORTOS[c] for c in COLS_ORIG]

fig2, ax2 = plt.subplots(figsize=(12, 9))
im = ax2.imshow(corr.values, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
plt.colorbar(im, ax=ax2, label='Coeficiente de Correlacion de Pearson')
ax2.set_xticks(range(len(labels_cortos)))
ax2.set_yticks(range(len(labels_cortos)))
ax2.set_xticklabels(labels_cortos, rotation=45, ha='right', fontsize=9)
ax2.set_yticklabels(labels_cortos, fontsize=9)
for i in range(len(COLS_ORIG)):
    for j in range(len(COLS_ORIG)):
        val = corr.values[i, j]
        color_txt = 'white' if abs(val) > 0.7 else 'black'
        ax2.text(j, i, f'{val:.2f}', ha='center', va='center',
                 fontsize=7, color=color_txt, fontweight='bold')
ax2.set_title(
    "Mapa de Calor de Correlaciones — Variables del Intercambiador\n"
    "Diagnostico: Relaciones entre variables de proceso",
    fontsize=12, fontweight='bold'
)
plt.tight_layout()
registrar(fig2, "Mapa de Calor de Correlaciones (Pearson)")
plt.close()
print("[OK] Grafica 2 generada: Mapa de correlaciones")

# KPI: Diferencial de presion (Delta-P) usando valores RUIDOSOS del SCADA
# Las presiones originales son constantes en este dataset de simulacion,
# por lo que el diferencial operacional se calcula sobre las lecturas SCADA
df['delta_p_pa'] = df['hot_outlet_pressure_pa_noisy'] - df['cold_outlet_pressure_pa_noisy']
dp_mean  = df['delta_p_pa'].mean()
dp_std   = df['delta_p_pa'].std()
dp_alarm = dp_mean + 2 * dp_std

print(f"\n[KPI] Delta-P medio (SCADA)    : {dp_mean:.0f} Pa")
print(f"[KPI] Desv. estandar           : {dp_std:.0f} Pa")
print(f"[KPI] Umbral de alarma         : {dp_alarm:.0f} Pa (media + 2 sigma)")
print(f"[KPI] Puntos sobre umbral      : {(df['delta_p_pa'] > dp_alarm).sum()} / {len(df)}")

# KPI: Eficiencia termica usando reduccion de temperatura corriente caliente
# eta = (T_hot_in - T_hot_out) / (T_hot_in - T_ref)
# T_ref = minima temperatura de salida observada (estado de referencia ideal)
T_hot_in   = df['hot_inlet_temperature_k']
T_hot_out  = df['hot_outlet_temperature_k']
T_hot_in_min = T_hot_in.min()

# Normalizacion: que fraccion de la caida maxima posible se logra
df['eficiencia_pct'] = (
    (T_hot_in - T_hot_out) / (T_hot_in - T_hot_in_min + 1e-9)
) * 100
df['eficiencia_pct'] = df['eficiencia_pct'].clip(0, 100)

print(f"\n[KPI] Eficiencia termica media  : {df['eficiencia_pct'].mean():.1f}%")
print(f"[KPI] Eficiencia minima         : {df['eficiencia_pct'].min():.1f}%")
print(f"[KPI] Puntos bajo 85% (alerta)  : {(df['eficiencia_pct'] < 85).sum()}")

# --- Grafica 3: Scatter T_hot_in vs Heat Load + Curva de operacion ---
fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))
fig3.suptitle(
    "Curva de Operacion del Intercambiador — Diagnostico de Desempeno\n"
    "Aplicacion: Deteccion de operacion fuera de diseno",
    fontsize=12, fontweight='bold'
)

ax3a = axes3[0]
sc = ax3a.scatter(df['hot_inlet_temperature_k'], df['hx_1_heat_load_kw'],
                  c=df['cold_inlet_mass_flow_kg_s'], cmap='plasma',
                  alpha=0.4, s=5)
plt.colorbar(sc, ax=ax3a, label='Flujo frio entrada (kg/s)')
ax3a.set_xlabel('Temperatura Entrada Corriente Caliente (K)', fontsize=10)
ax3a.set_ylabel('Carga Termica HX-1 (kW)', fontsize=10)
ax3a.set_title('T_hot_in vs Heat Load\n(color = caudal frio)', fontsize=10)
ax3a.grid(True, alpha=0.3)

ax3b = axes3[1]
sc2 = ax3b.scatter(df['cold_inlet_mass_flow_kg_s'], df['hx_1_heat_load_kw'],
                   c=df['hx_1_logarithmic_mean_temperature_difference_lmtd_k'],
                   cmap='coolwarm', alpha=0.4, s=5)
plt.colorbar(sc2, ax=ax3b, label='LMTD (K)')
ax3b.set_xlabel('Flujo Masa Entrada Corriente Fria (kg/s)', fontsize=10)
ax3b.set_ylabel('Carga Termica HX-1 (kW)', fontsize=10)
ax3b.set_title('Flujo Frio vs Heat Load\n(color = LMTD)', fontsize=10)
ax3b.grid(True, alpha=0.3)

plt.tight_layout()
registrar(fig3, "Curva de Operacion: T_hot_in y Flujo vs Heat Load")
plt.close()
print("[OK] Grafica 3 generada: Curva de operacion")

# --- Grafica 4: Delta-P y Eficiencia Termica ---
fig4, axes4 = plt.subplots(1, 2, figsize=(14, 5))
fig4.suptitle(
    "Indicadores Operacionales Clave (KPIs) — Intercambiador de Calor\n"
    "Aplicacion: Monitoreo continuo en sala de control",
    fontsize=12, fontweight='bold'
)

ax4a = axes4[0]
ax4a.plot(df.index, df['delta_p_pa'], color='steelblue', linewidth=0.4, alpha=0.6, label='Delta-P')
ax4a.axhline(dp_mean,  color='green',  linestyle='--', linewidth=1.5, label=f'Media: {dp_mean:.0f} Pa')
ax4a.axhline(dp_alarm, color='red',    linestyle='--', linewidth=1.5, label=f'Alarma: {dp_alarm:.0f} Pa')
ax4a.fill_between(df.index, dp_alarm, df['delta_p_pa'].max(),
                  where=df['delta_p_pa'] > dp_alarm,
                  color='red', alpha=0.15, label='Zona de alerta')
ax4a.set_title('Diferencial de Presion Hot vs Cold (Delta-P)\nIndicador de ensuciamiento/bloqueo', fontsize=10)
ax4a.set_xlabel('Muestra', fontsize=9)
ax4a.set_ylabel('Delta-P (Pa)', fontsize=9)
ax4a.legend(fontsize=8)
ax4a.grid(True, alpha=0.3)

ax4b = axes4[1]
ax4b.hist(df['eficiencia_pct'], bins=50, color='mediumseagreen', edgecolor='white', alpha=0.85)
ax4b.axvline(85, color='red', linestyle='--', linewidth=2, label='Umbral critico: 85%')
ax4b.axvline(df['eficiencia_pct'].mean(), color='orange', linestyle='--',
             linewidth=2, label=f'Media: {df["eficiencia_pct"].mean():.1f}%')
ax4b.set_title('Distribucion de Eficiencia Termica (%)\nUmbral operacional: 85%', fontsize=10)
ax4b.set_xlabel('Eficiencia Termica (%)', fontsize=9)
ax4b.set_ylabel('Frecuencia', fontsize=9)
ax4b.legend(fontsize=8)
ax4b.grid(True, alpha=0.3)

plt.tight_layout()
registrar(fig4, "KPIs: Delta-P y Eficiencia Termica")
plt.close()
print("[OK] Grafica 4 generada: KPIs operacionales")


# =============================================================================
# NIVEL 3 — COMPARACION SENAL ORIGINAL VS RUIDOSA (SCADA)
# =============================================================================
print("\n" + "="*65)
print("  NIVEL 3: COMPARACION SENAL ORIGINAL vs SCADA (RUIDOSA)")
print("="*65)

# Pares a comparar (los mas relevantes para operaciones)
PARES_COMP = [
    ('hot_inlet_temperature_k',
     'hot_inlet_temperature_k_noisy',
     'Temperatura Entrada Caliente (K)'),
    ('hx_1_heat_load_kw',
     'hx_1_heat_load_kw_noisy',
     'Carga Termica HX-1 (kW)'),
    ('hx_1_logarithmic_mean_temperature_difference_lmtd_k',
     'hx_1_logarithmic_mean_temperature_difference_lmtd_k_noisy',
     'LMTD (K) — Indice de Fouling'),
    ('cold_outlet_pressure_pa',
     'cold_outlet_pressure_pa_noisy',
     'Presion Salida Corriente Fria (Pa)'),
]

fig5, axes5 = plt.subplots(2, 2, figsize=(15, 9))
fig5.suptitle(
    "Comparacion Senal de Simulacion vs Lectura SCADA (con ruido gaussiano)\n"
    "Aplicacion: Calibracion y validacion de instrumentos en campo",
    fontsize=12, fontweight='bold'
)
axes5 = axes5.flatten()
muestra = np.arange(200)  # Primeras 200 muestras para visibilidad

for i, (col_orig, col_noisy, titulo) in enumerate(PARES_COMP):
    ax = axes5[i]
    ax.plot(muestra, df[col_orig].iloc[:200].values,
            color='steelblue', linewidth=1.5, label='Simulacion (referencia)', zorder=3)
    ax.plot(muestra, df[col_noisy].iloc[:200].values,
            color='tomato', linewidth=0.8, alpha=0.7, label='Lectura SCADA (ruidosa)', zorder=2)

    # Diferencia porcentual promedio
    diff_pct = np.abs((df[col_orig] - df[col_noisy]) / (df[col_orig].abs() + 1e-9)).mean() * 100
    print(f"  [{titulo[:30]}] Error relativo medio: {diff_pct:.2f}%")

    ax.set_title(f'{titulo}\nError relativo medio: {diff_pct:.2f}%', fontsize=9, fontweight='bold')
    ax.set_xlabel('Muestra', fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
registrar(fig5, "Comparacion Senal Original vs SCADA (Ruidosa)")
plt.close()
print("[OK] Grafica 5 generada: Comparacion original vs SCADA")

# --- Grafica 6: Indice de Fouling (LMTD_orig vs LMTD_noisy) ---
df['fouling_idx'] = np.abs(
    df['hx_1_logarithmic_mean_temperature_difference_lmtd_k'] -
    df['hx_1_logarithmic_mean_temperature_difference_lmtd_k_noisy']
) / (df['hx_1_logarithmic_mean_temperature_difference_lmtd_k'].abs() + 1e-9) * 100

fouling_umbral = df['fouling_idx'].mean() + 2 * df['fouling_idx'].std()

fig6, ax6 = plt.subplots(figsize=(14, 5))
ax6.plot(df.index, df['fouling_idx'], color='#B45309', linewidth=0.5, alpha=0.7, label='Indice de Fouling (%)')
mm50 = df['fouling_idx'].rolling(window=50, min_periods=1).mean()
ax6.plot(df.index, mm50, color='#92400E', linewidth=2.5, label='Tendencia (media movil 50 muestras)')
ax6.axhline(fouling_umbral, color='red', linestyle='--', linewidth=2,
            label=f'Umbral de limpieza: {fouling_umbral:.1f}%')
ax6.fill_between(df.index, fouling_umbral, df['fouling_idx'].max(),
                 where=df['fouling_idx'] > fouling_umbral,
                 color='red', alpha=0.12, label='Zona: Requiere inspeccion')
ax6.set_title(
    "Indice de Ensuciamiento (Fouling) del Intercambiador\n"
    "Diferencia relativa LMTD_simulado vs LMTD_SCADA — Critico para programar limpieza CIP",
    fontsize=11, fontweight='bold'
)
ax6.set_xlabel('Muestra (condicion operacional)', fontsize=10)
ax6.set_ylabel('Indice de Fouling (%)', fontsize=10)
ax6.legend(fontsize=9)
ax6.grid(True, alpha=0.3)
plt.tight_layout()
registrar(fig6, "Indice de Ensuciamiento (Fouling) — LMTD original vs SCADA")
plt.close()
print("[OK] Grafica 6 generada: Indice de fouling")


# =============================================================================
# NIVEL 4 — DETECCION DE ANOMALIAS (Z-SCORE SOBRE SENALES SCADA)
# =============================================================================
print("\n" + "="*65)
print("  NIVEL 4: DETECCION DE ANOMALIAS — Z-SCORE (SENALES SCADA)")
print("="*65)

UMBRAL_Z = 3.0  # Umbral estandar: 3 sigmas

# Calcular Z-Score para cada variable ruidosa
z_scores = {}
anomalias_count = {}

for col in COLS_NOISY:
    media = df[col].mean()
    std   = df[col].std()
    zs    = np.abs((df[col] - media) / (std + 1e-9))
    z_scores[col] = zs
    n_anomalias = (zs > UMBRAL_Z).sum()
    anomalias_count[col] = n_anomalias
    label = LABELS_CORTOS.get(col.replace('_noisy', ''), col)
    print(f"  {label:25s}: {n_anomalias:4d} anomalias ({n_anomalias/len(df)*100:.2f}%)")

# --- Grafica 7: Conteo de anomalias por variable ---
fig7, axes7 = plt.subplots(1, 2, figsize=(15, 6))
fig7.suptitle(
    "Deteccion de Anomalias en Senales SCADA — Metodo Z-Score (|Z| > 3)\n"
    "Aplicacion: Alertas de instrumentacion y fallas de sensores",
    fontsize=12, fontweight='bold'
)

labels_anom = [LABELS_CORTOS.get(c.replace('_noisy',''), c) for c in COLS_NOISY]
counts_anom = [anomalias_count[c] for c in COLS_NOISY]
colores_bar = ['#DC2626' if v > np.mean(counts_anom) else '#2563EB' for v in counts_anom]

ax7a = axes7[0]
bars = ax7a.barh(labels_anom, counts_anom, color=colores_bar, edgecolor='white', height=0.6)
ax7a.axvline(np.mean(counts_anom), color='orange', linestyle='--',
             linewidth=2, label=f'Media: {np.mean(counts_anom):.0f}')
for bar, val in zip(bars, counts_anom):
    ax7a.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
              f'{val}', va='center', fontsize=9, color='#1F2937')
ax7a.set_title('Cantidad de Anomalias por Variable\n(Rojo = sobre la media)', fontsize=10)
ax7a.set_xlabel('Numero de Anomalias', fontsize=9)
ax7a.legend(fontsize=9)
ax7a.grid(True, alpha=0.3, axis='x')

# Serie temporal Z-Score del heat load (variable critica)
col_critica = 'hx_1_heat_load_kw_noisy'
zs_critica  = z_scores[col_critica]

ax7b = axes7[1]
ax7b.plot(df.index, zs_critica, color='#6366F1', linewidth=0.5, alpha=0.7, label='|Z-Score| Heat Load')
ax7b.axhline(UMBRAL_Z, color='red', linestyle='--', linewidth=2, label=f'Umbral Z={UMBRAL_Z}')
mask_anom = zs_critica > UMBRAL_Z
ax7b.scatter(df.index[mask_anom], zs_critica[mask_anom],
             color='red', s=15, zorder=5, label=f'Anomalias ({mask_anom.sum()})')
ax7b.set_title('Z-Score Temporal — Carga Termica SCADA\nDeteccion de picos y caidas anomalas', fontsize=10)
ax7b.set_xlabel('Muestra', fontsize=9)
ax7b.set_ylabel('|Z-Score|', fontsize=9)
ax7b.legend(fontsize=8)
ax7b.grid(True, alpha=0.3)

plt.tight_layout()
registrar(fig7, "Deteccion de Anomalias Z-Score — Senales SCADA")
plt.close()
print("[OK] Grafica 7 generada: Deteccion de anomalias Z-Score")

# --- Grafica 8: Boxplot comparativo original vs ruidoso ---
VARS_BOX = [
    ('hot_inlet_temperature_k',       'T_hot_in (K)'),
    ('hx_1_heat_load_kw',             'Heat Load (kW)'),
    ('hx_1_logarithmic_mean_temperature_difference_lmtd_k', 'LMTD (K)'),
    ('cold_outlet_pressure_pa',       'P_cold_out (Pa)'),
]

fig8, axes8 = plt.subplots(1, 4, figsize=(16, 6))
fig8.suptitle(
    "Boxplot: Valores Atipicos — Simulacion vs SCADA\n"
    "Aplicacion: Validacion de calibracion de instrumentos en campo",
    fontsize=12, fontweight='bold'
)

for i, (col, label) in enumerate(VARS_BOX):
    ax = axes8[i]
    datos = [df[col].dropna().values, df[col + '_noisy'].dropna().values]
    bp = ax.boxplot(datos, patch_artist=True, widths=0.5,
                    medianprops=dict(color='black', linewidth=2))
    bp['boxes'][0].set_facecolor('#93C5FD')
    bp['boxes'][1].set_facecolor('#FCA5A5')
    ax.set_xticklabels(['Simulacion', 'SCADA'], fontsize=9)
    ax.set_title(label, fontsize=9, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
registrar(fig8, "Boxplot Valores Atipicos: Simulacion vs SCADA")
plt.close()
print("[OK] Grafica 8 generada: Boxplot outliers")


# =============================================================================
# NIVEL 5 — MODELO PREDICTIVO GRU
# Objetivo: Predecir hx_1_heat_load_kw a partir de variables de entrada
# Valor operacional: Mantenimiento predictivo, programacion de limpieza CIP
# =============================================================================
print("\n" + "="*65)
print("  NIVEL 5: MODELO GRU — PREDICCION DE CARGA TERMICA (kW)")
print("="*65)

TENSORFLOW_OK = False
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import GRU, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_absolute_error, r2_score
    tf.random.set_seed(42)
    np.random.seed(42)
    TENSORFLOW_OK = True
    print("[INFO] TensorFlow disponible:", tf.__version__)
except ImportError:
    print("[!] TensorFlow no instalado. Saltando Nivel 5.")
    print("    Para instalar: pip install tensorflow scikit-learn")

resultados_gru = {}

if TENSORFLOW_OK:

    # Caracteristicas de entrada para el modelo
    FEATURES = [
        'hot_inlet_temperature_k',
        'cold_inlet_mass_flow_kg_s',
        'hx_1_logarithmic_mean_temperature_difference_lmtd_k',
        'hot_outlet_temperature_k',
        'cold_outlet_temperature_k',
    ]
    WINDOW   = 20   # Ventana temporal: 20 condiciones de operacion previas
    N_FC     = 200  # Puntos a predecir en modo forecast

    print(f"\n[CONFIG] Features : {FEATURES}")
    print(f"[CONFIG] Ventana  : {WINDOW} pasos")
    print(f"[CONFIG] Forecast : {N_FC} puntos")

    # ---- Preparacion de datos ----
    datos_modelo = df[FEATURES + [TARGET]].dropna().copy()

    # Normalizar features y target por separado
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_all_sc = scaler_X.fit_transform(datos_modelo[FEATURES].values)
    y_all_sc = scaler_y.fit_transform(datos_modelo[[TARGET]].values)

    # Crear secuencias de ventana deslizante
    def crear_secuencias_multi(X_sc, y_sc, window):
        Xs, ys = [], []
        for i in range(len(X_sc) - window):
            Xs.append(X_sc[i:i + window])
            ys.append(y_sc[i + window])
        return np.array(Xs), np.array(ys)

    X_seq, y_seq = crear_secuencias_multi(X_all_sc, y_all_sc, WINDOW)

    split = int(len(X_seq) * 0.85)
    X_tr, X_te = X_seq[:split], X_seq[split:]
    y_tr, y_te = y_seq[:split], y_seq[split:]
    print(f"[INFO] Train={X_tr.shape[0]} muestras | Test={X_te.shape[0]} muestras")

    # ---- Arquitectura GRU ----
    modelo_gru = Sequential([
        GRU(64, return_sequences=True, input_shape=(WINDOW, len(FEATURES))),
        Dropout(0.20),
        GRU(32, return_sequences=False),
        Dropout(0.15),
        Dense(16, activation='relu'),
        Dense(1)
    ], name='GRU_HeatLoad_Predictor')

    modelo_gru.compile(optimizer='adam', loss='mse', metrics=['mae'])
    modelo_gru.summary()

    es = EarlyStopping(monitor='val_loss', patience=25,
                       restore_best_weights=True, verbose=0)

    print("\n[INFO] Entrenando modelo GRU (hasta 300 epocas con EarlyStopping)...")
    historia = modelo_gru.fit(
        X_tr, y_tr,
        epochs=300,
        batch_size=32,
        validation_data=(X_te, y_te),
        callbacks=[es],
        verbose=0
    )
    epocas = len(historia.history['loss'])
    print(f"[INFO] Entrenado en {epocas} epocas | val_loss={min(historia.history['val_loss']):.6f}")

    # ---- Evaluacion ----
    pred_sc   = modelo_gru.predict(X_te, verbose=0)
    pred_real = scaler_y.inverse_transform(pred_sc).flatten()
    y_real    = scaler_y.inverse_transform(y_te).flatten()

    mae  = mean_absolute_error(y_real, pred_real)
    rmse = np.sqrt(np.mean((y_real - pred_real)**2))
    mape = np.mean(np.abs((y_real - pred_real) / (y_real + 1e-8))) * 100
    r2   = r2_score(y_real, pred_real)

    clasificacion = ("EXCELENTE" if mape < 5 else
                     "ACEPTABLE" if mape < 15 else
                     "MEJORAR (mas datos recomendados)")

    print(f"\n[METRICAS] MAE   = {mae:.3f} kW")
    print(f"[METRICAS] RMSE  = {rmse:.3f} kW")
    print(f"[METRICAS] MAPE  = {mape:.2f}% — {clasificacion}")
    print(f"[METRICAS] R2    = {r2:.4f}")

    # ---- Forecast recursivo ----
    def forecast_recursivo_multi(modelo, ultima_ventana_sc, n_puntos,
                                  scaler_X, scaler_y, n_features):
        ventana = ultima_ventana_sc.copy()
        predicciones_sc = []
        for _ in range(n_puntos):
            X_in = ventana.reshape(1, len(ventana), n_features)
            pred = modelo.predict(X_in, verbose=0)[0, 0]
            predicciones_sc.append(pred)
            # Desplazar ventana: la nueva entrada usa la prediccion como proxy del target
            nueva_fila = ventana[-1].copy()
            nueva_fila[0] = pred  # Sustituir primera feature con el predicho (hot_inlet_temp proxy)
            ventana = np.roll(ventana, -1, axis=0)
            ventana[-1] = nueva_fila
        return scaler_y.inverse_transform(
            np.array(predicciones_sc).reshape(-1, 1)
        ).flatten()

    ultima_ventana = X_all_sc[-WINDOW:]
    forecast_vals  = forecast_recursivo_multi(
        modelo_gru, ultima_ventana, N_FC, scaler_X, scaler_y, len(FEATURES)
    )

    resultados_gru = {
        'y_real':     y_real,
        'pred_real':  pred_real,
        'forecast':   forecast_vals,
        'historia':   historia,
        'mae': mae, 'rmse': rmse, 'mape': mape, 'r2': r2,
        'epocas': epocas, 'clasificacion': clasificacion,
        'split': split,
    }

    # ---- Grafica 9: Historico + Prediccion Test + Forecast ----
    fig9, axes9 = plt.subplots(2, 1, figsize=(15, 10))
    fig9.suptitle(
        f"GRU — Prediccion y Forecast de Carga Termica HX-1 (kW)\n"
        f"MAPE={mape:.2f}% | R2={r2:.4f} | {clasificacion}\n"
        f"Aplicacion: Mantenimiento predictivo y programacion de limpieza CIP",
        fontsize=12, fontweight='bold'
    )

    ax9a = axes9[0]
    historico_real = scaler_y.inverse_transform(y_all_sc).flatten()
    ax9a.plot(np.arange(len(historico_real)), historico_real,
              color='#CBD5E1', linewidth=0.7, alpha=0.8, label='Historico real')
    idx_test_start = split + WINDOW
    ax9a.plot(np.arange(idx_test_start, idx_test_start + len(y_real)),
              y_real, color='#0EA5E9', linewidth=1.5,
              label='Real (periodo test)', zorder=3)
    ax9a.plot(np.arange(idx_test_start, idx_test_start + len(pred_real)),
              pred_real, color='#7C3AED', linewidth=1.5, linestyle='--',
              label='GRU — Prediccion test', zorder=4)
    fc_start = len(historico_real)
    ax9a.plot(np.arange(fc_start, fc_start + N_FC), forecast_vals,
              color='#EF4444', linewidth=2.5, linestyle='-.',
              marker='^', markersize=4, label='Forecast', zorder=5)
    ax9a.fill_between(np.arange(fc_start, fc_start + N_FC),
                      forecast_vals * 0.92, forecast_vals * 1.08,
                      alpha=0.15, color='#EF4444', label='Banda +/-8%')
    ax9a.axvline(x=idx_test_start, color='#F59E0B', linewidth=1.5,
                 linestyle=':', label='Corte Train/Test')
    ax9a.set_ylabel('Carga Termica HX-1 (kW)', fontsize=10)
    ax9a.set_title('Carga Termica: Historico, Prediccion en Test y Forecast', fontsize=10)
    ax9a.legend(fontsize=8, loc='upper left')
    ax9a.grid(True, alpha=0.3)

    ax9b = axes9[1]
    ep_rng = range(1, epocas + 1)
    ax9b.plot(ep_rng, historia.history['loss'],
              color='#7C3AED', linewidth=2, label='Loss Entrenamiento')
    ax9b.plot(ep_rng, historia.history['val_loss'],
              color='#EF4444', linewidth=2, linestyle='--', label='Loss Validacion')
    ax9b.set_xlabel('Epoca', fontsize=10)
    ax9b.set_ylabel('MSE Loss (escala log)', fontsize=10)
    ax9b.set_title('Curva de Aprendizaje — Convergencia del Modelo GRU', fontsize=10)
    ax9b.set_yscale('log')
    ax9b.legend(fontsize=9)
    ax9b.grid(True, alpha=0.3)

    plt.tight_layout()
    registrar(fig9, "GRU — Prediccion y Forecast de Carga Termica HX-1")
    plt.close()
    print("[OK] Grafica 9 generada: GRU historico + forecast")

    # ---- Grafica 10: Scatter prediccion vs real + residuos ----
    fig10, axes10 = plt.subplots(1, 2, figsize=(13, 5))
    fig10.suptitle(
        "Calidad del Modelo GRU — Prediccion vs Real y Distribucion de Residuos\n"
        "Aplicacion: Validacion del modelo antes de despliegue en produccion",
        fontsize=12, fontweight='bold'
    )

    ax10a = axes10[0]
    lim_min = min(y_real.min(), pred_real.min())
    lim_max = max(y_real.max(), pred_real.max())
    ax10a.scatter(y_real, pred_real, alpha=0.3, s=8, color='#7C3AED')
    ax10a.plot([lim_min, lim_max], [lim_min, lim_max],
               color='red', linewidth=2, linestyle='--', label='Linea ideal (y=x)')
    ax10a.set_xlabel('Valor Real (kW)', fontsize=10)
    ax10a.set_ylabel('Prediccion GRU (kW)', fontsize=10)
    ax10a.set_title(f'Prediccion vs Real\nR2 = {r2:.4f} | MAE = {mae:.2f} kW', fontsize=10)
    ax10a.legend(fontsize=9)
    ax10a.grid(True, alpha=0.3)

    ax10b = axes10[1]
    residuos = y_real - pred_real
    ax10b.hist(residuos, bins=50, color='#0EA5E9', edgecolor='white', alpha=0.85)
    ax10b.axvline(0, color='red', linestyle='--', linewidth=2, label='Residuo = 0')
    ax10b.axvline(residuos.mean(), color='orange', linestyle='--', linewidth=2,
                  label=f'Media: {residuos.mean():.2f} kW')
    ax10b.set_title(f'Distribucion de Residuos\nSesgo medio: {residuos.mean():.2f} kW', fontsize=10)
    ax10b.set_xlabel('Residuo (kW)', fontsize=9)
    ax10b.set_ylabel('Frecuencia', fontsize=9)
    ax10b.legend(fontsize=9)
    ax10b.grid(True, alpha=0.3)

    plt.tight_layout()
    registrar(fig10, "Calidad del Modelo GRU: Scatter y Residuos")
    plt.close()
    print("[OK] Grafica 10 generada: Scatter prediccion vs real")

    print("\n==== MODELOS GRU FINALIZADOS ====")


# =============================================================================
# NIVEL 6 — EXPORTAR TODAS LAS GRAFICAS A PDF
# =============================================================================
print("\n" + "="*65)
print("  NIVEL 6: EXPORTANDO GRAFICAS A PDF")
print("="*65)

pdf_path = os.path.join(OUTPUT_DIR, 'analisis_intercambiador_calor.pdf')

with PdfPages(pdf_path) as pdf:

    # Portada
    portada, ax_p = plt.subplots(figsize=(11, 8.5))
    ax_p.axis('off')
    ax_p.set_facecolor('#0F172A')
    portada.patch.set_facecolor('#0F172A')

    ax_p.text(0.5, 0.82,
              'SISTEMA DE ANALISIS Y REPORTE DE DATOS',
              ha='center', va='center', fontsize=18, fontweight='bold',
              color='white', transform=ax_p.transAxes)
    ax_p.text(0.5, 0.72,
              'Intercambiador de Calor — Preheat Train de Crudo',
              ha='center', va='center', fontsize=14, color='#93C5FD',
              transform=ax_p.transAxes)
    ax_p.text(0.5, 0.60,
              'Monitoreo continuo | Deteccion de anomalias | Modelo GRU predictivo',
              ha='center', va='center', fontsize=11, color='#CBD5E1',
              transform=ax_p.transAxes)
    ax_p.text(0.5, 0.48,
              'Dataset: Heat Exchanger Parametric Study (DWSIM / Kaggle)',
              ha='center', va='center', fontsize=10, color='#94A3B8',
              transform=ax_p.transAxes)
    ax_p.text(0.5, 0.38,
              'Aplicacion: Industria Petrolera — YPFB Refinacion',
              ha='center', va='center', fontsize=10, color='#94A3B8',
              transform=ax_p.transAxes)
    ax_p.text(0.5, 0.25,
              f'Generado: {date.today().strftime("%d/%m/%Y")}',
              ha='center', va='center', fontsize=10, color='#64748B',
              transform=ax_p.transAxes)

    # Lineas decorativas
    for y_pos in [0.90, 0.18]:
        portada.add_artist(
            plt.Line2D([0.05, 0.95], [y_pos, y_pos],
                       transform=portada.transFigure,
                       color='#3B82F6', linewidth=2)
        )

    pdf.savefig(portada, bbox_inches='tight', facecolor='#0F172A')
    plt.close(portada)

    for i, (fig, titulo) in enumerate(zip(figuras, titulos_fig), start=1):
        pdf.savefig(fig, bbox_inches='tight')
        print(f"  Figura {i}/{len(figuras)} anadida: {titulo[:55]}")

    d = pdf.infodict()
    d['Title']   = 'Analisis Intercambiador de Calor — Sistema Predictivo GRU'
    d['Author']  = 'Python — Analisis Automatico Industria Petrolera'
    d['Subject'] = 'Monitoreo, anomalias y forecast carga termica'

print(f"\n[OK] PDF exportado: {pdf_path}")
print(f"     Total paginas: {len(figuras) + 1} (1 portada + {len(figuras)} graficas)")


# =============================================================================
# NIVEL 7 — EXPORTAR A HTML AUTOCONTENIDO (BASE64)
# =============================================================================
print("\n" + "="*65)
print("  NIVEL 7: EXPORTANDO A HTML AUTOCONTENIDO")
print("="*65)

html_path = os.path.join(OUTPUT_DIR, 'analisis_intercambiador_calor.html')

def fig_a_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=110, bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

# Metricas GRU para el HTML
metricas_html = ""
if resultados_gru:
    r = resultados_gru
    metricas_html = f"""
    <div class="metricas">
      <span class="chip chip-blue">MAE: {r['mae']:.3f} kW</span>
      <span class="chip chip-blue">RMSE: {r['rmse']:.3f} kW</span>
      <span class="chip chip-blue">MAPE: {r['mape']:.2f}%</span>
      <span class="chip chip-green">R2: {r['r2']:.4f}</span>
      <span class="chip chip-blue">Epocas: {r['epocas']}</span>
      <span class="chip chip-orange">{r['clasificacion']}</span>
    </div>"""

secciones = [
    ("Nivel 1 — Estadistica Descriptiva", [0]),
    ("Nivel 2 — Correlaciones y KPIs Operacionales", [1, 2, 3]),
    ("Nivel 3 — Comparacion Senal Original vs SCADA", [4, 5]),
    ("Nivel 4 — Deteccion de Anomalias Z-Score", [6, 7]),
    ("Nivel 5 — Modelo GRU: Prediccion y Forecast", list(range(8, len(figuras)))),
]

tarjetas_html = ""
for titulo_sec, indices in secciones:
    tarjetas_html += f'<h2 class="sec-titulo">{titulo_sec}</h2>\n'
    # Metricas GRU solo en la seccion del modelo
    if "Nivel 5" in titulo_sec and metricas_html:
        tarjetas_html += f'<div class="tarjeta"><div class="tarjeta-titulo">Metricas del Modelo GRU</div>{metricas_html}</div>\n'
    for idx in indices:
        if idx < len(figuras):
            img64  = fig_a_base64(figuras[idx])
            titulo_g = titulos_fig[idx]
            tarjetas_html += f"""
<div class="tarjeta">
  <div class="tarjeta-titulo">Grafica {idx+1}: {titulo_g}</div>
  <img src="data:image/png;base64,{img64}" alt="{titulo_g}">
</div>
"""

html_final = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Analisis Intercambiador de Calor — Sistema Predictivo GRU</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #F1F5F9; color: #1E293B; }}
    header {{
      background: linear-gradient(135deg, #0F172A 0%, #1E3A5F 60%, #1D4ED8 100%);
      color: white; padding: 40px; text-align: center;
    }}
    header h1 {{ font-size: 1.9rem; margin-bottom: 10px; }}
    header p  {{ font-size: 1rem; opacity: 0.80; margin-top: 6px; }}
    .badge {{
      display: inline-block; background: rgba(255,255,255,0.12);
      border-radius: 20px; padding: 4px 14px; font-size: 0.85rem;
      margin: 4px; color: #BAE6FD;
    }}
    .container {{ max-width: 1200px; margin: 0 auto; padding: 30px 20px; }}
    .sec-titulo {{
      font-size: 1.3rem; font-weight: bold; color: #1E3A5F;
      margin: 36px 0 14px; border-left: 5px solid #1D4ED8;
      padding-left: 12px;
    }}
    .tarjeta {{
      background: white; border-radius: 12px;
      box-shadow: 0 2px 14px rgba(0,0,0,0.07);
      margin-bottom: 28px; overflow: hidden;
    }}
    .tarjeta-titulo {{
      background: #EFF6FF; border-bottom: 2px solid #BFDBFE;
      padding: 13px 20px; font-size: 0.95rem;
      font-weight: bold; color: #1E3A5F;
    }}
    .tarjeta img {{ width: 100%; display: block; padding: 14px; }}
    .metricas {{ display: flex; flex-wrap: wrap; gap: 10px; padding: 16px 20px; }}
    .chip {{
      border-radius: 20px; padding: 6px 16px;
      font-size: 0.85rem; font-weight: bold;
    }}
    .chip-blue   {{ background: #DBEAFE; color: #1E40AF; }}
    .chip-green  {{ background: #D1FAE5; color: #065F46; }}
    .chip-orange {{ background: #FEF3C7; color: #92400E; }}
    footer {{
      text-align: center; padding: 28px;
      background: #0F172A; color: #64748B;
      font-size: 0.85rem; margin-top: 50px;
    }}
  </style>
</head>
<body>
<header>
  <h1>Sistema de Analisis y Reporte de Datos</h1>
  <p>Intercambiador de Calor — Preheat Train de Crudo | Industria Petrolera</p>
  <div style="margin-top: 14px;">
    <span class="badge">Estadistica Descriptiva</span>
    <span class="badge">Deteccion de Anomalias</span>
    <span class="badge">Modelo GRU</span>
    <span class="badge">Fouling Index</span>
    <span class="badge">KPIs Operacionales</span>
    <span class="badge">Generado: {date.today().strftime('%d/%m/%Y')}</span>
  </div>
</header>
<div class="container">
{tarjetas_html}
</div>
<footer>
  Generado con Python · Pandas · TensorFlow/Keras · Matplotlib
  &nbsp;|&nbsp; Dataset: Heat Exchanger DWSIM (Kaggle)
  &nbsp;|&nbsp; {date.today().strftime('%d/%m/%Y')}
</footer>
</body>
</html>
"""

with open(html_path, 'w', encoding='utf-8') as f:
    f.write(html_final)

print(f"[OK] HTML exportado: {html_path}")
print(f"     Abre en cualquier navegador — sin conexion a internet requerida.")


# =============================================================================
# NIVEL 8 — GUIA DE DESPLIEGUE: GITHUB · STREAMLIT · RENDER
# =============================================================================
print("\n" + "="*65)
print("  NIVEL 8: GUIA DE DESPLIEGUE")
print("="*65)

guia = """
==========================================================================
          GUIA: PUBLICAR TU ANALISIS EN GITHUB, STREAMLIT Y RENDER
==========================================================================

--------------------------------------------------------------------------
ESTRUCTURA DE CARPETAS RECOMENDADA
--------------------------------------------------------------------------
  proyecto_intercambiador/
  |-- analisis_intercambiador_calor.py   <- script principal (este)
  |-- requirements.txt
  |-- README.md
  |-- .gitignore
  |-- data/
  |   |-- heat_exchanger_dataset.csv     <- NO subir si es confidencial
  |-- outputs/
      |-- analisis_intercambiador_calor.pdf
      |-- analisis_intercambiador_calor.html

--------------------------------------------------------------------------
REQUIREMENTS.TXT
--------------------------------------------------------------------------
  pandas
  numpy
  matplotlib
  scipy
  tensorflow
  scikit-learn

--------------------------------------------------------------------------
.GITIGNORE
--------------------------------------------------------------------------
  *.csv
  __pycache__/
  *.pyc
  .env
  outputs/

--------------------------------------------------------------------------
1) GITHUB — SUBIR CODIGO
--------------------------------------------------------------------------
  git init
  git add .
  git commit -m "feat: sistema analisis intercambiador calor GRU"
  git remote add origin https://github.com/TU_USUARIO/intercambiador-gru.git
  git branch -M main
  git push -u origin main

  Actualizaciones futuras:
    git add . && git commit -m "update: nuevos datos" && git push

--------------------------------------------------------------------------
2) STREAMLIT — APP WEB INTERACTIVA
--------------------------------------------------------------------------
  Crear app.py con este contenido minimo:

    import streamlit as st
    import pandas as pd

    st.title("Analisis Intercambiador de Calor")
    archivo = st.sidebar.file_uploader("Sube tu CSV", type="csv")
    if archivo:
        df = pd.read_csv(archivo)
        st.write(df.describe())

  Probar local:
    streamlit run app.py

  Publicar en Streamlit Cloud (GRATIS):
    https://share.streamlit.io -> conectar repo GitHub -> Deploy

--------------------------------------------------------------------------
3) RENDER — PUBLICAR HTML ESTATICO
--------------------------------------------------------------------------
  Opcion A (HTML estatico):
    1. Coloca analisis_intercambiador_calor.html en la raiz del repo
    2. Render -> New -> Static Site -> conectar repo
    3. Publish directory: .
    4. Create Static Site

  Opcion B (Streamlit en Render):
    Crear Procfile:
      web: streamlit run app.py --server.port $PORT --server.address 0.0.0.0
    Render -> New -> Web Service -> Python 3
    Build: pip install -r requirements.txt
    Start: (contenido del Procfile)

--------------------------------------------------------------------------
RESUMEN
--------------------------------------------------------------------------
  Solo versionar codigo   -> GitHub
  App interactiva facil   -> Streamlit Cloud
  HTML estatico online    -> Render (Static Site)
  App completa produccion -> Render (Web Service)

  ORDEN: GitHub -> Streamlit Cloud -> Render
==========================================================================
"""
print(guia)

# =============================================================================
# RESUMEN FINAL
# =============================================================================
print("="*65)
print("  SCRIPT COMPLETO FINALIZADO CON EXITO")
print("="*65)
print(f"\n  Graficas generadas : {len(figuras)}")
print(f"  PDF exportado      : analisis_intercambiador_calor.pdf")
print(f"  HTML exportado     : analisis_intercambiador_calor.html")
if resultados_gru:
    r = resultados_gru
    print(f"\n  MODELO GRU:")
    print(f"    MAE  = {r['mae']:.3f} kW")
    print(f"    RMSE = {r['rmse']:.3f} kW")
    print(f"    MAPE = {r['mape']:.2f}% — {r['clasificacion']}")
    print(f"    R2   = {r['r2']:.4f}")
print("\n==== FIN ====")
