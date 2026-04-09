# Sistema de Análisis y Reporte de Datos
## Intercambiador de Calor — Preheat Train de Crudo

---

## Descripción

Sistema de análisis de datos operacionales para un intercambiador de calor industrial, desarrollado sobre un dataset de simulación paramétrica generado con DWSIM. El sistema integra estadística descriptiva, detección de anomalías con Z-Score, modelo predictivo GRU (Deep Learning) y exportación automática de reportes en PDF y HTML autocontenido.

**Aplicación industrial:** Monitoreo continuo de intercambiadores en preheat train de crudo, detección de ensuciamiento (fouling) y mantenimiento predictivo.

---

## Dataset

- **Fuente:** [Heat Exchanger Parametric Study — Kaggle](https://www.kaggle.com/)
- **Filas:** 10,000 condiciones operacionales
- **Columnas:** 20 (10 originales + 10 con ruido gaussiano que simula lectura SCADA)
- **Variable objetivo (modelo GRU):** `hx_1_heat_load_kw` — Carga Térmica (kW)

| Variable | Descripción | Unidad |
|---|---|---|
| hot_inlet_temperature_k | Temperatura entrada corriente caliente | K |
| cold_inlet_mass_flow_kg_s | Flujo masa entrada corriente fría | kg/s |
| hx_1_heat_load_kw | Carga térmica HX-1 **(TARGET)** | kW |
| hx_1_lmtd_k | Diferencia de temperatura logarítmica (LMTD) | K |
| *_noisy | Señales con ruido gaussiano (SCADA simulado) | — |

---

## Estructura del Script

El sistema opera en **8 niveles** dentro de un único script Python:

```
NIVEL 1 — Ingesta, limpieza y estadística descriptiva
NIVEL 2 — Correlaciones, Delta-P y KPIs operacionales
NIVEL 3 — Comparación señal original vs SCADA (Fouling Index)
NIVEL 4 — Detección de anomalías Z-Score sobre señales SCADA
NIVEL 5 — Modelo GRU: predicción de carga térmica (kW)
NIVEL 6 — Exportación a PDF multipágina (portada + 8 gráficas)
NIVEL 7 — Exportación a HTML autocontenido (Base64)
NIVEL 8 — Guía de despliegue: GitHub / Streamlit / Render
```

---

## Instalación y Ejecución

### Requisitos

```bash
pip install -r requirements.txt
```

### Ejecutar

```bash
python analisis_intercambiador_calor.py
```

> Coloca `heat_exchanger_dataset.csv` en la misma carpeta o ajusta `FILE_PATH` al inicio del script.

### Salidas generadas

```
analisis_intercambiador_calor.pdf   — Reporte PDF (portada + 8 gráficas)
analisis_intercambiador_calor.html  — Dashboard HTML autocontenido
```

---

## Stack Tecnológico

| Capa | Herramienta | Función |
|---|---|---|
| Procesamiento | Python 3 + Pandas + NumPy | Ingesta, limpieza y transformación |
| Visualización | Matplotlib | Gráficas estadísticas en Base64 |
| IA Predictiva | TensorFlow 2 + Keras (GRU) | Red neuronal recurrente con ventana dinámica |
| Exportación | HTML5 + Base64 + PdfPages | Reportes autocontenidos sin dependencias |


---

## Gráficas Generadas

1. Histogramas de las 10 variables del proceso
2. Mapa de calor de correlaciones (Pearson)
3. Curva de operación: T_hot_in y Flujo vs Heat Load
4. KPIs: Delta-P diferencial y Eficiencia Térmica
5. Comparación señal original vs SCADA (4 variables)
6. Índice de ensuciamiento (Fouling Index) — LMTD
7. Detección de anomalías Z-Score por variable SCADA
8. Boxplot: valores atípicos simulación vs SCADA
9. GRU: histórico + predicción test + forecast *(requiere TensorFlow)*
10. GRU: scatter predicción vs real + residuos *(requiere TensorFlow)*

---

## Modelo GRU — Métricas Esperadas

| Métrica | Valor esperado | Descripción |
|---|---|---|
| MAE | < 5 kW | Error absoluto medio |
| RMSE | < 8 kW | Error cuadrático medio |
| MAPE | < 5% (Excelente) / < 15% (Aceptable) | Error porcentual |
| R² | > 0.95 | Coeficiente de determinación |

---

## Despliegue

### Streamlit Cloud (app interactiva)
1. Fork este repositorio
2. Ve a [share.streamlit.io](https://share.streamlit.io)
3. Conecta el repo → selecciona `analisis_intercambiador_calor.py` → Deploy

### Render.com (HTML estático)
1. New → Static Site → conecta el repo
2. Publish directory: `.`
3. Create Static Site

---

## Consideraciones de Seguridad

- **No subir el CSV al repositorio** si los datos son confidenciales (ver `.gitignore`)
- Verificar el reporte localmente antes de cada despliegue
- Los archivos de salida PDF/HTML no se versionan

---

## Licencia

MIT License — Dataset original: Heat Exchanger Parametric Study (Kaggle)
