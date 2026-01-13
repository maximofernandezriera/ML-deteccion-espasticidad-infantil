# 🧠 Aplicación de algoritmos de machine learning tradicional para el apoyo al diagnóstico temprano de espasticidad en recién nacidos

Mediante la **utilización la inteligencia artificial este proyecto pretende ayudar a detectar la espasticidad en los recién nacidos de forma temprana**.

## 🎯 ¿Por qué este proyecto?

Este sistema analiza el datset público de vídeo y, mediante técnicas de Machine Learning, identifica patrones que pueden indicar riesgo de espasticidad. Su objetivo es aproximarse al ojo clínico de un profesional especialista en el campo.

## 🛠️ ¿Cómo funciona?

```
Dataset público → Extracción de características → Modelos ML → Predicción de riesgo
```

El pipeline combina tres tipos de análisis:

- **Flujo óptico**: Detecta cómo es el movimiento frame a frame
- **Características temporales**: Analiza la dinámica del movimiento a lo largo del tiempo
- **Características espaciales**: Estudia la distribución del movimiento en diferentes partes del cuerpo

Cuatro modelos trabajan en conjunto para ofrecer predicciones robustas: 
- Regresión Logística
- Random Forest
- SVM (Support Vector Machine)
- XGBoost

## 🚀 Uso

```bash
# Ejecutar el pipeline completo
python main_pipeline.py
```

El sistema generará: 
- Modelos entrenados en `/models`
- Reportes de evaluación en `/reports`
- Visualizaciones explicativas con SHAP

## 📁 Estructura del proyecto

```
├── main_pipeline.py          # Pipeline principal
├── config.yaml               # Configuración centralizada
├── exportar_videos_npz.py    # Utilidad para exportar videos
├── src/
│   ├── data/                 # Carga y división de datos
│   ├── features/             # Extracción de características
│   ├── models/               # Implementación de modelos
│   ├── evaluation/           # Métricas clínicas
│   └── explainability/       # Análisis SHAP
└── requirements.txt
```

## 📊 Métricas de evaluación

El sistema prioriza métricas clínicamente relevantes:
- **Sensibilidad ≥ 90%**: Minimizar falsos negativos
- **AUC-ROC ≥ 85%**: Capacidad discriminativa general
- **Especificidad ≥ 75%**:  Reducir falsos positivos

## 🔬 Interpretabilidad

NMediante **SHAP (SHapley Additive exPlanations)**, cada predicción viene acompañada de una explicación visual de qué características influyeron en el resultado.

## 👨‍🎓 Contexto académico

Este proyecto forma parte de un **Trabajo de Fin de Máster (TFM)** en la Universitat Oberta de Catalunya (UOC), desarrollado por Máximo Fernández Riera. 

## 📄 Licencia

Este proyecto está disponible como código abierto para fines educativos y de investigación.

---

*"Tecnología al servicio de los pequeños por Máximo Fernández Riera"* 💙
