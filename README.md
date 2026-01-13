# 🧠 Detección de Espasticidad Infantil mediante Machine Learning

> *Porque cada segundo cuenta en el diagnóstico temprano*

Este proyecto nace de una idea simple pero poderosa: **utilizar la inteligencia artificial para ayudar a detectar espasticidad en bebés de forma temprana**, cuando la intervención terapéutica puede marcar la diferencia entre una vida con limitaciones y un desarrollo pleno.

## 🎯 ¿Por qué este proyecto?

La espasticidad infantil es una condición que afecta el control muscular en niños pequeños. Un diagnóstico tardío puede significar perder la ventana crítica de neuroplasticidad, ese período mágico donde el cerebro infantil tiene una capacidad extraordinaria de adaptación.

Este sistema analiza **videos de movimientos espontáneos** de bebés y, mediante técnicas de Machine Learning, identifica patrones que pueden indicar riesgo de espasticidad. 

## 🛠️ ¿Cómo funciona?

```
Video del bebé → Extracción de características → Modelo ML → Predicción de riesgo
```

El pipeline combina tres tipos de análisis:

- **Flujo óptico**: Detecta cómo se mueve el bebé frame a frame
- **Características temporales**: Analiza la dinámica del movimiento a lo largo del tiempo
- **Características espaciales**: Estudia la distribución del movimiento en diferentes partes del cuerpo

Cuatro modelos trabajan en conjunto para ofrecer predicciones robustas: 
- Regresión Logística
- Random Forest
- SVM (Support Vector Machine)
- XGBoost

## 📋 Requisitos

```bash
# Clonar el repositorio
git clone https://github.com/maximofernandezriera/ML-deteccion-espasticidad-infantil.git
cd ML-deteccion-espasticidad-infantil

# Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

**Dependencias principales**:  NumPy, Pandas, Scikit-learn, XGBoost, OpenCV, SHAP

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
- **Sensibilidad ≥ 90%**: Minimizar falsos negativos (no pasar por alto casos reales)
- **AUC-ROC ≥ 85%**: Capacidad discriminativa general
- **Especificidad ≥ 75%**:  Reducir falsos positivos

## 🔬 Interpretabilidad

No nos conformamos con un modelo "caja negra". Mediante **SHAP (SHapley Additive exPlanations)**, cada predicción viene acompañada de una explicación visual de qué características influyeron en el resultado.

## 👨‍🎓 Contexto académico

Este proyecto forma parte de un **Trabajo de Fin de Máster (TFM)** en la Universitat Oberta de Catalunya (UOC), desarrollado por Máximo Fernández Riera. 

## 📄 Licencia

Este proyecto está disponible como código abierto para fines educativos y de investigación.

---

*"La tecnología al servicio de los más pequeños"* 💙