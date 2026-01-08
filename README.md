# 🎯 SentimentAPI v3.0 - Análisis de Sentimientos en Español

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)

## 📋 Descripción del Proyecto

**SentimentAPI** es una solución completa de Machine Learning para el análisis de sentimientos de reseñas de Amazon en español. El sistema recibe textos y devuelve una clasificación de sentimiento con probabilidad calibrada y flag de revisión humana.

### Clasificaciones disponibles:
- 🟢 **Positivo** (estrellas 4-5)
- 🟡 **Neutro** (estrella 3)
- 🔴 **Negativo** (estrellas 1-2)

### Características principales:
- ✅ Probabilidad calibrada (0-1) interpretable
- ✅ Flag `review_required` para casos de baja confianza (<60%)
- ✅ Trazabilidad con versión del modelo y hash
- ✅ Endpoint batch para múltiples textos (hasta 100)
- ✅ Bundle único (`sentiment_bundle.joblib`) listo para producción

---

## 📊 Métricas del Modelo

| Métrica | Test Set |
|:--------|:--------:|
| **Accuracy** | 77.54% |
| **F1-macro** | 68.62% |
| **Recall Negativo** | 90.4% |
| **Recall Positivo** | 89.95% |

### Classification Report (Test)

```
              precision    recall  f1-score   support

    Negativo       0.78      0.90      0.84      2000
      Neutro       0.57      0.27      0.37      1000
    Positivo       0.81      0.90      0.85      2000

    accuracy                           0.78      5000
   macro avg       0.72      0.69      0.69      5000
weighted avg       0.75      0.78      0.75      5000
```

### Matriz de Confusión

|  | Negativo | Neutro | Positivo |
|:--|:--:|:--:|:--:|
| **Negativo** | 1808 | 109 | 83 |
| **Neutro** | 399 | 270 | 331 |
| **Positivo** | 104 | 97 | 1799 |

> 📝 **Nota:** El modelo prioriza alto recall en Negativo (90.4%) para no perder críticas importantes en atención al cliente.

---

## � Archivos Grandes (Descarga Requerida)

Los siguientes archivos superan el límite de GitHub y deben descargarse manualmente:

| Archivo | Descripción | 
|---------|-------------|
| `train.csv` | Dataset de entrenamiento (~200K reseñas de Amazon multilenguaje) |
| `sentiment_bundle.joblib` | Modelo ML serializado (pipeline completo + metadata) |

📎 **Descargar desde:** [Google Drive](https://drive.google.com/file/d/18Hd2lqwTytVHA7I5lbjQ6YzKWQ34ok36/view?usp=sharing)

> ⚠️ **Importante:** Coloca ambos archivos en la raíz del proyecto antes de ejecutar el notebook o la API.

---

## 🚀 Inicio Rápido

### 1. Descargar archivos grandes
Descarga `train.csv` y `sentiment_bundle.joblib` desde el [enlace de Google Drive](https://drive.google.com/file/d/18Hd2lqwTytVHA7I5lbjQ6YzKWQ34ok36/view?usp=sharing) y colócalos en la raíz del proyecto.

### 2. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 3. Ejecutar el notebook para entrenar el modelo
```bash
jupyter notebook Proyecto_final_v8.ipynb
# Ejecutar todas las celdas para generar sentiment_bundle.joblib
```

### 3. Iniciar la API
```bash
cd api
uvicorn main:app --reload --port 8000
```

### 4. Probar la API
```bash
# Health check
curl http://localhost:8000/health

# Análisis de sentimiento
curl -X POST "http://localhost:8000/sentiment" \
     -H "Content-Type: application/json" \
     -d '{"text": "El producto es excelente y llegó rápido"}'
```

### 5. Documentación interactiva
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## 🏗️ Arquitectura General

```
┌─────────────────────────────────────────────────────────────────────┐
│                     SentimentAPI v3.0                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────────┐         ┌──────────────────────────────┐ │
│  │   📊 Data Science    │         │      🐍 API (FastAPI)        │ │
│  │     (Notebook)       │         │                              │ │
│  │                      │         │  • POST /sentiment           │ │
│  │  • EDA               │  ────►  │  • POST /sentiment/batch     │ │
│  │  • Preprocesamiento  │         │  • GET /health               │ │
│  │  • TF-IDF            │         │  • GET /stats                │ │
│  │  • Logistic Regr.    │         │  • review_required flag      │ │
│  │  • Calibración       │         │  • Swagger UI                │ │
│  │  • Bundle joblib     │         │                              │ │
│  └──────────────────────┘         └──────────────────────────────┘ │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Pipeline de ML

```
Texto → TextCleaner → TF-IDF Vectorizer → Logistic Regression (Calibrada) → Predicción
```

**Componentes:**

1. **TextCleaner**: lowercase, eliminación de URLs/menciones/hashtags, conservación de caracteres españoles
2. **TF-IDF Vectorizer**: ngram_range=(1,2), min_df=3, max_df=0.95, max_features=200,000
3. **Logistic Regression**: solver=lbfgs, class_weight=balanced, C=1.0
4. **Calibración**: CalibratedClassifierCV con método sigmoid (3 folds)

---

## 📁 Estructura del Proyecto

```
📦 No Country/
├── 📓 Proyecto_final_v8.ipynb     # Notebook principal (MVP Final)
├── 📦 sentiment_bundle.joblib     # Bundle del modelo (pipeline + metadata)
├── 📊 train.csv                   # Dataset de entrenamiento (200K es)
├── 📊 validation.csv              # Dataset de validación (5K es)
├── 📊 test.csv                    # Dataset de test (5K es)
├── 📄 README.md                   # Esta documentación
├── 📄 requirements.txt            # Dependencias Python
└── 📁 api/
    └── main.py                    # API FastAPI v3.0
```

---

## 🔌 API Endpoints

### POST /sentiment
Analiza el sentimiento de un texto.

**Request:**
```json
{
  "text": "El producto es excelente y llegó rápido, muy recomendado."
}
```

**Response:**
```json
{
  "prevision": "Positivo",
  "probabilidad": 0.9249,
  "review_required": false,
  "threshold": 0.6,
  "model_version": "sentiment_es_tfidf_lr_calibrated_v1",
  "artifact_hash": "1c13b982c169"
}
```

### POST /sentiment/batch
Analiza múltiples textos en una sola petición (máx 100).

**Request:**
```json
{
  "texts": [
    "Excelente producto",
    "No funciona, llegó roto",
    "Está bien, cumple"
  ]
}
```

### GET /health
Verifica el estado de la API y del modelo.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "sentiment_es_tfidf_lr_calibrated_v1",
  "threshold": 0.6,
  "labels": ["Negativo", "Neutro", "Positivo"],
  "artifact_hash": "1c13b982c169"
}
```

### GET /stats
Obtiene estadísticas de uso.

**Response:**
```json
{
  "total_requests": 150,
  "positive_count": 95,
  "negative_count": 40,
  "neutral_count": 15,
  "review_required_count": 25,
  "review_required_percentage": 16.67,
  "avg_probability": 0.7823,
  "start_time": "2026-01-08T10:30:00"
}
```

### Campos de respuesta `/sentiment`

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `prevision` | string | Clase predicha: "Negativo", "Neutro", "Positivo" |
| `probabilidad` | float | Confianza del modelo (0.0 - 1.0) |
| `review_required` | bool | `true` si probabilidad < threshold |
| `threshold` | float | Umbral de confianza (0.6) |
| `model_version` | string | Versión del modelo |
| `artifact_hash` | string | Hash SHA256 del artefacto (12 chars) |

---

## 📓 Notebook: Pipeline Completo

El notebook `Proyecto_final_v8.ipynb` está organizado en **14 secciones**:

| Sección | Contenido |
|---------|-----------|
| **0-2** | Setup, carga de datos, EDA rápida |
| **3** | Preparación de texto y labels (stars → sentiment) |
| **4** | Baseline TF-IDF + Logistic Regression |
| **5** | Optimización con GridSearchCV (CV estratificada) |
| **6** | Calibración de probabilidades (sigmoid) |
| **7** | Política de revisión (`review_required`) |
| **8** | Evaluación final en test holdout |
| **9** | Explicabilidad (términos influyentes por clase) |
| **10** | Serialización productiva (bundle joblib) |
| **11** | Funciones para Back-End (`validate_text`, `predict_one`) |
| **12-14** | Contrato API, ejemplos cURL, notas de producción |

### Términos más influyentes por clase

| Clase | Términos Positivos | Términos Negativos |
|-------|-------------------|-------------------|
| **Negativo** | no, mala, no funciona, fatal, decepción, roto | perfecto, buena, genial, excelente, bien |
| **Neutro** | pero, tres estrellas, regular, aceptable, mejorable | excelente, recomiendo, perfecto, genial |
| **Positivo** | perfecto, genial, excelente, encantado, recomendable | no, mala, regular, no funciona, mal |

---

## 🛠️ Tecnologías Utilizadas

### Data Science (Notebook)
| Tecnología | Uso |
|------------|-----|
| 🐍 Python 3.10+ | Lenguaje principal |
| 📊 Pandas, NumPy | Manipulación de datos |
| 📈 Matplotlib, Seaborn | Visualizaciones |
| 🤖 scikit-learn | Modelo ML (TF-IDF + LogReg + Calibración) |
| 💾 joblib | Serialización del modelo |

### API (Producción)
| Tecnología | Uso |
|------------|-----|
| ⚡ FastAPI | Framework web REST |
| 📝 Pydantic | Validación de datos |
| 🔄 Uvicorn | Servidor ASGI |

---

## 📊 Ejemplos de Uso

### Python
```python
import requests

response = requests.post(
    "http://localhost:8000/sentiment",
    json={"text": "El producto es excelente y llegó rápido"}
)
result = response.json()
print(f"Sentimiento: {result['prevision']}")
print(f"Confianza: {result['probabilidad']:.2%}")
print(f"Requiere revisión: {result['review_required']}")
```

### cURL
```bash
# Positivo
curl -X POST "http://localhost:8000/sentiment" \
     -H "Content-Type: application/json" \
     -d '{"text": "El producto es excelente, llegó rápido y funciona perfecto."}'

# Negativo
curl -X POST "http://localhost:8000/sentiment" \
     -H "Content-Type: application/json" \
     -d '{"text": "Llegó roto, pésimo servicio y nadie responde."}'

# Neutro
curl -X POST "http://localhost:8000/sentiment" \
     -H "Content-Type: application/json" \
     -d '{"text": "Está bien, cumple lo prometido, nada especial."}'

# Batch
curl -X POST "http://localhost:8000/sentiment/batch" \
     -H "Content-Type: application/json" \
     -d '{"texts": ["Excelente", "Malo", "Normal"]}'
```

### JavaScript (fetch)
```javascript
const response = await fetch('http://localhost:8000/sentiment', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({text: 'Está bien, cumple lo prometido'})
});
const data = await response.json();
console.log(data.prevision);        // "Neutro"
console.log(data.review_required);  // true (baja confianza)
```

### Cargar el modelo directamente en Python
```python
import joblib
import numpy as np

# Cargar el bundle
bundle = joblib.load("sentiment_bundle.joblib")
pipeline = bundle["pipeline"]
meta = bundle["meta"]

# Predecir
text = "El producto es excelente"
proba = pipeline.predict_proba([text])[0]
pred = pipeline.classes_[np.argmax(proba)]
print(f"Sentimiento: {pred}, Confianza: {max(proba):.2%}")
```

---

## 🎯 Casos de Uso

| Área | Aplicación |
|------|------------|
| 📞 **Atención al Cliente** | Clasificación automática de tickets y priorización |
| 📈 **Marketing** | Análisis de campañas y percepción de marca |
| 📊 **Monitoreo** | Dashboard de sentimientos en tiempo real |
| 🛒 **E-commerce** | Análisis de reseñas de productos |
| 📱 **Redes Sociales** | Monitoreo de menciones y tweets |

---

## 🚀 Despliegue

### Local
```bash
# Instalar dependencias
pip install -r requirements.txt

# Ejecutar notebook para entrenar modelo (genera sentiment_bundle.joblib)
jupyter notebook Proyecto_final_v8.ipynb

# Ejecutar API
cd api
uvicorn main:app --reload --port 8000
```

### Docker (Opcional)
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY sentiment_bundle.joblib .
COPY api/ ./api/
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t sentiment-api .
docker run -p 8000:8000 sentiment-api
```

---

## 📌 Notas de Producción

### Ventajas del enfoque

| Aspecto | Beneficio |
|---------|-----------|
| **Eficiencia** | TF-IDF + LR es rápido y CPU-friendly |
| **Escalabilidad** | Modelo ligero, se carga una vez al iniciar |
| **Probabilidad confiable** | Calibración sigmoid mejora interpretación |
| **Robustez operacional** | `review_required` deriva casos dudosos a humanos |
| **Reproducibilidad** | Semilla fija + metadata con versión y hash |
| **Integración simple** | Un solo archivo `.joblib` contiene todo |

### Limitaciones conocidas

- **Clase Neutro**: Menor precisión (57%) y recall (27%) — típico en clasificación ternaria
- **Idioma**: Solo español (filtrado por `language == "es"`)
- **Longitud**: Textos entre 5 y 2000 caracteres

### Recomendaciones

1. **Monitoreo**: Trackear % de `review_required` en producción
2. **Feedback loop**: Usar casos revisados para re-entrenar
3. **Umbral ajustable**: Modificar `threshold` según tolerancia al riesgo
4. **Cache**: Considerar cache para textos repetidos

---

## ✅ Funcionalidades Implementadas

- [x] Notebook completo con EDA, preprocesamiento y entrenamiento
- [x] Clasificación ternaria (Positivo, Neutro, Negativo)
- [x] Modelo calibrado con probabilidades interpretables
- [x] Flag `review_required` para revisión humana
- [x] Endpoint POST /sentiment con clasificación y probabilidad
- [x] Endpoint POST /sentiment/batch para múltiples textos
- [x] Bundle único (pipeline + metadata) en joblib
- [x] Validación de input (5-2000 caracteres)
- [x] Trazabilidad (model_version, artifact_hash)
- [x] Endpoint GET /health con info del modelo
- [x] Endpoint GET /stats para estadísticas
- [x] Documentación Swagger automática

## 🔮 Funcionalidades Opcionales (Para Extender)

- [ ] Persistencia en base de datos (PostgreSQL)
- [ ] Interfaz web con Streamlit
- [ ] Explicabilidad (top features por predicción)
- [ ] Contenerización completa con docker-compose
- [ ] Tests automatizados con pytest
- [ ] Ajuste dinámico de threshold

---

## 👥 Equipo

Proyecto desarrollado por **"No Data - No Code"** en el marco del Hackatón **No Country** 🌎

## 📄 Licencia

Este proyecto está bajo licencia MIT.

---

<p align="center">
  <i>💡 Transformando feedback en insights accionables</i>
</p>

