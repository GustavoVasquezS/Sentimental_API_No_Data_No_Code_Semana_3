"""
SentimentAPI v3.0 - API de Análisis de Sentimientos en Español
==============================================================

API REST para clasificación de sentimientos en reseñas de Amazon en español.
Compatible con el modelo `sentiment_bundle.joblib` del notebook Proyecto_final_v8.

Clasificaciones: Positivo, Neutro, Negativo
- Negativo: estrellas 1-2
- Neutro: estrella 3
- Positivo: estrellas 4-5

Características:
- Probabilidad calibrada (0-1)
- Flag `review_required` para casos de baja confianza
- Metadata del modelo incluida en cada respuesta

Ejecutar con: uvicorn main:app --reload --port 8000
Documentación: http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
import joblib
import numpy as np
import os
from typing import Optional, Dict, Any, List
from datetime import datetime
import logging

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURACIÓN DE LA API
# =============================================================================

app = FastAPI(
    title="SentimentAPI v3.0 - Amazon ES",
    description="""
API de Análisis de Sentimientos para reseñas de Amazon en español.

## Características
- **Modelo calibrado**: Probabilidades interpretables (0-1)
- **Review automático**: Flag `review_required` cuando confianza < 60%
- **Trazabilidad**: Versión del modelo y hash en cada respuesta

## Clasificación
| Estrellas | Sentimiento |
|:---------:|:-----------:|
| 4-5 ⭐ | Positivo |
| 3 ⭐ | Neutro |
| 1-2 ⭐ | Negativo |
    """,
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configurar CORS para permitir peticiones desde cualquier origen
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# RUTAS DE ARCHIVOS DEL MODELO
# =============================================================================

# El bundle está en la raíz del proyecto (un nivel arriba de api/)
BUNDLE_PATH = os.path.join(os.path.dirname(__file__), "..", "sentiment_bundle.joblib")

# Variables globales para el modelo
bundle: Optional[Dict[str, Any]] = None
pipeline = None
model_meta: Optional[Dict[str, Any]] = None

# Estadísticas de uso
stats = {
    "total_requests": 0,
    "positive_count": 0,
    "negative_count": 0,
    "neutral_count": 0,
    "review_required_count": 0,
    "avg_probability": 0.0,
    "start_time": datetime.now().isoformat()
}

# =============================================================================
# MODELOS PYDANTIC (VALIDACIÓN)
# =============================================================================

class TextInput(BaseModel):
    """Modelo de entrada para el análisis de sentimiento."""
    text: str = Field(
        ..., 
        min_length=5, 
        max_length=2000, 
        description="Texto a analizar (5-2000 caracteres)"
    )
    
    @field_validator('text')
    @classmethod
    def text_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('El texto no puede estar vacío')
        stripped = v.strip()
        if len(stripped) < 5:
            raise ValueError('El texto debe tener al menos 5 caracteres')
        return stripped


class SentimentResponse(BaseModel):
    """Modelo de respuesta del análisis de sentimiento."""
    prevision: str = Field(..., description="Sentimiento predicho: Positivo, Neutro o Negativo")
    probabilidad: float = Field(..., description="Probabilidad/confianza de la predicción (0-1)")
    review_required: bool = Field(..., description="True si probabilidad < threshold (requiere revisión humana)")
    threshold: float = Field(..., description="Umbral de confianza configurado")
    model_version: str = Field(..., description="Versión del modelo")
    artifact_hash: str = Field(..., description="Hash del artefacto para trazabilidad")


class ErrorResponse(BaseModel):
    """Modelo de respuesta de error."""
    error: str = Field(..., description="Tipo de error")
    detail: str = Field(..., description="Descripción del error")


class HealthResponse(BaseModel):
    """Modelo de respuesta del health check."""
    status: str
    model_loaded: bool
    model_version: Optional[str] = None
    threshold: Optional[float] = None
    labels: Optional[List[str]] = None
    artifact_hash: Optional[str] = None


class StatsResponse(BaseModel):
    """Modelo de respuesta de estadísticas."""
    total_requests: int
    positive_count: int
    negative_count: int
    neutral_count: int
    review_required_count: int
    review_required_percentage: float
    avg_probability: float
    start_time: str


# =============================================================================
# FUNCIONES DE CARGA DEL MODELO
# =============================================================================

def load_bundle():
    """Carga el bundle (pipeline + metadata) al iniciar la API."""
    global bundle, pipeline, model_meta
    
    try:
        if not os.path.exists(BUNDLE_PATH):
            raise FileNotFoundError(f"No se encontró el archivo: {BUNDLE_PATH}")
        
        bundle = joblib.load(BUNDLE_PATH)
        pipeline = bundle["pipeline"]
        model_meta = bundle["meta"]
        
        logger.info("✅ Bundle cargado exitosamente")
        logger.info(f"   Versión: {model_meta['model_version']}")
        logger.info(f"   Labels: {model_meta['labels']}")
        logger.info(f"   Threshold: {model_meta['threshold']}")
        logger.info(f"   Hash: {model_meta['artifact_hash']}")
        
    except Exception as e:
        logger.error(f"❌ Error cargando bundle: {e}")
        raise


# =============================================================================
# FUNCIÓN DE PREDICCIÓN
# =============================================================================

def predict_sentiment(text: str) -> Dict[str, Any]:
    """
    Predice el sentimiento de un texto.
    
    Args:
        text: Texto a analizar (ya validado)
    
    Returns:
        dict con predicción, probabilidad y metadata
    """
    global stats
    
    # El pipeline ya incluye TextCleaner, no necesitamos preprocesar
    proba = pipeline.predict_proba([text])[0]
    classes = pipeline.classes_
    
    # Obtener la clase con mayor probabilidad
    idx = int(np.argmax(proba))
    pred = str(classes[idx])
    max_prob = float(proba[idx])
    
    # Determinar si requiere revisión humana
    threshold = float(model_meta["threshold"])
    needs_review = max_prob < threshold
    
    # Actualizar estadísticas
    stats["total_requests"] += 1
    n = stats["total_requests"]
    stats["avg_probability"] = stats["avg_probability"] + (max_prob - stats["avg_probability"]) / n
    
    if needs_review:
        stats["review_required_count"] += 1
    
    if pred == "Positivo":
        stats["positive_count"] += 1
    elif pred == "Negativo":
        stats["negative_count"] += 1
    else:
        stats["neutral_count"] += 1
    
    return {
        "prevision": pred,
        "probabilidad": round(max_prob, 4),
        "review_required": needs_review,
        "threshold": threshold,
        "model_version": str(model_meta["model_version"]),
        "artifact_hash": str(model_meta["artifact_hash"])
    }


# =============================================================================
# EVENTOS DE CICLO DE VIDA
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Evento ejecutado al iniciar la API."""
    logger.info("🚀 Iniciando SentimentAPI v3.0...")
    load_bundle()
    logger.info("✅ API lista para recibir peticiones")


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/", tags=["General"])
async def root():
    """Endpoint raíz con información de la API."""
    return {
        "nombre": "SentimentAPI v3.0 - Amazon ES",
        "version": "3.0.0",
        "descripcion": "API de Análisis de Sentimientos para reseñas de Amazon en español",
        "modelo": {
            "version": model_meta["model_version"] if model_meta else None,
            "labels": model_meta["labels"] if model_meta else None,
            "threshold": model_meta["threshold"] if model_meta else None,
            "artifact_hash": model_meta["artifact_hash"] if model_meta else None
        },
        "endpoints": {
            "POST /sentiment": "Analizar sentimiento de un texto",
            "GET /health": "Estado de la API y del modelo",
            "GET /stats": "Estadísticas de uso"
        },
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Verifica el estado de la API y del modelo."""
    return HealthResponse(
        status="healthy" if pipeline is not None else "unhealthy",
        model_loaded=pipeline is not None,
        model_version=model_meta["model_version"] if model_meta else None,
        threshold=model_meta["threshold"] if model_meta else None,
        labels=model_meta["labels"] if model_meta else None,
        artifact_hash=model_meta["artifact_hash"] if model_meta else None
    )


@app.get("/stats", response_model=StatsResponse, tags=["General"])
async def get_stats():
    """Obtiene estadísticas de uso de la API."""
    total = stats["total_requests"]
    review_pct = (stats["review_required_count"] / total * 100) if total > 0 else 0
    
    return StatsResponse(
        total_requests=total,
        positive_count=stats["positive_count"],
        negative_count=stats["negative_count"],
        neutral_count=stats["neutral_count"],
        review_required_count=stats["review_required_count"],
        review_required_percentage=round(review_pct, 2),
        avg_probability=round(stats["avg_probability"], 4),
        start_time=stats["start_time"]
    )


@app.post(
    "/sentiment", 
    response_model=SentimentResponse,
    responses={
        200: {"description": "Predicción exitosa"},
        400: {"model": ErrorResponse, "description": "Error de validación"},
        500: {"model": ErrorResponse, "description": "Error interno"}
    },
    tags=["Predicción"]
)
async def analyze_sentiment(input_data: TextInput):
    """
    Analiza el sentimiento de un texto en español.
    
    ## Request
    ```json
    { "text": "El producto es excelente y llegó rápido" }
    ```
    
    ## Response
    - **prevision**: Sentimiento predicho (Positivo, Neutro, Negativo)
    - **probabilidad**: Confianza del modelo (0.0 - 1.0)
    - **review_required**: `true` si probabilidad < threshold
    - **threshold**: Umbral de confianza (0.6 por defecto)
    - **model_version**: Versión del modelo para trazabilidad
    - **artifact_hash**: Hash único del artefacto
    
    ## Interpretación de `review_required`
    - `false`: El modelo está seguro, se puede usar la predicción automáticamente
    - `true`: Baja confianza, se recomienda revisión humana
    """
    # Verificar que el modelo está cargado
    if pipeline is None or model_meta is None:
        raise HTTPException(
            status_code=500, 
            detail="Modelo no cargado. Reinicie la API."
        )
    
    try:
        # Realizar predicción
        result = predict_sentiment(input_data.text)
        
        # Log de la predicción
        emoji = "🟢" if result["prevision"] == "Positivo" else ("🔴" if result["prevision"] == "Negativo" else "🟡")
        review_flag = "⚠️" if result["review_required"] else "✓"
        logger.info(
            f"{emoji} {result['prevision']} | "
            f"Prob: {result['probabilidad']:.2%} {review_flag} | "
            f"'{input_data.text[:50]}...'"
        )
        
        return SentimentResponse(**result)
        
    except Exception as e:
        logger.error(f"Error en predicción: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error procesando el texto: {str(e)}"
        )


# =============================================================================
# ENDPOINT BATCH (PARA MÚLTIPLES TEXTOS)
# =============================================================================

class BatchInput(BaseModel):
    """Entrada para análisis en lote."""
    texts: List[str] = Field(..., min_length=1, max_length=100, description="Lista de textos (máx 100)")


class BatchResponse(BaseModel):
    """Respuesta del análisis en lote."""
    results: List[SentimentResponse]
    total: int
    successful: int
    failed: int


@app.post("/sentiment/batch", response_model=BatchResponse, tags=["Predicción"])
async def analyze_sentiment_batch(input_data: BatchInput):
    """
    Analiza el sentimiento de múltiples textos en una sola petición.
    
    Máximo 100 textos por petición.
    """
    if pipeline is None or model_meta is None:
        raise HTTPException(
            status_code=500, 
            detail="Modelo no cargado. Reinicie la API."
        )
    
    results = []
    failed = 0
    
    for text in input_data.texts:
        try:
            # Validar longitud mínima
            if not text or len(text.strip()) < 5:
                failed += 1
                continue
            
            result = predict_sentiment(text.strip())
            results.append(SentimentResponse(**result))
            
        except Exception:
            failed += 1
    
    return BatchResponse(
        results=results,
        total=len(input_data.texts),
        successful=len(results),
        failed=failed
    )


# =============================================================================
# EJECUCIÓN DIRECTA
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
