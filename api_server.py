# api_server.py - Servidor API Empresarial para PromptGen
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import sys
import os
import logging
import time

# Configuración de logging empresarial
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Añadir el directorio actual al path para importar módulos empresariales
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Importar el nuevo sistema empresarial
try:
    from promptgen_enterprise import (
        analyze_prompt_quality_bart,
        get_structural_feedback,
        generate_variations,
        generate_ideas,
        EnterpriseModelManager,
        AdvancedQualityAnalyzer,
        ProgressiveImprovementEngine
    )
    from monitoring_system import get_monitoring_system
    
    logger.info("✅ Sistema empresarial PromptGen cargado exitosamente")
    
    # Inicializar componentes empresariales
    model_manager = EnterpriseModelManager()
    quality_analyzer = AdvancedQualityAnalyzer()
    improvement_engine = ProgressiveImprovementEngine(model_manager, quality_analyzer)
    monitoring = get_monitoring_system()
    
except ImportError as e:
    logger.error(f"❌ Error al importar sistema empresarial: {e}")
    # Funciones dummy para que el servidor pueda arrancar
    def analyze_prompt_quality_bart(prompt: str):
        return {"error": "Sistema empresarial no disponible"}
    def get_structural_feedback(prompt: str, model_name: str):
        return {"error": "Sistema empresarial no disponible"}
    def generate_variations(prompt: str, model_name: str, num_variations: int):
        return {"error": "Sistema empresarial no disponible"}
    def generate_ideas(prompt: str, model_name: str, num_ideas: int):
        return {"error": "Sistema empresarial no disponible"}
    monitoring = None

# Importar sistema real como alternativa
try:
    from promptgen_real_system import (
        RealIterativeImprover, RealQualityAnalyzer, RealModelManager,
        improve_iteratively_real, analyze_quality_real
    )
    REAL_SYSTEM_AVAILABLE = True
    logger.info("✅ Sistema real de mejora cargado como alternativa")
    
    # Inicializar sistema real
    real_improver = RealIterativeImprover()
    real_analyzer = RealQualityAnalyzer()
    
except ImportError as e:
    REAL_SYSTEM_AVAILABLE = False
    logger.warning(f"⚠️ Sistema real no disponible: {e}")
    real_improver = None
    real_analyzer = None

app = FastAPI(
    title="PromptGen Enterprise API",
    description="API empresarial para análisis y mejora iterativa de prompts con modelos reales de Hugging Face.",
    version="2.0.0"
)

# Configuración de CORS empresarial
origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:3001",  # Puerto alternativo
    "https://promptgen.enterprise.com",  # Dominio de producción
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Modelos Pydantic Empresariales ---

class PromptRequest(BaseModel):
    prompt: str
    
    class Config:
        schema_extra = {
            "example": {
                "prompt": "Quiero crear una página web para una cafetería"
            }
        }

class ModelNameRequest(PromptRequest):
    model_name: str = "gpt2"
    
    class Config:
        schema_extra = {
            "example": {
                "prompt": "Desarrollar un sistema de gestión empresarial",
                "model_name": "gpt2"
            }
        }

class VariationRequest(ModelNameRequest):
    num_variations: int = 3
    
class IdeaRequest(ModelNameRequest):
    num_ideas: int = 3

class IterativeImprovementRequest(BaseModel):
    prompt: str
    model_name: str = "gpt2"
    max_iterations: int = 5
    target_quality: float = 85.0
    
    class Config:
        schema_extra = {
            "example": {
                "prompt": "Crear una aplicación móvil",
                "model_name": "gpt2",
                "max_iterations": 3,
                "target_quality": 80.0
            }
        }

# --- Middleware Empresarial de Monitoreo ---

@app.middleware("http")
async def monitoring_middleware(request: Request, call_next):
    """Middleware empresarial para monitoreo de requests"""
    start_time = time.time()
    
    # Registrar inicio de request
    if monitoring:
        monitoring.record_session_activity(str(request.client.host if request.client else "unknown"))
    
    try:
        response = await call_next(request)
        
        # Calcular tiempo de respuesta
        response_time = time.time() - start_time
        
        # Registrar métricas
        if monitoring:
            monitoring.record_request(
                endpoint=str(request.url.path),
                response_time=response_time,
                success=response.status_code < 400
            )
        
        # Añadir headers de monitoreo
        response.headers["X-Response-Time"] = f"{response_time:.3f}s"
        response.headers["X-Server-Version"] = "PromptGen-Enterprise-2.0.0"
        
        return response
        
    except Exception as e:
        # Registrar error
        response_time = time.time() - start_time
        if monitoring:
            monitoring.record_request(
                endpoint=str(request.url.path),
                response_time=response_time,
                success=False
            )
        
        logger.error(f"❌ Error en request {request.url.path}: {e}")
        raise

# --- Endpoints Empresariales ---

@app.post("/api/analyze_quality")
async def api_analyze_quality(request_data: PromptRequest):
    """
    Análisis avanzado de calidad del prompt con métricas empresariales.
    
    Utiliza el analizador empresarial que evalúa:
    - Completitud
    - Claridad  
    - Especificidad
    - Estructura
    - Coherencia
    - Accionabilidad
    """
    if not request_data.prompt or not request_data.prompt.strip():
        raise HTTPException(status_code=400, detail="El prompt no puede estar vacío.")
    
    try:
        start_time = time.time()
        logger.info(f"📊 Analizando calidad del prompt: {request_data.prompt[:50]}...")
        
        # Usar el sistema real si está disponible
        if REAL_SYSTEM_AVAILABLE and real_analyzer:
            logger.info("🤖 Usando análisis REAL de calidad")
            result = analyze_quality_real(request_data.prompt)
        else:
            logger.warning("⚠️ Usando análisis empresarial como fallback")
            result = analyze_prompt_quality_bart(request_data.prompt)
        
        # Registrar métricas de análisis
        analysis_time = time.time() - start_time
        if monitoring:
            monitoring._update_performance_metrics(quality_analysis_time=analysis_time)
        
        if result.get("error"):
            raise HTTPException(status_code=500, detail=result.get("error"))
            
        logger.info("✅ Análisis de calidad completado")
        return result
        
    except Exception as e:
        logger.error(f"❌ Error en análisis de calidad: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {str(e)}")

@app.post("/api/generate_feedback")
async def api_structural_feedback(request_data: ModelNameRequest):
    """
    Genera feedback estructural inteligente basado en análisis de calidad.
    """
    if not request_data.prompt or not request_data.prompt.strip():
        raise HTTPException(status_code=400, detail="El prompt no puede estar vacío.")
        
    try:
        logger.info(f"💡 Generando feedback para: {request_data.prompt[:50]}...")
        result = get_structural_feedback(request_data.prompt, request_data.model_name)
        
        if result.get("error"):
            raise HTTPException(status_code=500, detail=result.get("error"))
            
        logger.info("✅ Feedback generado exitosamente")
        return result
        
    except Exception as e:
        logger.error(f"❌ Error generando feedback: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {str(e)}")

@app.post("/api/generate_variations")
async def api_generate_variations(request_data: VariationRequest):
    """
    Genera variaciones mejoradas del prompt usando modelos reales de Hugging Face.
    """
    if not request_data.prompt or not request_data.prompt.strip():
        raise HTTPException(status_code=400, detail="El prompt no puede estar vacío.")
        
    try:
        start_time = time.time()
        logger.info(f"🔄 Generando {request_data.num_variations} variaciones...")
        
        result = generate_variations(
            request_data.prompt, 
            request_data.model_name, 
            request_data.num_variations
        )
        
        # Registrar uso del modelo
        model_time = time.time() - start_time
        if monitoring:
            monitoring.record_model_usage(request_data.model_name, model_time)
        
        if result.get("error"):
            raise HTTPException(status_code=500, detail=result.get("error"))
            
        logger.info("✅ Variaciones generadas exitosamente")
        return result
        
    except Exception as e:
        logger.error(f"❌ Error generando variaciones: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {str(e)}")

@app.post("/api/generate_ideas")
async def api_generate_ideas(request_data: IdeaRequest):
    """
    Genera ideas creativas basadas en el prompt usando modelos reales de Hugging Face.
    """
    if not request_data.prompt or not request_data.prompt.strip():
        raise HTTPException(status_code=400, detail="El prompt no puede estar vacío.")
        
    try:
        start_time = time.time()
        logger.info(f"💡 Generando {request_data.num_ideas} ideas...")
        
        result = generate_ideas(
            request_data.prompt, 
            request_data.model_name, 
            request_data.num_ideas
        )
        
        # Registrar uso del modelo
        model_time = time.time() - start_time
        if monitoring:
            monitoring.record_model_usage(request_data.model_name, model_time)
        
        if result.get("error"):
            raise HTTPException(status_code=500, detail=result.get("error"))
            
        logger.info("✅ Ideas generadas exitosamente")
        return result
        
    except Exception as e:
        logger.error(f"❌ Error generando ideas: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {str(e)}")

@app.post("/api/improve_iteratively")
async def api_improve_iteratively(request_data: IterativeImprovementRequest):
    """
    Mejora iterativa empresarial con aprendizaje contextual y métricas avanzadas.
    
    Utiliza el motor de mejora progresiva que:
    - Analiza calidad con 6 métricas empresariales
    - Aplica mejoras inteligentes basadas en deficiencias
    - Aprende de iteraciones anteriores
    - Valida mejoras reales vs regresiones
    """
    if not request_data.prompt or not request_data.prompt.strip():
        raise HTTPException(status_code=400, detail="El prompt no puede estar vacío.")
    
    try:
        start_time = time.time()
        logger.info(f"🚀 Iniciando mejora iterativa empresarial...")
        logger.info(f"   Prompt: {request_data.prompt[:50]}...")
        logger.info(f"   Modelo: {request_data.model_name}")
        logger.info(f"   Max iteraciones: {request_data.max_iterations}")
        logger.info(f"   Calidad objetivo: {request_data.target_quality}%")
        
        # Usar el sistema real si está disponible
        if REAL_SYSTEM_AVAILABLE and real_improver:
            logger.info("🤖 Usando sistema REAL de mejora con modelos HuggingFace")
            result = real_improver.improve_prompt_iteratively(
                original_prompt=request_data.prompt,
                model_name=request_data.model_name,
                max_iterations=request_data.max_iterations,
                target_quality=request_data.target_quality
            )
        elif 'improvement_engine' in globals():
            logger.warning("⚠️ Usando sistema empresarial como fallback")
            result = improvement_engine.improve_iteratively(
                prompt=request_data.prompt,
                model_name=request_data.model_name,
                max_iterations=request_data.max_iterations,
                target_quality=request_data.target_quality
            )
        else:
            raise HTTPException(status_code=500, detail="Ningún motor de mejora disponible")
        
        # Registrar métricas de mejora
        total_time = time.time() - start_time
        if monitoring and result.get('iterations'):
            # Calcular mejora de calidad
            iterations = result['iterations']
            if len(iterations) > 0:
                original_quality = iterations[0].get('quality_score', 0)
                final_quality = iterations[-1].get('quality_score', 0)
                monitoring.record_prompt_improvement(original_quality, final_quality)
        
        if monitoring:
            monitoring.record_model_usage(request_data.model_name, total_time)
        
        if result.get("error"):
            raise HTTPException(status_code=500, detail=result.get("error"))
            
        logger.info(f"✅ Mejora iterativa completada en {total_time:.2f}s")
        return result
        
    except Exception as e:
        logger.error(f"❌ Error en mejora iterativa: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {str(e)}")

# --- Endpoints de Monitoreo y Observabilidad ---

@app.get("/api/health")
async def health_check():
    """
    Endpoint de health check empresarial con métricas detalladas.
    """
    try:
        health_data = {
            "status": "healthy",
            "timestamp": time.time(),
            "version": "2.0.0-enterprise",
            "components": {
                "model_manager": "operational" if 'model_manager' in globals() else "unavailable",
                "quality_analyzer": "operational" if 'quality_analyzer' in globals() else "unavailable", 
                "improvement_engine": "operational" if 'improvement_engine' in globals() else "unavailable",
                "monitoring_system": "operational" if monitoring else "unavailable"
            }
        }
        
        # Añadir métricas de monitoreo si está disponible
        if monitoring:
            dashboard_data = monitoring.get_dashboard_data()
            health_data.update({
                "system_health": dashboard_data.get("system_health", "unknown"),
                "uptime": dashboard_data.get("uptime", 0),
                "active_alerts": dashboard_data.get("total_alerts", 0),
                "critical_alerts": dashboard_data.get("critical_alerts", 0)
            })
        
        return health_data
        
    except Exception as e:
        logger.error(f"❌ Error en health check: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time()
            }
        )

@app.get("/api/metrics/dashboard")
async def get_dashboard_metrics():
    """
    Endpoint para obtener métricas del dashboard empresarial.
    """
    if not monitoring:
        raise HTTPException(status_code=503, detail="Sistema de monitoreo no disponible")
    
    try:
        dashboard_data = monitoring.get_dashboard_data()
        return dashboard_data
        
    except Exception as e:
        logger.error(f"❌ Error obteniendo métricas de dashboard: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

@app.post("/api/metrics/export")
async def export_metrics():
    """
    Endpoint para exportar métricas históricas.
    """
    if not monitoring:
        raise HTTPException(status_code=503, detail="Sistema de monitoreo no disponible")
    
    try:
        timestamp = int(time.time())
        filename = f"promptgen_metrics_{timestamp}.json"
        filepath = os.path.join(current_dir, "exports", filename)
        
        # Crear directorio de exports si no existe
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        monitoring.export_metrics(filepath)
        
        return {
            "status": "success",
            "message": "Métricas exportadas exitosamente",
            "filename": filename,
            "filepath": filepath
        }
        
    except Exception as e:
        logger.error(f"❌ Error exportando métricas: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

@app.get("/api/models")
async def get_available_models():
    """
    Obtiene la lista de modelos disponibles con información detallada.
    """
    try:
        if 'model_manager' not in globals():
            raise HTTPException(status_code=503, detail="Gestor de modelos no disponible")
        
        models_info = {
            "available_models": list(model_manager.model_configs.keys()),
            "model_details": {}
        }
        
        for model_key, config in model_manager.model_configs.items():
            models_info["model_details"][model_key] = {
                "name": config["name"],
                "parameters": config.get("parameters", "Unknown"),
                "type": config.get("type", "causal-lm"),
                "description": config.get("description", "Modelo de lenguaje para generación de texto"),
                "loaded": model_key in model_manager.model_cache
            }
        
        return models_info
        
    except Exception as e:
        logger.error(f"❌ Error obteniendo modelos disponibles: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {str(e)}")

@app.get("/")
async def root():
    """
    Endpoint raíz con información del sistema empresarial.
    """
    return {
        "message": "PromptGen Enterprise API v2.0.0",
        "description": "Sistema empresarial de mejora iterativa de prompts con IA",
        "features": [
            "Análisis avanzado de calidad con 6 métricas",
            "Mejora iterativa con aprendizaje contextual", 
            "Modelos reales de Hugging Face",
            "Sistema de monitoreo empresarial",
            "Métricas de rendimiento y negocio",
            "Alertas inteligentes",
            "Dashboard de observabilidad"
        ],
        "endpoints": {
            "health": "/api/health",
            "models": "/api/models", 
            "analyze": "/api/analyze_quality",
            "feedback": "/api/get_feedback",
            "variations": "/api/generate_variations",
            "ideas": "/api/generate_ideas",
            "improve": "/api/improve_iteratively",
            "dashboard": "/api/metrics/dashboard",
            "export": "/api/metrics/export"
        },
        "documentation": "/docs"
    }

# --- Configuración de Servidor ---

if __name__ == "__main__":
    logger.info("🚀 Iniciando PromptGen Enterprise API Server...")
    
    try:
        uvicorn.run(
            "api_server:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        logger.info("🛑 Servidor detenido por usuario")
        if monitoring:
            monitoring.stop_monitoring()
    except Exception as e:
        logger.error(f"❌ Error crítico del servidor: {e}")
        if monitoring:
            monitoring.stop_monitoring()
        raise 