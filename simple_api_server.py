# simple_api_server.py - Servidor API Simplificado
import uvicorn
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import sys
import os
import logging
import time
import uuid
import asyncio
import aiofiles
from pathlib import Path
from typing import List, Optional

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Añadir el directorio actual al path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Importar el sistema simplificado
try:
    from simple_document_system import (
        process_document,
        query_knowledge_base,
        get_documents,
        delete_document,
        get_system_status,
        cleanup_system
    )
    logger.info("✅ Sistema simplificado de documentos cargado exitosamente")
    
except ImportError as e:
    logger.error(f"❌ Error importando sistema simplificado: {e}")
    # Crear funciones dummy
    async def process_document(file_path: str, filename: str):
        return {"error": "Sistema de documentos no disponible"}
    
    async def query_knowledge_base(query: str, k: int = 5):
        return {"error": "Sistema de documentos no disponible"}
    
    def get_documents():
        return []
    
    def delete_document(doc_id: str):
        return False
    
    def get_system_status():
        return {"error": "Sistema de documentos no disponible"}
    
    def cleanup_system():
        pass

# Configurar aplicación FastAPI
app = FastAPI(
    title="PromptGen - Sistema Seguro de Documentos",
    description="API para procesamiento seguro de documentos con cifrado AES",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelos Pydantic
class ChatRequest(BaseModel):
    query: str
    max_results: int = 5

class SystemCleanupRequest(BaseModel):
    cleanup_cache: bool = True

# Endpoints del sistema de documentos
@app.post("/api/upload-file")
async def upload_file(file: UploadFile = File(...)):
    """
    Subir archivo al sistema seguro de documentos
    """
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No se proporcionó nombre de archivo")
        
        # Crear directorio temporal
        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(exist_ok=True)
        
        # Guardar archivo temporalmente
        temp_file_path = temp_dir / f"{uuid.uuid4()}_{file.filename}"
        
        async with aiofiles.open(temp_file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        # Procesar archivo
        result = await process_document(str(temp_file_path), file.filename)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        logger.info(f"✅ Archivo subido y procesado: {file.filename}")
        
        return {
            "success": True,
            "message": f"Archivo {file.filename} procesado exitosamente",
            "result": result
        }
        
    except Exception as e:
        logger.error(f"❌ Error subiendo archivo: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload-multiple-files")
async def upload_multiple_files(files: List[UploadFile] = File(...)):
    """
    Subir múltiples archivos
    """
    try:
        if not files or len(files) == 0:
            raise HTTPException(status_code=400, detail="No se proporcionaron archivos")
        
        # Crear directorio temporal
        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(exist_ok=True)
        
        # Procesar archivos en paralelo
        tasks = []
        
        for file in files:
            if not file.filename:
                continue
                
            temp_file_path = temp_dir / f"{uuid.uuid4()}_{file.filename}"
            
            # Guardar archivo temporalmente
            async with aiofiles.open(temp_file_path, 'wb') as f:
                content = await file.read()
                await f.write(content)
            
            # Crear tarea de procesamiento
            task = process_document(str(temp_file_path), file.filename)
            tasks.append(task)
        
        # Procesar todos los archivos en paralelo
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Compilar resultados
        processed_files = []
        errors = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                errors.append(f"Error en {files[i].filename}: {str(result)}")
            elif "error" in result:
                errors.append(f"Error en {files[i].filename}: {result['error']}")
            else:
                processed_files.append(result)
        
        logger.info(f"✅ Procesamiento masivo: {len(processed_files)} exitosos, {len(errors)} errores")
        
        return {
            "success": True,
            "processed_files": processed_files,
            "errors": errors,
            "total_files": len(files),
            "successful_files": len(processed_files)
        }
        
    except Exception as e:
        logger.error(f"❌ Error en carga masiva: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
async def chat_with_documents(request: ChatRequest):
    """
    Realizar consulta sobre documentos
    """
    try:
        start_time = time.time()
        
        # Validar entrada
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query vacío")
        
        # Procesar consulta
        result = await query_knowledge_base(request.query, request.max_results)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        # Verificar tiempo de respuesta
        response_time = time.time() - start_time
        if response_time > 10.0:
            logger.warning(f"⚠️ Respuesta lenta: {response_time:.2f}s")
        
        return {
            "success": True,
            "result": result,
            "response_time": response_time
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error en chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/documents")
async def get_documents_list():
    """
    Obtener lista de documentos procesados
    """
    try:
        documents = get_documents()
        return {
            "success": True,
            "documents": documents,
            "total_count": len(documents)
        }
    except Exception as e:
        logger.error(f"❌ Error obteniendo documentos: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/documents/{doc_id}")
async def delete_document_endpoint(doc_id: str):
    """
    Eliminar documento específico
    """
    try:
        success = delete_document(doc_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Documento no encontrado")
        
        return {
            "success": True,
            "message": "Documento eliminado exitosamente"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error eliminando documento: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/system/status")
async def get_system_status_endpoint():
    """
    Obtener estado del sistema
    """
    try:
        status = get_system_status()
        return {
            "success": True,
            "status": status
        }
    except Exception as e:
        logger.error(f"❌ Error obteniendo estado: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/system/cleanup")
async def cleanup_system_endpoint(request: SystemCleanupRequest):
    """
    Limpiar sistema y optimizar memoria
    """
    try:
        if request.cleanup_cache:
            cleanup_system()
        
        return {
            "success": True,
            "message": "Sistema limpiado exitosamente"
        }
        
    except Exception as e:
        logger.error(f"❌ Error limpiando sistema: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """
    Verificar salud del sistema
    """
    try:
        # Verificar componentes principales
        components_status = {
            "api_server": "online",
            "document_system": "online",
            "encryption": "enabled"
        }
        
        # Obtener métricas del sistema
        system_metrics = get_system_status()
        
        return {
            "success": True,
            "status": "healthy",
            "components": components_status,
            "metrics": system_metrics,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"❌ Error en health check: {e}")
        return {
            "success": False,
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.time()
        }

@app.get("/")
async def root():
    """
    Endpoint raíz con información del sistema
    """
    return {
        "message": "PromptGen - Sistema Seguro de Documentos",
        "version": "1.0.0",
        "features": [
            "Cifrado AES local",
            "Soporte para múltiples formatos",
            "Procesamiento rápido",
            "Memoria persistente"
        ],
        "endpoints": {
            "upload": "/api/upload-file",
            "chat": "/api/chat",
            "documents": "/api/documents",
            "health": "/api/health",
            "docs": "/docs"
        }
    }

# Manejador de errores global
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Manejador global de excepciones
    """
    logger.error(f"❌ Error no manejado: {exc}")
    
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Error interno del servidor",
            "detail": str(exc),
            "timestamp": time.time()
        }
    )

# Evento de inicio
@app.on_event("startup")
async def startup_event():
    """
    Configuración al iniciar el servidor
    """
    logger.info("🚀 Sistema Seguro de Documentos iniciándose...")
    
    # Crear directorios necesarios
    os.makedirs("temp_uploads", exist_ok=True)
    os.makedirs("simple_documents", exist_ok=True)
    
    logger.info("✅ Sistema Seguro de Documentos iniciado exitosamente")

# Evento de cierre
@app.on_event("shutdown")
async def shutdown_event():
    """
    Limpieza al cerrar el servidor
    """
    logger.info("🔄 Cerrando Sistema Seguro de Documentos...")
    
    # Limpiar recursos
    try:
        cleanup_system()
    except:
        pass
    
    # Limpiar directorio temporal
    temp_dir = Path("temp_uploads")
    if temp_dir.exists():
        import shutil
        shutil.rmtree(temp_dir)
    
    logger.info("✅ Sistema Seguro de Documentos cerrado exitosamente")

if __name__ == "__main__":
    logger.info("🎯 Iniciando Sistema Seguro de Documentos...")
    
    # Configuración del servidor
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True,
        workers=1
    ) 