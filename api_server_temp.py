from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import json
import os
import uuid
from datetime import datetime

app = FastAPI()

# Configuración CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    response: str
    sources: List[str] = []

class DocumentInfo(BaseModel):
    id: str
    filename: str
    upload_date: str
    file_size: int

# Simulamos una base de datos temporal
documents_db = []

@app.get("/")
async def root():
    return {"message": "PromptGen API - Versión Temporal (Sin Ollama)"}

@app.post("/api/upload_document", response_model=dict)
async def upload_document_temp(file: UploadFile = File(...)):
    """Simulación de subida de documento sin procesamiento real"""
    
    # Crear directorio si no existe
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    
    # Guardar archivo
    file_path = os.path.join(upload_dir, file.filename)
    content = await file.read()
    
    with open(file_path, "wb") as f:
        f.write(content)
    
    # Simular documento procesado
    doc_id = str(uuid.uuid4())
    doc_info = {
        "id": doc_id,
        "filename": file.filename,
        "upload_date": datetime.now().isoformat(),
        "file_size": len(content)
    }
    
    documents_db.append(doc_info)
    
    return {
        "success": True,
        "file_info": {
            "filename": file.filename,
            "doc_id": doc_id,
            "upload_date": datetime.now().isoformat()
        },
        "processing_result": {
            "chunks_count": 1,
            "status": "processed_temp"
        },
        "message": "Documento subido exitosamente (modo temporal)",
        "note": "⚠️ Ollama no está instalado. Instálalo para procesamiento real."
    }

@app.post("/api/query")
async def query_documents_temp(request: QueryRequest):
    """Simulación de consulta sin Ollama"""
    
    if not documents_db:
        return {
            "success": True,
            "result": {
                "answer": "No hay documentos subidos para consultar. Por favor, sube algunos documentos primero.",
                "source_documents": [],
                "query": request.query,
                "timestamp": datetime.now().isoformat()
            }
        }
    
    # Respuesta simulada
    response = f"""⚠️ **Modo Temporal (Sin Ollama)**

Tu pregunta: "{request.query}"

Respuesta simulada: Para obtener respuestas reales basadas en tus documentos, necesitas:

1. **Instalar Ollama**: Ve a https://ollama.com/download
2. **Descargar modelos**: Ejecuta estos comandos después de instalar Ollama:
   ```
   ollama pull llama3.2:3b
   ollama pull mxbai-embed-large
   ```
3. **Reiniciar el sistema**: Usa `start_system.bat` nuevamente

Documentos disponibles: {len(documents_db)} archivo(s)
"""
    
    # Crear source_documents en el formato esperado
    source_documents = [
        {
            "filename": doc["filename"],
            "content": f"Documento: {doc['filename']} (subido el {doc['upload_date']})",
            "chunk_index": 0
        }
        for doc in documents_db
    ]
    
    return {
        "success": True,
        "result": {
            "answer": response,
            "source_documents": source_documents,
            "query": request.query,
            "timestamp": datetime.now().isoformat()
        }
    }

@app.get("/api/documents", response_model=List[DocumentInfo])
async def get_documents_temp():
    """Obtener lista de documentos"""
    return [DocumentInfo(**doc) for doc in documents_db]

@app.delete("/api/documents/{document_id}")
async def delete_document_temp(document_id: str):
    """Eliminar documento"""
    global documents_db
    documents_db = [doc for doc in documents_db if doc["id"] != document_id]
    return {"message": "Documento eliminado", "document_id": document_id}

if __name__ == "__main__":
    import uvicorn
    print("🚀 Iniciando servidor temporal (sin Ollama)")
    print("⚠️  Para funcionalidad completa, instala Ollama desde: https://ollama.com/download")
    uvicorn.run(app, host="0.0.0.0", port=8000) 