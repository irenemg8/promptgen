# simple_document_system.py - Sistema RAG Simplificado sin LangChain
import os
import json
import uuid
import hashlib
import base64
import asyncio
import time
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import logging

# Cifrado y seguridad
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# Procesamiento de archivos
import PyPDF2 as pypdf2
from docx import Document
import pandas as pd
from bs4 import BeautifulSoup
import json as jsonlib

# Comunicación con Ollama
import ollama

# Optimización y cache
from diskcache import Cache
from cachetools import TTLCache
import lz4.frame
import psutil

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleDocumentSystem:
    """Sistema RAG Simplificado sin LangChain"""
    
    def __init__(self, 
                 storage_path: str = "./simple_documents",
                 model_name: str = "llama3.2:3b",
                 embeddings_model: str = "mxbai-embed-large",
                 encryption_key: Optional[str] = None):
        """
        Inicializa el sistema simplificado
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        # Configurar cifrado
        self.encryption_key = encryption_key or os.environ.get("ENCRYPTION_KEY", "simple_secure_key_2024")
        self.cipher = self._setup_encryption()
        
        # Directorios
        self.encrypted_files_path = self.storage_path / "encrypted_files"
        self.encrypted_files_path.mkdir(exist_ok=True)
        
        # Archivo de metadatos cifrado
        self.metadata_file = self.storage_path / "metadata.enc"
        
        # Configurar modelos
        self.model_name = model_name
        self.embeddings_model = embeddings_model
        
        # Sistema de cache
        self.memory_cache = TTLCache(maxsize=1000, ttl=3600)
        self.disk_cache = Cache(str(self.storage_path / "cache"))
        
        # Cargar metadatos
        self.documents_metadata = self._load_encrypted_metadata()
        
        # Estadísticas
        self.stats = {
            "total_documents": 0,
            "cache_hits": 0,
            "processing_times": []
        }
        
        logger.info(f"✅ Sistema Simplificado inicializado en {self.storage_path}")
    
    def _setup_encryption(self) -> Fernet:
        """Configurar cifrado AES"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'simple_salt_2024',
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.encryption_key.encode()))
        return Fernet(key)
    
    def _encrypt_data(self, data: Union[str, bytes]) -> bytes:
        """Cifrar datos"""
        if isinstance(data, str):
            data = data.encode('utf-8')
        return self.cipher.encrypt(data)
    
    def _decrypt_data(self, encrypted_data: bytes) -> str:
        """Descifrar datos"""
        return self.cipher.decrypt(encrypted_data).decode('utf-8')
    
    def _load_encrypted_metadata(self) -> Dict[str, Any]:
        """Cargar metadatos cifrados"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'rb') as f:
                    encrypted_data = f.read()
                decrypted_data = self._decrypt_data(encrypted_data)
                return jsonlib.loads(decrypted_data)
            except Exception as e:
                logger.error(f"Error cargando metadatos: {e}")
                return {}
        return {}
    
    def _save_encrypted_metadata(self):
        """Guardar metadatos cifrados"""
        try:
            json_data = jsonlib.dumps(self.documents_metadata, indent=2, ensure_ascii=False)
            encrypted_data = self._encrypt_data(json_data)
            with open(self.metadata_file, 'wb') as f:
                f.write(encrypted_data)
        except Exception as e:
            logger.error(f"Error guardando metadatos: {e}")
    
    def _extract_text_from_file(self, file_path: str, file_type: str) -> str:
        """Extraer texto de archivo según su tipo"""
        try:
            if file_type.lower() == 'pdf':
                return self._extract_pdf_text(file_path)
            elif file_type.lower() == 'docx':
                return self._extract_docx_text(file_path)
            elif file_type.lower() == 'txt':
                return self._extract_txt_text(file_path)
            elif file_type.lower() == 'json':
                return self._extract_json_text(file_path)
            elif file_type.lower() == 'csv':
                return self._extract_csv_text(file_path)
            else:
                return self._extract_generic_text(file_path)
        except Exception as e:
            logger.error(f"Error extrayendo texto de {file_path}: {e}")
            return f"Error procesando archivo {file_path}"
    
    def _extract_pdf_text(self, file_path: str) -> str:
        """Extraer texto de PDF"""
        try:
            with open(file_path, 'rb') as f:
                reader = pypdf2.PdfReader(f)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            logger.error(f"Error leyendo PDF: {e}")
            return f"Error leyendo PDF: {str(e)}"
    
    def _extract_docx_text(self, file_path: str) -> str:
        """Extraer texto de DOCX"""
        try:
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            logger.error(f"Error leyendo DOCX: {e}")
            return f"Error leyendo DOCX: {str(e)}"
    
    def _extract_txt_text(self, file_path: str) -> str:
        """Extraer texto de TXT"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error leyendo TXT: {e}")
            return f"Error leyendo TXT: {str(e)}"
    
    def _extract_json_text(self, file_path: str) -> str:
        """Extraer texto de JSON"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = jsonlib.load(f)
            return f"Contenido JSON:\n{jsonlib.dumps(data, indent=2, ensure_ascii=False)}"
        except Exception as e:
            logger.error(f"Error leyendo JSON: {e}")
            return f"Error leyendo JSON: {str(e)}"
    
    def _extract_csv_text(self, file_path: str) -> str:
        """Extraer texto de CSV"""
        try:
            df = pd.read_csv(file_path)
            return f"Datos CSV:\n{df.to_string()}\n\nResumen:\n{df.describe()}"
        except Exception as e:
            logger.error(f"Error leyendo CSV: {e}")
            return f"Error leyendo CSV: {str(e)}"
    
    def _extract_generic_text(self, file_path: str) -> str:
        """Extraer texto genérico"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            return f"Contenido del archivo:\n{content}"
        except Exception as e:
            logger.error(f"Error leyendo archivo genérico: {e}")
            return f"Error leyendo archivo: {str(e)}"
    
    def _split_text_into_chunks(self, text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
        """Dividir texto en chunks"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            if end > len(text):
                end = len(text)
            
            chunk = text[start:end]
            chunks.append(chunk)
            
            if end == len(text):
                break
            
            start = end - overlap
        
        return chunks
    
    async def process_uploaded_file(self, file_path: str, filename: str) -> Dict[str, Any]:
        """Procesar archivo subido"""
        start_time = time.time()
        
        try:
            # Detectar tipo de archivo
            file_type = filename.split('.')[-1].lower()
            
            # Generar ID único
            doc_id = str(uuid.uuid4())
            
            # Leer y cifrar archivo
            with open(file_path, 'rb') as f:
                file_data = f.read()
            
            encrypted_data = self._encrypt_data(file_data)
            
            # Guardar archivo cifrado
            encrypted_filename = f"{doc_id}.enc"
            encrypted_path = self.encrypted_files_path / encrypted_filename
            
            with open(encrypted_path, 'wb') as f:
                f.write(encrypted_data)
            
            # Extraer texto
            text_content = self._extract_text_from_file(file_path, file_type)
            
            # Dividir en chunks
            chunks = self._split_text_into_chunks(text_content)
            
            # Guardar metadatos
            processing_time = time.time() - start_time
            
            self.documents_metadata[doc_id] = {
                "filename": filename,
                "file_type": file_type,
                "encrypted_path": str(encrypted_path),
                "upload_date": datetime.now().isoformat(),
                "file_size": len(file_data),
                "chunks_count": len(chunks),
                "processing_time": processing_time,
                "text_content": text_content[:1000] + "..." if len(text_content) > 1000 else text_content,
                "chunks": chunks,
                "status": "processed"
            }
            
            self._save_encrypted_metadata()
            
            # Actualizar estadísticas
            self.stats["total_documents"] += 1
            self.stats["processing_times"].append(processing_time)
            
            logger.info(f"✅ Archivo procesado: {filename} ({len(chunks)} chunks, {processing_time:.2f}s)")
            
            return {
                "doc_id": doc_id,
                "filename": filename,
                "file_type": file_type,
                "chunks_count": len(chunks),
                "processing_time": processing_time,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error procesando archivo {filename}: {e}")
            return {"error": str(e)}
        finally:
            # Limpiar archivo temporal
            if os.path.exists(file_path):
                os.remove(file_path)
    
    async def query_documents(self, query: str, k: int = 5) -> Dict[str, Any]:
        """Consultar documentos"""
        start_time = time.time()
        
        try:
            # Verificar caché
            cache_key = f"query_{hashlib.md5(query.encode()).hexdigest()}"
            
            if cache_key in self.memory_cache:
                self.stats["cache_hits"] += 1
                result = self.memory_cache[cache_key]
                result["from_cache"] = True
                result["response_time"] = time.time() - start_time
                return result
            
            # Buscar documentos relevantes (búsqueda simple por texto)
            relevant_chunks = []
            
            for doc_id, metadata in self.documents_metadata.items():
                if "chunks" in metadata:
                    for i, chunk in enumerate(metadata["chunks"]):
                        # Búsqueda simple por palabras clave
                        if any(word.lower() in chunk.lower() for word in query.split()):
                            relevant_chunks.append({
                                "doc_id": doc_id,
                                "filename": metadata["filename"],
                                "chunk_index": i,
                                "content": chunk,
                                "file_type": metadata["file_type"]
                            })
            
            # Limitar resultados
            relevant_chunks = relevant_chunks[:k]
            
            # Construir contexto
            context = "\n\n".join([
                f"Documento: {chunk['filename']}\n{chunk['content'][:500]}..."
                for chunk in relevant_chunks
            ])
            
            # Generar respuesta con Ollama
            if context:
                prompt = f"""
                Contexto de documentos:
                {context}
                
                Pregunta: {query}
                
                Responde de manera precisa basándote únicamente en la información de los documentos proporcionados.
                """
                
                try:
                    response = ollama.generate(
                        model=self.model_name,
                        prompt=prompt,
                        stream=False
                    )
                    answer = response['response']
                except Exception as e:
                    logger.error(f"Error con Ollama: {e}")
                    answer = f"Error generando respuesta: {str(e)}"
            else:
                answer = "No encontré información relevante en los documentos para responder tu pregunta."
            
            # Procesar fuentes
            sources = []
            for chunk in relevant_chunks:
                sources.append({
                    "filename": chunk["filename"],
                    "content": chunk["content"][:200] + "...",
                    "chunk_index": chunk["chunk_index"],
                    "file_type": chunk["file_type"]
                })
            
            response_time = time.time() - start_time
            
            result = {
                "answer": answer,
                "sources": sources,
                "query": query,
                "response_time": response_time,
                "timestamp": datetime.now().isoformat(),
                "from_cache": False
            }
            
            # Guardar en caché
            if response_time < 8.0:
                self.memory_cache[cache_key] = result
            
            logger.info(f"🔍 Consulta procesada: {response_time:.2f}s ({len(sources)} fuentes)")
            
            return result
            
        except Exception as e:
            logger.error(f"Error en consulta: {e}")
            return {
                "error": str(e),
                "query": query,
                "response_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat()
            }
    
    def get_documents_list(self) -> List[Dict[str, Any]]:
        """Obtener lista de documentos"""
        return [
            {
                "doc_id": doc_id,
                **{k: v for k, v in metadata.items() if k != "chunks"}  # Excluir chunks para eficiencia
            }
            for doc_id, metadata in self.documents_metadata.items()
        ]
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del sistema"""
        memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        cpu_percent = psutil.cpu_percent()
        
        avg_processing_time = (
            sum(self.stats["processing_times"]) / len(self.stats["processing_times"])
            if self.stats["processing_times"] else 0
        )
        
        return {
            "total_documents": self.stats["total_documents"],
            "total_chunks": sum(metadata.get("chunks_count", 0) for metadata in self.documents_metadata.values()),
            "cache_hits": self.stats["cache_hits"],
            "memory_usage_mb": memory_usage,
            "cpu_usage_percent": cpu_percent,
            "avg_processing_time": avg_processing_time,
            "storage_path": str(self.storage_path),
            "encryption_enabled": True,
            "supported_formats": ["PDF", "DOCX", "TXT", "JSON", "CSV", "Otros"]
        }
    
    def delete_document(self, doc_id: str) -> bool:
        """Eliminar documento"""
        try:
            if doc_id in self.documents_metadata:
                metadata = self.documents_metadata[doc_id]
                
                # Eliminar archivo cifrado
                encrypted_path = Path(metadata["encrypted_path"])
                if encrypted_path.exists():
                    encrypted_path.unlink()
                
                # Eliminar metadatos
                del self.documents_metadata[doc_id]
                self._save_encrypted_metadata()
                
                # Actualizar estadísticas
                self.stats["total_documents"] -= 1
                
                logger.info(f"🗑️ Documento eliminado: {metadata['filename']}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error eliminando documento {doc_id}: {e}")
            return False
    
    def cleanup_cache(self):
        """Limpiar caché"""
        self.memory_cache.clear()
        self.disk_cache.clear()
        logger.info("🧹 Caché limpiado")

# Instancia global del sistema simplificado
simple_system = SimpleDocumentSystem()

# Funciones helper para la API
async def process_document(file_path: str, filename: str) -> Dict[str, Any]:
    """Procesar documento"""
    return await simple_system.process_uploaded_file(file_path, filename)

async def query_knowledge_base(query: str, k: int = 5) -> Dict[str, Any]:
    """Consultar base de conocimiento"""
    return await simple_system.query_documents(query, k)

def get_documents() -> List[Dict[str, Any]]:
    """Obtener lista de documentos"""
    return simple_system.get_documents_list()

def delete_document(doc_id: str) -> bool:
    """Eliminar documento"""
    return simple_system.delete_document(doc_id)

def get_system_status() -> Dict[str, Any]:
    """Obtener estado del sistema"""
    return simple_system.get_system_stats()

def cleanup_system():
    """Limpiar sistema"""
    simple_system.cleanup_cache() 