# secure_document_system.py - Sistema RAG Seguro con Cifrado y Procesamiento Universal
import os
import json
import uuid
import hashlib
import base64
import mimetypes
import asyncio
import time
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# Cifrado y seguridad
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# Procesamiento de archivos
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader, Docx2txtLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.schema import Document
import chromadb

# Procesamiento de archivos adicionales
from PIL import Image
import pandas as pd
import openpyxl
from bs4 import BeautifulSoup
import markdown
from pygments import highlight
from pygments.lexers import get_lexer_by_name, guess_lexer
from pygments.formatters import TextFormatter
import easyocr
import magic

# Optimización y cache
from diskcache import Cache
from cachetools import TTLCache
import msgpack
import lz4.frame
import psutil

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecureDocumentSystem:
    """Sistema RAG Seguro con Cifrado y Procesamiento Universal de Archivos"""
    
    def __init__(self, 
                 storage_path: str = "./secure_documents",
                 model_name: str = "llama3.2:3b",
                 embeddings_model: str = "mxbai-embed-large",
                 encryption_key: Optional[str] = None,
                 max_memory_cache_mb: int = 1024):
        """
        Inicializa el sistema de documentos seguro
        
        Args:
            storage_path: Ruta de almacenamiento cifrado
            model_name: Modelo de Ollama para chat
            embeddings_model: Modelo de embeddings
            encryption_key: Clave de cifrado personalizada
            max_memory_cache_mb: Límite de caché en memoria (MB)
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        # Configurar cifrado
        self.encryption_key = encryption_key or os.environ.get("ENCRYPTION_KEY", "secure_promptgen_key_2024")
        self.cipher = self._setup_encryption()
        
        # Directorios seguros
        self.encrypted_files_path = self.storage_path / "encrypted_files"
        self.encrypted_files_path.mkdir(exist_ok=True)
        
        self.vectordb_path = self.storage_path / "vectordb"
        self.vectordb_path.mkdir(exist_ok=True)
        
        # Archivos de configuración cifrados
        self.metadata_file = self.storage_path / "metadata.enc"
        self.config_file = self.storage_path / "config.enc"
        
        # Configurar modelos
        self.model_name = model_name
        self.embeddings_model = embeddings_model
        
        # Configurar procesamiento de texto
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,  # Optimizado para velocidad
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Inicializar modelos con optimización
        self.embeddings = OllamaEmbeddings(
            model=self.embeddings_model,
            num_thread=psutil.cpu_count()
        )
        
        self.llm = OllamaLLM(
            model=self.model_name,
            num_thread=psutil.cpu_count(),
            num_ctx=4096,  # Contexto optimizado
            temperature=0.1  # Respuestas más precisas
        )
        
        # Sistema de cache multi-nivel
        self.memory_cache = TTLCache(maxsize=1000, ttl=3600)  # 1 hora
        self.disk_cache = Cache(str(self.storage_path / "cache"))
        
        # Cache de embeddings en memoria
        self.embeddings_cache = {}
        
        # Inicializar OCR
        self.ocr_reader = easyocr.Reader(['es', 'en'])
        
        # Inicializar base de datos vectorial
        self.vectorstore = Chroma(
            embedding_function=self.embeddings,
            persist_directory=str(self.vectordb_path),
            collection_name="secure_documents"
        )
        
        # Cargar metadatos
        self.documents_metadata = self._load_encrypted_metadata()
        
        # Configurar executor para procesamiento paralelo
        self.executor = ThreadPoolExecutor(max_workers=psutil.cpu_count())
        
        # Estadísticas del sistema
        self.stats = {
            "total_documents": 0,
            "total_chunks": 0,
            "cache_hits": 0,
            "processing_times": []
        }
        
        logger.info(f"✅ Sistema Seguro de Documentos inicializado")
        logger.info(f"📁 Almacenamiento: {self.storage_path}")
        logger.info(f"🔐 Cifrado: Activado")
        logger.info(f"💾 Caché: {max_memory_cache_mb}MB")
        logger.info(f"🔧 Procesadores: {psutil.cpu_count()}")
    
    def _setup_encryption(self) -> Fernet:
        """Configurar sistema de cifrado AES"""
        # Generar clave derivada de la contraseña
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'secure_salt_promptgen_2024',
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
        decrypted = self.cipher.decrypt(encrypted_data)
        return decrypted.decode('utf-8')
    
    def _compress_data(self, data: str) -> bytes:
        """Comprimir datos con LZ4"""
        return lz4.frame.compress(data.encode('utf-8'))
    
    def _decompress_data(self, compressed_data: bytes) -> str:
        """Descomprimir datos LZ4"""
        return lz4.frame.decompress(compressed_data).decode('utf-8')
    
    def _load_encrypted_metadata(self) -> Dict[str, Any]:
        """Cargar metadatos cifrados"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'rb') as f:
                    encrypted_data = f.read()
                decrypted_data = self._decrypt_data(encrypted_data)
                return json.loads(decrypted_data)
            except Exception as e:
                logger.error(f"Error cargando metadatos: {e}")
                return {}
        return {}
    
    def _save_encrypted_metadata(self):
        """Guardar metadatos cifrados"""
        try:
            json_data = json.dumps(self.documents_metadata, indent=2, ensure_ascii=False)
            encrypted_data = self._encrypt_data(json_data)
            with open(self.metadata_file, 'wb') as f:
                f.write(encrypted_data)
        except Exception as e:
            logger.error(f"Error guardando metadatos: {e}")
    
    def _detect_file_type(self, file_path: str) -> Tuple[str, str]:
        """Detectar tipo de archivo usando magic"""
        try:
            mime_type = magic.from_file(file_path, mime=True)
            file_type = mimetypes.guess_type(file_path)[0] or mime_type
            
            # Mapear tipos específicos
            type_mapping = {
                'application/pdf': 'pdf',
                'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'docx',
                'application/msword': 'doc',
                'text/plain': 'txt',
                'application/json': 'json',
                'text/csv': 'csv',
                'application/vnd.ms-excel': 'xlsx',
                'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'xlsx',
                'text/html': 'html',
                'text/markdown': 'md',
                'image/jpeg': 'jpg',
                'image/png': 'png',
                'image/gif': 'gif',
                'image/bmp': 'bmp',
                'image/tiff': 'tiff'
            }
            
            detected_type = type_mapping.get(file_type, 'unknown')
            return detected_type, mime_type
        except Exception as e:
            logger.warning(f"Error detectando tipo de archivo: {e}")
            return 'unknown', 'unknown'
    
    async def _process_file_content(self, file_path: str, file_type: str) -> List[Document]:
        """Procesar contenido de archivo según su tipo"""
        try:
            if file_type == 'pdf':
                return await self._process_pdf(file_path)
            elif file_type in ['docx', 'doc']:
                return await self._process_docx(file_path)
            elif file_type == 'txt':
                return await self._process_text(file_path)
            elif file_type == 'json':
                return await self._process_json(file_path)
            elif file_type == 'csv':
                return await self._process_csv(file_path)
            elif file_type == 'xlsx':
                return await self._process_excel(file_path)
            elif file_type == 'html':
                return await self._process_html(file_path)
            elif file_type == 'md':
                return await self._process_markdown(file_path)
            elif file_type in ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff']:
                return await self._process_image(file_path)
            elif file_type in ['py', 'js', 'ts', 'java', 'cpp', 'c', 'cs', 'php', 'rb', 'go', 'rs']:
                return await self._process_code(file_path, file_type)
            else:
                return await self._process_unknown(file_path)
        except Exception as e:
            logger.error(f"Error procesando archivo {file_path}: {e}")
            return []
    
    async def _process_pdf(self, file_path: str) -> List[Document]:
        """Procesar archivo PDF"""
        loader = PDFPlumberLoader(file_path)
        return loader.load()
    
    async def _process_docx(self, file_path: str) -> List[Document]:
        """Procesar archivo DOCX"""
        loader = Docx2txtLoader(file_path)
        return loader.load()
    
    async def _process_text(self, file_path: str) -> List[Document]:
        """Procesar archivo de texto"""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        return [Document(page_content=content, metadata={"source": file_path})]
    
    async def _process_json(self, file_path: str) -> List[Document]:
        """Procesar archivo JSON"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convertir JSON a texto legible
        content = json.dumps(data, indent=2, ensure_ascii=False)
        return [Document(
            page_content=f"Contenido JSON:\n{content}",
            metadata={"source": file_path, "type": "json"}
        )]
    
    async def _process_csv(self, file_path: str) -> List[Document]:
        """Procesar archivo CSV"""
        df = pd.read_csv(file_path)
        
        # Convertir DataFrame a texto
        content = f"Datos CSV:\n{df.to_string()}\n\nResumen:\n{df.describe()}"
        return [Document(
            page_content=content,
            metadata={"source": file_path, "type": "csv", "rows": len(df)}
        )]
    
    async def _process_excel(self, file_path: str) -> List[Document]:
        """Procesar archivo Excel"""
        df = pd.read_excel(file_path)
        
        # Convertir DataFrame a texto
        content = f"Datos Excel:\n{df.to_string()}\n\nResumen:\n{df.describe()}"
        return [Document(
            page_content=content,
            metadata={"source": file_path, "type": "excel", "rows": len(df)}
        )]
    
    async def _process_html(self, file_path: str) -> List[Document]:
        """Procesar archivo HTML"""
        with open(file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Extraer texto limpio
        soup = BeautifulSoup(html_content, 'html.parser')
        text_content = soup.get_text(separator='\n', strip=True)
        
        return [Document(
            page_content=text_content,
            metadata={"source": file_path, "type": "html"}
        )]
    
    async def _process_markdown(self, file_path: str) -> List[Document]:
        """Procesar archivo Markdown"""
        with open(file_path, 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        # Convertir a HTML y extraer texto
        html_content = markdown.markdown(md_content)
        soup = BeautifulSoup(html_content, 'html.parser')
        text_content = soup.get_text(separator='\n', strip=True)
        
        return [Document(
            page_content=f"Contenido Markdown:\n{text_content}",
            metadata={"source": file_path, "type": "markdown"}
        )]
    
    async def _process_image(self, file_path: str) -> List[Document]:
        """Procesar imagen con OCR"""
        try:
            # Usar EasyOCR para extraer texto
            result = self.ocr_reader.readtext(file_path)
            
            # Combinar texto extraído
            extracted_text = "\n".join([item[1] for item in result if item[2] > 0.5])
            
            if extracted_text.strip():
                return [Document(
                    page_content=f"Texto extraído de imagen:\n{extracted_text}",
                    metadata={"source": file_path, "type": "image_ocr"}
                )]
            else:
                return [Document(
                    page_content=f"Imagen procesada: {os.path.basename(file_path)} (sin texto extraíble)",
                    metadata={"source": file_path, "type": "image"}
                )]
        except Exception as e:
            logger.error(f"Error procesando imagen {file_path}: {e}")
            return []
    
    async def _process_code(self, file_path: str, file_type: str) -> List[Document]:
        """Procesar archivo de código"""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            code_content = f.read()
        
        try:
            # Detectar lexer automáticamente
            lexer = get_lexer_by_name(file_type)
            
            # Formatear código para mejor legibilidad
            formatted_code = highlight(code_content, lexer, TextFormatter())
            
            return [Document(
                page_content=f"Código {file_type.upper()}:\n{formatted_code}",
                metadata={"source": file_path, "type": "code", "language": file_type}
            )]
        except Exception as e:
            logger.warning(f"Error formateando código: {e}")
            return [Document(
                page_content=f"Código {file_type.upper()}:\n{code_content}",
                metadata={"source": file_path, "type": "code", "language": file_type}
            )]
    
    async def _process_unknown(self, file_path: str) -> List[Document]:
        """Procesar archivo de tipo desconocido"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            return [Document(
                page_content=f"Contenido de archivo:\n{content}",
                metadata={"source": file_path, "type": "unknown"}
            )]
        except Exception as e:
            logger.warning(f"No se pudo procesar archivo {file_path}: {e}")
            return []
    
    async def process_uploaded_file(self, file_path: str, filename: str) -> Dict[str, Any]:
        """
        Procesar archivo subido de forma segura y rápida
        
        Args:
            file_path: Ruta del archivo temporal
            filename: Nombre original del archivo
            
        Returns:
            Dict con información del procesamiento
        """
        start_time = time.time()
        
        try:
            # Detectar tipo de archivo
            file_type, mime_type = self._detect_file_type(file_path)
            
            # Generar ID único
            doc_id = str(uuid.uuid4())
            
            # Crear ruta cifrada
            encrypted_filename = f"{doc_id}.enc"
            encrypted_path = self.encrypted_files_path / encrypted_filename
            
            # Cifrar y guardar archivo
            with open(file_path, 'rb') as f:
                file_data = f.read()
            
            encrypted_data = self._encrypt_data(file_data)
            
            with open(encrypted_path, 'wb') as f:
                f.write(encrypted_data)
            
            # Procesar contenido
            documents = await self._process_file_content(file_path, file_type)
            
            if not documents:
                return {"error": "No se pudo procesar el archivo"}
            
            # Dividir en chunks optimizados
            text_chunks = self.text_splitter.split_documents(documents)
            
            # Agregar metadatos enriquecidos
            for i, chunk in enumerate(text_chunks):
                chunk.metadata.update({
                    "doc_id": doc_id,
                    "filename": filename,
                    "file_type": file_type,
                    "mime_type": mime_type,
                    "chunk_index": i,
                    "upload_date": datetime.now().isoformat(),
                    "file_size": len(file_data),
                    "chunk_size": len(chunk.page_content)
                })
            
            # Agregar a base de datos vectorial con procesamiento paralelo
            await self._add_chunks_parallel(text_chunks)
            
            # Guardar metadatos
            processing_time = time.time() - start_time
            
            self.documents_metadata[doc_id] = {
                "filename": filename,
                "file_type": file_type,
                "mime_type": mime_type,
                "encrypted_path": str(encrypted_path),
                "upload_date": datetime.now().isoformat(),
                "file_size": len(file_data),
                "chunks_count": len(text_chunks),
                "processing_time": processing_time,
                "status": "processed"
            }
            
            self._save_encrypted_metadata()
            
            # Actualizar estadísticas
            self.stats["total_documents"] += 1
            self.stats["total_chunks"] += len(text_chunks)
            self.stats["processing_times"].append(processing_time)
            
            logger.info(f"✅ Archivo procesado: {filename} ({len(text_chunks)} chunks, {processing_time:.2f}s)")
            
            return {
                "doc_id": doc_id,
                "filename": filename,
                "file_type": file_type,
                "chunks_count": len(text_chunks),
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
    
    async def _add_chunks_parallel(self, chunks: List[Document]):
        """Agregar chunks a la base de datos con procesamiento paralelo"""
        batch_size = 10
        batches = [chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)]
        
        for batch in batches:
            self.vectorstore.add_documents(batch)
    
    async def query_documents(self, query: str, k: int = 5) -> Dict[str, Any]:
        """
        Consultar documentos con optimización de velocidad
        
        Args:
            query: Pregunta del usuario
            k: Número de documentos relevantes
            
        Returns:
            Dict con respuesta y fuentes
        """
        start_time = time.time()
        
        try:
            # Verificar caché
            cache_key = f"query_{hashlib.md5(query.encode()).hexdigest()}_{k}"
            
            if cache_key in self.memory_cache:
                self.stats["cache_hits"] += 1
                result = self.memory_cache[cache_key]
                result["from_cache"] = True
                result["response_time"] = time.time() - start_time
                return result
            
            # Crear retriever optimizado
            retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": k * 2}  # Obtener más para filtrar
            )
            
            # Obtener documentos relevantes
            relevant_docs = retriever.get_relevant_documents(query)
            
            # Filtrar duplicados y seleccionar mejores
            unique_docs = []
            seen_content = set()
            
            for doc in relevant_docs:
                content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
                if content_hash not in seen_content:
                    unique_docs.append(doc)
                    seen_content.add(content_hash)
                
                if len(unique_docs) >= k:
                    break
            
            # Construir contexto optimizado
            context = "\n\n".join([
                f"Documento: {doc.metadata.get('filename', 'Unknown')}\n{doc.page_content[:500]}..."
                for doc in unique_docs
            ])
            
            # Generar respuesta con prompt optimizado
            prompt = f"""
            Contexto de documentos:
            {context}
            
            Pregunta: {query}
            
            Responde de manera precisa y detallada basándote únicamente en la información de los documentos proporcionados.
            Si no encuentras información relevante, indícalo claramente.
            """
            
            # Llamada optimizada al modelo
            response = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self.llm.invoke,
                prompt
            )
            
            # Procesar fuentes
            sources = []
            for doc in unique_docs:
                sources.append({
                    "filename": doc.metadata.get("filename", "Unknown"),
                    "content": doc.page_content[:200] + "...",
                    "chunk_index": doc.metadata.get("chunk_index", 0),
                    "file_type": doc.metadata.get("file_type", "unknown")
                })
            
            response_time = time.time() - start_time
            
            result = {
                "answer": response,
                "sources": sources,
                "query": query,
                "response_time": response_time,
                "timestamp": datetime.now().isoformat(),
                "from_cache": False
            }
            
            # Guardar en caché si la respuesta es buena
            if response_time < 8.0:  # Solo cachear respuestas rápidas
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
        """Obtener lista completa de documentos"""
        return [
            {
                "doc_id": doc_id,
                **metadata
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
            "total_chunks": self.stats["total_chunks"],
            "cache_hits": self.stats["cache_hits"],
            "memory_usage_mb": memory_usage,
            "cpu_usage_percent": cpu_percent,
            "avg_processing_time": avg_processing_time,
            "storage_path": str(self.storage_path),
            "encryption_enabled": True,
            "supported_formats": [
                "PDF", "DOCX", "DOC", "TXT", "JSON", "CSV", "XLSX", 
                "HTML", "Markdown", "Images (OCR)", "Code files"
            ]
        }
    
    def delete_document(self, doc_id: str) -> bool:
        """Eliminar documento de forma segura"""
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
                self.stats["total_chunks"] -= metadata.get("chunks_count", 0)
                
                logger.info(f"🗑️ Documento eliminado: {metadata['filename']}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error eliminando documento {doc_id}: {e}")
            return False
    
    def cleanup_cache(self):
        """Limpiar caché y optimizar memoria"""
        self.memory_cache.clear()
        self.disk_cache.clear()
        self.embeddings_cache.clear()
        logger.info("🧹 Caché limpiado")

# Instancia global del sistema seguro
secure_system = SecureDocumentSystem()

# Funciones helper para la API
async def process_document(file_path: str, filename: str) -> Dict[str, Any]:
    """Procesar documento de forma segura"""
    return await secure_system.process_uploaded_file(file_path, filename)

async def query_knowledge_base(query: str, k: int = 5) -> Dict[str, Any]:
    """Consultar base de conocimiento"""
    return await secure_system.query_documents(query, k)

def get_documents() -> List[Dict[str, Any]]:
    """Obtener lista de documentos"""
    return secure_system.get_documents_list()

def delete_document(doc_id: str) -> bool:
    """Eliminar documento"""
    return secure_system.delete_document(doc_id)

def get_system_status() -> Dict[str, Any]:
    """Obtener estado del sistema"""
    return secure_system.get_system_stats()

def cleanup_system():
    """Limpiar sistema"""
    secure_system.cleanup_cache() 