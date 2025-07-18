# document_rag_system.py - Sistema RAG con Ollama para consultas sobre documentos
import os
import json
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
import asyncio
from pathlib import Path

# Imports para procesamiento de documentos
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader, Docx2txtLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.schema import Document
import chromadb
from chromadb.config import Settings

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentRAGSystem:
    def __init__(self, 
                 storage_path: str = "./documents_storage",
                 model_name: str = "llama3.2:3b",
                 embeddings_model: str = "mxbai-embed-large"):
        """
        Inicializa el sistema RAG con Ollama
        
        Args:
            storage_path: Ruta donde se almacenan los documentos
            model_name: Nombre del modelo de Ollama para chat
            embeddings_model: Modelo de embeddings de Ollama
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        # Crear directorio para archivos subidos
        self.uploads_path = self.storage_path / "uploads"
        self.uploads_path.mkdir(exist_ok=True)
        
        # Crear directorio para la base de datos vectorial
        self.vectordb_path = self.storage_path / "vectordb"
        self.vectordb_path.mkdir(exist_ok=True)
        
        # Archivo de metadatos
        self.metadata_file = self.storage_path / "documents_metadata.json"
        
        # Configurar modelos
        self.model_name = model_name
        self.embeddings_model = embeddings_model
        
        # Inicializar componentes
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # Inicializar embeddings y modelo
        self.embeddings = OllamaEmbeddings(model=self.embeddings_model)
        self.llm = OllamaLLM(model=self.model_name)
        
        # Inicializar base de datos vectorial
        self.vectorstore = Chroma(
            embedding_function=self.embeddings,
            persist_directory=str(self.vectordb_path),
            collection_name="documents"
        )
        
        # Cargar metadatos existentes
        self.documents_metadata = self.load_metadata()
    
    def load_metadata(self) -> Dict[str, Any]:
        """Cargar metadatos de documentos existentes"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error cargando metadatos: {e}")
                return {}
        return {}
    
    def save_metadata(self):
        """Guardar metadatos de documentos"""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.documents_metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error guardando metadatos: {e}")
    
    async def process_uploaded_file(self, file_path: str, filename: str, file_type: str) -> Dict[str, Any]:
        """
        Procesar un archivo subido y agregarlo a la base de datos vectorial
        
        Args:
            file_path: Ruta del archivo subido
            filename: Nombre original del archivo
            file_type: Tipo de archivo (pdf, docx, txt)
            
        Returns:
            Dict con información del procesamiento
        """
        try:
            # Generar ID único para el documento
            doc_id = str(uuid.uuid4())
            
            # Cargar documento según el tipo
            documents = await self.load_document(file_path, file_type)
            
            if not documents:
                return {"error": "No se pudo procesar el documento"}
            
            # Dividir texto en chunks
            text_chunks = self.text_splitter.split_documents(documents)
            
            # Agregar metadatos a cada chunk
            for i, chunk in enumerate(text_chunks):
                chunk.metadata.update({
                    "doc_id": doc_id,
                    "filename": filename,
                    "file_type": file_type,
                    "chunk_index": i,
                    "upload_date": datetime.now().isoformat()
                })
            
            # Agregar a la base de datos vectorial
            self.vectorstore.add_documents(text_chunks)
            
            # Guardar metadatos
            self.documents_metadata[doc_id] = {
                "filename": filename,
                "file_type": file_type,
                "file_path": file_path,
                "upload_date": datetime.now().isoformat(),
                "chunks_count": len(text_chunks),
                "status": "processed"
            }
            
            self.save_metadata()
            
            logger.info(f"Documento procesado: {filename} ({len(text_chunks)} chunks)")
            
            return {
                "doc_id": doc_id,
                "filename": filename,
                "chunks_count": len(text_chunks),
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error procesando archivo {filename}: {e}")
            return {"error": str(e)}
    
    async def load_document(self, file_path: str, file_type: str) -> List[Document]:
        """Cargar documento según su tipo"""
        try:
            if file_type.lower() == 'pdf':
                loader = PDFPlumberLoader(file_path)
                return loader.load()
            elif file_type.lower() in ['docx', 'doc']:
                loader = Docx2txtLoader(file_path)
                return loader.load()
            elif file_type.lower() == 'txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                return [Document(page_content=content, metadata={"source": file_path})]
            else:
                logger.warning(f"Tipo de archivo no soportado: {file_type}")
                return []
        except Exception as e:
            logger.error(f"Error cargando documento {file_path}: {e}")
            return []
    
    async def query_documents(self, query: str, k: int = 5) -> Dict[str, Any]:
        """
        Realizar consulta sobre los documentos usando RAG
        
        Args:
            query: Pregunta del usuario
            k: Número de documentos relevantes a recuperar
            
        Returns:
            Dict con la respuesta y documentos relevantes
        """
        try:
            # Crear retriever
            retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": k}
            )
            
            # Crear cadena de QA
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True
            )
            
            # Ejecutar consulta
            result = qa_chain({"query": query})
            
            # Procesar documentos fuente
            source_docs = []
            for doc in result["source_documents"]:
                source_docs.append({
                    "content": doc.page_content[:200] + "...",
                    "filename": doc.metadata.get("filename", "Unknown"),
                    "chunk_index": doc.metadata.get("chunk_index", 0)
                })
            
            return {
                "answer": result["result"],
                "source_documents": source_docs,
                "query": query,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error en consulta: {e}")
            return {
                "error": str(e),
                "query": query,
                "timestamp": datetime.now().isoformat()
            }
    
    def get_documents_list(self) -> List[Dict[str, Any]]:
        """Obtener lista de documentos procesados"""
        return [
            {
                "doc_id": doc_id,
                **metadata
            }
            for doc_id, metadata in self.documents_metadata.items()
        ]
    
    def delete_document(self, doc_id: str) -> bool:
        """Eliminar documento de la base de datos"""
        try:
            if doc_id in self.documents_metadata:
                # Eliminar de metadatos
                metadata = self.documents_metadata[doc_id]
                del self.documents_metadata[doc_id]
                
                # Eliminar archivo físico
                file_path = Path(metadata["file_path"])
                if file_path.exists():
                    file_path.unlink()
                
                # Eliminar de base de datos vectorial
                # (Chroma no tiene método directo para eliminar por metadatos específicos)
                # Necesitaríamos recrear la base de datos sin este documento
                
                self.save_metadata()
                return True
            return False
        except Exception as e:
            logger.error(f"Error eliminando documento {doc_id}: {e}")
            return False
    
    def get_system_info(self) -> Dict[str, Any]:
        """Obtener información del sistema"""
        return {
            "total_documents": len(self.documents_metadata),
            "storage_path": str(self.storage_path),
            "model_name": self.model_name,
            "embeddings_model": self.embeddings_model,
            "vectordb_path": str(self.vectordb_path)
        }

# Instancia global del sistema RAG
rag_system = DocumentRAGSystem()

# Funciones helper para usar en la API
async def process_document(file_path: str, filename: str, file_type: str) -> Dict[str, Any]:
    """Procesar documento y agregarlo al sistema RAG"""
    return await rag_system.process_uploaded_file(file_path, filename, file_type)

async def query_knowledge_base(query: str, k: int = 5) -> Dict[str, Any]:
    """Consultar la base de conocimiento"""
    return await rag_system.query_documents(query, k)

def get_documents() -> List[Dict[str, Any]]:
    """Obtener lista de documentos"""
    return rag_system.get_documents_list()

def delete_document(doc_id: str) -> bool:
    """Eliminar documento"""
    return rag_system.delete_document(doc_id)

def get_system_status() -> Dict[str, Any]:
    """Obtener estado del sistema"""
    return rag_system.get_system_info() 