# 🤖 PromptGen - Sistema de Chat con Documentos

## 📋 Resumen

Se ha añadido una nueva funcionalidad de **chat con documentos** que permite:

- ✅ **Subir documentos** (PDF, Word, TXT) y almacenarlos localmente
- ✅ **Procesamiento automático** con embeddings para búsqueda semántica
- ✅ **Chat inteligente** que responde preguntas basándose en el contenido de los documentos
- ✅ **Almacenamiento persistente** local (sin envío a APIs externas)
- ✅ **Interfaz moderna** con historial de conversaciones y fuentes citadas

## 🛠️ Instalación y Configuración

### 1. Instalar Dependencias Python

```bash
# Instalar las nuevas dependencias
pip install -r requirements.txt

# Dependencias principales añadidas:
# - ollama (cliente para Ollama)
# - chromadb (base de datos vectorial)
# - langchain (framework para RAG)
# - pypdf2 (procesamiento de PDFs)
# - python-docx (procesamiento de Word)
```

### 2. Configurar Ollama

#### Opción A: Instalación Automática
```bash
# Ejecutar script de configuración automática
python setup_ollama.py
```

#### Opción B: Instalación Manual

**Windows:**
1. Descarga Ollama desde: https://ollama.com/download
2. Ejecuta el instalador
3. Abre CMD/PowerShell y ejecuta:
```bash
ollama pull llama3.2:3b
ollama pull mxbai-embed-large
```

**macOS/Linux:**
```bash
# Instalar Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Descargar modelos
ollama pull llama3.2:3b
ollama pull mxbai-embed-large
```

### 3. Verificar Instalación

```bash
# Verificar que Ollama está ejecutándose
ollama list

# Iniciar servidor si no está corriendo
ollama serve

# Verificar API (debería devolver JSON con modelos)
curl http://localhost:11434/api/tags
```

## 🚀 Uso del Sistema

### 1. Iniciar Aplicación

```bash
# Terminal 1: Backend (API)
python api_server.py

# Terminal 2: Frontend (Next.js)
npm run dev

# Terminal 3: Ollama (si no está como servicio)
ollama serve
```

### 2. Acceder al Chat

1. Visita: http://localhost:3000
2. Haz clic en el botón **"Chat"** en el header
3. O accede directamente a: http://localhost:3000/chat

### 3. Subir Documentos

1. **Seleccionar archivo:** Haz clic en "Seleccionar Archivo"
2. **Formatos soportados:** PDF, Word (.docx, .doc), TXT
3. **Subir:** Haz clic en "Subir y Procesar"
4. **Esperar:** El sistema procesará el documento automáticamente
5. **Confirmación:** Verás un mensaje de éxito en el chat

### 4. Hacer Consultas

1. **Escribir pregunta:** En el campo de texto del chat
2. **Enviar:** Presiona Enter o haz clic en el botón de enviar
3. **Respuesta:** El sistema buscará en los documentos y generará una respuesta
4. **Fuentes:** Verás las fuentes citadas debajo de cada respuesta

## 📂 Estructura de Archivos

```
promptgen/
├── documents_storage/          # Almacenamiento local de documentos
│   ├── uploads/               # Archivos subidos
│   ├── vectordb/              # Base de datos vectorial (ChromaDB)
│   └── documents_metadata.json # Metadatos de documentos
├── document_rag_system.py      # Sistema RAG principal
├── setup_ollama.py            # Script de configuración
├── components/chat-interface.tsx # Componente de chat
├── app/chat/page.tsx          # Página de chat
└── api_server.py              # Backend con endpoints RAG
```

## 🔧 Configuración Avanzada

### Variables de Entorno

```bash
# Configurar modelos (opcional)
OLLAMA_MODEL=llama3.2:3b
OLLAMA_EMBEDDING_MODEL=mxbai-embed-large

# Configurar paths (opcional)
DOCUMENTS_STORAGE_PATH=./documents_storage
VECTOR_DB_PATH=./documents_storage/vectordb
```

### Personalización de Modelos

Puedes cambiar los modelos editando `document_rag_system.py`:

```python
# Cambiar modelo de chat
model_name = "llama3.1:8b"  # Modelo más grande pero más lento

# Cambiar modelo de embeddings
embeddings_model = "nomic-embed-text"  # Alternativa ligera
```

## 🎯 Funcionalidades Disponibles

### Chat Inteligente
- ✅ Responde preguntas basándose en documentos subidos
- ✅ Cita fuentes específicas
- ✅ Mantiene contexto conversacional
- ✅ Manejo de errores y respuestas informativas

### Gestión de Documentos
- ✅ Subida con drag & drop
- ✅ Vista previa de archivos seleccionados
- ✅ Lista de documentos procesados
- ✅ Eliminación de documentos
- ✅ Conteo de fragmentos procesados

### Procesamiento Inteligente
- ✅ División automática en fragmentos (chunks)
- ✅ Embeddings semánticos
- ✅ Búsqueda por similitud
- ✅ Metadatos enriquecidos

## 📊 Endpoints API

### Chat y Consultas
- `POST /api/query` - Realizar consulta sobre documentos
- `POST /api/upload_document` - Subir y procesar documento
- `GET /api/documents` - Obtener lista de documentos
- `DELETE /api/documents/{id}` - Eliminar documento
- `GET /api/rag_status` - Estado del sistema RAG

### Ejemplo de Uso API

```bash
# Subir documento
curl -X POST -F "file=@documento.pdf" http://localhost:8000/api/upload_document

# Hacer consulta
curl -X POST -H "Content-Type: application/json" \
  -d '{"query": "¿Cuál es el tema principal?", "k": 5}' \
  http://localhost:8000/api/query

# Listar documentos
curl http://localhost:8000/api/documents
```

## 🛡️ Seguridad y Privacidad

### Procesamiento Local
- ✅ **Sin APIs externas:** Todo el procesamiento es local
- ✅ **Datos privados:** Los documentos no salen de tu máquina
- ✅ **Offline:** Funciona sin conexión a internet
- ✅ **Control total:** Tú controlas tus datos

### Almacenamiento Seguro
- ✅ **Encriptación:** Base de datos vectorial local
- ✅ **Metadatos:** Información estructurada en JSON
- ✅ **Cleanup:** Funcionalidad de eliminación completa

## 🔍 Solución de Problemas

### Ollama no se conecta
```bash
# Verificar que Ollama esté corriendo
ollama ps

# Iniciar manualmente
ollama serve

# Verificar puerto
netstat -an | grep 11434
```

### Modelos no encontrados
```bash
# Listar modelos instalados
ollama list

# Descargar modelos faltantes
ollama pull llama3.2:3b
ollama pull mxbai-embed-large
```

### Error de memoria
```bash
# Usar modelo más pequeño
ollama pull llama3.2:1b

# Verificar memoria disponible
free -h  # Linux
vm_stat  # macOS
```

### Archivos no se procesan
- ✅ Verificar formato soportado (PDF, DOCX, TXT)
- ✅ Verificar tamaño del archivo (< 50MB recomendado)
- ✅ Verificar permisos de la carpeta documents_storage/
- ✅ Revisar logs del servidor para errores específicos

## 📈 Rendimiento

### Recomendaciones de Hardware
- **CPU:** 4+ cores para procesamiento eficiente
- **RAM:** 8GB mínimo, 16GB recomendado
- **Almacenamiento:** SSD para mejor rendimiento de la base de datos vectorial

### Optimizaciones
- **Tamaño de chunks:** Ajustar chunk_size en document_rag_system.py
- **Número de resultados:** Modificar parámetro k en las consultas
- **Modelo más rápido:** Usar llama3.2:1b para respuestas más rápidas

## 🤝 Contribuir

### Añadir nuevos formatos
1. Editar `load_document()` en `document_rag_system.py`
2. Añadir loader apropiado de langchain
3. Actualizar `allowed_types` en el API

### Mejorar embeddings
1. Cambiar modelo de embeddings en la configuración
2. Probar con diferentes modelos de Ollama
3. Ajustar parámetros de similitud

## 📞 Soporte

Para problemas técnicos o consultas:

- **Email:** irenebati4@gmail.com
- **Issues:** Crear issue en el repositorio
- **Documentación:** Revisar ollama_config.md

¡Disfruta del nuevo sistema de chat con documentos! 🎉 