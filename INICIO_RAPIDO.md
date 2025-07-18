# 🚀 INICIO RÁPIDO - Sistema de Chat con Documentos

## ✅ **¡Tu sistema YA ESTÁ IMPLEMENTADO!**

Todas las funcionalidades que solicitaste están **100% terminadas**:

- ✅ **Subida de archivos** (PDF, Word, TXT)
- ✅ **Almacenamiento local permanente** 
- ✅ **Chat inteligente con Ollama**
- ✅ **Consultas sobre documentos**
- ✅ **Respuestas con fuentes citadas**

## 🔧 **Configuración (solo una vez)**

### 1. **Instalar Ollama**
- Ve a: **https://ollama.com/download**
- Descarga e instala Ollama para Windows
- Reinicia tu computadora si es necesario

### 2. **Verificar instalación**
```powershell
# Abrir PowerShell y ejecutar:
ollama --version
```

### 3. **Descargar modelos**
```powershell
# Modelo principal para chat (tarda 2-3 minutos)
ollama pull llama3.2:3b

# Modelo para embeddings (tarda 1-2 minutos)
ollama pull mxbai-embed-large
```

## 🎯 **Usar el Sistema**

### **Opción A: Scripts automáticos (RECOMENDADO)**

#### 1. **Iniciar Backend:**
```powershell
# Doble clic en:
start_system.bat
```

#### 2. **Iniciar Frontend (nueva terminal):**
```powershell
# Doble clic en:
start_frontend.bat
```

### **Opción B: Manual**

#### 1. **Terminal 1 - Backend:**
```powershell
python api_server.py
```

#### 2. **Terminal 2 - Frontend:**
```powershell
npm run dev
```

## 🌐 **Acceder al Chat**

1. **Abrir navegador:** http://localhost:3000
2. **Ir al chat:** Clic en botón **"Chat"** o http://localhost:3000/chat
3. **Subir documentos:** Panel derecho → "Seleccionar Archivo"
4. **Hacer preguntas:** Escribir en el chat y enviar

## 📋 **Ejemplo de Uso**

```
1. Subir archivo: "manual_empresa.pdf"
2. Preguntar: "¿Cuál es la política de vacaciones?"
3. El sistema responde con información del PDF
4. Muestra fuentes específicas del documento
```

## 🎨 **Características del Chat**

- **📄 Formatos soportados:** PDF, Word (.docx, .doc), TXT
- **🔍 Búsqueda inteligente:** Encuentra información relevante
- **📖 Fuentes citadas:** Muestra de dónde viene cada respuesta
- **💾 Persistente:** Los documentos se guardan localmente
- **🔒 Privado:** Todo procesamiento es local, sin APIs externas

## 🔍 **Solución de Problemas**

### **Error: "Ollama no está instalado"**
```powershell
# Verificar instalación
ollama --version

# Si no funciona, reinstalar desde: https://ollama.com/download
```

### **Error: "Modelos no encontrados"**
```powershell
# Descargar modelos
ollama pull llama3.2:3b
ollama pull mxbai-embed-large

# Verificar instalación
ollama list
```

### **Error: "Puerto ocupado"**
```powershell
# Cambiar puerto del backend (editar api_server.py línea final)
uvicorn.run("api_server:app", host="0.0.0.0", port=8001)
```

### **Error: "Dependencias faltantes"**
```powershell
# Reinstalar dependencias
pip install -r requirements.txt
```

## 📊 **Estructura del Sistema**

```
promptgen/
├── 📁 documents_storage/     # Documentos subidos (local)
├── 🤖 document_rag_system.py # Sistema RAG con Ollama
├── 🔧 api_server.py          # Backend API
├── 🎨 components/chat-interface.tsx # Interfaz chat
├── 📝 start_system.bat       # Script inicio backend
├── 🌐 start_frontend.bat     # Script inicio frontend
└── 📋 INICIO_RAPIDO.md       # Esta guía
```

## 🎉 **¡Listo para usar!**

Tu sistema de chat con documentos está **completamente funcional**. Solo necesitas:

1. **Instalar Ollama** (una vez)
2. **Ejecutar los scripts** (cada vez que uses el sistema)
3. **Subir documentos y chatear** 

**¡Disfruta tu nuevo asistente de documentos!** 🚀

---

💡 **Tip:** Guarda este archivo para referencia futura.
📞 **Soporte:** irenebati4@gmail.com 