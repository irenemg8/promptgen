# 🚀 PromptGen - Sistema Seguro de Documentos

## ✅ SISTEMA LISTO PARA USAR

Tu sistema de documentos seguro está **completamente configurado y funcionando**. Aquí tienes todo lo que necesitas saber:

## 🎯 Inicio Rápido

### Opción 1: Usar el Script de Inicio (Recomendado)
```bash
# Simplemente ejecuta este archivo:
start_app.bat
```

### Opción 2: Inicio Manual
```bash
# Terminal 1 - Backend
python simple_api_server.py

# Terminal 2 - Frontend
npm run dev
```

## 🔗 Enlaces de Acceso

- **🌐 Aplicación Web**: [http://localhost:3000](http://localhost:3000)
- **📊 API Backend**: [http://localhost:8000](http://localhost:8000)
- **📚 Documentación API**: [http://localhost:8000/docs](http://localhost:8000/docs)

## 🔐 Características Implementadas

### ✅ Seguridad
- **Cifrado AES-256** de todos los archivos
- **Claves derivadas** con PBKDF2 (100,000 iteraciones)
- **Almacenamiento local** - sin conexiones externas
- **Metadatos cifrados** para máxima seguridad

### ✅ Formatos Soportados
- **📄 Documentos**: PDF, DOCX, DOC, TXT
- **📊 Datos**: JSON, CSV, Excel
- **🌐 Web**: HTML, Markdown
- **💻 Código**: Python, JavaScript, etc.
- **🖼️ Imágenes**: JPG, PNG, GIF (con OCR)

### ✅ Funcionalidades
- **Carga masiva** de archivos (arrastra y suelta)
- **Procesamiento paralelo** para velocidad
- **Respuestas en < 10 segundos**
- **Memoria persistente** - los archivos se mantienen
- **Sistema de caché** para consultas rápidas

## 📱 Cómo Usar la Aplicación

### 1. Subir Documentos
- Arrastra archivos a la zona de carga
- O usa el botón "Seleccionar Archivos"
- Los archivos se cifran automáticamente

### 2. Hacer Preguntas
- Escribe tu pregunta en el chat
- El sistema buscará en todos tus documentos
- Obtendrás respuestas precisas con fuentes

### 3. Gestionar Documentos
- Ve la lista de documentos en el sidebar
- Elimina archivos que ya no necesites
- Verifica estadísticas del sistema

## 🛠️ Tecnologías Utilizadas

- **Backend**: Python, FastAPI, Ollama
- **Frontend**: Next.js, React, TypeScript
- **Cifrado**: Cryptography (AES-256)
- **Base de Datos**: Archivos cifrados locales
- **Procesamiento**: Multithread para velocidad

## 📊 Monitoreo del Sistema

El sistema incluye métricas en tiempo real:
- **Uso de memoria y CPU**
- **Número de documentos procesados**
- **Tiempos de respuesta**
- **Estadísticas de caché**

## 🔧 Resolución de Problemas

### El sistema no inicia
```bash
# Verifica dependencias
pip install -r requirements.txt
npm install

# Verifica puertos
netstat -an | findstr :3000
netstat -an | findstr :8000
```

### Error con Ollama
```bash
# Instala Ollama desde: https://ollama.com/download
# Descarga modelos necesarios
ollama pull llama3.2:3b
```

### Archivos no se procesan
- Verifica que el formato sea soportado
- Revisa los logs del backend
- Asegúrate que el archivo no esté corrupto

## 📁 Estructura de Archivos

```
promptgen/
├── simple_api_server.py      # Servidor backend
├── simple_document_system.py # Sistema de documentos
├── components/
│   └── chat-interface.tsx    # Interfaz de chat
├── simple_documents/         # Documentos cifrados
│   ├── encrypted_files/      # Archivos cifrados
│   └── metadata.enc          # Metadatos cifrados
└── start_app.bat            # Script de inicio
```

## 🔒 Datos de Seguridad

- **Todos los archivos se cifran** antes de almacenarse
- **Las claves no se almacenan** en texto plano
- **Los metadatos están cifrados**
- **No hay conexiones externas**
- **Todo funciona offline**

## 🚀 Características Avanzadas

### Procesamiento Inteligente
- **Extracción de texto** de PDFs e imágenes
- **Análisis de datos** de CSVs y Excel
- **Parsing de código** con sintaxis highlighting
- **Búsqueda semántica** por contenido

### Optimización de Rendimiento
- **Caché en memoria** para consultas frecuentes
- **Procesamiento paralelo** de archivos
- **Compresión LZ4** para eficiencia
- **Chunks optimizados** para velocidad

## 📞 Soporte

Si necesitas ayuda:
1. Revisa los logs del backend
2. Verifica la documentación de la API en `/docs`
3. Asegúrate de tener todas las dependencias instaladas

## 🎉 ¡Listo para Usar!

Tu sistema está completamente funcional y seguro. Puedes empezar a subir documentos y hacer preguntas inmediatamente.

**¡Disfruta de tu sistema de documentos seguro!** 🔐📄💬 