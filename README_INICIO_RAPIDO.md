# 🚀 INICIO RÁPIDO - SISTEMA PROMPTGEN

## ❌ **PROBLEMA**: "Failed to connect to Ollama"

---

## ✅ **SOLUCIÓN RÁPIDA** (3 pasos):

### 1. **INSTALAR OLLAMA**
```bash
# Opción A: Script automático
instalar_ollama.bat

# Opción B: Manual
# Ve a: https://ollama.com/download
# Descarga e instala OllamaSetup.exe
```

### 2. **INICIAR TODO EL SISTEMA**
```bash
# Script todo-en-uno
inicio_completo.bat

# Este script:
# ✅ Verifica todas las dependencias
# ✅ Inicia Ollama
# ✅ Inicia el backend Python
# ✅ Inicia el frontend Next.js
# ✅ Abre el navegador automáticamente
```

### 3. **USAR EL CHAT**
Una vez ejecutado el script, el navegador se abrirá automáticamente en:
- **http://localhost:3000/chat** (o puerto 3001)

---

## 🎯 **PASOS MANUALES** (si los scripts no funcionan):

### 1. **Terminal 1** - Ollama:
```bash
ollama serve
```

### 2. **Terminal 2** - Backend:
```bash
python simple_api_server.py
```

### 3. **Terminal 3** - Frontend:
```bash
npm run dev
```

### 4. **Navegador**:
```
http://localhost:3001/chat
```

---

## 🛠️ **MEJORAS IMPLEMENTADAS**:

### **✨ Botón de Sidebar Mejorado**:
- **Flecha animada** para expandir/contraer
- **Sidebar comprimido** en desktop con iconos
- **Botón flotante** en móvil
- **Mejor responsive design**

### **📱 Interfaz Optimizada**:
- **Completamente responsive**
- **Sin scroll manual necesario**
- **Auto-scroll automático** a mensajes nuevos
- **Lista de documentos** optimizada para no desbordarse

### **🔧 Scripts de Automatización**:
- `instalar_ollama.bat` - Instalación automática de Ollama
- `inicio_completo.bat` - Inicia todo el sistema
- `detener_sistema.bat` - Detiene todos los servicios

---

## 🆘 **SOLUCIÓN DE PROBLEMAS**:

### **Error: "Ollama no está instalado"**
```bash
# Ejecutar:
instalar_ollama.bat

# O manual:
# 1. Ir a https://ollama.com/download
# 2. Descargar OllamaSetup.exe
# 3. Instalar con opciones por defecto
```

### **Error: "Puerto en uso"**
```bash
# Detener servicios:
detener_sistema.bat

# Luego reiniciar:
inicio_completo.bat
```

### **Error: "Model not found"**
```bash
# Instalar modelo básico:
ollama pull llama3.2:1b

# Verificar:
ollama list
```

### **Error: Frontend no carga**
```bash
# Verificar puertos:
# Si 3000 no funciona, intenta 3001
http://localhost:3001/chat
```

---

## 🎉 **CARACTERÍSTICAS DEL SISTEMA**:

### **🔐 Seguridad**:
- Cifrado AES-256 de todos los archivos
- Operación completamente local
- Sin conexiones externas

### **⚡ Rendimiento**:
- Respuestas en menos de 10 segundos
- Cache inteligente
- Procesamiento paralelo

### **📁 Formatos Soportados**:
- PDF, DOCX, TXT, JSON, CSV
- Imágenes (con OCR)
- Archivos de código
- Excel, HTML, Markdown

### **🎨 Interfaz Moderna**:
- Dark theme con gradientes
- Diseño tipo ChatGPT
- Sidebar inteligente
- Drag & drop para archivos

---

## 🔄 **COMANDOS ÚTILES**:

```bash
# Verificar estado
curl http://localhost:11434/api/tags    # Ollama
curl http://localhost:8000/api/health   # Backend
curl http://localhost:3001              # Frontend

# Instalar modelo específico
ollama pull llama3.2:3b

# Ver procesos
ollama ps

# Detener todo
detener_sistema.bat
```

---

## 📞 **CONTACTO**:
Si tienes problemas, revisa:
- `SOLUCION_OLLAMA.md` - Guía detallada
- `setup_ollama.py` - Script de instalación avanzado
- Logs en las terminales para más detalles

---

¡Listo! Tu sistema debería estar funcionando perfectamente. 🎉 