# 🧪 Prueba de Nuevas Funcionalidades del Chatbot

## ✅ ¿Qué se ha mejorado?

### 1. **Manejo de Errores de Ollama**
- El chatbot ahora detecta si Ollama no está instalado
- Proporciona instrucciones claras para solucionarlo
- Funciona incluso sin Ollama para preguntas básicas

### 2. **Preguntas sobre Archivos Cargados**
- Puedes preguntar qué archivos tienes cargados
- Buscar archivos específicos
- Obtener información sobre los documentos

## 🔧 Cómo Probar

### **Paso 1: Solucionar el Error de Ollama**
```bash
# Ejecuta este archivo (doble clic):
SOLUCION_OLLAMA_RAPIDA.bat
```

### **Paso 2: Iniciar el Sistema**
```bash
# Terminal 1:
python simple_api_server.py

# Terminal 2:
npm run dev
```

### **Paso 3: Probar las Nuevas Funcionalidades**

#### **🗂️ Preguntas sobre Archivos**
Prueba estas preguntas en el chat:

1. **"¿Qué archivos tienes cargados?"**
   - Respuesta: Lista completa de archivos con detalles

2. **"¿Cuántos documentos tienes?"**
   - Respuesta: Número total + información de cada archivo

3. **"¿Existe el archivo manual.pdf?"**
   - Respuesta: Confirma si existe + detalles del archivo

4. **"Listar todos los archivos"**
   - Respuesta: Lista detallada con tipos y fragmentos

5. **"Mostrar documentos disponibles"**
   - Respuesta: Vista completa de documentos + consejos

#### **🔍 Búsqueda de Archivos Específicos**
Si tienes archivos cargados, prueba:

1. **"¿Tienes algún archivo de manual?"**
   - Encuentra archivos que contengan "manual" en el nombre

2. **"Existe documento corporativo?"**
   - Busca archivos relacionados con "corporativo"

3. **"Hay archivos PDF?"**
   - Muestra archivos por tipo específico

#### **🤖 Funcionamiento Sin Ollama**
Si Ollama no está instalado:

1. **Cualquier pregunta sobre contenido**
   - Respuesta: Información del contexto + instrucciones de instalación

2. **Preguntas sobre archivos**
   - Funcionan perfectamente sin Ollama

## 📋 Lista de Pruebas

### ✅ Pruebas Básicas
- [ ] El chatbot responde a "¿qué archivos tienes?"
- [ ] Muestra lista de archivos correctamente
- [ ] Detecta archivos específicos mencionados
- [ ] Proporciona información detallada de cada archivo

### ✅ Pruebas con Ollama
- [ ] Responde preguntas sobre contenido de archivos
- [ ] Cita fuentes correctamente
- [ ] Genera respuestas coherentes

### ✅ Pruebas sin Ollama
- [ ] Muestra información del contexto
- [ ] Proporciona instrucciones de instalación
- [ ] Sigue funcionando para preguntas sobre archivos

## 🆘 Solución de Problemas

### **Error: "model not found"**
```bash
# Ejecuta:
SOLUCION_OLLAMA_RAPIDA.bat
```

### **El chatbot no responde**
1. Verifica que `simple_api_server.py` esté ejecutándose
2. Verifica que el frontend esté en `http://localhost:3000/chat`
3. Revisa la consola para errores

### **Las preguntas sobre archivos no funcionan**
1. Verifica que hayas subido archivos
2. Usa preguntas específicas como "¿qué archivos tienes?"
3. Revisa que los archivos se hayan procesado correctamente

## 🎯 Preguntas de Prueba Rápida

Copia y pega estas preguntas en el chat:

```
¿Qué archivos tienes cargados?
¿Cuántos documentos hay disponibles?
Muestra todos los archivos
¿Existe algún archivo PDF?
Listar documentos disponibles
¿Hay archivos de manual?
¿Tienes documentos corporativos?
```

## 💡 Consejos para Mejores Resultados

1. **Sube algunos archivos primero** para probar todas las funcionalidades
2. **Usa preguntas específicas** para obtener mejores respuestas
3. **Instala Ollama** para funcionalidad completa de IA
4. **Prueba diferentes tipos de preguntas** para ver la versatilidad del sistema

---

## 🎉 ¡Listo!

El chatbot ahora es mucho más inteligente y útil. Puede:
- ✅ Responder preguntas sobre archivos cargados
- ✅ Funcionar sin Ollama para tareas básicas
- ✅ Proporcionar información detallada de documentos
- ✅ Ayudarte a solucionar problemas de configuración

¡Prueba todas las funcionalidades y disfruta del chatbot mejorado! 🚀 