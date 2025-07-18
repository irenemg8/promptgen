# 🚀 SOLUCIÓN RÁPIDA PARA OLLAMA

## ❌ **Problema**: "Failed to connect to Ollama"

## ✅ **Solución en 3 pasos**:

### 1. **INSTALACIÓN AUTOMÁTICA** (Recomendado)
```bash
# Ejecutar el script de instalación automática
python setup_ollama.py
```

### 2. **INSTALACIÓN MANUAL**

#### **Windows:**
1. Descargar desde: https://ollama.com/download
2. Ejecutar el instalador `OllamaSetup.exe`
3. Seguir las instrucciones por defecto

#### **macOS:**
```bash
# Con Homebrew
brew install ollama

# O manual
curl -fsSL https://ollama.com/install.sh | sh
```

#### **Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### 3. **INICIAR OLLAMA**

#### **Opción A: Servicio en segundo plano**
```bash
# Iniciar servicio
ollama serve

# Mantener abierta esta terminal
```

#### **Opción B: Script automático**
```bash
# Windows
start_ollama.bat

# macOS/Linux
./start_ollama.sh
```

---

## 🔧 **VERIFICACIÓN**

### Comprobar si funciona:
```bash
# Verificar versión
ollama --version

# Verificar servicio
curl http://localhost:11434/api/tags
```

### Instalar modelo básico:
```bash
# Modelo pequeño y rápido (1.3GB)
ollama pull llama3.2:1b

# Modelo estándar (2GB)
ollama pull llama3.2:3b
```

---

## 🚀 **INICIO RÁPIDO**

### Una vez instalado:
1. **Terminal 1**: `ollama serve`
2. **Terminal 2**: `python simple_api_server.py`
3. **Terminal 3**: `npm run dev`
4. **Navegador**: `http://localhost:3001/chat`

---

## 🆘 **SOLUCIÓN DE PROBLEMAS**

### **Error: "command not found"**
```bash
# Reiniciar terminal o agregar al PATH
export PATH=$PATH:/usr/local/bin
```

### **Error: "port 11434 already in use"**
```bash
# Encontrar proceso
lsof -i :11434
# Matar proceso
kill -9 <PID>
```

### **Error: "model not found"**
```bash
# Listar modelos instalados
ollama list

# Instalar modelo necesario
ollama pull llama3.2:1b
```

### **Error: "connection refused"**
```bash
# Verificar que el servicio esté corriendo
ps aux | grep ollama

# Si no está corriendo, iniciar
ollama serve
```

---

## 🎯 **COMANDOS ÚTILES**

```bash
# Listar modelos instalados
ollama list

# Descargar modelo
ollama pull <modelo>

# Ejecutar modelo interactivo
ollama run <modelo>

# Ver procesos de Ollama
ollama ps

# Detener servicio
pkill ollama
```

---

## 📚 **MODELOS RECOMENDADOS**

| Modelo | Tamaño | Uso | Comando |
|--------|--------|-----|---------|
| llama3.2:1b | 1.3GB | Rápido, básico | `ollama pull llama3.2:1b` |
| llama3.2:3b | 2GB | Equilibrado | `ollama pull llama3.2:3b` |
| llama3.1:8b | 4.7GB | Completo | `ollama pull llama3.1:8b` |
| qwen2.5:7b | 4.4GB | Alternativo | `ollama pull qwen2.5:7b` |

---

## 🌐 **RECURSOS ADICIONALES**

- **Documentación oficial**: https://ollama.com/download
- **Modelos disponibles**: https://ollama.com/library
- **GitHub**: https://github.com/ollama/ollama

---

## ⚡ **SOLUCIÓN ULTRA RÁPIDA**

Si tienes prisa, ejecuta estos comandos en orden:

```bash
# 1. Instalar (según tu sistema)
curl -fsSL https://ollama.com/install.sh | sh

# 2. Iniciar servicio
ollama serve &

# 3. Instalar modelo básico
ollama pull llama3.2:1b

# 4. Probar
curl http://localhost:11434/api/tags
```

---

¡Listo! Ahora tu sistema debería conectarse correctamente a Ollama. 🎉 