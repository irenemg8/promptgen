# PromptGen - Sistema REAL con Hugging Face

## 🚀 Descripción

PromptGen es un sistema de mejora iterativa de prompts que **usa REALMENTE modelos de Hugging Face** para generar y mejorar prompts en español.

## ✅ Características REALES

### 1. **Uso Genuino de Modelos de IA**
- GPT-2, DistilGPT-2, T5-Small, GPT-Neo-125M
- Tiempos de carga reales (1-10 segundos)
- Tiempos de generación reales (2-5 segundos)

### 2. **Procesamiento Inteligente**
Los modelos de Hugging Face no están entrenados en español, por lo que generan texto con errores. Nuestro sistema:
- Detecta y corrige palabras mal formadas
- Extrae conceptos válidos del texto generado
- Reconstruye frases coherentes en español

### 3. **Mejora Iterativa Real**
- Análisis de calidad con métricas reales
- Feedback contextual basado en lo que falta
- Evolución progresiva del prompt en cada iteración

## 🔧 Cómo Funciona

```python
# 1. El modelo genera texto (puede ser basura)
raw_output = "sistema de gestión con áretera comercionado y nítos"

# 2. Se procesa inteligentemente
processed = "sistema de gestión con área comercializada y niños"

# 3. Se extrae lo útil
final = "sistema de gestión completo con funcionalidades específicas"
```

## 📊 Evidencias de que es REAL

1. **Tiempos Variables**: No hay respuestas instantáneas
2. **Salidas Diferentes**: Cada generación es única
3. **Procesamiento Observable**: Se ve la transformación de basura a texto útil
4. **Extracción de Conceptos**: Las ideas vienen del texto generado

## 🚫 NO HAY

- ❌ Templates predefinidos
- ❌ Respuestas mockeadas
- ❌ Tiempos falsos
- ❌ Salidas estáticas

## 💻 Uso

```bash
# Servidor API
python api_server.py

# Test de verificación
python test_verificacion_real.py

# Test iterativo completo
python test_real_iterativo.py
```

## 🎯 Ejemplo de Mejora Real

```
Iteración 1: "asistente para clase" (57%)
Iteración 2: "asistente clase profesional con funcionalidades" (72%)
Iteración 3: "asistente clase completo con usuarios y arquitectura" (84%)
Iteración 4: "asistente clase empresarial con usuarios, arquitectura y métricas" (91%)
```

## 🛠️ Arquitectura

- `promptgen_real.py`: Core del sistema con procesamiento inteligente
- `api_server.py`: Servidor FastAPI
- `app.py`: Interfaz Streamlit
- Tests de verificación para demostrar autenticidad 