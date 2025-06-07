# Documentación del Proyecto: PromptGen

Este documento detalla el diseño, desarrollo y despliegue de la aplicación **PromptGen**, un asistente de ingeniería de prompts basado en modelos de Hugging Face.

## 1. Justificación y Descripción del Caso de Uso

### 1.1. Problema a Resolver

La efectividad de los grandes modelos de lenguaje (LLMs) depende en gran medida de la calidad de los *prompts* (las instrucciones) que reciben. Un prompt vago, incompleto o mal estructurado suele producir resultados pobres, genéricos o irrelevantes. Este problema afecta tanto a usuarios noveles como a desarrolladores experimentados, quienes a menudo deben recurrir a un proceso de prueba y error para encontrar la formulación adecuada.

### 1.2. Caso de Uso Elegido: Asistente de Ingeniería de Prompts

Para abordar este problema, hemos desarrollado **PromptGen**, una aplicación diseñada para funcionar como un **asistente inteligente para la creación y refinamiento de prompts**.

En lugar de ser simplemente una interfaz para generar texto, PromptGen analiza la idea inicial del usuario y la enriquece de forma automática. El objetivo es transformar un concepto básico en un prompt detallado, estructurado y optimizado para obtener los mejores resultados posibles de un modelo de IA generativa.

### 1.3. Público Objetivo

La aplicación está pensada para un público amplio:

*   **Desarrolladores y Programadores:** Que utilizan LLMs a través de APIs (como en Cursor) y necesitan generar prompts de alta calidad para sus aplicaciones.
*   **Creadores de Contenido y Escritores:** Que usan herramientas como ChatGPT o Claude para generar borradores, ideas o textos completos.
*   **Diseñadores y Artistas:** Que interactúan con modelos de generación de imágenes como Sora o Adobe Firefly y necesitan descripciones textuales muy precisas.
*   **Estudiantes e Investigadores:** Que exploran las capacidades de los LLMs para sus trabajos académicos.

## 2. Modelos Utilizados y Comparación

La aplicación se basa exclusivamente en modelos pre-entrenados y de código abierto disponibles en el **Model Hub de Hugging Face**, ejecutándose de forma local para garantizar la privacidad y el control. Se han seleccionado varios modelos, cada uno con un propósito específico.

### 2.1. Modelo Principal para Análisis de Calidad

*   **Modelo:** `facebook/bart-large-mnli`
*   **Tarea:** *Zero-Shot Classification*
*   **Justificación:** Este modelo es fundamental para la funcionalidad principal de PromptGen. Lo utilizamos para analizar el prompt del usuario según una serie de etiquetas predefinidas (claridad, especificidad, completitud) sin necesidad de re-entrenamiento. Su capacidad para entender la inferencia del lenguaje natural nos permite evaluar si un prompt contiene la información necesaria para ser efectivo.

### 2.2. Modelos para Generación de Texto (Variaciones e Ideas)

Se ha integrado un sistema que permite al usuario elegir entre varios modelos generativos. Esto no solo ofrece flexibilidad, sino que también sirve como una comparativa implícita, ya que el usuario puede observar cómo diferentes arquitecturas responden al mismo prompt.

*   **`gpt2` y `distilgpt2`:**
    *   **Descripción:** Modelos autorregresivos de propósito general. `distilgpt2` es una versión más ligera y rápida.
    *   **Reflexión:** Son excelentes como modelos base. Ofrecen una buena relación entre velocidad y calidad para generar variaciones y sugerencias rápidas. Son menos propensos a "alucinar" en tareas creativas sencillas, pero pueden ser menos coherentes en prompts muy largos o complejos. Son la opción por defecto por su bajo consumo de recursos.

*   **`google-t5/t5-small`:**
    *   **Descripción:** Un modelo de tipo *encoder-decoder* (texto a texto).
    *   **Reflexión:** T5 funciona muy bien cuando la tarea se puede formular como una traducción o una instrucción directa. Lo usamos internamente para reestructurar frases. Su rendimiento es bueno para refinar la estructura de un prompt, pero puede ser menos "creativo" que los modelos GPT para la generación de ideas abiertas.

*   **`EleutherAI/gpt-neo-125M`:**
    *   **Descripción:** Una alternativa de código abierto a la familia GPT, entrenada por EleutherAI.
    *   **Reflexión:** En nuestras pruebas, `gpt-neo` a menudo produce resultados más creativos y diversos que `gpt2`. Es una excelente opción cuando el usuario busca ideas más originales o variaciones menos literales de su prompt inicial. Representa una alternativa sólida que enriquece la capacidad de la herramienta.

### 2.3. Modelo para Análisis Semántico

*   **Modelo:** `all-MiniLM-L6-v2`
*   **Tarea:** *Sentence Similarity*
*   **Justificación:** Este modelo se utiliza para tareas internas como la extracción de palabras clave y la detección de conceptos centrales en el prompt del usuario. Es extremadamente rápido y eficiente para calcular la similitud semántica, lo que nos permite interpretar la intención del usuario de manera más precisa antes de pasar el prompt a los modelos más pesados.

## 3. Descripción de la Interfaz y Funcionamiento

Se ha desarrollado una interfaz gráfica web utilizando **Next.js (React)** para el frontend y **FastAPI (Python)** para el backend, comunicándose a través de una API REST. Se ha priorizado una experiencia de usuario clara e intuitiva.

### 3.1. Flujo de Trabajo del Usuario

1.  **Entrada de la Idea:** El usuario introduce su idea o prompt inicial en el área de texto principal.
2.  **Selección de Opciones:** Opcionalmente, puede seleccionar el modelo generativo que desea utilizar y etiquetar su prompt para una plataforma específica (ej. `ChatGPT`, `Sora`) para su propia organización.
3.  **Análisis y Generación:** Al pulsar el botón "Generar", la aplicación inicia un proceso secuencial:
    *   **Paso 1: Análisis de Calidad:** Se envía el prompt al backend, que utiliza `bart-large-mnli` para realizar un análisis completo.
    *   **Paso 2: Generación de Variaciones:** El prompt se envía al modelo generativo seleccionado (`gpt2`, `gpt-neo`, etc.) para crear 3 versiones mejoradas.
    *   **Paso 3: Generación de Ideas:** Se utiliza el mismo modelo para generar ideas creativas relacionadas con el tema del prompt.
4.  **Visualización de Resultados:** Los resultados se muestran en una nueva tarjeta en la interfaz, que contiene:
    *   El **prompt original** y el **prompt mejorado** (la primera y mejor variación generada).
    *   Una sección desplegable con el **Informe de Calidad**, que detalla puntuaciones y sugerencias sobre claridad, especificidad y estructura.
    *   Secciones desplegables con las **otras variaciones** y las **ideas generadas**.
5.  **Historial:** Todas las generaciones se guardan en un historial en la barra lateral, permitiendo al usuario revisar y comparar resultados anteriores.

### 3.2. Otras Herramientas

*   **Uvicorn:** Servidor ASGI para ejecutar la API de FastAPI.
*   **shadcn/ui y Tailwind CSS:** Librerías utilizadas para construir una interfaz de usuario moderna, responsive y accesible de forma rápida.
*   **Lucide-React:** Para los iconos, mejorando la usabilidad de la interfaz.

## 4. Pasos para Desplegar y Utilizar la Aplicación

La aplicación consta de dos componentes principales: el **servidor de API (backend)** y la **aplicación cliente (frontend)**. Ambos deben ejecutarse simultáneamente.

### 4.1. Prerrequisitos

*   Python 3.8 o superior.
*   Node.js 18 o superior.
*   Una GPU con al menos 6 GB de VRAM es **muy recomendable** para ejecutar los modelos de forma eficiente.

### 4.2. Instalación del Backend (Servidor de API)

1.  **Abrir una terminal** en la raíz del proyecto.
2.  **Crear un entorno virtual (recomendado):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # En Windows: venv\Scripts\activate
    ```
3.  **Instalar las dependencias de Python:**
    ```bash
    pip install -r requirements.txt
    ```
    *Nota: Esto descargará PyTorch, Transformers, Sentence-Transformers y otras librerías necesarias.*

### 4.3. Instalación del Frontend

1.  **Abrir una segunda terminal** en la raíz del proyecto.
2.  **Instalar las dependencias de Node.js:**
    ```bash
    npm install
    ```

### 4.4. Ejecución de la Aplicación

1.  **Iniciar el servidor de backend:**
    *   En la primera terminal (con el entorno virtual de Python activado), ejecuta:
    ```bash
    python api_server.py
    ```
    *   El servidor se iniciará en `http://127.0.0.1:5000`. La primera vez que se inicie, **tardará varios minutos** en descargar los modelos de Hugging Face del hub. Este proceso solo ocurre una vez.

2.  **Iniciar la aplicación de frontend:**
    *   En la segunda terminal, ejecuta:
    ```bash
    npm run dev
    ```
    *   La aplicación web se iniciará y estará accesible en `http://localhost:3000`.

3.  **Uso:**
    *   Abre tu navegador y ve a `http://localhost:3000`.
    *   La aplicación estará lista para usarse.
