# api_server.py
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import os

# Añadir el directorio actual al path para importar promptgen_app
# Esto es útil si ejecutas api_server.py directamente y está en el mismo directorio que promptgen_app.py
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Intentar importar las funciones de lógica de promptgen_app
try:
    from promptgen_real import (
    analyze_prompt_quality_bart,
    get_structural_feedback,
    generate_variations,
    generate_ideas
)
    print("✅ Funciones importadas exitosamente desde promptgen_core.py")
    print("🤖 Usando el sistema corregido sin mockups")
except ImportError as e:
    print(f"Error al importar desde promptgen_app.py: {e}")
    print("Asegúrate de que promptgen_app.py está en el mismo directorio o en el PYTHONPATH.")
    # Definir funciones dummy para que el servidor pueda arrancar y señalar el problema
    def analyze_prompt_quality_bart(prompt: str):
        return {"error": "Función 'analyze_prompt_quality_bart' no importada correctamente desde promptgen_app.py"}
    def get_structural_feedback(prompt: str, model_name: str):
        return {"error": "Función 'get_structural_feedback' no importada correctamente"}
    def generate_variations(prompt: str, model_name: str, num_variations: int):
        return {"error": "Función 'generate_variations' no importada correctamente"}
    def generate_ideas(prompt: str, model_name: str, num_ideas: int):
        return {"error": "Función 'generate_ideas' no importada correctamente"}

app = FastAPI(
    title="PromptGen API",
    description="API para analizar y mejorar prompts de lenguaje natural.",
    version="0.1.0"
)

# Configuración de CORS
# Permite que tu frontend Next.js (ej. localhost:3000) haga peticiones a esta API
origins = [
    "http://localhost",         # Para desarrollo local general
    "http://localhost:3000",    # Puerto común para Next.js/React
    "http://127.0.0.1:3000",
    # Añade aquí la URL de tu aplicación Next.js si está en un dominio diferente en producción
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, # Lista de orígenes permitidos
    allow_credentials=True,
    allow_methods=["*"],    # Permite todos los métodos (GET, POST, etc.)
    allow_headers=["*"],    # Permite todas las cabeceras
)

# --- Modelos Pydantic para Validación de Peticiones/Respuestas ---

class PromptRequest(BaseModel):
    prompt: str
    # Podríamos añadir model_name aquí si fuera general para todas las peticiones
    # o en modelos de petición específicos para tareas que lo requieran.

class ModelNameRequest(PromptRequest):
    model_name: str = "gpt2"

class VariationRequest(ModelNameRequest):
    num_variations: int = 3

class IdeaRequest(ModelNameRequest):
    num_ideas: int = 3

# --- Endpoints de la API ---

@app.post("/api/analyze_quality")
async def api_analyze_quality(request_data: PromptRequest):
    """
    Analiza la calidad de un prompt dado utilizando el modelo BART MNLI.
    Espera un JSON con {"prompt": "tu texto de prompt"}
    """
    if not request_data.prompt or not request_data.prompt.strip():
        raise HTTPException(status_code=400, detail="El prompt no puede estar vacío.")
    try:
        result = analyze_prompt_quality_bart(request_data.prompt)
        if result.get("error"):
            raise HTTPException(status_code=500, detail=result.get("error"))
        return result
    except Exception as e:
        print(f"Error en el endpoint /api/analyze_quality: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {str(e)}")

# Endpoints para las otras funcionalidades (usando placeholders por ahora)
@app.post("/api/get_feedback")
async def api_structural_feedback(request_data: ModelNameRequest):
    if not request_data.prompt or not request_data.prompt.strip():
        raise HTTPException(status_code=400, detail="El prompt no puede estar vacío.")
    try:
        result = get_structural_feedback(request_data.prompt, request_data.model_name)
        if result.get("error"):
            raise HTTPException(status_code=500, detail=result.get("error"))
        return result
    except Exception as e:
        print(f"Error en el endpoint /api/get_feedback: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {str(e)}")

@app.post("/api/generate_variations")
async def api_generate_variations(request_data: VariationRequest): # Usar VariationRequest
    if not request_data.prompt or not request_data.prompt.strip():
        raise HTTPException(status_code=400, detail="El prompt no puede estar vacío.")
    try:
        result = generate_variations(request_data.prompt, request_data.model_name, request_data.num_variations)
        if result.get("error"):
            raise HTTPException(status_code=500, detail=result.get("error"))
        return result
    except Exception as e:
        print(f"Error en el endpoint /api/generate_variations: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {str(e)}")

@app.post("/api/generate_ideas")
async def api_generate_ideas(request_data: IdeaRequest): # Usar IdeaRequest
    if not request_data.prompt or not request_data.prompt.strip():
        raise HTTPException(status_code=400, detail="El prompt no puede estar vacío.")
    try:
        result = generate_ideas(request_data.prompt, request_data.model_name, request_data.num_ideas)
        if result.get("error"):
            raise HTTPException(status_code=500, detail=result.get("error"))
        return result
    except Exception as e:
        print(f"Error en el endpoint /api/generate_ideas: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Bienvenido a la API de PromptGen. Accede a /docs para ver la documentación de la API."}

# --- Ejecución del Servidor ---
if __name__ == "__main__":
    print("Iniciando servidor FastAPI con Uvicorn...")
    # Para desarrollo, Uvicorn es una buena opción.
    # host="0.0.0.0" para que sea accesible desde fuera del contenedor si usas Docker, o desde la red local.
    # reload=True para que el servidor se reinicie automáticamente con los cambios en el código.
    uvicorn.run("api_server:app", host="127.0.0.1", port=5000, reload=True)
    # Nota: 'api_server:app' se refiere al archivo api_server.py y la instancia 'app' de FastAPI dentro de él.
    # Necesitarás instalar FastAPI y Uvicorn: pip install fastapi "uvicorn[standard]" pydantic 