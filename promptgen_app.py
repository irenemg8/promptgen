import os
from dotenv import load_dotenv
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer, util
import numpy as np
import warnings
import re

# --- Nuevas importaciones para APIs ---
import openai
import anthropic
import google.generativeai as genai
from groq import Groq

# --- Configuración de APIs ---
load_dotenv()

# Configura los clientes de las API. Si la clave no está, el cliente no se crea.
try:
    openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except Exception as e:
    openai_client = None
    print(f"Advertencia: No se pudo inicializar el cliente de OpenAI. Error: {e}")

try:
    anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
except Exception as e:
    anthropic_client = None
    print(f"Advertencia: No se pudo inicializar el cliente de Anthropic. Error: {e}")

try:
    if os.getenv("GOOGLE_API_KEY"):
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        google_client = genai.GenerativeModel('gemini-1.5-flash-latest')
    else:
        google_client = None
except Exception as e:
    google_client = None
    print(f"Advertencia: No se pudo inicializar el cliente de Google Gemini. Error: {e}")

try:
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
except Exception as e:
    groq_client = None
    print(f"Advertencia: No se pudo inicializar el cliente de Groq. Error: {e}")

# Ignorar advertencias específicas de transformers
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.models.bart.modeling_bart")

# --- Carga de Modelos Locales ---
# Diccionario para cachear los pipelines de modelos locales
local_pipelines = {}
active_local_model_name = None

def get_local_text_generation_pipeline(model_name="gpt2", force_reload=False):
    global active_local_model_name
    
    if model_name in local_pipelines and not force_reload and active_local_model_name == model_name:
        return local_pipelines[model_name]

    print(f"Cargando el modelo local: {model_name}...")
    
    # Liberar memoria de la GPU si hay un modelo cargado
    if active_local_model_name and active_local_model_name in local_pipelines:
        print(f"Liberando memoria del modelo anterior: {active_local_model_name}")
        del local_pipelines[active_local_model_name]
        torch.cuda.empty_cache()

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Para modelos más grandes, usar cuantización para reducir el uso de memoria
        if "7b" in model_name.lower() or "6.7b" in model_name.lower():
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                quantization_config=bnb_config,
                trust_remote_code=True
            )
        else: # Modelos más pequeños no necesitan cuantización obligatoriamente
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True
            ).to('cuda' if torch.cuda.is_available() else 'cpu')

        # Asegurarse de que el token de padding está definido
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = model.config.eos_token_id

        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
        local_pipelines[model_name] = pipe
        active_local_model_name = model_name
        
        print(f"Modelo local '{model_name}' cargado exitosamente.")
        return pipe
    except Exception as e:
        print(f"Error crítico al cargar el modelo local '{model_name}': {e}")
        local_pipelines[model_name] = None
        return None

# Carga del modelo para análisis de calidad (BART)
try:
    print("Cargando modelo de análisis de calidad (BART MNLI)...")
    quality_analyzer_pipeline = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    print("Modelo de análisis de calidad cargado.")
except Exception as e:
    print(f"Error al cargar el modelo de análisis de calidad: {e}")
    quality_analyzer_pipeline = None

# Carga del modelo para similitud semántica (opcional, para palabras clave)
try:
    print("Cargando modelo de similitud semántica...")
    similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Modelo de similitud semántica cargado.")
except Exception as e:
    print(f"Error al cargar el modelo de similitud semántica: {e}")
    similarity_model = None

# --- Lógica de Generación y Análisis (Refactorizada) ---

def call_api_model(model_name, system_prompt, user_prompt, max_tokens=500):
    """Función para llamar a los diferentes clientes de API."""
    try:
        if model_name.startswith("api/gpt"):
            if not openai_client: return "Error: Cliente de OpenAI no configurado."
            model_id = model_name.split('/')[1]
            response = openai_client.chat.completions.create(
                model=f"gpt-3.5-turbo" if model_id == "gpt-3.5-turbo" else "gpt-4o", # Asegurar modelo correcto
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content

        elif model_name.startswith("api/claude"):
            if not anthropic_client: return "Error: Cliente de Anthropic no configurado."
            model_id = model_name.split('/')[1].replace('-', '_') # ej. claude-3-5-sonnet -> claude_3_5_sonnet
            response = anthropic_client.messages.create(
                model=f"claude-3-haiku-20240307" if "haiku" in model_id else "claude-3-5-sonnet-20240620",
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                max_tokens=max_tokens,
            )
            return response.content[0].text
        
        elif model_name.startswith("api/gemini"):
            if not google_client: return "Error: Cliente de Google Gemini no configurado."
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            response = google_client.generate_content(full_prompt)
            return response.text

        elif model_name.startswith("api/groq"):
            if not groq_client: return "Error: Cliente de Groq no configurado."
            model_id = 'llama3-8b-8192' # Mapear nuestro nombre al de Groq
            response = groq_client.chat.completions.create(
                model=model_id,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
            
        else:
            return f"Error: Modelo API '{model_name}' no reconocido."

    except Exception as e:
        return f"Error al llamar a la API para el modelo {model_name}: {e}"

def generate_text_dispatcher(model_name, prompt, max_length=150):
    """Despachador que decide si usar un modelo local o una API."""
    if model_name.startswith("api/"):
        # Para las APIs, el prompt puede ser más complejo (system + user)
        # Aquí simplificamos y lo pasamos directamente. Las funciones específicas construirán el prompt completo.
        return call_api_model(model_name, "Eres un asistente experto en la creación de prompts.", prompt)
    else:
        # Lógica para modelos locales
        pipe = get_local_text_generation_pipeline(model_name)
        if pipe is None:
            return {"error": f"Modelo local '{model_name}' no pudo ser cargado."}
        
        # Algunos modelos usan un formato de prompt específico
        if "instruct" in model_name.lower() or "chat" in model_name.lower():
            # Formato simple para modelos instruct/chat
            full_prompt = f"### Instruction:\n{prompt}\n\n### Response:"
        else:
            full_prompt = prompt

        try:
            sequences = pipe(
                full_prompt,
                max_new_tokens=max_length,
                num_return_sequences=1,
                eos_token_id=pipe.tokenizer.eos_token_id,
                pad_token_id=pipe.tokenizer.pad_token_id,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.7,
            )
            # Limpiar la salida para quitar el prompt original
            generated_text = sequences[0]['generated_text']
            # Encontrar el inicio de la respuesta
            response_start = generated_text.find("### Response:")
            if response_start != -1:
                cleaned_text = generated_text[response_start + len("### Response:"):].strip()
            else:
                # Para modelos que no usan el formato, quitar el prompt inicial
                cleaned_text = generated_text[len(full_prompt):].strip()
            return cleaned_text
        except Exception as e:
            return {"error": f"Error durante la generación de texto con '{model_name}': {str(e)}"}

# --- Funciones de la API (Endpoint Logic) ---

def analyze_prompt_quality_bart(prompt: str):
    if quality_analyzer_pipeline is None:
        return {"error": "Modelo de análisis de calidad no cargado."}
    
    candidate_labels = ["claro y específico", "ambiguo y vago", "demasiado corto", "demasiado largo", "bien estructurado", "mal estructurado", "creativo", "técnico"]
    try:
        analysis = quality_analyzer_pipeline(prompt, candidate_labels, multi_label=True)
        scores = dict(zip(analysis['labels'], np.round(analysis['scores'], 2)))
        
        # Interpretación simple
        report = f"Análisis de calidad para el prompt:\n"
        report += f"- Claridad y especificidad: {scores.get('claro y específico', 0)*100:.0f}%\n"
        report += f"- Ambigüedad: {scores.get('ambiguo y vago', 0)*100:.0f}%\n"
        report += f"- Potencial creativo: {scores.get('creativo', 0)*100:.0f}%\n"
        report += f"- Estructura: {scores.get('bien estructurado', 0)*100:.0f}%\n"

        # Extracción de palabras clave (usando un método simple y el modelo de similitud)
        keywords = "N/A"
        if similarity_model:
            try:
                # Usar regex para encontrar "palabras" y luego el modelo para encontrar las más significativas
                words = list(set(re.findall(r'\b\w+\b', prompt.lower())))
                if len(words) > 1:
                    prompt_embedding = similarity_model.encode(prompt, convert_to_tensor=True)
                    word_embeddings = similarity_model.encode(words, convert_to_tensor=True)
                    similarities = util.pytorch_cos_sim(prompt_embedding, word_embeddings)[0]
                    # Tomar las 5 palabras con mayor similitud al prompt completo
                    top_indices = torch.topk(similarities, k=min(5, len(words))).indices
                    keywords = ", ".join([words[i] for i in top_indices])
            except Exception as e:
                keywords = f"Error en extracción: {e}"

        return {
            "quality_report": report,
            "interpreted_keywords": keywords,
            "raw_scores": scores
        }
    except Exception as e:
        return {"error": f"Error en análisis de calidad: {str(e)}"}

def get_structural_feedback(prompt: str, model_name: str = "gpt2"):
    system_prompt = "Eres un experto en ingeniería de prompts. Analiza el siguiente prompt de usuario y proporciona feedback conciso y accionable para mejorarlo. Céntrate en la estructura, claridad y elementos que podrían faltar. No generes un nuevo prompt, solo da el feedback. Sé breve, usa viñetas."
    user_prompt = f"Analiza este prompt y dame feedback para mejorarlo:\n\n---\n{prompt}\n---"
    
    feedback = generate_text_dispatcher(model_name, user_prompt)
    if isinstance(feedback, dict) and 'error' in feedback:
        return {"error": feedback['error']}
        
    return {"feedback": feedback}

def generate_variations(prompt: str, model_name: str = "gpt2", num_variations: int = 3):
    system_prompt = f"Eres un asistente de IA experto en reescribir y mejorar prompts. Dado el siguiente prompt, genera {num_variations} variaciones mejoradas y creativas. Cada variación debe ser distinta. Devuelve solo las variaciones, una por línea, sin numeración ni texto adicional."
    user_prompt = f"Genera {num_variations} variaciones mejoradas para este prompt:\n\n---\n{prompt}\n---"

    response_text = generate_text_dispatcher(model_name, user_prompt, max_length=150 * num_variations)
    if isinstance(response_text, dict) and 'error' in response_text:
        return {"error": response_text['error']}

    # Limpiar y separar las variaciones
    variations = [v.strip() for v in response_text.split('\n') if v.strip() and not v.strip().startswith('-')]
    # Si el split falla, puede que el modelo devuelva una lista numerada, intentar limpiarla
    if len(variations) < num_variations:
        variations = [re.sub(r'^\d+\.\s*', '', v).strip() for v in response_text.split('\n') if v.strip()]

    return {"variations": variations[:num_variations]}

def generate_ideas(prompt: str, model_name: str = "gpt2", num_ideas: int = 3):
    system_prompt = f"Eres un generador de ideas creativo. Basándote en el siguiente concepto o prompt, genera una lista de {num_ideas} ideas relacionadas o conceptos derivados. Sé conciso y presenta las ideas claramente, separadas por saltos de línea y sin numeración."
    user_prompt = f"Basado en este concepto, genera {num_ideas} ideas nuevas:\n\n---\n{prompt}\n---"

    ideas_text = generate_text_dispatcher(model_name, user_prompt, max_length=100 * num_ideas)
    if isinstance(ideas_text, dict) and 'error' in ideas_text:
        return {"error": ideas_text['error']}
        
    return {"ideas": ideas_text}

def main():
    print("Módulo promptgen_app cargado. Funciones listas para ser usadas por el servidor API.")
    # Prueba rápida opcional
    # test_model = "gpt2" # o "api/gpt-3.5-turbo"
    # test_prompt = "crea una imagen de un astronauta montando un caballo en marte"
    # print(f"\n--- Probando Feedback con {test_model} ---")
    # print(get_structural_feedback(test_prompt, model_name=test_model))
    # print(f"\n--- Probando Variaciones con {test_model} ---")
    # print(generate_variations(test_prompt, model_name=test_model))
    # print(f"\n--- Probando Ideas con {test_model} ---")
    # print(generate_ideas(test_prompt, model_name=test_model))

if __name__ == '__main__':
    main() 