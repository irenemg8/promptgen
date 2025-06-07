import os
from dotenv import load_dotenv
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer, util
import numpy as np
import warnings
import re

# Se eliminan todas las referencias a APIs externas y carga de variables de entorno (dotenv).
# El proyecto ahora se basa exclusivamente en modelos de Hugging Face que no requieren API keys.

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
        
        # Diferenciar la carga según la arquitectura del modelo
        if "t5" in model_name.lower():
            # Modelos T5 son Encoder-Decoder (Seq2Seq)
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                trust_remote_code=True
            ).to('cuda' if torch.cuda.is_available() else 'cpu')
            pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
        else:
            # Modelos como GPT-2 o GPT-Neo son Decoder-Only (CausalLM)
            # Para modelos más grandes, usar cuantización para reducir el uso de memoria
            if "7b" in model_name.lower() or "1.3b" in model_name.lower(): # Ampliado para otros tamaños
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

# --- Lógica de Generación y Análisis (Refactorizada y Simplificada) ---

def generate_text_dispatcher(model_name, prompt, max_length=150):
    """
    Generador de texto que utiliza exclusivamente modelos locales de Hugging Face.
    """
    pipe = get_local_text_generation_pipeline(model_name)
    if pipe is None:
        return {"error": f"Modelo local '{model_name}' no pudo ser cargado."}

    # El prompt ya viene pre-formateado con ejemplos (few-shot)
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
        # Limpieza de la salida: quitar el prompt original para obtener solo la respuesta generada
        generated_text = sequences[0]['generated_text']
        
        if generated_text.startswith(full_prompt):
            cleaned_text = generated_text[len(full_prompt):].strip()
        else:
            # Para modelos text2text que no repiten el prompt
            cleaned_text = generated_text.strip()

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
    """
    Genera feedback sobre la estructura y claridad de un prompt usando un modelo local.
    """
    # Prompt simple y directo, asignando un rol y una tarea clara al modelo.
    instruction = f"Eres un experto en ingeniería de prompts. Analiza el siguiente prompt y proporciona feedback conciso y accionable para mejorarlo en una lista de viñetas. Céntrate en la estructura, claridad y elementos que podrían faltar. No generes un nuevo prompt, solo da el feedback.\n\nPrompt a analizar:\n---\n{prompt}\n---\n\nFeedback útil:"
    
    feedback = generate_text_dispatcher(model_name, instruction, max_length=100)
    if isinstance(feedback, dict) and 'error' in feedback:
        return {"error": feedback['error']}
        
    # Limpieza para quitar cualquier texto que no sea una lista de viñetas
    feedback_lines = [line.strip() for line in feedback.split('\n') if line.strip().startswith('-')]
    return {"feedback": "\n".join(feedback_lines) if feedback_lines else feedback}

def generate_variations(prompt: str, model_name: str = "gpt2", num_variations: int = 3):
    """
    Genera variaciones de un prompt usando un modelo local.
    """
    # Instrucción directa para reescribir, enfocada en la tarea.
    instruction = f"Tu tarea es mejorar el siguiente prompt. Reescríbelo en {num_variations} versiones que sean más detalladas, claras y efectivas. Proporciona únicamente las nuevas versiones del prompt, una por línea, sin añadir texto extra.\n\nPrompt original:\n---\n{prompt}\n---\n\nPrompts mejorados:"

    response_text = generate_text_dispatcher(model_name, instruction, max_length=100 * num_variations)
    if isinstance(response_text, dict) and 'error' in response_text:
        return {"error": response_text['error']}

    # Limpiar y separar las variaciones, eliminando cualquier línea que no sea un prompt.
    variations = [v.strip() for v in response_text.split('\n') if v.strip() and len(v.split()) > 3] # Filtrar líneas muy cortas/sin sentido
    if variations:
        variations = [re.sub(r'^\d+\.\s*|-\s*', '', v).strip() for v in variations]

    return {"variations": variations[:num_variations]}

def generate_ideas(prompt: str, model_name: str = "gpt2", num_ideas: int = 3):
    """
    Genera ideas basadas en un prompt usando un modelo local.
    """
    # Instrucción directa para generar ideas relacionadas con el concepto del usuario.
    instruction = f"Actúa como un generador de ideas creativo. Basándote en el siguiente concepto, genera una lista de {num_ideas} ideas para prompts que lo expandan o lo exploren desde diferentes ángulos. Proporciona solo la lista de ideas, una por línea.\n\nConcepto base:\n---\n{prompt}\n---\n\nNuevas ideas de prompt:"

    ideas_text = generate_text_dispatcher(model_name, instruction, max_length=70 * num_ideas)
    if isinstance(ideas_text, dict) and 'error' in ideas_text:
        return {"error": ideas_text['error']}
        
    # Limpia la salida para obtener una lista de ideas.
    ideas = [idea.strip() for idea in ideas_text.split('\n') if idea.strip() and len(idea.split()) > 3]
    if ideas:
        ideas = [re.sub(r'^\d+\.\s*|-\s*', '', idea).strip() for idea in ideas]
        
    return {"ideas": ideas}

def main():
    print("Módulo promptgen_app cargado. Funciones listas para ser usadas por el servidor API.")
    # Prueba rápida opcional
    # test_model = "gpt2" # o "google-t5/t5-small"
    # test_prompt = "crea una imagen de un astronauta montando un caballo en marte"
    # print(f"\n--- Probando Feedback con {test_model} ---")
    # print(get_structural_feedback(test_prompt, model_name=test_model))
    # print(f"\n--- Probando Variaciones con {test_model} ---")
    # print(generate_variations(test_prompt, model_name=test_model))
    # print(f"\n--- Probando Ideas con {test_model} ---")
    # print(generate_ideas(test_prompt, model_name=test_model))

if __name__ == '__main__':
    main() 