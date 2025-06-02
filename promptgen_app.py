import os
# Evitar un warning de tokenizers_parallelism que no es relevante para la inferencia básica
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import pipeline
import torch # Es bueno importarlo para verificar si CUDA está disponible si se quiere usar GPU

# --- Carga de Modelos (Hacerlo una vez) ---
# Usamos un diccionario para cargar los modelos bajo demanda o al inicio.
# Por ahora, cargaremos BART directamente.
device = 0 if torch.cuda.is_available() else -1 # 0 para GPU, -1 para CPU

try:
    quality_analyzer_pipeline = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=device
    )
    print(f"Modelo BART (quality_analyzer) cargado en: {'GPU' if device == 0 else 'CPU'}")
except Exception as e:
    print(f"Error al cargar el modelo BART (quality_analyzer): {e}")
    quality_analyzer_pipeline = None

try:
    # Usaremos distilgpt2 para las tareas de generación por ser más ligero
    # Podríamos cambiar a "gpt2" o "t5-small"/	5-base" si se necesita más potencia o capacidades diferentes
    text_generator_pipeline = pipeline(
        "text-generation", 
        model="gpt2",
        device=device,
        # Es crucial establecer pad_token_id si el modelo no lo tiene por defecto o si es igual a eos_token_id
        # Para GPT-2, eos_token_id (50256) se usa a menudo como pad_token_id si no hay uno específico.
        pad_token_id=50256 # pipeline intentará usar tokenizer.eos_token_id si es None
    )
    print(f"Modelo GPT-2 (text_generator) cargado en: {'GPU' if device == 0 else 'CPU'}")
except Exception as e:
    print(f"Error al cargar el modelo GPT-2 (text_generator): {e}")
    text_generator_pipeline = None

# Definición de etiquetas para la calidad del prompt
QUALITY_CANDIDATE_LABELS = [
    "prompt claro y específico",
    "prompt bien formulado",
    "prompt necesita más detalles",
    "prompt vago o ambiguo",
    "prompt podría ser más conciso",
    "prompt accionable y directo",
    "prompt creativo e inspirador"
]

# --- Funciones de Lógica de Modelos ---

def analyze_prompt_quality_bart(prompt_text: str):
    if not quality_analyzer_pipeline:
        return {
            "error": "Modelo de análisis de calidad no cargado.",
            "quality_report": "N/A",
            "interpreted_keywords": "N/A"
        }
    if not prompt_text or not prompt_text.strip():
        return {
            "quality_report": "El prompt está vacío.",
            "interpreted_keywords": "N/A"
        }

    try:
        # Realizar la clasificación zero-shot
        results = quality_analyzer_pipeline(prompt_text, QUALITY_CANDIDATE_LABELS, multi_label=True)
        
        # Formatear el reporte de calidad
        quality_report_parts = ["Calidad del Prompt (Análisis con BART-MNLI):"]
        scores_for_report = []

        for label, score in zip(results['labels'], results['scores']):
            scores_for_report.append(f"- {label}: {score:.2%}")
        
        quality_report_parts.extend(sorted(scores_for_report, key=lambda x: float(x.split(": ")[1].replace('%','')), reverse=True))
        
        # "Palabras clave interpretadas" serán las etiquetas con mayor puntuación
        # Tomamos las top 3 etiquetas como "palabras clave interpretadas"
        interpreted_keywords = [label for label, score in sorted(zip(results['labels'], results['scores']), key=lambda x: x[1], reverse=True)[:3]]

        return {
            "quality_report": "\n".join(quality_report_parts),
            "interpreted_keywords": ", ".join(interpreted_keywords)
        }
    except Exception as e:
        print(f"Error durante el análisis de calidad: {e}")
        return {
            "error": f"Error durante el análisis: {str(e)}",
            "quality_report": "Error al procesar.",
            "interpreted_keywords": "Error al procesar."
        }

def get_structural_feedback(prompt_text: str, model_name: str = "GPT-2"):
    if not text_generator_pipeline:
        return {"error": "Modelo generador de texto no cargado.", "feedback": "N/A", "keywords": "N/A"}
    if not prompt_text or not prompt_text.strip():
        return {"feedback": "El prompt está vacío.", "keywords": "N/A"}

    system_prompt = (
        f"Analiza la estructura del siguiente prompt y ofrece feedback constructivo y conciso sobre su claridad, especificidad y si es accionable para una IA. "
        f"Proporciona únicamente el feedback en formato de lista, donde cada punto comience con un guion (-). No incluyas encabezados, introducciones, conclusiones ni ningún otro texto fuera de la lista de feedback. "
        f"Evita generar URLs o cualquier contenido no relacionado con el análisis estructural del prompt. "
        f"Prompt a analizar: \\\"\"\"{prompt_text}\"\"\\n"
        f"Feedback Estructural Detallado:\\n- " # Mantenemos el inicio con guion para guiar al modelo
    )
    try:
        # Usar eos_token_id del tokenizer si está disponible
        eos_token_id = text_generator_pipeline.tokenizer.eos_token_id if text_generator_pipeline.tokenizer.eos_token_id is not None else 50256
        
        generated_outputs = text_generator_pipeline(
            system_prompt,
            max_new_tokens=100, # Reducido para feedback más conciso
            num_return_sequences=1,
            pad_token_id=text_generator_pipeline.tokenizer.pad_token_id if text_generator_pipeline.tokenizer.pad_token_id is not None else eos_token_id,
            eos_token_id=eos_token_id,
            no_repeat_ngram_size=2, 
            temperature=0.7,
            top_k=50
        )
        # Extraer solo el texto generado después del prompt de sistema
        full_generated_text = generated_outputs[0]['generated_text']
        feedback_text = "- " + full_generated_text[len(system_prompt):].strip()
        
        # Limpieza básica para remover cualquier repetición del prompt o encabezados no deseados
        feedback_text = feedback_text.split("Prompt a analizar:")[0].strip() # Mejorado para quitar repetición del prompt
        feedback_text = feedback_text.split("Feedback Estructural Detallado:")[0].strip() # Mejorado
        feedback_text = feedback_text.split("http://")[0].split("https://")[0].strip() # Eliminar URLs
        
        # Asegurar que el feedback no esté vacío y comience con un guion si es posible
        if not feedback_text or not feedback_text.strip() or feedback_text.strip() == "-":
            feedback_text = "No se pudo generar feedback estructural claro."
        elif not feedback_text.startswith("-"):
            feedback_text = "- " + feedback_text

        keywords = [model_name.lower(), "feedback", "estructura"]
        return {"feedback": feedback_text if feedback_text and feedback_text.strip() != "-" else "No se pudo generar feedback.", "keywords": ", ".join(keywords)}
    except Exception as e:
        print(f"Error en get_structural_feedback: {e}")
        return {"error": f"Error al generar feedback: {str(e)}", "feedback": "Error", "keywords": "Error"}

def generate_variations(prompt_text: str, model_name: str = "GPT-2", num_variations: int = 3):
    if not text_generator_pipeline:
        return {"error": "Modelo generador de texto no cargado.", "variations": [], "keywords": "N/A"}
    if not prompt_text or not prompt_text.strip():
        return {"variations": ["El prompt está vacío."], "keywords": "N/A"}

    system_prompt = (
        f"Genera {num_variations} variaciones efectivas del siguiente prompt. "
        f"Cada variación debe ser un prompt completo y usable, presentado en una nueva línea y comenzando con un guion. "
        f"Prompt Original: \"""{prompt_text}""\n"
        f"Variaciones Sugeridas:\n- "
    )
    try:
        eos_token_id = text_generator_pipeline.tokenizer.eos_token_id if text_generator_pipeline.tokenizer.eos_token_id is not None else 50256

        generated_outputs = text_generator_pipeline(
            system_prompt,
            max_new_tokens=60 * num_variations, 
            num_return_sequences=1, 
            pad_token_id=text_generator_pipeline.tokenizer.pad_token_id if text_generator_pipeline.tokenizer.pad_token_id is not None else eos_token_id,
            eos_token_id=eos_token_id,
            temperature=0.75,
            top_k=50,
            top_p=0.95,
            no_repeat_ngram_size=2
        )
        full_generated_text = generated_outputs[0]['generated_text']
        variations_block = "- " + full_generated_text[len(system_prompt):].strip()
        
        # Parseo simplificado: cada línea no vacía después del guion es una variación
        variations = [v.strip() for v in variations_block.split('\n- ') if v.strip()]
        # Quitar el primer guion si es necesario y luego los guiones de las líneas siguientes
        cleaned_variations = []
        for var in variations:
            cleaned_var = var.lstrip('- ').strip()
            if cleaned_var:
                 cleaned_variations.append(cleaned_var)
        
        variations = cleaned_variations[:num_variations]
                   
        keywords = [model_name.lower(), "variaciones", "generación"]
        return {"variations": variations if variations else ["No se pudieron generar variaciones claras."], "keywords": ", ".join(keywords)}
    except Exception as e:
        print(f"Error en generate_variations: {e}")
        return {"error": f"Error al generar variaciones: {str(e)}", "variations": [], "keywords": "Error"}

def generate_ideas(prompt_text: str, model_name: str = "GPT-2", num_ideas: int = 3):
    if not text_generator_pipeline:
        return {"error": "Modelo generador de texto no cargado.", "ideas": "N/A", "keywords": "N/A"}
    if not prompt_text or not prompt_text.strip():
        return {"ideas": "El prompt está vacío.", "keywords": "N/A"}

    system_prompt = (
        f"Genera {num_ideas} ideas creativas basadas en el siguiente tema/prompt. "
        f"Cada idea debe ser concisa, en una nueva línea, y comenzar con un guion. "
        f"Tema/Prompt: \"""{prompt_text}""\n"
        f"Ideas Sugeridas:\n- "
    )
    try:
        eos_token_id = text_generator_pipeline.tokenizer.eos_token_id if text_generator_pipeline.tokenizer.eos_token_id is not None else 50256
        
        generated_outputs = text_generator_pipeline(
            system_prompt,
            max_new_tokens=40 * num_ideas, 
            num_return_sequences=1, 
            pad_token_id=text_generator_pipeline.tokenizer.pad_token_id if text_generator_pipeline.tokenizer.pad_token_id is not None else eos_token_id,
            eos_token_id=eos_token_id,
            temperature=0.8,
            top_k=50,
            no_repeat_ngram_size=2
        )
        full_generated_text = generated_outputs[0]['generated_text']
        ideas_block = "- " + full_generated_text[len(system_prompt):].strip()

        ideas_list = [idea.strip() for idea in ideas_block.split('\n- ') if idea.strip()]
        cleaned_ideas = []
        for idea_item in ideas_list:
            cleaned_item = idea_item.lstrip('- ').strip()
            if cleaned_item:
                cleaned_ideas.append(cleaned_item)

        ideas_list = cleaned_ideas[:num_ideas]
        ideas_text = "\n".join(ideas_list) if ideas_list else "No se pudieron generar ideas claras."

        keywords = [model_name.lower(), "ideas", "generación"]
        return {"ideas": ideas_text, "keywords": ", ".join(keywords)}
    except Exception as e:
        print(f"Error en generate_ideas: {e}")
        return {"error": f"Error al generar ideas: {str(e)}", "ideas": "Error", "keywords": "Error"}

# --- Ejemplo de uso de las funciones directamente (sin API) ---
if __name__ == '__main__':
    print("--- Iniciando Pruebas Directas de Funciones de promptgen_app.py ---")
    # Test de Calidad (BART)
    if quality_analyzer_pipeline:
        test_prompt_quality = "Describe una escena de batalla épica entre dragones y unicornios en un castillo flotante."
        print(f"\nProbando análisis de calidad para: '{test_prompt_quality}'")
        analysis_result = analyze_prompt_quality_bart(test_prompt_quality)
        print("Reporte de Calidad:")
        print(analysis_result.get("quality_report"))
        print("Palabras Clave Interpretadas (Calidad):", analysis_result.get("interpreted_keywords"))

        test_prompt_vago = "cosas"
        print(f"\nProbando análisis de calidad para: '{test_prompt_vago}'")
        analysis_result_vago = analyze_prompt_quality_bart(test_prompt_vago)
        print("Reporte de Calidad:")
        print(analysis_result_vago.get("quality_report"))
        print("Palabras Clave Interpretadas (Calidad):", analysis_result_vago.get("interpreted_keywords"))
    else:
        print("\nquality_analyzer_pipeline no disponible. Sáltandose pruebas de calidad.")

    # Tests de Generación (GPT-2)
    if text_generator_pipeline:
        print("\n--- Pruebas con text_generator_pipeline (GPT-2) ---")

        test_prompt_feedback = "Escribe una historia sobre un perro que viaja en el tiempo."
        print(f"\nProbando feedback estructural para: '{test_prompt_feedback}'")
        feedback_result = get_structural_feedback(test_prompt_feedback)
        print("Feedback Estructural:", feedback_result.get("feedback"))
        
        test_prompt_variations = "Crea un eslogan publicitario para una nueva bebida energética natural."
        print(f"\nProbando generación de variaciones para: '{test_prompt_variations}' (pidiendo 3)")
        variations_result = generate_variations(test_prompt_variations, num_variations=3)
        print("Variaciones Generadas:")
        if isinstance(variations_result.get("variations"), list):
            for i, var in enumerate(variations_result.get("variations")):
                print(f"{i+1}. {var}")
        else:
            print(variations_result.get("variations"))

        test_prompt_ideas = "Formas innovadoras de usar la realidad aumentada en el turismo."
        print(f"\nProbando generación de ideas para: '{test_prompt_ideas}' (pidiendo 3)")
        ideas_result = generate_ideas(test_prompt_ideas, num_ideas=3)
        print("Ideas Generadas:") # Esperamos un string con ideas separadas por 

        print(ideas_result.get("ideas"))

        print("\n--- Prueba Adicional Variaciones (Plan de marketing) --- ")
        variations_result_2 = generate_variations("Plan de marketing para el lanzamiento de un nuevo videojuego indie de puzzles.", num_variations=2)
        print("Prompt: Plan de marketing para el lanzamiento de un nuevo videojuego indie de puzzles.")
        print("Variaciones:")
        if isinstance(variations_result_2.get("variations"), list):
            for i, var in enumerate(variations_result_2.get("variations")):
                print(f"{i+1}. {var}")
        else:
            print(variations_result_2.get("variations"))


        print("\n--- Prueba Adicional Ideas (Productividad) --- ")
        ideas_result_2 = generate_ideas("Cómo los equipos remotos pueden mejorar la colaboración y la comunicación.", num_ideas=4)
        print("Prompt: Cómo los equipos remotos pueden mejorar la colaboración y la comunicación.")
        print("Ideas:")
        print(ideas_result_2.get("ideas"))
    else:
        print("\ntext_generator_pipeline no disponible. Sáltandose pruebas de generación.")
    print("\n--- Pruebas Directas Finalizadas ---") 