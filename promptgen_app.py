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
        # Parámetros optimizados para cada tipo de modelo
        if "t5" in model_name.lower():
            # T5 es un modelo seq2seq que funciona mejor con instrucciones directas
            sequences = pipe(
                full_prompt,
                max_length=max_length,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.8,
                top_k=40,
                top_p=0.9
            )
        elif "gpt2" in model_name.lower() or "distilgpt2" in model_name.lower():
            # GPT-2 necesita parámetros más conservadores
            sequences = pipe(
                full_prompt,
                max_new_tokens=max_length,
                num_return_sequences=1,
                pad_token_id=pipe.tokenizer.pad_token_id,
                eos_token_id=pipe.tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.7,
                top_k=30,
                top_p=0.85,
                repetition_penalty=1.2
            )
        elif "neo" in model_name.lower():
            # GPT-Neo puede manejar mejor creatividad
            sequences = pipe(
                full_prompt,
                max_new_tokens=max_length,
                num_return_sequences=1,
                pad_token_id=pipe.tokenizer.pad_token_id,
                eos_token_id=pipe.tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.75,
                top_k=40,
                top_p=0.9,
                repetition_penalty=1.1
            )
        else:
            # Configuración por defecto
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
    """
    Analiza la calidad de un prompt de manera comprehensiva y proporciona feedback detallado.
    """
    # Análisis básico de estructura
    words = prompt.split()
    word_count = len(words)
    
    # Extraer concepto para contexto
    concept = extract_core_concept(prompt)
    project_type = detect_project_type(concept)
    
    # 1. ANÁLISIS DE PALABRAS CLAVE
    keywords = extract_enhanced_keywords(prompt)
    
    # 2. ANÁLISIS DE COMPLETITUD
    completeness_score, completeness_issues = analyze_completeness(prompt, project_type)
    
    # 3. ANÁLISIS DE CLARIDAD
    clarity_score, clarity_issues = analyze_clarity(prompt, word_count)
    
    # 4. ANÁLISIS DE ESPECIFICIDAD
    specificity_score, specificity_suggestions = analyze_specificity(prompt, project_type)
    
    # 5. ANÁLISIS DE ESTRUCTURA
    structure_score, structure_feedback = analyze_structure(prompt)
    
    # 6. CREAR REPORTE DETALLADO
    report = create_detailed_quality_report(
        prompt, word_count, completeness_score, clarity_score, 
        specificity_score, structure_score, completeness_issues,
        clarity_issues, specificity_suggestions, structure_feedback, project_type
    )
    
    return {
        "quality_report": report,
        "interpreted_keywords": keywords,
        "raw_scores": {
            "completeness": completeness_score,
            "clarity": clarity_score,
            "specificity": specificity_score,
            "structure": structure_score
        }
    }

def extract_enhanced_keywords(prompt: str):
    """
    Extrae palabras clave de manera más inteligente del prompt.
    """
    import re
    
    # Palabras vacías en español más completa
    stopwords = {
        'el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'una', 'para', 'con', 
        'por', 'los', 'las', 'del', 'al', 'es', 'su', 'se', 'como', 'más',
        'me', 'te', 'le', 'nos', 'os', 'les', 'mi', 'tu', 'si', 'no', 'lo',
        'quiero', 'hacer', 'crear', 'generar', 'ayudar', 'puedes', 'prompt',
        'sobre', 'acerca', 'ayúdame', 'necesito', 'gustaría'
    }
    
    # Extraer palabras significativas
    words = re.findall(r'\b[a-záéíóúñüA-ZÁÉÍÓÚÑÜ]{3,}\b', prompt.lower())
    significant_words = [w for w in words if w not in stopwords and len(w) > 2]
    
    # Detectar frases importantes (sustantivo + adjetivo/complemento)
    phrases = re.findall(r'\b(?:aplicación|página|sitio|logo|cartel|video|artículo|campaña|plan|estrategia|sistema|plataforma)\s+[a-záéíóúñü\s]{1,20}(?:para|de|sobre|con)', prompt.lower())
    
    # Combinar palabras individuales y frases
    keywords_list = []
    
    # Añadir frases importantes primero
    for phrase in phrases[:2]:  # Máximo 2 frases
        clean_phrase = re.sub(r'\s+', ' ', phrase.strip())
        if clean_phrase:
            keywords_list.append(clean_phrase)
    
    # Añadir palabras individuales más relevantes
    word_importance = {}
    for word in significant_words:
        # Dar más peso a palabras técnicas y específicas
        weight = 1
        if word in ['diseño', 'desarrollo', 'marketing', 'contenido', 'estrategia', 'aplicación', 'sistema']:
            weight = 2
        if len(word) > 6:  # Palabras más largas suelen ser más específicas
            weight += 1
        word_importance[word] = weight
    
    # Ordenar por importancia y tomar las mejores
    sorted_words = sorted(word_importance.items(), key=lambda x: x[1], reverse=True)
    for word, _ in sorted_words[:4]:  # Máximo 4 palabras individuales
        if word not in ' '.join(keywords_list).lower():
            keywords_list.append(word)
    
    return ', '.join(keywords_list) if keywords_list else "proyecto, diseño"

def analyze_completeness(prompt: str, project_type: str):
    """
    Analiza qué tan completo es el prompt según el tipo de proyecto.
    """
    elements_by_type = {
        'diseño_grafico': ['estilo', 'color', 'tamaño', 'formato', 'público', 'uso'],
        'web_desarrollo': ['funcionalidad', 'audiencia', 'contenido', 'tecnología', 'responsive'],
        'aplicacion_movil': ['plataforma', 'funcionalidad', 'usuarios', 'monetización'],
        'contenido_escrito': ['audiencia', 'tono', 'extensión', 'propósito', 'formato'],
        'marketing': ['objetivo', 'audiencia', 'presupuesto', 'canales', 'kpis'],
        'video_multimedia': ['duración', 'estilo', 'audiencia', 'plataforma', 'mensaje'],
        'general': ['objetivo', 'audiencia', 'recursos', 'contexto']
    }
    
    expected_elements = elements_by_type.get(project_type, elements_by_type['general'])
    prompt_lower = prompt.lower()
    
    present_elements = []
    missing_elements = []
    
    element_indicators = {
        'estilo': ['estilo', 'diseño', 'visual', 'moderno', 'vintage', 'minimalista'],
        'color': ['color', 'colores', 'paleta', 'cromático'],
        'tamaño': ['tamaño', 'dimensión', 'formato', 'resolución'],
        'público': ['audiencia', 'público', 'usuarios', 'clientes', 'target'],
        'funcionalidad': ['función', 'característica', 'feature', 'capacidad'],
        'contenido': ['contenido', 'información', 'texto', 'datos'],
        'objetivo': ['objetivo', 'meta', 'propósito', 'fin'],
        'presupuesto': ['presupuesto', 'costo', 'precio', 'inversión'],
        'duración': ['duración', 'tiempo', 'largo', 'minutos']
    }
    
    for element in expected_elements:
        indicators = element_indicators.get(element, [element])
        if any(indicator in prompt_lower for indicator in indicators):
            present_elements.append(element)
        else:
            missing_elements.append(element)
    
    completeness_score = round((len(present_elements) / len(expected_elements)) * 100)
    
    return completeness_score, missing_elements[:3]  # Máximo 3 elementos faltantes

def analyze_clarity(prompt: str, word_count: int):
    """
    Analiza la claridad del prompt.
    """
    issues = []
    
    # Análisis de longitud
    if word_count < 3:
        issues.append("Prompt demasiado corto")
        clarity_score = 20
    elif word_count < 5:
        issues.append("Necesita más contexto")
        clarity_score = 40
    elif word_count > 25:
        issues.append("Podría ser más conciso")
        clarity_score = 70
    else:
        clarity_score = 85
    
    # Análisis de ambigüedad
    vague_words = ['algo', 'cosa', 'tipo', 'bueno', 'bonito', 'interesante', 'creativo']
    vague_count = sum(1 for word in vague_words if word in prompt.lower())
    if vague_count > 0:
        issues.append("Contiene términos vagos")
        clarity_score -= vague_count * 15
    
    # Análisis de especificidad
    specific_indicators = ['específico', 'exacto', 'preciso', 'detallado', 'particular']
    if any(indicator in prompt.lower() for indicator in specific_indicators):
        clarity_score += 10
    
    return max(0, min(100, clarity_score)), issues

def analyze_specificity(prompt: str, project_type: str):
    """
    Analiza qué tan específico es el prompt y sugiere mejoras.
    """
    suggestions = []
    
    # Sugerencias específicas por tipo de proyecto
    type_suggestions = {
        'diseño_grafico': [
            "Especifica el estilo visual (moderno, vintage, minimalista, etc.)",
            "Define las dimensiones y formato final",
            "Menciona la paleta de colores preferida",
            "Indica el público objetivo"
        ],
        'web_desarrollo': [
            "Define las funcionalidades principales",
            "Especifica el tipo de contenido",
            "Menciona la audiencia objetivo",
            "Indica si necesita ser responsive"
        ],
        'aplicacion_movil': [
            "Especifica las plataformas (iOS, Android)",
            "Define las funcionalidades clave",
            "Menciona el tipo de usuarios",
            "Indica el modelo de monetización"
        ],
        'contenido_escrito': [
            "Define la extensión aproximada",
            "Especifica el tono y estilo",
            "Menciona el público objetivo",
            "Indica el propósito del contenido"
        ],
        'marketing': [
            "Define los objetivos específicos",
            "Especifica la audiencia target",
            "Menciona el presupuesto disponible",
            "Indica los canales preferidos"
        ],
        'video_multimedia': [
            "Especifica la duración deseada",
            "Define el estilo visual",
            "Menciona la plataforma de distribución",
            "Indica el mensaje principal"
        ],
        'general': [
            "Define objetivos específicos",
            "Especifica la audiencia",
            "Menciona los recursos disponibles",
            "Indica el contexto de uso"
        ]
    }
    
    project_suggestions = type_suggestions.get(project_type, type_suggestions['general'])
    
    # Verificar qué elementos ya están presentes
    prompt_lower = prompt.lower()
    present_elements = 0
    
    checklist = {
        'audiencia': ['audiencia', 'público', 'usuarios', 'clientes', 'target'],
        'estilo': ['estilo', 'diseño', 'visual', 'tono'],
        'objetivo': ['objetivo', 'propósito', 'meta', 'fin'],
        'formato': ['formato', 'tamaño', 'dimensión', 'extensión']
    }
    
    for element, indicators in checklist.items():
        if any(indicator in prompt_lower for indicator in indicators):
            present_elements += 1
    
    specificity_score = (present_elements / len(checklist)) * 100
    
    # Seleccionar las 3 sugerencias más relevantes que no estén presentes
    final_suggestions = []
    for suggestion in project_suggestions:
        if len(final_suggestions) < 3:
            suggestion_keywords = suggestion.lower()
            if not any(keyword in prompt_lower for keyword in ['audiencia', 'objetivo', 'estilo', 'formato'] if keyword in suggestion_keywords):
                final_suggestions.append(suggestion)
    
    return round(specificity_score), final_suggestions

def analyze_structure(prompt: str):
    """
    Analiza la estructura del prompt.
    """
    feedback = []
    
    # Verificar si tiene estructura básica
    has_action = any(verb in prompt.lower() for verb in ['crear', 'diseñar', 'desarrollar', 'hacer', 'generar', 'escribir'])
    has_object = any(noun in prompt.lower() for noun in ['logo', 'web', 'app', 'artículo', 'video', 'campaña', 'aplicación'])
    has_context = any(prep in prompt.lower() for prep in ['para', 'sobre', 'de', 'con'])
    
    structure_elements = sum([has_action, has_object, has_context])
    structure_score = (structure_elements / 3) * 100
    
    if not has_action:
        feedback.append("Incluye una acción clara (crear, diseñar, desarrollar, etc.)")
    if not has_object:
        feedback.append("Especifica qué tipo de elemento necesitas")
    if not has_context:
        feedback.append("Añade contexto sobre el propósito o tema")
    
    if structure_score == 100:
        feedback.append("Estructura bien definida")
    
    return round(structure_score), feedback[:2]  # Máximo 2 comentarios

def create_detailed_quality_report(prompt, word_count, completeness_score, clarity_score, 
                                 specificity_score, structure_score, completeness_issues,
                                 clarity_issues, specificity_suggestions, structure_feedback, project_type):
    """
    Crea un reporte detallado de calidad del prompt.
    """
    # Calcular puntuación general
    overall_score = round((completeness_score + clarity_score + specificity_score + structure_score) / 4)
    
    # Crear reporte
    report = f"📊 Análisis detallado del prompt ({word_count} palabras)\n\n"
    
    # Puntuación general con emoji
    if overall_score >= 80:
        report += f"🏆 Calidad general: {overall_score}% - Excelente\n"
    elif overall_score >= 60:
        report += f"✅ Calidad general: {overall_score}% - Buena\n"
    elif overall_score >= 40:
        report += f"⚠️ Calidad general: {overall_score}% - Mejorable\n"
    else:
        report += f"❌ Calidad general: {overall_score}% - Necesita mejoras\n"
    
    report += f"🎯 Tipo de proyecto detectado: {project_type.replace('_', ' ').title()}\n\n"
    
    # Análisis detallado
    report += "📈 Análisis por categorías:\n"
    report += f"• Completitud: {completeness_score}%\n"
    report += f"• Claridad: {clarity_score}%\n"
    report += f"• Especificidad: {specificity_score}%\n"
    report += f"• Estructura: {structure_score}%\n\n"
    
    # Recomendaciones específicas
    if completeness_issues or clarity_issues or specificity_suggestions or structure_feedback:
        report += "💡 Recomendaciones de mejora:\n"
        
        # Mostrar los problemas más importantes primero
        all_feedback = []
        
        if completeness_issues:
            all_feedback.extend([f"Incluye información sobre: {', '.join(completeness_issues)}"])
        
        if clarity_issues:
            all_feedback.extend(clarity_issues)
            
        if structure_feedback and structure_score < 80:
            all_feedback.extend(structure_feedback)
            
        if specificity_suggestions:
            all_feedback.extend(specificity_suggestions[:2])  # Solo las 2 más importantes
        
        # Mostrar máximo 4 recomendaciones
        for i, feedback in enumerate(all_feedback[:4], 1):
            report += f"{i}. {feedback}\n"
    
    if overall_score >= 80:
        report += "\n🎉 ¡Tu prompt tiene una calidad excelente!"
    elif overall_score >= 60:
        report += "\n👍 Tu prompt tiene buena calidad, con algunas mejoras será perfecto"
    
    return report

def extract_core_concept(prompt: str):
    """
    Extrae el concepto principal del prompt del usuario, eliminando meta-texto.
    """
    # Limpiar el prompt
    clean_prompt = prompt.lower().strip()
    
    # Remover frases meta como "me puedes generar un prompt" o "quiero"
    meta_phrases = [
        r'me puedes?\s+(?:generar|crear|hacer|ayudar|dar|proporcionar)',
        r'puedes?\s+(?:generar|crear|hacer|ayudar|dar|proporcionar)',
        r'genera?r?\s+un\s+prompt',
        r'crear?\s+un\s+prompt',
        r'prompt\s+que\s+me\s+ayude?',
        r'ayúdame\s+a',
        r'que\s+me\s+ayude',
        r'^quiero\s+',  # Nuevo: remover "quiero" al inicio
        r'^necesito\s+',
        r'^me\s+gustaría\s+',
    ]
    
    for pattern in meta_phrases:
        clean_prompt = re.sub(pattern, '', clean_prompt, flags=re.IGNORECASE)
    
    # Patrones más específicos y precisos para extraer el concepto central
    patterns = [
        # "diseñar un cartel de una película futurista"
        r'(?:diseñar|crear|hacer|desarrollar|generar|construir|elaborar|producir)\s+(?:una?|un)?\s*([^?]+?)(?:\?|$)',
        # "cartel de una película futurista" (directo)
        r'^([a-záéíóúñ\s]+(?:de|para|sobre|con)\s+[^?]+?)(?:\?|$)',
        # "sobre inteligencia artificial"
        r'sobre\s+(.+?)(?:\?|$)',
        # "acerca de machine learning"
        r'(?:acerca\s+de|relacionado\s+con)\s+(.+?)(?:\?|$)',
        # Cualquier sustantivo + complemento
        r'^([a-záéíóúñ]+(?:\s+[a-záéíóúñ]+)*(?:\s+de\s+[^?]+)?)(?:\?|$)',
        # Fallback general
        r'^\s*(.+?)(?:\?|$)'
    ]
    
    # Extraer el concepto usando patrones
    for pattern in patterns:
        match = re.search(pattern, clean_prompt.strip(), re.IGNORECASE)
        if match:
            concept = match.group(1).strip()
            
            # Limpiar preposiciones innecesarias al inicio
            concept = re.sub(r'^(a|con|para|de|que|una?|el|la|los|las)\s+', '', concept)
            
            # No cortar en preposiciones que son parte del concepto
            # Solo limpiar palabras sobrantes muy específicas al final
            concept = re.sub(r'\s+(que\s+(?:me\s+)?ayude|para\s+(?:mi|el)).*$', '', concept)
            
            if len(concept.split()) >= 1 and len(concept) > 3:  # Más flexible
                return concept
    
    # Fallback: tomar palabras significativas sin filtrar demasiado
    words = clean_prompt.split()
    significant_words = [w for w in words if len(w) > 2 and w not in ['crear', 'generar', 'hacer', 'ayude', 'puedes', 'prompt', 'ayudar', 'para', 'que', 'una', 'una']]
    
    if significant_words:
        return ' '.join(significant_words[:6])  # Más palabras para preservar contexto
    
    return "proyecto creativo"  # Fallback más genérico

def detect_project_type(concept: str):
    """
    Detecta el tipo de proyecto basado en el concepto para usar templates más específicos.
    """
    concept_lower = concept.lower()
    
    # Patrones para diferentes tipos de proyectos
    project_types = {
        'diseño_grafico': ['cartel', 'póster', 'logo', 'logotipo', 'banner', 'flyer', 'folleto', 'portada', 'diseño gráfico', 'ilustración'],
        'web_desarrollo': ['página web', 'sitio web', 'aplicación web', 'website', 'plataforma online', 'portal web'],
        'aplicacion_movil': ['aplicación móvil', 'app móvil', 'aplicación', 'app', 'móvil'],
        'contenido_escrito': ['artículo', 'blog', 'ensayo', 'libro', 'novela', 'cuento', 'historia', 'texto', 'redacción'],
        'video_multimedia': ['video', 'película', 'documental', 'animación', 'cortometraje', 'trailer'],
        'marketing': ['campaña', 'marketing', 'publicidad', 'anuncio', 'promoción', 'estrategia comercial'],
        'educacion': ['curso', 'tutorial', 'guía', 'manual', 'lección', 'capacitación', 'entrenamiento'],
        'evento': ['evento', 'conferencia', 'seminario', 'taller', 'workshop', 'presentación'],
        'negocio': ['plan de negocio', 'startup', 'empresa', 'emprendimiento', 'proyecto empresarial'],
        'juego': ['juego', 'videojuego', 'game', 'aplicación de juego']
    }
    
    for project_type, keywords in project_types.items():
        if any(keyword in concept_lower for keyword in keywords):
            return project_type
    
    return 'general'  # Tipo genérico para proyectos no clasificados

def generate_adaptive_fallback(concept: str, task: str):
    """
    Genera respuestas de respaldo adaptativas basadas en el tipo de proyecto detectado.
    """
    project_type = detect_project_type(concept)
    
    if task == "improve":
        templates_by_type = {
            'diseño_grafico': [
                f"Diseña un {concept} impactante y profesional, incluyendo composición visual llamativa, tipografía creativa, paleta de colores atractiva y elementos gráficos modernos",
                f"Crea un {concept} con estilo visual único, incorporando tendencias de diseño actuales, jerarquía visual clara y elementos que capten la atención del público objetivo",
                f"Desarrolla un {concept} memorable que combine creatividad y funcionalidad, con alta resolución, formato optimizado para impresión y elementos visuales cohesivos"
            ],
            'web_desarrollo': [
                f"Desarrolla una {concept} completa y profesional, incluyendo diseño moderno, contenido detallado, funcionalidades interactivas y optimización para dispositivos móviles",
                f"Crea una {concept} estructurada con secciones claramente definidas, navegación intuitiva, contenido de calidad y elementos visuales atractivos",
                f"Diseña una {concept} que incluya información detallada, galería de imágenes, testimonios de usuarios y formularios de contacto funcionales"
            ],
            'aplicacion_movil': [
                f"Desarrolla una {concept} innovadora con interfaz intuitiva, experiencia de usuario fluida, funcionalidades útiles y compatibilidad multiplataforma",
                f"Crea una {concept} que incluya diseño responsive, navegación simple, notificaciones push y integración con servicios populares",
                f"Diseña una {concept} con arquitectura escalable, rendimiento optimizado, seguridad robusta y características que resuelvan problemas reales"
            ],
            'contenido_escrito': [
                f"Redacta un {concept} cautivador y bien estructurado, con investigación profunda, estilo narrativo envolvente y contenido original de alta calidad",
                f"Crea un {concept} informativo que incluya introducción impactante, desarrollo lógico de ideas, ejemplos relevantes y conclusiones sólidas",
                f"Desarrolla un {concept} único con enfoque específico, fuentes confiables, lenguaje apropiado para la audiencia y formato atractivo"
            ],
            'video_multimedia': [
                f"Produce un {concept} visualmente impresionante con narrativa convincente, efectos visuales de calidad, audio profesional y edición dinámica",
                f"Crea un {concept} que combine storytelling efectivo, cinematografía atractiva, banda sonora adecuada y ritmo envolvente",
                f"Desarrolla un {concept} memorable con concepto original, personajes interesantes, mensaje claro y producción de alta calidad"
            ],
            'marketing': [
                f"Diseña una {concept} estratégica con investigación de mercado, targeting preciso, mensajes persuasivos y canales de distribución efectivos",
                f"Crea una {concept} innovadora que incluya análisis de competencia, propuesta de valor única, creatividad publicitaria y métricas de éxito",
                f"Desarrolla una {concept} integral con objetivos claros, presupuesto optimizado, cronograma realista y estrategias de engagement"
            ],
            'educacion': [
                f"Desarrolla un {concept} didáctico y completo con objetivos de aprendizaje claros, metodología interactiva, recursos variados y evaluaciones efectivas",
                f"Crea un {concept} estructurado que incluya contenido progresivo, ejercicios prácticos, ejemplos reales y herramientas de seguimiento",
                f"Diseña un {concept} engaging con formato multimedia, actividades participativas, feedback continuo y adaptación a diferentes estilos de aprendizaje"
            ],
            'evento': [
                f"Organiza un {concept} memorable con programa atractivo, speakers relevantes, logística impecable y experiencia participativa única",
                f"Planifica un {concept} exitoso que incluya objetivos claros, audiencia definida, contenido valioso y networking efectivo",
                f"Diseña un {concept} impactante con formato innovador, tecnología integrada, engagement del público y seguimiento post-evento"
            ],
            'negocio': [
                f"Desarrolla un {concept} sólido con análisis de mercado, modelo de negocio viable, estrategia financiera y plan de crecimiento escalable",
                f"Crea un {concept} innovador que incluya propuesta de valor diferenciada, análisis de competencia, equipo competente y proyecciones realistas",
                f"Diseña un {concept} estratégico con validación de mercado, recursos necesarios, cronograma de implementación y métricas de éxito"
            ],
            'juego': [
                f"Desarrolla un {concept} adictivo con mecánicas innovadoras, historia envolvente, gráficos atractivos y experiencia de usuario excepcional",
                f"Crea un {concept} divertido que incluya gameplay balanceado, progresión satisfactoria, elementos sociales y rejugabilidad alta",
                f"Diseña un {concept} memorable con concepto original, controles intuitivos, desafíos graduales y sistema de recompensas motivador"
            ],
            'general': [
                f"Desarrolla un {concept} excepcional y profesional, incorporando las mejores prácticas de la industria, innovación creativa y atención al detalle",
                f"Crea un {concept} único y de alta calidad que se destaque por su originalidad, funcionalidad y impacto en la audiencia objetivo",
                f"Diseña un {concept} completo que combine creatividad, técnica profesional y enfoque estratégico para obtener resultados sobresalientes"
            ]
        }
        return templates_by_type.get(project_type, templates_by_type['general'])
    
    elif task == "feedback":
        feedback_by_type = {
            'diseño_grafico': [
                f"Define el estilo visual y la estética deseada para el {concept}",
                f"Especifica las dimensiones, formato y uso final del diseño",
                f"Incluye referências visuales o inspiraciones estilísticas",
                f"Considera el público objetivo y el mensaje que quieres transmitir"
            ],
            'web_desarrollo': [
                f"Especifica el público objetivo para la {concept}",
                f"Define las funcionalidades principales que necesitas",
                f"Incluye el tipo de contenido y estructura deseada",
                f"Considera la experiencia de usuario y accesibilidad"
            ],
            'aplicacion_movil': [
                f"Define las funcionalidades principales de la {concept}",
                f"Especifica las plataformas objetivo (iOS, Android, etc.)",
                f"Incluye el tipo de usuarios y sus necesidades",
                f"Considera la monetización y modelo de negocio"
            ],
            'contenido_escrito': [
                f"Especifica el público objetivo y tono deseado",
                f"Define la extensión y formato del {concept}",
                f"Incluye los temas principales a cubrir",
                f"Considera el propósito y objetivos del contenido"
            ],
            'video_multimedia': [
                f"Define el estilo visual y duración del {concept}",
                f"Especifica el público objetivo y plataforma de distribución",
                f"Incluye el mensaje principal y tono narrativo",
                f"Considera el presupuesto y recursos disponibles"
            ],
            'marketing': [
                f"Define el público objetivo y segmentación",
                f"Especifica los objetivos y KPIs de la {concept}",
                f"Incluye el presupuesto y canales preferidos",
                f"Considera el timing y duración de la campaña"
            ],
            'educacion': [
                f"Define el público objetivo y nivel de conocimiento",
                f"Especifica los objetivos de aprendizaje del {concept}",
                f"Incluye la metodología y formato preferido",
                f"Considera la duración y recursos necesarios"
            ],
            'evento': [
                f"Define el público objetivo y número de asistentes",
                f"Especifica los objetivos y tipo de {concept}",
                f"Incluye el presupuesto y ubicación preferida",
                f"Considera la fecha, duración y logística necesaria"
            ],
            'negocio': [
                f"Define el mercado objetivo y propuesta de valor",
                f"Especifica el modelo de negocio del {concept}",
                f"Incluye el análisis de competencia y diferenciación",
                f"Considera los recursos y capital necesario"
            ],
            'juego': [
                f"Define el género y plataforma objetivo del {concept}",
                f"Especifica las mecánicas principales de gameplay",
                f"Incluye el público objetivo y rating",
                f"Considera el estilo visual y temática del juego"
            ],
            'general': [
                f"Define los objetivos principales del {concept}",
                f"Especifica el público objetivo y sus necesidades",
                f"Incluye los recursos y limitaciones disponibles",
                f"Considera el contexto y propósito del proyecto"
            ]
        }
        return feedback_by_type.get(project_type, feedback_by_type['general'])
    
    elif task == "ideas":
        ideas_by_type = {
            'diseño_grafico': [
                f"Crear una serie de variaciones estilísticas del {concept}",
                f"Desarrollar una guía de estilo y elementos gráficos complementarios",
                f"Diseñar adaptaciones del concepto para diferentes formatos y medios",
                f"Producir un mockup realista mostrando el {concept} en contexto",
                f"Elaborar un proceso creativo documentado del desarrollo del diseño"
            ],
            'web_desarrollo': [
                f"Crear un prototipo interactivo de la {concept}",
                f"Desarrollar una estrategia de contenido y SEO",
                f"Diseñar un sistema de analíticas y métricas",
                f"Implementar funcionalidades de accesibilidad avanzadas",
                f"Crear una versión móvil optimizada"
            ],
            'aplicacion_movil': [
                f"Desarrollar un MVP (Producto Mínimo Viable) de la {concept}",
                f"Crear wireframes y prototipos de la interfaz",
                f"Diseñar un plan de testing con usuarios reales",
                f"Implementar un sistema de analytics y feedback",
                f"Elaborar una estrategia de lanzamiento en app stores"
            ],
            'contenido_escrito': [
                f"Crear un plan editorial completo para el {concept}",
                f"Desarrollar una serie de contenidos relacionados",
                f"Diseñar una estrategia de distribución y promoción",
                f"Implementar SEO y optimización para buscadores",
                f"Crear formatos multimedia complementarios"
            ],
            'video_multimedia': [
                f"Desarrollar un storyboard detallado del {concept}",
                f"Crear un plan de producción y cronograma",
                f"Diseñar una estrategia de distribución multiplataforma",
                f"Implementar elementos interactivos o de realidad aumentada",
                f"Producir contenido adicional (behind the scenes, extras)"
            ],
            'marketing': [
                f"Desarrollar una estrategia de marketing digital integral",
                f"Crear contenido viral y campañas en redes sociales",
                f"Diseñar un sistema de métricas y ROI",
                f"Implementar marketing automation y segmentación",
                f"Elaborar partnerships y colaboraciones estratégicas"
            ],
            'educacion': [
                f"Crear materiales de apoyo interactivos para el {concept}",
                f"Desarrollar evaluaciones y sistemas de certificación",
                f"Diseñar una comunidad de aprendizaje online",
                f"Implementar gamificación y elementos motivacionales",
                f"Elaborar programas de mentoría y seguimiento"
            ],
            'evento': [
                f"Crear una experiencia digital complementaria al {concept}",
                f"Desarrollar un programa de networking estructurado",
                f"Diseñar actividades interactivas y workshops",
                f"Implementar tecnología para engagement en tiempo real",
                f"Elaborar un plan de seguimiento post-evento"
            ],
            'negocio': [
                f"Desarrollar un plan de validación de mercado",
                f"Crear prototipos y productos mínimos viables",
                f"Diseñar una estrategia de financiación y investment",
                f"Implementar sistemas de control y métricas",
                f"Elaborar un plan de escalabilidad y crecimiento"
            ],
            'juego': [
                f"Crear un documento de diseño de juego completo",
                f"Desarrollar un prototipo jugable y mecánicas básicas",
                f"Diseñar la progresión del jugador y sistema de recompensas",
                f"Implementar elementos multijugador o sociales",
                f"Elaborar una estrategia de monetización ética"
            ],
            'general': [
                f"Crear una guía completa de mejores prácticas para {concept}",
                f"Desarrollar un tutorial paso a paso para implementar {concept}",
                f"Diseñar una estrategia de contenido y comunicación",
                f"Implementar herramientas de medición y análisis",
                f"Elaborar un plan de mejora continua y evolución"
            ]
        }
        return ideas_by_type.get(project_type, ideas_by_type['general'])[:5]  # Límite de 5 ideas
    
    else:
        return [f"Mejora la descripción de {concept} con más detalles específicos"]

def generate_smart_fallback(concept: str, task: str):
    """
    Genera respuestas de respaldo inteligentes basadas en el concepto extraído.
    DEPRECATED: Usar generate_adaptive_fallback en su lugar.
    """
    return generate_adaptive_fallback(concept, task)

def is_coherent_spanish(text: str, min_words: int = 3):
    """
    Verifica si el texto es coherente y está en español.
    """
    if not text or len(text.strip()) < 10:
        return False
    
    words = text.split()
    if len(words) < min_words:
        return False
    
    # Verificar que no esté en inglés
    english_indicators = ['the', 'and', 'or', 'but', 'with', 'from', 'they', 'this', 'that', 'have', 'been', 'will', 'would', 'could', 'should']
    english_count = sum(1 for word in words if word.lower() in english_indicators)
    if english_count > len(words) * 0.2:  # Más del 20% en inglés
        return False
    
    # Verificar que no sea repetitivo
    unique_words = set(words)
    if len(unique_words) < len(words) * 0.6:  # Menos del 60% de palabras únicas
        return False
    
    # Verificar que tenga estructura básica de oración
    if not any(char in text for char in '.,!?'):
        # Si no tiene puntuación, verificar que al menos parezca una oración
        if not re.search(r'\b(el|la|un|una|de|para|con|en|que|como|sobre)\b', text.lower()):
            return False
    
    return True

def get_model_specific_prompt(base_prompt: str, model_name: str, task: str):
    """
    Genera prompts optimizados para cada modelo según la tarea.
    """
    # Extraer el concepto principal
    concept = extract_core_concept(base_prompt)
    
    if "t5" in model_name.lower():
        # T5 funciona mejor con tareas estructuradas en inglés pero podemos forzar español
        if task == "improve":
            return f"paraphrase in Spanish with more details: Create a detailed {concept}"
        elif task == "feedback":
            return f"analyze in Spanish: What can be improved about {concept}"
        elif task == "ideas":
            return f"generate ideas in Spanish for: {concept}"
    
    elif "gpt2" in model_name.lower() or "distilgpt2" in model_name.lower():
        # GPT-2 con ejemplos muy específicos en español
        if task == "improve":
            return f"""Ejemplo 1:
Concepto: página sobre perros
Versión mejorada: Desarrolla una página web completa sobre razas caninas, incluyendo fichas detalladas de cada raza, consejos de cuidado, galería fotográfica y directorio de veterinarios especializados

Ejemplo 2:
Concepto: tienda online
Versión mejorada: Crea una tienda online moderna con catálogo de productos, carrito de compras, sistema de pagos seguro, reseñas de clientes y soporte en tiempo real

Concepto: {concept}
Versión mejorada:"""
        
        elif task == "feedback":
            return f"""Para mejorar '{concept}', considera estos aspectos:
1. Público objetivo específico
2."""
        
        elif task == "ideas":
            return f"""Ideas para desarrollar '{concept}':
1. Tutorial interactivo paso a paso
2."""
    
    elif "neo" in model_name.lower():
        # GPT-Neo con instrucciones claras en español
        if task == "improve":
            return f"""Instrucción: Reescribe la siguiente idea de manera más detallada y profesional.

Idea original: {concept}

Versión mejorada y detallada:"""
        
        elif task == "feedback":
            return f"""Analiza qué se puede mejorar en esta idea: '{concept}'

Sugerencias de mejora:
-"""
        
        elif task == "ideas":
            return f"""Genera 3 ideas creativas relacionadas con: '{concept}'

Ideas:
1."""
    
    # Default con concepto extraído
    return f"Mejora esta idea: {concept}"

def clean_generated_output(text: str, model_name: str, task: str, original_concept: str):
    """
    Limpia y valida la salida generada según el modelo y la tarea.
    """
    if not text:
        return ""
    
    # Eliminar líneas vacías y espacios extra
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    # Verificar coherencia en español
    full_text = " ".join(lines)
    
    if not is_coherent_spanish(full_text):
        # Si no es coherente, usar fallback adaptativo
        fallbacks = generate_adaptive_fallback(original_concept, task)
        if task == "feedback":
            return "\n".join([f"- {fb}" for fb in fallbacks])
        return fallbacks[0] if fallbacks else f"Desarrolla {original_concept} con más detalles específicos"
    
    # Limpieza específica por modelo y tarea
    if "gpt2" in model_name.lower() or "distilgpt2" in model_name.lower():
        if task == "improve":
            # Buscar la línea con la mejora
            for line in lines:
                if (len(line.split()) > 8 and 
                    not line.lower().startswith(('ejemplo', 'concepto', 'versión', 'idea')) and
                    any(verb in line.lower() for verb in ['desarrolla', 'crea', 'diseña', 'incluye', 'contiene'])):
                    return line
        
        elif task == "feedback":
            feedback_lines = []
            for line in lines:
                if (line and len(line.split()) > 3 and
                    not line.lower().startswith(('para', 'considera', 'aspectos'))):
                    if not line.startswith('-'):
                        line = f"- {line}"
                    feedback_lines.append(line)
            return "\n".join(feedback_lines[:4]) if feedback_lines else "\n".join([f"- {fb}" for fb in generate_adaptive_fallback(original_concept, "feedback")])
        
        elif task == "ideas":
            ideas = []
            for line in lines:
                if (line and len(line.split()) > 4 and
                    not line.lower().startswith(('ideas', 'para', 'desarrollar'))):
                    clean_line = re.sub(r'^\d+\.?\s*', '', line)
                    ideas.append(clean_line)
            return ideas[:3] if ideas else generate_adaptive_fallback(original_concept, "ideas")[:3]
    
    elif "t5" in model_name.lower():
        # T5 devuelve respuestas más directas
        if task == "improve":
            result = " ".join(lines)
            if len(result.split()) < 8:
                return generate_adaptive_fallback(original_concept, "improve")[0]
            return result
        elif task in ["feedback", "ideas"]:
            return " ".join(lines) if lines else generate_adaptive_fallback(original_concept, task)[0]
    
    elif "neo" in model_name.lower():
        if task == "improve":
            # Buscar líneas que parezcan mejoras
            for line in lines:
                if (len(line.split()) > 10 and
                    any(verb in line.lower() for verb in ['desarrolla', 'crea', 'diseña', 'implementa', 'incluye'])):
                    return line
        
        elif task == "feedback":
            feedback = []
            for line in lines:
                if (line and len(line.split()) > 3 and
                    not any(skip in line.lower() for skip in ['analiza', 'sugerencias', 'mejora:'])):
                    if not line.startswith('-'):
                        line = f"- {line}"
                    feedback.append(line)
            return "\n".join(feedback[:4]) if feedback else "\n".join([f"- {fb}" for fb in generate_adaptive_fallback(original_concept, "feedback")])
        
        elif task == "ideas":
            ideas = []
            for line in lines:
                if (line and len(line.split()) > 4 and
                    not any(skip in line.lower() for skip in ['genera', 'ideas:', 'relacionadas'])):
                    clean_line = re.sub(r'^\d+\.?\s*', '', line)
                    ideas.append(clean_line)
            return ideas[:3] if ideas else generate_adaptive_fallback(original_concept, "ideas")[:3]
    
    # Fallback general
    if task == "improve":
        return generate_adaptive_fallback(original_concept, "improve")[0]
    elif task == "feedback":
        return "\n".join([f"- {fb}" for fb in generate_adaptive_fallback(original_concept, "feedback")])
    elif task == "ideas":
        return generate_adaptive_fallback(original_concept, "ideas")[:3]
    
    return full_text

def get_structural_feedback(prompt: str, model_name: str = "gpt2"):
    """
    Genera feedback sobre la estructura y claridad de un prompt usando un modelo local.
    """
    concept = extract_core_concept(prompt)
    optimized_prompt = get_model_specific_prompt(prompt, model_name, "feedback")
    
    feedback = generate_text_dispatcher(model_name, optimized_prompt, max_length=120)
    if isinstance(feedback, dict) and 'error' in feedback:
        # Usar fallback inteligente
        fallback_feedback = generate_adaptive_fallback(concept, "feedback")
        return {"feedback": "\n".join([f"- {fb}" for fb in fallback_feedback])}
    
    # Limpiar y estructurar el feedback
    cleaned_feedback = clean_generated_output(feedback, model_name, "feedback", concept)
    
    # Validar que el feedback sea útil
    if not cleaned_feedback or len(cleaned_feedback.split()) < 10:
        fallback_feedback = generate_adaptive_fallback(concept, "feedback")
        cleaned_feedback = "\n".join([f"- {fb}" for fb in fallback_feedback])
    
    return {"feedback": cleaned_feedback}

def generate_variations(prompt: str, model_name: str = "gpt2", num_variations: int = 3):
    """
    Genera variaciones de un prompt usando un modelo local.
    """
    concept = extract_core_concept(prompt)
    variations = []
    
    if "gpt2" in model_name.lower() or "distilgpt2" in model_name.lower():
        # Para GPT-2, intentar generar una vez con few-shot
        optimized_prompt = get_model_specific_prompt(prompt, model_name, "improve")
        
        response_text = generate_text_dispatcher(model_name, optimized_prompt, max_length=100)
        
        if (not isinstance(response_text, dict) and 
            response_text and 
            is_coherent_spanish(response_text)):
            
            cleaned = clean_generated_output(response_text, model_name, "improve", concept)
            if cleaned and len(cleaned.split()) > 8:
                variations.append(cleaned)
        
        # Completar con fallbacks inteligentes
        fallbacks = generate_adaptive_fallback(concept, "improve")
        while len(variations) < num_variations:
            idx = len(variations)
            if idx < len(fallbacks):
                variations.append(fallbacks[idx])
            else:
                variations.append(f"Desarrolla {concept} de manera {['profesional', 'creativa', 'detallada'][idx % 3]}")
    
    elif "t5" in model_name.lower():
        # T5 con diferentes enfoques
        approaches = ["detailed", "professional", "creative"]
        for i in range(num_variations):
            task_prompt = f"paraphrase in Spanish with {approaches[i % len(approaches)]} style: Create {concept}"
            response_text = generate_text_dispatcher(model_name, task_prompt, max_length=80)
            
            if (not isinstance(response_text, dict) and 
                response_text and 
                is_coherent_spanish(response_text)):
                
                cleaned = clean_generated_output(response_text, model_name, "improve", concept)
                variations.append(cleaned if cleaned else generate_adaptive_fallback(concept, "improve")[0])
            else:
                variations.append(generate_adaptive_fallback(concept, "improve")[i % 3])
    
    else:
        # GPT-Neo y otros
        optimized_prompt = get_model_specific_prompt(prompt, model_name, "improve")
        response_text = generate_text_dispatcher(model_name, optimized_prompt, max_length=120)
        
        if (not isinstance(response_text, dict) and 
            response_text and 
            is_coherent_spanish(response_text)):
            
            cleaned = clean_generated_output(response_text, model_name, "improve", concept)
            if cleaned:
                variations.append(cleaned)
        
        # Completar con fallbacks
        fallbacks = generate_adaptive_fallback(concept, "improve")
        while len(variations) < num_variations:
            idx = len(variations)
            variations.append(fallbacks[idx % len(fallbacks)])
    
    # Asegurar que todas las variaciones sean únicas y válidas
    unique_variations = []
    seen = set()
    for var in variations:
        if var and var not in seen and len(var.split()) > 5:
            seen.add(var)
            unique_variations.append(var)
    
    # Completar si es necesario
    fallbacks = generate_adaptive_fallback(concept, "improve")
    while len(unique_variations) < num_variations:
        idx = len(unique_variations)
        fallback = fallbacks[idx % len(fallbacks)]
        if fallback not in seen:
            unique_variations.append(fallback)
            seen.add(fallback)
    
    return {"variations": unique_variations[:num_variations]}

def generate_ideas(prompt: str, model_name: str = "gpt2", num_ideas: int = 3):
    """
    Genera ideas basadas en un prompt usando un modelo local.
    """
    concept = extract_core_concept(prompt)
    ideas = []
    
    if "gpt2" in model_name.lower() or "distilgpt2" in model_name.lower():
        # Intentar con el modelo
        optimized_prompt = get_model_specific_prompt(prompt, model_name, "ideas")
        response_text = generate_text_dispatcher(model_name, optimized_prompt, max_length=150)
        
        if (not isinstance(response_text, dict) and 
            response_text and 
            is_coherent_spanish(response_text)):
            
            extracted_ideas = clean_generated_output(response_text, model_name, "ideas", concept)
            if isinstance(extracted_ideas, list):
                ideas.extend(extracted_ideas[:num_ideas])
        
        # Completar con templates inteligentes
        smart_ideas = generate_adaptive_fallback(concept, "ideas")
        while len(ideas) < num_ideas:
            idx = len(ideas)
            ideas.append(smart_ideas[idx % len(smart_ideas)])
    
    elif "t5" in model_name.lower():
        # T5 con diferentes enfoques
        approaches = ["tutorial", "guide", "tool", "resource", "strategy"]
        for i in range(num_ideas):
            approach = approaches[i % len(approaches)]
            task_prompt = f"generate idea in Spanish: {approach} for {concept}"
            response_text = generate_text_dispatcher(model_name, task_prompt, max_length=60)
            
            if (not isinstance(response_text, dict) and 
                response_text and 
                is_coherent_spanish(response_text)):
                
                cleaned = response_text.strip()
                ideas.append(f"Crear {cleaned}" if not cleaned.lower().startswith('crear') else cleaned)
            else:
                smart_ideas = generate_adaptive_fallback(concept, "ideas")
                ideas.append(smart_ideas[i % len(smart_ideas)])
    
    else:
        # Otros modelos
        optimized_prompt = get_model_specific_prompt(prompt, model_name, "ideas")
        response_text = generate_text_dispatcher(model_name, optimized_prompt, max_length=150)
        
        if (not isinstance(response_text, dict) and 
            response_text and 
            is_coherent_spanish(response_text)):
            
            extracted_ideas = clean_generated_output(response_text, model_name, "ideas", concept)
            if isinstance(extracted_ideas, list):
                ideas.extend(extracted_ideas[:num_ideas])
        
        # Completar con templates
        smart_ideas = generate_smart_fallback(concept, "ideas")
        while len(ideas) < num_ideas:
            idx = len(ideas)
            ideas.append(smart_ideas[idx % len(smart_ideas)])
    
    # Formatear y limpiar ideas finales
    formatted_ideas = []
    for i, idea in enumerate(ideas[:num_ideas]):
        # Limpiar numeración y formatear
        clean_idea = re.sub(r'^\d+\.?\s*[-\s]*', '', str(idea)).strip()
        if clean_idea and len(clean_idea.split()) > 2:
            formatted_ideas.append(clean_idea)
    
    # Asegurar que tengamos suficientes ideas
    smart_fallbacks = generate_smart_fallback(concept, "ideas")
    while len(formatted_ideas) < num_ideas:
        idx = len(formatted_ideas)
        formatted_ideas.append(smart_fallbacks[idx % len(smart_fallbacks)])
    
    return {"ideas": formatted_ideas[:num_ideas]}

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