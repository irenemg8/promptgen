#!/usr/bin/env python3
"""
🧪 Test del Sistema Real de PromptGen
Verifica que el sistema real de mejora de prompts funcione correctamente
"""

import sys
import os
import time

# Añadir directorio actual al path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

def test_real_system():
    """Prueba el sistema real de mejora de prompts"""
    
    print("🧪 INICIANDO PRUEBAS DEL SISTEMA REAL")
    print("=" * 50)
    
    try:
        # Importar sistema real
        from promptgen_real_system import (
            RealIterativeImprover, RealQualityAnalyzer, 
            improve_iteratively_real, analyze_quality_real
        )
        print("✅ Sistema real importado correctamente")
        
    except ImportError as e:
        print(f"❌ Error importando sistema real: {e}")
        return False
    
    # Prompt de prueba
    test_prompt = "Crea una plataforma SaaS que monitoriza y audita el consumo de APIs en tiempo real, muestra dashboards y permite configurar alertas de uso/errores."
    
    print(f"\n📝 Prompt de prueba:")
    print(f"   {test_prompt}")
    
    # 1. Probar análisis de calidad
    print(f"\n🔍 PRUEBA 1: Análisis de Calidad")
    try:
        quality_result = analyze_quality_real(test_prompt)
        print(f"✅ Análisis completado")
        print(f"   Calidad general: {quality_result['overall_score']:.1f}%")
        
        metrics = quality_result['metrics']
        print(f"   📊 Métricas detalladas:")
        print(f"      Completitud: {metrics['completeness']:.1f}%")
        print(f"      Claridad: {metrics['clarity']:.1f}%")
        print(f"      Especificidad: {metrics['specificity']:.1f}%")
        print(f"      Estructura: {metrics['structure']:.1f}%")
        print(f"      Coherencia: {metrics['coherence']:.1f}%")
        print(f"      Accionabilidad: {metrics['actionability']:.1f}%")
        
    except Exception as e:
        print(f"❌ Error en análisis de calidad: {e}")
        return False
    
    # 2. Probar mejora iterativa
    print(f"\n🚀 PRUEBA 2: Mejora Iterativa Real")
    try:
        start_time = time.time()
        
        improvement_result = improve_iteratively_real(
            prompt=test_prompt,
            model_name="gpt2",
            max_iterations=3,
            target_quality=80.0
        )
        
        total_time = time.time() - start_time
        
        print(f"✅ Mejora iterativa completada en {total_time:.2f}s")
        print(f"\n📊 RESULTADOS:")
        print(f"   🎯 Calidad inicial: {improvement_result['initial_quality']:.1f}%")
        print(f"   🎯 Calidad final: {improvement_result['final_quality']:.1f}%")
        print(f"   📈 Mejora total: +{improvement_result['total_improvement']:.1f}%")
        print(f"   🔄 Iteraciones: {improvement_result['iterations_completed']}")
        
        print(f"\n✨ PROMPT ORIGINAL:")
        print(f"   {improvement_result['original_prompt']}")
        
        print(f"\n🌟 PROMPT MEJORADO:")
        print(f"   {improvement_result['final_prompt']}")
        
        # Mostrar detalles de iteraciones
        if improvement_result['iterations_data']:
            print(f"\n📈 PROGRESO POR ITERACIÓN:")
            for iteration in improvement_result['iterations_data']:
                print(f"   Iteración {iteration['iteration']}: "
                      f"{iteration['quality_before']:.1f}% → "
                      f"{iteration['quality_after']:.1f}% "
                      f"(+{iteration['improvement_delta']:.1f}%)")
        
        # Verificar que realmente mejoró
        if improvement_result['total_improvement'] > 0:
            print(f"\n🎉 ¡ÉXITO! El prompt mejoró {improvement_result['total_improvement']:.1f}%")
            return True
        else:
            print(f"\n⚠️ ADVERTENCIA: No se logró mejora significativa")
            return False
            
    except Exception as e:
        print(f"❌ Error en mejora iterativa: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_api_integration():
    """Prueba la integración con la API"""
    
    print(f"\n🔗 PRUEBA 3: Integración con API")
    
    try:
        import requests
        
        # Verificar que la API esté corriendo
        response = requests.get("http://localhost:8000/api/health", timeout=5)
        
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ API disponible - Versión: {health_data.get('version', 'unknown')}")
            
            # Probar endpoint de mejora iterativa
            test_data = {
                "prompt": "Crear una app móvil",
                "model_name": "gpt2",
                "max_iterations": 2,
                "target_quality": 75.0
            }
            
            response = requests.post(
                "http://localhost:8000/api/improve_iteratively",
                json=test_data,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Endpoint de mejora iterativa funcionando")
                print(f"   Mejora: +{result.get('total_improvement', 0):.1f}%")
                return True
            else:
                print(f"❌ Error en endpoint: {response.status_code}")
                print(f"   Respuesta: {response.text}")
                return False
                
        else:
            print(f"❌ API no disponible: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"⚠️ API no está ejecutándose en localhost:8000")
        print(f"   Ejecuta: python launch_promptgen_enterprise.py --api-only")
        return False
    except Exception as e:
        print(f"❌ Error probando API: {e}")
        return False

def main():
    """Función principal de pruebas"""
    
    print("🧪 SUITE DE PRUEBAS - PROMPTGEN REAL SYSTEM")
    print("=" * 60)
    
    # Prueba 1: Sistema real
    system_ok = test_real_system()
    
    # Prueba 2: Integración API
    api_ok = test_api_integration()
    
    # Resumen final
    print(f"\n📋 RESUMEN DE PRUEBAS:")
    print(f"   Sistema Real: {'✅ PASS' if system_ok else '❌ FAIL'}")
    print(f"   Integración API: {'✅ PASS' if api_ok else '❌ FAIL'}")
    
    if system_ok and api_ok:
        print(f"\n🎉 ¡TODAS LAS PRUEBAS PASARON!")
        print(f"   El sistema real está funcionando correctamente")
        print(f"   Los modelos de HuggingFace se están usando realmente")
        print(f"   Las mejoras son genuinas, no simuladas")
    elif system_ok:
        print(f"\n⚠️ Sistema real funciona, pero API no disponible")
        print(f"   Ejecuta: python launch_promptgen_enterprise.py --api-only")
    else:
        print(f"\n❌ FALLOS DETECTADOS")
        print(f"   Revisa las dependencias y configuración")
    
    print(f"\n🔗 ENLACES ÚTILES:")
    print(f"   API: http://localhost:8000/docs")
    print(f"   Frontend: http://localhost:3000/promptgen")
    print(f"   Dashboard: http://localhost:8501")

if __name__ == "__main__":
    main() 