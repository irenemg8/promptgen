# setup_ollama.py - Script para instalar y configurar Ollama
import subprocess
import sys
import platform
import os
import time
import requests
import json
from pathlib import Path

def run_command(command, shell=True):
    """Ejecutar comando del sistema"""
    try:
        result = subprocess.run(command, shell=shell, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error ejecutando comando: {command}")
        print(f"Error: {e.stderr}")
        return None

def check_ollama_installed():
    """Verificar si Ollama está instalado"""
    try:
        result = subprocess.run(['ollama', '--version'], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False

def install_ollama():
    """Instalar Ollama según el sistema operativo"""
    system = platform.system().lower()
    
    print("🚀 Instalando Ollama...")
    
    if system == "linux":
        print("📦 Instalando Ollama en Linux...")
        command = "curl -fsSL https://ollama.com/install.sh | sh"
    elif system == "darwin":  # macOS
        print("📦 Instalando Ollama en macOS...")
        command = "curl -fsSL https://ollama.com/install.sh | sh"
    elif system == "windows":
        print("📦 Para Windows, descarga Ollama desde: https://ollama.com/download")
        print("🔗 Ejecuta el instalador y luego ejecuta este script nuevamente.")
        return False
    else:
        print(f"❌ Sistema operativo no soportado: {system}")
        return False
    
    result = run_command(command)
    if result is None:
        return False
    
    print("✅ Ollama instalado exitosamente")
    return True

def start_ollama_service():
    """Iniciar el servicio de Ollama"""
    print("🔄 Iniciando servicio Ollama...")
    
    system = platform.system().lower()
    
    if system == "linux":
        # Intentar iniciar con systemd
        run_command("sudo systemctl start ollama")
        run_command("sudo systemctl enable ollama")
    elif system == "darwin":
        # En macOS, ollama se inicia automáticamente
        pass
    elif system == "windows":
        # En Windows, el servicio se inicia automáticamente
        pass
    
    # Verificar si el servicio está ejecutándose
    time.sleep(2)
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Servicio Ollama iniciado correctamente")
            return True
    except requests.RequestException:
        pass
    
    print("❌ Error iniciando el servicio Ollama")
    return False

def download_models():
    """Descargar modelos necesarios"""
    models_to_download = [
        "llama3.2:3b",           # Modelo para chat
        "mxbai-embed-large",     # Modelo para embeddings
    ]
    
    print(f"📥 Descargando {len(models_to_download)} modelos...")
    
    for model in models_to_download:
        print(f"🔄 Descargando {model}...")
        
        # Descargar modelo
        result = run_command(f"ollama pull {model}")
        if result is None:
            print(f"❌ Error descargando {model}")
            return False
        
        print(f"✅ {model} descargado exitosamente")
    
    return True

def verify_models():
    """Verificar que los modelos están disponibles"""
    print("🔍 Verificando modelos instalados...")
    
    result = run_command("ollama list")
    if result is None:
        return False
    
    print("📋 Modelos instalados:")
    print(result)
    
    required_models = ["llama3.2:3b", "mxbai-embed-large"]
    available_models = result.lower()
    
    for model in required_models:
        if model.lower() in available_models:
            print(f"✅ {model} disponible")
        else:
            print(f"❌ {model} no encontrado")
            return False
    
    return True

def test_ollama_connection():
    """Probar conexión con Ollama"""
    print("🧪 Probando conexión con Ollama...")
    
    try:
        # Probar API de modelos
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        if response.status_code == 200:
            models = response.json()
            print(f"✅ Conexión exitosa. Modelos disponibles: {len(models.get('models', []))}")
            return True
        else:
            print(f"❌ Error en conexión: {response.status_code}")
            return False
    except requests.RequestException as e:
        print(f"❌ Error de conexión: {e}")
        return False

def create_ollama_config():
    """Crear archivo de configuración para Ollama"""
    config_content = """# Configuración de Ollama para PromptGen

# Modelos requeridos:
# - llama3.2:3b (modelo principal para chat)
# - mxbai-embed-large (modelo para embeddings)

# Comandos útiles:
# ollama list                    # Listar modelos instalados
# ollama pull <modelo>           # Descargar modelo
# ollama run <modelo>            # Ejecutar modelo
# ollama ps                      # Ver modelos en ejecución
# ollama serve                   # Iniciar servidor

# URL del servidor: http://localhost:11434
"""
    
    with open("ollama_config.md", "w", encoding="utf-8") as f:
        f.write(config_content)
    
    print("✅ Archivo de configuración creado: ollama_config.md")

def main():
    """Función principal del script"""
    print("🤖 PromptGen - Configurador de Ollama")
    print("=" * 50)
    
    # Verificar si Ollama ya está instalado
    if check_ollama_installed():
        print("✅ Ollama ya está instalado")
    else:
        print("📦 Ollama no está instalado. Instalando...")
        if not install_ollama():
            print("❌ Error durante la instalación")
            return False
    
    # Verificar/iniciar servicio
    if not start_ollama_service():
        print("🔄 Intentando iniciar Ollama manualmente...")
        print("💡 Ejecuta 'ollama serve' en otra terminal y luego ejecuta este script nuevamente")
        return False
    
    # Descargar modelos
    if not download_models():
        print("❌ Error descargando modelos")
        return False
    
    # Verificar modelos
    if not verify_models():
        print("❌ Error verificando modelos")
        return False
    
    # Probar conexión
    if not test_ollama_connection():
        print("❌ Error en conexión con Ollama")
        return False
    
    # Crear configuración
    create_ollama_config()
    
    print("\n🎉 ¡Configuración completada exitosamente!")
    print("=" * 50)
    print("✅ Ollama instalado y configurado")
    print("✅ Modelos descargados")
    print("✅ Servicio ejecutándose")
    print("\n🚀 Ahora puedes ejecutar el sistema PromptGen:")
    print("   1. python api_server.py")
    print("   2. npm run dev")
    print("   3. Visita http://localhost:3000/chat")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n❌ Instalación cancelada por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        sys.exit(1) 