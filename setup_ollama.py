#!/usr/bin/env python3
"""
Script para instalar y configurar Ollama automáticamente
Funciona en Windows, macOS y Linux
"""

import os
import sys
import subprocess
import platform
import requests
import time
from pathlib import Path

def print_colored(text, color='white'):
    """Imprime texto con colores"""
    colors = {
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'magenta': '\033[95m',
        'cyan': '\033[96m',
        'white': '\033[97m',
        'reset': '\033[0m'
    }
    print(f"{colors.get(color, '')}{text}{colors['reset']}")

def check_internet():
    """Verifica conexión a internet"""
    try:
        requests.get("https://www.google.com", timeout=5)
        return True
    except:
        return False

def is_ollama_installed():
    """Verifica si Ollama está instalado"""
    try:
        subprocess.run(['ollama', '--version'], capture_output=True, check=True)
        return True
    except:
        return False

def is_ollama_running():
    """Verifica si Ollama está ejecutándose"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False

def install_ollama_windows():
    """Instala Ollama en Windows"""
    print_colored("🔧 Instalando Ollama en Windows...", 'blue')
    
    # Descargar instalador
    url = "https://ollama.com/download/OllamaSetup.exe"
    installer_path = Path("OllamaSetup.exe")
    
    try:
        print_colored("📥 Descargando instalador...", 'yellow')
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(installer_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print_colored("🚀 Ejecutando instalador...", 'blue')
        print_colored("⚠️  IMPORTANTE: Acepta todas las opciones por defecto en el instalador", 'yellow')
        
        # Ejecutar instalador
        subprocess.run([str(installer_path)], check=True)
        
        # Limpiar
        if installer_path.exists():
            installer_path.unlink()
            
        print_colored("✅ Ollama instalado exitosamente", 'green')
        return True
        
    except Exception as e:
        print_colored(f"❌ Error instalando Ollama: {e}", 'red')
        return False

def install_ollama_macos():
    """Instala Ollama en macOS"""
    print_colored("🔧 Instalando Ollama en macOS...", 'blue')
    
    try:
        # Verificar si Homebrew está instalado
        subprocess.run(['brew', '--version'], capture_output=True, check=True)
        
        # Instalar con Homebrew
        print_colored("📦 Instalando con Homebrew...", 'yellow')
        subprocess.run(['brew', 'install', 'ollama'], check=True)
        
        print_colored("✅ Ollama instalado exitosamente", 'green')
        return True
        
    except subprocess.CalledProcessError:
        print_colored("⚠️  Homebrew no encontrado. Usando instalador manual...", 'yellow')
        
        # Instalación manual
        try:
            cmd = 'curl -fsSL https://ollama.com/install.sh | sh'
            subprocess.run(cmd, shell=True, check=True)
            print_colored("✅ Ollama instalado exitosamente", 'green')
            return True
        except Exception as e:
            print_colored(f"❌ Error instalando Ollama: {e}", 'red')
            return False

def install_ollama_linux():
    """Instala Ollama en Linux"""
    print_colored("🔧 Instalando Ollama en Linux...", 'blue')
    
    try:
        cmd = 'curl -fsSL https://ollama.com/install.sh | sh'
        subprocess.run(cmd, shell=True, check=True)
        print_colored("✅ Ollama instalado exitosamente", 'green')
        return True
    except Exception as e:
        print_colored(f"❌ Error instalando Ollama: {e}", 'red')
        return False

def start_ollama():
    """Inicia el servicio de Ollama"""
    print_colored("🚀 Iniciando servicio de Ollama...", 'blue')
    
    system = platform.system().lower()
    
    try:
        if system == 'windows':
            # En Windows, Ollama se inicia automáticamente como servicio
            subprocess.Popen(['ollama', 'serve'], 
                           creationflags=subprocess.CREATE_NO_WINDOW)
        else:
            # En macOS y Linux
            subprocess.Popen(['ollama', 'serve'], 
                           stdout=subprocess.DEVNULL, 
                           stderr=subprocess.DEVNULL)
        
        # Esperar a que el servicio esté listo
        print_colored("⏳ Esperando a que el servicio esté listo...", 'yellow')
        for i in range(10):
            if is_ollama_running():
                print_colored("✅ Servicio de Ollama iniciado exitosamente", 'green')
                return True
            time.sleep(2)
        
        print_colored("⚠️  El servicio tardó en iniciarse, pero debería estar funcionando", 'yellow')
        return True
        
    except Exception as e:
        print_colored(f"❌ Error iniciando Ollama: {e}", 'red')
        return False

def install_model(model_name="llama3.2"):
    """Instala un modelo de Ollama"""
    print_colored(f"📚 Instalando modelo {model_name}...", 'blue')
    
    try:
        # Verificar modelos disponibles
        available_models = ["llama3.2", "llama3.2:1b", "llama3.1", "qwen2.5", "mistral"]
        
        if model_name not in available_models:
            print_colored(f"⚠️  Modelo {model_name} no encontrado. Usando llama3.2:1b", 'yellow')
            model_name = "llama3.2:1b"
        
        print_colored(f"📥 Descargando {model_name}... (esto puede tardar varios minutos)", 'yellow')
        
        # Ejecutar comando de instalación
        process = subprocess.Popen(
            ['ollama', 'pull', model_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Mostrar progreso
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(f"  {output.strip()}")
        
        if process.returncode == 0:
            print_colored(f"✅ Modelo {model_name} instalado exitosamente", 'green')
            return True
        else:
            print_colored(f"❌ Error instalando modelo {model_name}", 'red')
            return False
            
    except Exception as e:
        print_colored(f"❌ Error instalando modelo: {e}", 'red')
        return False

def test_ollama():
    """Prueba la instalación de Ollama"""
    print_colored("🧪 Probando instalación de Ollama...", 'blue')
    
    try:
        # Verificar que está corriendo
        if not is_ollama_running():
            print_colored("❌ Ollama no está ejecutándose", 'red')
            return False
        
        # Probar una consulta simple
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2",
                "prompt": "Hello! Say 'Ollama is working' if you can read this.",
                "stream": False
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print_colored(f"✅ Respuesta: {result.get('response', 'Sin respuesta')}", 'green')
            return True
        else:
            print_colored(f"❌ Error en la respuesta: {response.status_code}", 'red')
            return False
            
    except Exception as e:
        print_colored(f"❌ Error probando Ollama: {e}", 'red')
        return False

def create_start_script():
    """Crea un script para iniciar Ollama fácilmente"""
    system = platform.system().lower()
    
    if system == 'windows':
        script_content = '''@echo off
echo Iniciando Ollama...
ollama serve
pause
'''
        script_path = Path("start_ollama.bat")
        
    else:  # macOS y Linux
        script_content = '''#!/bin/bash
echo "Iniciando Ollama..."
ollama serve &
echo "Ollama iniciado en segundo plano"
echo "Para detener: pkill ollama"
'''
        script_path = Path("start_ollama.sh")
    
    try:
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        if system != 'windows':
            os.chmod(script_path, 0o755)
        
        print_colored(f"✅ Script creado: {script_path}", 'green')
        return True
        
    except Exception as e:
        print_colored(f"❌ Error creando script: {e}", 'red')
        return False

def main():
    """Función principal"""
    print_colored("🔧 INSTALADOR AUTOMÁTICO DE OLLAMA", 'cyan')
    print_colored("=" * 50, 'cyan')
    
    # Verificar conexión a internet
    if not check_internet():
        print_colored("❌ No hay conexión a internet", 'red')
        return
    
    # Detectar sistema operativo
    system = platform.system().lower()
    print_colored(f"🖥️  Sistema operativo detectado: {system}", 'blue')
    
    # Verificar si ya está instalado
    if is_ollama_installed():
        print_colored("✅ Ollama ya está instalado", 'green')
    else:
        print_colored("📦 Ollama no está instalado. Instalando...", 'yellow')
        
        # Instalar según el sistema
        if system == 'windows':
            success = install_ollama_windows()
        elif system == 'darwin':  # macOS
            success = install_ollama_macos()
        elif system == 'linux':
            success = install_ollama_linux()
        else:
            print_colored(f"❌ Sistema operativo no soportado: {system}", 'red')
            return
        
        if not success:
            print_colored("❌ Error durante la instalación", 'red')
            return
    
    # Verificar si está corriendo
    if not is_ollama_running():
        print_colored("🚀 Iniciando Ollama...", 'blue')
        if not start_ollama():
            print_colored("❌ Error iniciando Ollama", 'red')
            return
    else:
        print_colored("✅ Ollama ya está ejecutándose", 'green')
    
    # Verificar modelos instalados
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        if response.status_code == 200:
            models = response.json().get('models', [])
            if models:
                print_colored(f"📚 Modelos instalados: {len(models)}", 'green')
                for model in models[:3]:  # Mostrar solo los primeros 3
                    print_colored(f"  • {model['name']}", 'white')
            else:
                print_colored("📚 No hay modelos instalados. Instalando modelo por defecto...", 'yellow')
                install_model("llama3.2:1b")
        else:
            print_colored("⚠️  No se pudieron verificar los modelos", 'yellow')
    except:
        print_colored("⚠️  Error verificando modelos", 'yellow')
    
    # Crear script de inicio
    create_start_script()
    
    # Probar instalación
    if test_ollama():
        print_colored("\n🎉 ¡INSTALACIÓN COMPLETADA EXITOSAMENTE!", 'green')
        print_colored("=" * 50, 'green')
        print_colored("🚀 Ollama está funcionando correctamente", 'green')
        print_colored("🌐 Puedes usar el chat ahora", 'green')
        print_colored("📝 Si necesitas reiniciar Ollama, usa el script creado", 'blue')
    else:
        print_colored("\n⚠️  INSTALACIÓN COMPLETADA CON ADVERTENCIAS", 'yellow')
        print_colored("=" * 50, 'yellow')
        print_colored("🔧 Ollama está instalado pero puede necesitar configuración manual", 'yellow')
        print_colored("📖 Consulta la documentación en https://ollama.com/download", 'blue')

if __name__ == "__main__":
    main() 