#!/usr/bin/env python3
"""
Sistema de Configuración Completo para PromptGen Enterprise
Con sistema seguro de documentos cifrados y procesamiento local
"""

import os
import sys
import subprocess
import time
import json
from pathlib import Path

def print_banner():
    """Mostrar banner del sistema"""
    print("\n" + "="*70)
    print("🚀 PROMPTGEN ENTERPRISE - SISTEMA SEGURO DE DOCUMENTOS")
    print("="*70)
    print("✅ Sistema completamente local")
    print("🔐 Cifrado AES de archivos")
    print("📁 Soporte para múltiples formatos")
    print("⚡ Procesamiento en menos de 10 segundos")
    print("🧠 Memoria persistente")
    print("="*70)

def check_python_version():
    """Verificar versión de Python"""
    if sys.version_info < (3, 8):
        print("❌ Se requiere Python 3.8 o superior")
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detectado")

def install_system_dependencies():
    """Instalar dependencias del sistema"""
    print("\n📦 Instalando dependencias del sistema...")
    
    # Dependencias para diferentes sistemas
    system_deps = {
        'linux': [
            'apt-get update',
            'apt-get install -y python3-dev python3-pip build-essential',
            'apt-get install -y tesseract-ocr libtesseract-dev',
            'apt-get install -y libmagic1',
            'apt-get install -y ffmpeg libsm6 libxext6'  # Para OpenCV
        ],
        'darwin': [  # macOS
            'brew install tesseract',
            'brew install libmagic',
            'brew install ffmpeg'
        ],
        'win32': [
            'echo "Para Windows, instala manualmente: tesseract-ocr, visual studio build tools"'
        ]
    }
    
    system = sys.platform
    if system in system_deps:
        for cmd in system_deps[system]:
            try:
                if system == 'win32':
                    subprocess.run(cmd, shell=True)
                else:
                    subprocess.run(cmd.split(), check=True)
            except subprocess.CalledProcessError:
                print(f"⚠️ Error instalando dependencia: {cmd}")
                print("   Continúa con instalación manual si es necesario")

def create_directories():
    """Crear directorios necesarios"""
    print("\n📁 Creando directorios...")
    
    directories = [
        "secure_documents",
        "secure_documents/encrypted_files",
        "secure_documents/vectordb",
        "secure_documents/cache",
        "temp_uploads",
        "logs",
        "exports"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   ✅ {directory}")

def install_python_dependencies():
    """Instalar dependencias de Python"""
    print("\n🐍 Instalando dependencias de Python...")
    
    try:
        # Actualizar pip
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=True)
        
        # Instalar dependencias
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        
        print("✅ Dependencias de Python instaladas")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error instalando dependencias: {e}")
        print("   Intenta instalar manualmente: pip install -r requirements.txt")
        return False
    
    return True

def setup_ollama():
    """Configurar Ollama para procesamiento local"""
    print("\n🤖 Configurando Ollama...")
    
    try:
        # Verificar si Ollama está instalado
        result = subprocess.run(["ollama", "--version"], capture_output=True, text=True)
        print(f"✅ Ollama detectado: {result.stdout.strip()}")
        
        # Descargar modelos necesarios
        models_to_download = [
            "llama3.2:3b",  # Modelo principal
            "mxbai-embed-large"  # Modelo de embeddings
        ]
        
        for model in models_to_download:
            print(f"📥 Descargando modelo {model}...")
            result = subprocess.run(["ollama", "pull", model], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ Modelo {model} descargado")
            else:
                print(f"⚠️ Error descargando {model}: {result.stderr}")
        
        return True
        
    except FileNotFoundError:
        print("❌ Ollama no encontrado")
        print("   Instala Ollama desde: https://ollama.com/download")
        return False
    except Exception as e:
        print(f"❌ Error configurando Ollama: {e}")
        return False

def create_env_file():
    """Crear archivo de configuración de entorno"""
    print("\n⚙️ Creando configuración de entorno...")
    
    env_content = """# Configuración del Sistema Seguro PromptGen
ENCRYPTION_KEY=secure_promptgen_enterprise_2024
OLLAMA_MODEL=llama3.2:3b
EMBEDDINGS_MODEL=mxbai-embed-large
STORAGE_PATH=./secure_documents
MAX_MEMORY_CACHE_MB=1024
MAX_UPLOAD_SIZE_MB=500
ENABLE_MONITORING=true
LOG_LEVEL=INFO
"""
    
    with open(".env", "w") as f:
        f.write(env_content)
    
    print("✅ Archivo .env creado")

def create_startup_scripts():
    """Crear scripts de inicio"""
    print("\n🚀 Creando scripts de inicio...")
    
    # Script para Windows
    windows_script = """@echo off
title PromptGen Enterprise - Sistema Seguro
echo Iniciando PromptGen Enterprise...
echo.

echo Iniciando servidor backend...
start "Backend" cmd /k "python api_server.py"

timeout /t 5 /nobreak

echo Iniciando frontend...
start "Frontend" cmd /k "npm run dev"

echo.
echo Sistema iniciado correctamente!
echo Backend: http://localhost:8000
echo Frontend: http://localhost:3000
echo.
pause
"""
    
    with open("start_secure_system.bat", "w") as f:
        f.write(windows_script)
    
    # Script para Linux/Mac
    unix_script = """#!/bin/bash
clear
echo "🚀 PromptGen Enterprise - Sistema Seguro"
echo "========================================"
echo

echo "🔧 Iniciando servidor backend..."
python3 api_server.py &
BACKEND_PID=$!

sleep 5

echo "🌐 Iniciando frontend..."
npm run dev &
FRONTEND_PID=$!

echo
echo "✅ Sistema iniciado correctamente!"
echo "📊 Backend: http://localhost:8000"
echo "🌐 Frontend: http://localhost:3000"
echo "📚 Documentación: http://localhost:8000/docs"
echo
echo "Para detener el sistema, presiona Ctrl+C"
echo

# Función para limpiar procesos al salir
cleanup() {
    echo
    echo "🔄 Deteniendo sistema..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    echo "✅ Sistema detenido"
    exit 0
}

trap cleanup SIGINT SIGTERM

# Esperar a que los procesos terminen
wait
"""
    
    with open("start_secure_system.sh", "w") as f:
        f.write(unix_script)
    
    # Hacer ejecutable en Unix
    if sys.platform != 'win32':
        os.chmod("start_secure_system.sh", 0o755)
    
    print("✅ Scripts de inicio creados")

def install_node_dependencies():
    """Instalar dependencias de Node.js"""
    print("\n📦 Instalando dependencias de Node.js...")
    
    try:
        # Verificar si Node.js está instalado
        result = subprocess.run(["node", "--version"], capture_output=True, text=True)
        print(f"✅ Node.js detectado: {result.stdout.strip()}")
        
        # Instalar dependencias
        subprocess.run(["npm", "install"], check=True)
        print("✅ Dependencias de Node.js instaladas")
        
        return True
        
    except FileNotFoundError:
        print("❌ Node.js no encontrado")
        print("   Instala Node.js desde: https://nodejs.org/")
        return False
    except subprocess.CalledProcessError as e:
        print(f"❌ Error instalando dependencias de Node.js: {e}")
        return False

def test_system():
    """Probar el sistema"""
    print("\n🧪 Probando sistema...")
    
    try:
        # Test de importación de módulos críticos
        import secure_document_system
        print("✅ Sistema de documentos seguro importado")
        
        # Test de configuración
        from secure_document_system import secure_system
        stats = secure_system.get_system_stats()
        print(f"✅ Sistema inicializado: {stats['total_documents']} documentos")
        
        # Test de cifrado
        test_data = "test encryption"
        encrypted = secure_system._encrypt_data(test_data)
        decrypted = secure_system._decrypt_data(encrypted)
        assert decrypted == test_data
        print("✅ Sistema de cifrado funcionando")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en pruebas: {e}")
        return False

def create_documentation():
    """Crear documentación del sistema"""
    print("\n📚 Creando documentación...")
    
    readme_content = """# PromptGen Enterprise - Sistema Seguro de Documentos

## 🔐 Características Principales

- **Cifrado AES Local**: Todos los archivos se cifran automáticamente
- **Soporte Universal**: PDF, DOCX, TXT, JSON, CSV, Excel, HTML, Markdown, imágenes, código
- **Procesamiento Rápido**: Respuestas en menos de 10 segundos
- **Memoria Persistente**: Los documentos permanecen en memoria para acceso rápido
- **Sistema Local**: No hay conexiones externas, todo funciona offline

## 🚀 Inicio Rápido

### Windows
```bash
start_secure_system.bat
```

### Linux/Mac
```bash
./start_secure_system.sh
```

### Manual
```bash
# Backend
python api_server.py

# Frontend (en otra terminal)
npm run dev
```

## 📊 Endpoints de la API

- `POST /api/upload-file` - Subir archivo individual
- `POST /api/upload-multiple-files` - Subir múltiples archivos
- `POST /api/chat` - Chat con documentos
- `GET /api/documents` - Listar documentos
- `DELETE /api/documents/{id}` - Eliminar documento
- `GET /api/system/status` - Estado del sistema
- `POST /api/system/cleanup` - Limpiar cache

## 🔧 Configuración

El sistema se configura a través del archivo `.env`:

```env
ENCRYPTION_KEY=secure_promptgen_enterprise_2024
OLLAMA_MODEL=llama3.2:3b
EMBEDDINGS_MODEL=mxbai-embed-large
STORAGE_PATH=./secure_documents
MAX_MEMORY_CACHE_MB=1024
```

## 📁 Estructura de Archivos

```
secure_documents/
├── encrypted_files/    # Archivos cifrados
├── vectordb/          # Base de datos vectorial
├── cache/             # Cache del sistema
└── metadata.enc       # Metadatos cifrados
```

## 🛡️ Seguridad

- Cifrado AES-256 con PBKDF2
- Claves derivadas con 100,000 iteraciones
- Archivos nunca se almacenan en texto plano
- Metadatos cifrados
- Procesamiento completamente local

## 📈 Monitoreo

- Métricas de rendimiento en tiempo real
- Estadísticas de uso de memoria y CPU
- Tracking de cache hits
- Tiempos de procesamiento

## 🔧 Solución de Problemas

### Ollama no encontrado
```bash
# Instalar Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Descargar modelos
ollama pull llama3.2:3b
ollama pull mxbai-embed-large
```

### Dependencias faltantes
```bash
pip install -r requirements.txt
npm install
```

### Problemas de permisos
```bash
chmod +x start_secure_system.sh
```

## 🤝 Soporte

Para soporte técnico, consulta los logs del sistema o revisa la documentación de la API en `http://localhost:8000/docs`
"""
    
    with open("README_SISTEMA_SEGURO.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
    
    print("✅ Documentación creada")

def main():
    """Función principal de configuración"""
    print_banner()
    
    # Verificaciones previas
    check_python_version()
    
    # Crear directorios
    create_directories()
    
    # Instalar dependencias del sistema
    install_system_dependencies()
    
    # Instalar dependencias de Python
    if not install_python_dependencies():
        print("❌ Error en instalación de dependencias de Python")
        return False
    
    # Instalar dependencias de Node.js
    if not install_node_dependencies():
        print("❌ Error en instalación de dependencias de Node.js")
        return False
    
    # Configurar Ollama
    ollama_ok = setup_ollama()
    if not ollama_ok:
        print("⚠️ Ollama no configurado - algunas funciones pueden no funcionar")
    
    # Crear archivos de configuración
    create_env_file()
    create_startup_scripts()
    create_documentation()
    
    # Probar sistema
    if test_system():
        print("\n🎉 SISTEMA CONFIGURADO EXITOSAMENTE!")
        print("\n📋 Próximos pasos:")
        print("1. Ejecutar: start_secure_system.bat (Windows) o ./start_secure_system.sh (Linux/Mac)")
        print("2. Abrir: http://localhost:3000")
        print("3. Subir documentos y comenzar a chatear")
        print("\n🔐 Tu sistema está completamente seguro y cifrado!")
        return True
    else:
        print("\n❌ ERRORES EN LA CONFIGURACIÓN")
        print("   Revisa los errores anteriores y ejecuta el script nuevamente")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 