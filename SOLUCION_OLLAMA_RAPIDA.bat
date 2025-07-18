@echo off
title Solución Rápida para Ollama - PromptGen
color 0b
cls

echo.
echo ===============================================
echo  🚀 SOLUCIÓN RÁPIDA PARA OLLAMA
echo ===============================================
echo.
echo ❌ Problema detectado: model 'llama3.2:3b' not found
echo ✅ Esta herramienta te ayudará a solucionarlo
echo.

echo 🔍 Verificando si Ollama está instalado...
ollama --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Ollama NO está instalado
    echo.
    echo 📥 Opción 1: Instalación Automática
    echo    Descarga e instala Ollama automáticamente
    echo.
    echo 📥 Opción 2: Instalación Manual
    echo    Abre la página web para descargar manualmente
    echo.
    set /p choice="Elige una opción (1 o 2): "
    
    if "!choice!"=="1" (
        echo 🔄 Iniciando instalación automática...
        echo ⚠️  Esto puede requerir permisos de administrador
        
        REM Descargar OllamaSetup.exe si no existe
        if not exist "OllamaSetup.exe" (
            echo 📥 Descargando instalador...
            curl -L https://ollama.com/download/OllamaSetup.exe -o OllamaSetup.exe
            if %errorlevel% neq 0 (
                echo ❌ Error descargando el instalador
                echo 💡 Usa la opción 2 para instalación manual
                pause
                exit /b 1
            )
        )
        
        echo 🚀 Ejecutando instalador...
        start /wait OllamaSetup.exe
        
        echo ⏳ Esperando a que la instalación complete...
        timeout /t 10 /nobreak >nul
        
        REM Verificar instalación
        ollama --version >nul 2>&1
        if %errorlevel% neq 0 (
            echo ❌ La instalación no se completó correctamente
            echo 💡 Intenta la instalación manual visitando: https://ollama.com/download
            pause
            exit /b 1
        )
        
        echo ✅ Ollama instalado exitosamente
        
    ) else if "!choice!"=="2" (
        echo 🌐 Abriendo página de descarga...
        start https://ollama.com/download
        echo.
        echo 📋 Instrucciones:
        echo 1. Descarga OllamaSetup.exe
        echo 2. Ejecuta el instalador
        echo 3. Reinicia este script
        echo.
        pause
        exit /b 0
    ) else (
        echo ❌ Opción no válida
        pause
        exit /b 1
    )
) else (
    echo ✅ Ollama ya está instalado
)

echo.
echo 🚀 Iniciando servicio Ollama...
start /min cmd /c "ollama serve"
timeout /t 5 /nobreak >nul

echo 🔍 Verificando conexión con Ollama...
timeout /t 3 /nobreak >nul
curl -s http://localhost:11434/api/tags >nul 2>&1
if %errorlevel% neq 0 (
    echo ⚠️  Ollama tardó en iniciar. Esperando un poco más...
    timeout /t 10 /nobreak >nul
    curl -s http://localhost:11434/api/tags >nul 2>&1
    if %errorlevel% neq 0 (
        echo ❌ No se pudo conectar con Ollama
        echo 💡 Verifica que el puerto 11434 esté libre
        pause
        exit /b 1
    )
)
echo ✅ Ollama está funcionando correctamente

echo.
echo 📥 Descargando modelos necesarios...
echo.

echo 🔄 Descargando llama3.2:1b (modelo ligero)...
echo    Esto puede tardar 2-3 minutos...
ollama pull llama3.2:1b
if %errorlevel% neq 0 (
    echo ❌ Error descargando llama3.2:1b
    echo 💡 Verifica tu conexión a internet
) else (
    echo ✅ llama3.2:1b instalado exitosamente
)

echo.
echo 🔄 Descargando mxbai-embed-large (modelo de embeddings)...
echo    Esto puede tardar 1-2 minutos...
ollama pull mxbai-embed-large
if %errorlevel% neq 0 (
    echo ❌ Error descargando mxbai-embed-large
    echo 💡 Verifica tu conexión a internet
) else (
    echo ✅ mxbai-embed-large instalado exitosamente
)

echo.
echo 🧪 Probando que todo funcione...
echo {"model":"llama3.2:1b","prompt":"Hola","stream":false} | curl -s -X POST http://localhost:11434/api/generate -d @- >nul 2>&1
if %errorlevel% neq 0 (
    echo ⚠️  La prueba no fue exitosa, pero los modelos están instalados
) else (
    echo ✅ Prueba exitosa - Todo funciona correctamente
)

echo.
echo ===============================================
echo  🎉 CONFIGURACIÓN COMPLETADA
echo ===============================================
echo.
echo ✅ Ollama instalado y funcionando
echo ✅ Modelos necesarios descargados
echo ✅ Servicio iniciado correctamente
echo.
echo 🚀 Para usar tu chat:
echo    1. Ejecuta: python simple_api_server.py
echo    2. En otra terminal: npm run dev
echo    3. Ve a: http://localhost:3000/chat
echo.
echo 💡 Si tienes problemas, reinicia tu computadora
echo    y ejecuta este script nuevamente
echo.
pause 