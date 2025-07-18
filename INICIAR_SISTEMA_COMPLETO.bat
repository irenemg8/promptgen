@echo off
title PromptGen - Sistema Completo con Funcionalidades Mejoradas
color 0a
cls

echo.
echo ===============================================
echo  🚀 PROMPTGEN - SISTEMA COMPLETO MEJORADO
echo ===============================================
echo.
echo ✅ Nuevas funcionalidades añadidas:
echo    🗂️  Preguntas sobre archivos cargados
echo    🤖 Manejo inteligente de errores
echo    📋 Respuestas informativas sin IA
echo    🔍 Búsqueda de archivos específicos
echo.

echo 🔍 Verificando Ollama...
ollama --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Ollama no está instalado
    echo.
    echo 🛠️  Opciones:
    echo    1. Ejecutar SOLUCION_OLLAMA_RAPIDA.bat
    echo    2. Continuar sin Ollama (funcionalidad limitada)
    echo.
    set /p choice="¿Qué deseas hacer? (1 o 2): "
    
    if "!choice!"=="1" (
        echo 🚀 Ejecutando solución rápida...
        call SOLUCION_OLLAMA_RAPIDA.bat
        echo.
        echo 🔄 Continuando con el inicio del sistema...
    ) else (
        echo ⚠️  Iniciando sin Ollama - Funcionalidad limitada
        echo    El chatbot podrá responder preguntas sobre archivos
        echo    pero no podrá analizar contenido con IA
        echo.
        timeout /t 3 /nobreak >nul
    )
) else (
    echo ✅ Ollama instalado
    
    echo 🔄 Iniciando servicio Ollama...
    start /min cmd /c "ollama serve"
    timeout /t 3 /nobreak >nul
    
    echo 🔍 Verificando modelos...
    ollama list | findstr llama3.2 >nul 2>&1
    if %errorlevel% neq 0 (
        echo ⚠️  Modelo no encontrado, descargando...
        ollama pull llama3.2:1b
    ) else (
        echo ✅ Modelos disponibles
    )
)

echo.
echo 🚀 Iniciando sistema...

echo 📊 Iniciando Backend (API)...
start "PromptGen Backend" cmd /k "echo 🔧 Backend iniciado - No cerrar esta ventana && python simple_api_server.py"

echo ⏳ Esperando a que el backend se inicie...
timeout /t 5 /nobreak >nul

echo 🔍 Verificando conexión con backend...
curl -s http://localhost:8000/api/health >nul 2>&1
if %errorlevel% neq 0 (
    echo ⚠️  Backend tardando en iniciar...
    timeout /t 5 /nobreak >nul
)

echo 🌐 Iniciando Frontend (Next.js)...
start "PromptGen Frontend" cmd /k "echo 🎯 Frontend iniciado - No cerrar esta ventana && npm run dev"

echo ⏳ Esperando a que el frontend se inicie...
timeout /t 8 /nobreak >nul

echo 🔍 Verificando puertos...
netstat -an | findstr :8000 >nul 2>&1
if %errorlevel% neq 0 (
    echo ⚠️  Backend puede no estar listo
)

netstat -an | findstr :3000 >nul 2>&1
if %errorlevel% neq 0 (
    echo ⚠️  Frontend puede estar en puerto 3001
)

echo.
echo ===============================================
echo  🎉 SISTEMA INICIADO EXITOSAMENTE
echo ===============================================
echo.
echo 🌐 Accesos disponibles:
echo    Chat: http://localhost:3000/chat
echo    Alt:  http://localhost:3001/chat
echo    API:  http://localhost:8000/docs
echo.
echo 🧪 Prueba las nuevas funcionalidades:
echo.
echo 📋 Preguntas que puedes hacer:
echo    "¿Qué archivos tienes cargados?"
echo    "¿Cuántos documentos hay?"
echo    "¿Existe el archivo manual.pdf?"
echo    "Listar todos los archivos"
echo    "Mostrar documentos disponibles"
echo.
echo 💡 Consejos:
echo    1. Sube algunos archivos primero
echo    2. Prueba preguntas sobre archivos específicos
echo    3. El sistema funciona incluso sin Ollama
echo.

echo 🚀 Abriendo aplicación...
timeout /t 2 /nobreak >nul

REM Intentar abrir en el puerto correcto
start http://localhost:3000/chat >nul 2>&1
timeout /t 2 /nobreak >nul
start http://localhost:3001/chat >nul 2>&1

echo.
echo ===============================================
echo  📋 INSTRUCCIONES DE USO
echo ===============================================
echo.
echo 🗂️  Para probar gestión de archivos:
echo    1. Sube archivos (PDF, Word, TXT)
echo    2. Pregunta: "¿qué archivos tienes?"
echo    3. Busca archivos específicos
echo.
echo 🤖 Para consultas con IA:
echo    1. Asegúrate de que Ollama esté instalado
echo    2. Haz preguntas sobre el contenido
echo    3. El sistema citará fuentes
echo.
echo 🔧 Para más información:
echo    - Lee: PRUEBA_NUEVAS_FUNCIONALIDADES.md
echo    - Problemas: SOLUCION_OLLAMA_RAPIDA.bat
echo.
echo ⚠️  IMPORTANTE: No cerrar las ventanas del Backend y Frontend
echo.
pause 