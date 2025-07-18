@echo off
title PromptGen - Sistema Seguro de Documentos
color 0a
cls

echo.
echo   =========================================================
echo   🚀 PROMPTGEN - SISTEMA SEGURO DE DOCUMENTOS
echo   =========================================================
echo   ✅ Sistema completamente local
echo   🔐 Cifrado AES de archivos
echo   📁 Soporte para multiples formatos
echo   ⚡ Procesamiento rapido
echo   🧠 Memoria persistente
echo   =========================================================
echo.

echo 🔧 Iniciando servidor backend...
start "Backend" /min cmd /k "python simple_api_server.py"

echo ⏳ Esperando 5 segundos para que el backend se inicie...
timeout /t 5 /nobreak > nul

echo 🌐 Iniciando servidor frontend...
start "Frontend" /min cmd /k "npm run dev"

echo.
echo   =========================================================
echo   ✅ SISTEMA INICIADO CORRECTAMENTE
echo   =========================================================
echo   📊 Backend:  http://localhost:8000
echo   🌐 Frontend: http://localhost:3000
echo   📚 API Docs: http://localhost:8000/docs
echo   =========================================================
echo.

echo 🎯 Abriendo aplicacion en el navegador...
timeout /t 3 /nobreak > nul
start "" "http://localhost:3000"

echo.
echo 📋 INSTRUCCIONES:
echo   1. La aplicacion se abrira automaticamente en tu navegador
echo   2. Puedes subir archivos arrastrando y soltando
echo   3. Haz preguntas sobre el contenido de tus documentos
echo   4. Todos los archivos se cifran automaticamente
echo.
echo ⚠️  Para detener el sistema, cierra esta ventana y las ventanas
echo    de Backend y Frontend que se abrieron.
echo.
pause 