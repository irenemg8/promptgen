@echo off
echo =============================================
echo    🚀 PromptGen - Backend Temporal (Sin Ollama)
echo =============================================
echo ⚠️  Este es un modo temporal para probar la interfaz
echo 📥 Puedes subir archivos pero no se procesarán con IA
echo 🔗 Instala Ollama desde: https://ollama.com/download
echo.
echo 🌐 Servidor iniciando en: http://localhost:8000
echo.
uvicorn api_server_temp:app --host 0.0.0.0 --port 8000 --reload
pause 