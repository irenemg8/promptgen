// lib/api-config.ts - Configuración de API
export const API_BASE_URL = process.env.NODE_ENV === 'production' 
  ? 'http://localhost:8000'
  : 'http://localhost:8000';

export const API_ENDPOINTS = {
  UPLOAD_FILE: `${API_BASE_URL}/api/upload-file`,
  UPLOAD_MULTIPLE_FILES: `${API_BASE_URL}/api/upload-multiple-files`,
  CHAT: `${API_BASE_URL}/api/chat`,
  DOCUMENTS: `${API_BASE_URL}/api/documents`,
  SYSTEM_STATUS: `${API_BASE_URL}/api/system/status`,
  SYSTEM_CLEANUP: `${API_BASE_URL}/api/system/cleanup`,
  HEALTH: `${API_BASE_URL}/api/health`,
};

export const fetchApi = async (url: string, options: RequestInit = {}) => {
  const response = await fetch(url, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  return response.json();
}; 