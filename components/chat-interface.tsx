"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { Progress } from "@/components/ui/progress";
import { API_ENDPOINTS } from "@/lib/api-config";
import { 
  Send, 
  Upload, 
  FileText, 
  MessageSquare, 
  Trash2, 
  Bot,
  User,
  Paperclip,
  X,
  RefreshCw,
  Shield,
  Clock,
  Files,
  Database,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Zap,
  HardDrive,
  Cpu,
  Sparkles,
  Lock,
  Activity,
  Menu,
  ArrowUp,
  ChevronLeft,
  ChevronRight
} from "lucide-react";

interface Message {
  id: string;
  type: "user" | "assistant";
  content: string;
  timestamp: string;
  response_time?: number;
  sources?: {
    filename: string;
    content: string;
    chunk_index: number;
    file_type?: string;
  }[];
  from_cache?: boolean;
}

interface Document {
  doc_id: string;
  filename: string;
  file_type: string;
  mime_type?: string;
  upload_date: string;
  file_size?: number;
  chunks_count: number;
  processing_time?: number;
  status: string;
}

interface ChatResponse {
  success: boolean;
  result: {
    answer: string;
    sources: {
      filename: string;
      content: string;
      chunk_index: number;
      file_type?: string;
    }[];
    response_time: number;
    from_cache?: boolean;
  };
  response_time?: number;
}

interface SystemStats {
  total_documents: number;
  total_chunks: number;
  memory_usage_mb: number;
  cpu_usage_percent: number;
  encryption_enabled: boolean;
  supported_formats: string[];
  cache_hits: number;
  avg_processing_time: number;
}

interface UploadProgress {
  filename: string;
  progress: number;
  status: 'uploading' | 'processing' | 'completed' | 'error';
  error?: string;
}

export default function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [documents, setDocuments] = useState<Document[]>([]);
  const [systemStats, setSystemStats] = useState<SystemStats | null>(null);
  const [uploadProgress, setUploadProgress] = useState<UploadProgress[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [selectedFiles, setSelectedFiles] = useState<FileList | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false); // Cerrado por defecto para mobile
  const [isMobile, setIsMobile] = useState(false);
  
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const dropZoneRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const chatContainerRef = useRef<HTMLDivElement>(null);

  // Detectar tamaño de pantalla
  useEffect(() => {
    const checkMobile = () => {
      const mobile = window.innerWidth < 768;
      setIsMobile(mobile);
      // En mobile, sidebar cerrado por defecto
      if (mobile) {
        setSidebarOpen(false);
      } else {
        setSidebarOpen(true);
      }
    };
    
    checkMobile();
    window.addEventListener('resize', checkMobile);
    return () => window.removeEventListener('resize', checkMobile);
  }, []);

  // Scroll automático mejorado
  const scrollToBottom = useCallback(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ 
        behavior: "smooth", 
        block: "end",
        inline: "nearest"
      });
    }
  }, []);

  useEffect(() => {
    // Delay para asegurar que el DOM se actualice
    const timer = setTimeout(() => {
      scrollToBottom();
    }, 100);
    
    return () => clearTimeout(timer);
  }, [messages, scrollToBottom]);

  // Auto-resize textarea
  const handleTextareaChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value);
    
    // Auto-resize
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 120)}px`;
    }
  };

  // Cargar datos al iniciar
  useEffect(() => {
    loadDocuments();
    loadSystemStats();
    
    // Actualizar estadísticas cada 30 segundos
    const interval = setInterval(() => {
      loadSystemStats();
    }, 30000);
    
    return () => clearInterval(interval);
  }, []);

  const loadDocuments = async () => {
    try {
      const response = await fetch(API_ENDPOINTS.DOCUMENTS);
      const data = await response.json();
      
      if (data.success) {
        setDocuments(data.documents);
      }
    } catch (error) {
      console.error("Error cargando documentos:", error);
    }
  };

  const loadSystemStats = async () => {
    try {
      const response = await fetch(API_ENDPOINTS.SYSTEM_STATUS);
      const data = await response.json();
      
      if (data.success) {
        setSystemStats(data.status);
      }
    } catch (error) {
      console.error("Error cargando estadísticas:", error);
    }
  };

  const handleSendMessage = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      type: "user",
      content: input,
      timestamp: new Date().toLocaleTimeString(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);

    // Reset textarea height
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }

    try {
      const response = await fetch(API_ENDPOINTS.CHAT, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query: input,
          max_results: 5,
        }),
      });

      const data: ChatResponse = await response.json();

      if (data.success) {
        const assistantMessage: Message = {
          id: (Date.now() + 1).toString(),
          type: "assistant",
          content: data.result.answer,
          timestamp: new Date().toLocaleTimeString(),
          response_time: data.result.response_time,
          sources: data.result.sources,
          from_cache: data.result.from_cache,
        };

        setMessages(prev => [...prev, assistantMessage]);
      } else {
        throw new Error("Error en la respuesta del servidor");
      }
    } catch (error) {
      console.error("Error enviando mensaje:", error);
      
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: "assistant",
        content: "Lo siento, ocurrió un error al procesar tu consulta. Asegúrate de que el backend esté funcionando en el puerto 8000.",
        timestamp: new Date().toLocaleTimeString(),
      };

      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleFileUpload = async (files: FileList) => {
    if (files.length === 0) return;

    setIsUploading(true);
    const formData = new FormData();
    
    // Inicializar progress tracking
    const initialProgress: UploadProgress[] = Array.from(files).map(file => ({
      filename: file.name,
      progress: 0,
      status: 'uploading'
    }));
    
    setUploadProgress(initialProgress);

    try {
      // Agregar todos los archivos al FormData
      Array.from(files).forEach(file => {
        formData.append('files', file);
      });

      const response = await fetch(API_ENDPOINTS.UPLOAD_MULTIPLE_FILES, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (data.success) {
        // Actualizar progress con resultados
        const updatedProgress = initialProgress.map(progress => {
          const result = data.processed_files.find((f: any) => f.filename === progress.filename);
          const error = data.errors.find((e: string) => e.includes(progress.filename));
          
          return {
            ...progress,
            progress: 100,
            status: error ? 'error' as const : 'completed' as const,
            error: error
          };
        });
        
        setUploadProgress(updatedProgress);
        
        // Recargar documentos
        await loadDocuments();
        await loadSystemStats();
        
        // Limpiar progress después de 3 segundos
        setTimeout(() => {
          setUploadProgress([]);
        }, 3000);
        
        // Mostrar mensaje de éxito
        const successMessage: Message = {
          id: Date.now().toString(),
          type: "assistant",
          content: `✅ Procesamiento completado: ${data.successful_files} archivos procesados exitosamente${data.errors.length > 0 ? `, ${data.errors.length} errores encontrados` : ''}.`,
          timestamp: new Date().toLocaleTimeString(),
        };
        
        setMessages(prev => [...prev, successMessage]);
        
      } else {
        throw new Error("Error procesando archivos");
      }
    } catch (error) {
      console.error("Error subiendo archivos:", error);
      
      // Marcar todos como error
      const errorProgress = initialProgress.map(progress => ({
        ...progress,
        progress: 100,
        status: 'error' as const,
        error: 'Error de conexión - Verifica que el backend esté funcionando'
      }));
      
      setUploadProgress(errorProgress);
      
      const errorMessage: Message = {
        id: Date.now().toString(),
        type: "assistant",
        content: "❌ Error al procesar archivos. Verifica que el backend esté funcionando en el puerto 8000.",
        timestamp: new Date().toLocaleTimeString(),
      };
      
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsUploading(false);
    }
  };

  const handleDeleteDocument = async (docId: string) => {
    try {
      const response = await fetch(`${API_ENDPOINTS.DOCUMENTS}/${docId}`, {
        method: "DELETE",
      });

      const data = await response.json();

      if (data.success) {
        await loadDocuments();
        await loadSystemStats();
        
        const successMessage: Message = {
          id: Date.now().toString(),
          type: "assistant",
          content: "✅ Documento eliminado exitosamente.",
          timestamp: new Date().toLocaleTimeString(),
        };
        
        setMessages(prev => [...prev, successMessage]);
      } else {
        throw new Error("Error eliminando documento");
      }
    } catch (error) {
      console.error("Error eliminando documento:", error);
      
      const errorMessage: Message = {
        id: Date.now().toString(),
        type: "assistant",
        content: "❌ Error al eliminar documento.",
        timestamp: new Date().toLocaleTimeString(),
      };
      
      setMessages(prev => [...prev, errorMessage]);
    }
  };

  const handleCleanupSystem = async () => {
    try {
      const response = await fetch(API_ENDPOINTS.SYSTEM_CLEANUP, {
        method: "POST",
      });

      const data = await response.json();

      if (data.success) {
        await loadDocuments();
        await loadSystemStats();
        
        const successMessage: Message = {
          id: Date.now().toString(),
          type: "assistant",
          content: "✅ Sistema limpiado exitosamente.",
          timestamp: new Date().toLocaleTimeString(),
        };
        
        setMessages(prev => [...prev, successMessage]);
      }
    } catch (error) {
      console.error("Error limpiando sistema:", error);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setDragActive(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setDragActive(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragActive(false);
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleFileUpload(files);
    }
  };

  const formatFileSize = (bytes: number) => {
    const sizes = ["Bytes", "KB", "MB", "GB"];
    if (bytes === 0) return "0 Bytes";
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return Math.round((bytes / Math.pow(1024, i)) * 100) / 100 + " " + sizes[i];
  };

  const getFileTypeIcon = (fileType: string) => {
    switch (fileType.toLowerCase()) {
      case 'pdf':
        return <FileText className="h-4 w-4 text-red-500" />;
      case 'docx':
      case 'doc':
        return <FileText className="h-4 w-4 text-blue-500" />;
      case 'xlsx':
      case 'csv':
        return <FileText className="h-4 w-4 text-green-500" />;
      case 'json':
        return <FileText className="h-4 w-4 text-yellow-500" />;
      case 'jpg':
      case 'jpeg':
      case 'png':
      case 'gif':
        return <FileText className="h-4 w-4 text-purple-500" />;
      default:
        return <FileText className="h-4 w-4 text-gray-500" />;
    }
  };

  return (
    <div className="flex h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 overflow-hidden">
      {/* Sidebar - Responsive */}
      <div className={`${
        sidebarOpen ? (isMobile ? 'w-full' : 'w-80') : (isMobile ? 'w-0' : 'w-16')
      } transition-all duration-300 bg-slate-800/50 backdrop-blur-xl border-r border-slate-700/50 flex flex-col ${
        isMobile ? 'absolute inset-0 z-50' : 'relative'
      }`}>
        <div className="p-4 border-b border-slate-700/50">
          <div className="flex items-center justify-between">
            <div className={`${sidebarOpen ? 'block' : 'hidden'} transition-all duration-300`}>
              <h2 className="text-lg font-semibold text-white flex items-center gap-2">
                <div className="p-2 bg-gradient-to-r from-blue-500 to-purple-500 rounded-lg">
                  <Shield className="h-5 w-5 text-white" />
                </div>
                Sistema Seguro
              </h2>
              <p className="text-sm text-slate-400 mt-1">Documentos cifrados localmente</p>
            </div>
            
            {/* Icono comprimido centrado */}
            <div className={`${!sidebarOpen && !isMobile ? 'flex' : 'hidden'} flex-col items-center justify-center w-full`}>
              <div className="p-2 bg-gradient-to-r from-blue-500 to-purple-500 rounded-lg mb-2">
                <Shield className="h-5 w-5 text-white" />
              </div>
              <span className="text-xs text-slate-400 font-medium">Seguro</span>
            </div>
            
            {/* Botón de expandir/contraer mejorado */}
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className={`${
                sidebarOpen ? 'text-slate-400 hover:text-white hover:bg-slate-700' : 'text-slate-300 hover:text-white hover:bg-gradient-to-r hover:from-blue-500 hover:to-purple-500'
              } transition-all duration-300 rounded-lg p-2 ${
                !sidebarOpen && !isMobile ? 'mt-2 shadow-lg ring-1 ring-slate-600/50' : ''
              }`}
              title={sidebarOpen ? "Comprimir sidebar" : "Expandir sidebar"}
            >
              {sidebarOpen ? (
                <ChevronLeft className="h-4 w-4" />
              ) : (
                <ChevronRight className="h-4 w-4 animate-pulse" />
              )}
            </Button>
          </div>
        </div>

        {/* Iconos de navegación comprimidos mejorados */}
        {!sidebarOpen && !isMobile && (
          <div className="flex-1 flex flex-col items-center py-4 space-y-3">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setSidebarOpen(true)}
              className="text-slate-400 hover:text-white hover:bg-gradient-to-r hover:from-blue-500 hover:to-purple-500 w-10 h-10 p-0 rounded-lg transition-all duration-300 shadow-md hover:shadow-lg"
              title="Documentos"
            >
              <Files className="h-5 w-5" />
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setSidebarOpen(true)}
              className="text-slate-400 hover:text-white hover:bg-gradient-to-r hover:from-green-500 hover:to-teal-500 w-10 h-10 p-0 rounded-lg transition-all duration-300 shadow-md hover:shadow-lg"
              title="Subir archivos"
            >
              <Upload className="h-5 w-5" />
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setSidebarOpen(true)}
              className="text-slate-400 hover:text-white hover:bg-gradient-to-r hover:from-purple-500 hover:to-pink-500 w-10 h-10 p-0 rounded-lg transition-all duration-300 shadow-md hover:shadow-lg"
              title="Estadísticas"
            >
              <Activity className="h-5 w-5" />
            </Button>
            
            {/* Indicador de estado 
            <div className="mt-4 flex flex-col items-center space-y-2">
              <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
              <span className="text-xs text-slate-500 font-medium transform rotate-90 whitespace-nowrap">
                Online
              </span>
            </div>*/}
          </div>
        )}

        {/* Botón flotante para expandir (móvil) */}
        {!sidebarOpen && isMobile && (
          <div className="absolute top-4 left-4 z-10">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setSidebarOpen(true)}
              className="text-white bg-gradient-to-r from-blue-500 to-purple-500 hover:from-blue-600 hover:to-purple-600 w-12 h-12 p-0 rounded-full shadow-lg hover:shadow-xl transition-all duration-300"
              title="Abrir menú"
            >
              <Menu className="h-5 w-5" />
            </Button>
          </div>
        )}

        {sidebarOpen && (
          <Tabs defaultValue="documents" className="flex-1 flex flex-col overflow-hidden">
            <TabsList className="grid w-full grid-cols-3 m-2 bg-slate-700/50">
              <TabsTrigger value="documents" className="text-slate-300 data-[state=active]:bg-slate-600 data-[state=active]:text-white">
                Docs
              </TabsTrigger>
              <TabsTrigger value="upload" className="text-slate-300 data-[state=active]:bg-slate-600 data-[state=active]:text-white">
                Subir
              </TabsTrigger>
              <TabsTrigger value="stats" className="text-slate-300 data-[state=active]:bg-slate-600 data-[state=active]:text-white">
                Stats
              </TabsTrigger>
            </TabsList>

            <TabsContent value="documents" className="flex-1 overflow-hidden">
              <div className="p-4 h-full flex flex-col">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="font-medium text-white">Documentos ({documents.length})</h3>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={loadDocuments}
                    disabled={isLoading}
                    className="text-slate-400 hover:text-white hover:bg-slate-700"
                  >
                    <RefreshCw className="h-4 w-4" />
                  </Button>
                </div>

                <ScrollArea className="flex-1 h-0">
                  <div className="space-y-2 pr-2">
                    {documents.length === 0 ? (
                      <div className="text-center py-8 text-slate-400">
                        <Files className="h-12 w-12 mx-auto mb-2 opacity-50" />
                        <p>No hay documentos</p>
                        <p className="text-sm">Sube archivos para comenzar</p>
                      </div>
                    ) : (
                      documents.map((doc) => (
                        <Card key={doc.doc_id} className="p-2 bg-slate-700/30 border-slate-600/50 hover:bg-slate-700/50 transition-colors">
                          <div className="flex items-start justify-between gap-2">
                            <div className="flex-1 min-w-0 overflow-hidden">
                              <div className="flex items-center gap-2 mb-1">
                                <div className="flex-shrink-0">
                                  {getFileTypeIcon(doc.file_type)}
                                </div>
                                <div className="flex-1 min-w-0 overflow-x-auto">
                                  <span className="font-medium text-xs text-white whitespace-nowrap" title={doc.filename}>
                                    {doc.filename}
                                  </span>
                                </div>
                              </div>
                              <div className="flex items-center gap-2 mb-1">
                                <Badge variant="secondary" className="text-xs bg-slate-600 text-slate-200 flex-shrink-0">
                                  {doc.file_type.toUpperCase()}
                                </Badge>
                                <span className="text-xs text-slate-400 flex-shrink-0">
                                  {doc.chunks_count} chunks
                                </span>
                              </div>
                              {doc.file_size && (
                                <p className="text-xs text-slate-400 truncate">
                                  {formatFileSize(doc.file_size)}
                                </p>
                              )}
                            </div>
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => handleDeleteDocument(doc.doc_id)}
                              className="text-red-400 hover:text-red-300 hover:bg-red-500/10 flex-shrink-0 w-8 h-8 p-0"
                            >
                              <Trash2 className="h-3 w-3" />
                            </Button>
                          </div>
                        </Card>
                      ))
                    )}
                  </div>
                </ScrollArea>
              </div>
            </TabsContent>

            <TabsContent value="upload" className="flex-1 overflow-hidden">
              <div className="p-4 h-full flex flex-col">
                <h3 className="font-medium mb-4 text-white">Subir Documentos</h3>
                
                {/* Drag and Drop Zone */}
                <div
                  ref={dropZoneRef}
                  onDragOver={handleDragOver}
                  onDragLeave={handleDragLeave}
                  onDrop={handleDrop}
                  className={`border-2 border-dashed rounded-lg p-6 text-center transition-all duration-300 ${
                    dragActive 
                      ? 'border-blue-400 bg-blue-500/10' 
                      : 'border-slate-600 hover:border-slate-500 bg-slate-700/20'
                  }`}
                >
                  <div className="p-3 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full w-fit mx-auto mb-4">
                    <Upload className="h-6 w-6 text-white" />
                  </div>
                  <p className="text-sm text-slate-300 mb-2">
                    Arrastra archivos aquí
                  </p>
                  <p className="text-xs text-slate-400 mb-4">
                    PDF, DOCX, TXT, JSON, CSV, imágenes, etc.
                  </p>
                  
                  <input
                    ref={fileInputRef}
                    type="file"
                    multiple
                    className="hidden"
                    onChange={(e) => {
                      if (e.target.files) {
                        setSelectedFiles(e.target.files);
                        handleFileUpload(e.target.files);
                      }
                    }}
                  />
                  
                  <Button
                    onClick={() => fileInputRef.current?.click()}
                    disabled={isUploading}
                    className="w-full bg-gradient-to-r from-blue-500 to-purple-500 hover:from-blue-600 hover:to-purple-600 text-white border-none"
                  >
                    <Paperclip className="h-4 w-4 mr-2" />
                    {isUploading ? 'Procesando...' : 'Seleccionar'}
                  </Button>
                </div>

                {/* Upload Progress */}
                {uploadProgress.length > 0 && (
                  <div className="mt-4 flex-1 overflow-hidden">
                    <h4 className="font-medium text-sm text-white mb-2">Progreso</h4>
                    <ScrollArea className="h-full">
                      <div className="space-y-2 pr-2">
                        {uploadProgress.map((progress, index) => (
                          <div key={index} className="bg-slate-700/30 p-3 rounded border border-slate-600/50">
                            <div className="flex items-center justify-between mb-1">
                              <span className="text-sm truncate text-white" title={progress.filename}>
                                {progress.filename}
                              </span>
                              {progress.status === 'completed' && (
                                <CheckCircle className="h-4 w-4 text-green-400" />
                              )}
                              {progress.status === 'error' && (
                                <XCircle className="h-4 w-4 text-red-400" />
                              )}
                            </div>
                            <Progress value={progress.progress} className="h-2" />
                            {progress.error && (
                              <p className="text-xs text-red-400 mt-1">{progress.error}</p>
                            )}
                          </div>
                        ))}
                      </div>
                    </ScrollArea>
                  </div>
                )}
              </div>
            </TabsContent>

            <TabsContent value="stats" className="flex-1 overflow-hidden">
              <div className="p-4 h-full flex flex-col">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="font-medium text-white">Estadísticas</h3>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={handleCleanupSystem}
                    className="text-slate-400 hover:text-white hover:bg-slate-700"
                  >
                    <RefreshCw className="h-4 w-4" />
                  </Button>
                </div>

                {systemStats && (
                                     <ScrollArea className="flex-1 h-0">
                     <div className="space-y-4 pr-2">
                      <Card className="p-4 bg-slate-700/30 border-slate-600/50">
                        <h4 className="font-medium mb-3 flex items-center gap-2 text-white">
                          <Database className="h-4 w-4" />
                          Almacenamiento
                        </h4>
                        <div className="space-y-2 text-sm">
                          <div className="flex justify-between">
                            <span className="text-slate-400">Docs:</span>
                            <span className="font-medium text-white">{systemStats.total_documents}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-slate-400">Chunks:</span>
                            <span className="font-medium text-white">{systemStats.total_chunks}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-slate-400">Cifrado:</span>
                            <Badge className={systemStats.encryption_enabled ? "bg-green-500" : "bg-red-500"}>
                              {systemStats.encryption_enabled ? "ON" : "OFF"}
                            </Badge>
                          </div>
                        </div>
                      </Card>

                      <Card className="p-4 bg-slate-700/30 border-slate-600/50">
                        <h4 className="font-medium mb-3 flex items-center gap-2 text-white">
                          <Zap className="h-4 w-4" />
                          Rendimiento
                        </h4>
                        <div className="space-y-2 text-sm">
                          <div className="flex justify-between">
                            <span className="text-slate-400">Caché:</span>
                            <span className="font-medium text-white">{systemStats.cache_hits}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-slate-400">Tiempo:</span>
                            <span className="font-medium text-white">{systemStats.avg_processing_time.toFixed(1)}s</span>
                          </div>
                        </div>
                      </Card>

                      <Card className="p-4 bg-slate-700/30 border-slate-600/50">
                        <h4 className="font-medium mb-3 flex items-center gap-2 text-white">
                          <Activity className="h-4 w-4" />
                          Sistema
                        </h4>
                        <div className="space-y-2 text-sm">
                          <div className="flex justify-between">
                            <span className="text-slate-400">RAM:</span>
                            <span className="font-medium text-white">{systemStats.memory_usage_mb.toFixed(0)} MB</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-slate-400">CPU:</span>
                            <span className="font-medium text-white">{systemStats.cpu_usage_percent.toFixed(1)}%</span>
                          </div>
                        </div>
                      </Card>
                    </div>
                  </ScrollArea>
                )}
              </div>
            </TabsContent>
          </Tabs>
        )}
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Header */}
        <div className="bg-slate-800/50 backdrop-blur-xl border-b border-slate-700/50 p-4 flex-shrink-0">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              {/* Mobile menu button */}
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setSidebarOpen(!sidebarOpen)}
                className="text-slate-400 hover:text-white hover:bg-slate-700 md:hidden"
              >
                <Menu className="h-4 w-4" />
              </Button>
              
              <div>
                <h1 className="text-xl font-semibold text-white flex items-center gap-2">
                  <div className="p-2 bg-gradient-to-r from-green-500 to-blue-500 rounded-lg">
                    <MessageSquare className="h-5 w-5 text-white" />
                  </div>
                  <span className="hidden sm:block">Chat con Documentos</span>
                  <span className="sm:hidden">Chat</span>
                </h1>
                <p className="text-sm text-slate-400 hidden sm:block">
                  Haz preguntas sobre tus documentos cifrados
                </p>
              </div>
            </div>
            
            <div className="flex items-center gap-2 sm:gap-4">
              {systemStats && (
                <div className="flex items-center gap-2 sm:gap-4 text-sm text-slate-400">
                  <div className="flex items-center gap-1">
                    <Database className="h-4 w-4" />
                    <span>{systemStats.total_documents}</span>
                  </div>
                  <div className="hidden sm:flex items-center gap-1">
                    <Lock className="h-4 w-4 text-green-400" />
                    <span>Seguro</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                    <span className="hidden sm:block">Online</span>
                  </div>
                </div>
              )}
              
              {/* Desktop sidebar toggle 
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setSidebarOpen(!sidebarOpen)}
                className="text-slate-400 hover:text-white hover:bg-slate-700 hidden md:flex"
              >
                <Files className="h-4 w-4" />
              </Button>*/}
            </div>
          </div>
        </div>

        {/* Messages - Using proper viewport height */}
        <div 
          ref={chatContainerRef}
          className="flex-1 overflow-hidden"
          style={{ height: 'calc(100vh - 140px)' }}
        >
          <ScrollArea className="h-full">
            <div className="p-4 space-y-4 sm:space-y-6">
              {messages.length === 0 ? (
                <div className="text-center py-8 sm:py-16 px-4">
                  <div className="mb-6">
                    <div className="p-4 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full w-fit mx-auto mb-4">
                      <Sparkles className="h-6 w-6 sm:h-8 sm:w-8 text-white" />
                    </div>
                    <h3 className="text-xl sm:text-2xl font-bold text-white mb-2">
                      ¡Bienvenido!
                    </h3>
                    <p className="text-slate-400 text-sm sm:text-lg mb-6 sm:mb-8">
                      Sube documentos y hazme preguntas sobre su contenido.
                    </p>
                  </div>
                  
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 sm:gap-4 max-w-2xl mx-auto">
                    <div className="bg-slate-800/50 backdrop-blur-xl p-4 sm:p-6 rounded-xl border border-slate-700/50">
                      <div className="p-2 sm:p-3 bg-gradient-to-r from-blue-500 to-cyan-500 rounded-lg w-fit mx-auto mb-3">
                        <Shield className="h-4 w-4 sm:h-6 sm:w-6 text-white" />
                      </div>
                      <h4 className="font-semibold text-white mb-2 text-sm sm:text-base">Cifrado AES</h4>
                      <p className="text-xs sm:text-sm text-slate-400">Archivos cifrados automáticamente</p>
                    </div>
                    
                    <div className="bg-slate-800/50 backdrop-blur-xl p-4 sm:p-6 rounded-xl border border-slate-700/50">
                      <div className="p-2 sm:p-3 bg-gradient-to-r from-green-500 to-teal-500 rounded-lg w-fit mx-auto mb-3">
                        <Zap className="h-4 w-4 sm:h-6 sm:w-6 text-white" />
                      </div>
                      <h4 className="font-semibold text-white mb-2 text-sm sm:text-base">Respuestas Rápidas</h4>
                      <p className="text-xs sm:text-sm text-slate-400">Procesamiento en menos de 10s</p>
                    </div>
                    
                    <div className="bg-slate-800/50 backdrop-blur-xl p-4 sm:p-6 rounded-xl border border-slate-700/50">
                      <div className="p-2 sm:p-3 bg-gradient-to-r from-purple-500 to-pink-500 rounded-lg w-fit mx-auto mb-3">
                        <Files className="h-4 w-4 sm:h-6 sm:w-6 text-white" />
                      </div>
                      <h4 className="font-semibold text-white mb-2 text-sm sm:text-base">Múltiples Formatos</h4>
                      <p className="text-xs sm:text-sm text-slate-400">PDF, DOCX, imágenes, código</p>
                    </div>
                    
                    <div className="bg-slate-800/50 backdrop-blur-xl p-4 sm:p-6 rounded-xl border border-slate-700/50">
                      <div className="p-2 sm:p-3 bg-gradient-to-r from-orange-500 to-red-500 rounded-lg w-fit mx-auto mb-3">
                        <HardDrive className="h-4 w-4 sm:h-6 sm:w-6 text-white" />
                      </div>
                      <h4 className="font-semibold text-white mb-2 text-sm sm:text-base">Completamente Local</h4>
                      <p className="text-xs sm:text-sm text-slate-400">Sin conexiones externas</p>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="max-w-4xl mx-auto space-y-4 sm:space-y-6">
                  {messages.map((message) => (
                    <div
                      key={message.id}
                      className={`flex gap-3 sm:gap-4 ${
                        message.type === "user" ? "justify-end" : "justify-start"
                      }`}
                    >
                      {message.type === "assistant" && (
                        <div className="flex-shrink-0">
                          <div className="w-8 h-8 sm:w-10 sm:h-10 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full flex items-center justify-center">
                            <Bot className="h-4 w-4 sm:h-5 sm:w-5 text-white" />
                          </div>
                        </div>
                      )}
                      
                      <div
                        className={`max-w-[85%] sm:max-w-[80%] ${
                          message.type === "user"
                            ? "bg-gradient-to-r from-blue-500 to-purple-500 text-white"
                            : "bg-slate-800/50 backdrop-blur-xl border border-slate-700/50 text-white"
                        } rounded-2xl p-3 sm:p-4 shadow-lg`}
                      >
                        <div className="prose prose-invert max-w-none">
                          <p className="mb-0 whitespace-pre-wrap text-sm sm:text-base">{message.content}</p>
                        </div>
                        
                        <div className="flex items-center gap-2 mt-2 sm:mt-3 text-xs opacity-70">
                          <Clock className="h-3 w-3" />
                          <span>{message.timestamp}</span>
                          {message.response_time && (
                            <>
                              <span>•</span>
                              <span>{message.response_time.toFixed(1)}s</span>
                            </>
                          )}
                          {message.from_cache && (
                            <Badge variant="secondary" className="text-xs bg-slate-600 text-slate-200">
                              Caché
                            </Badge>
                          )}
                        </div>
                        
                        {message.sources && message.sources.length > 0 && (
                          <div className="mt-3 pt-3 border-t border-slate-600/50">
                            <h4 className="font-medium text-sm mb-2 text-slate-300">Fuentes:</h4>
                            <div className="space-y-1">
                              {message.sources.map((source, index) => (
                                <div key={index} className="flex items-center gap-2 text-xs bg-slate-700/30 p-2 rounded">
                                  {getFileTypeIcon(source.file_type || 'unknown')}
                                  <div className="flex-1 min-w-0 overflow-x-auto">
                                    <span className="font-medium text-slate-300 whitespace-nowrap" title={source.filename}>{source.filename}</span>
                                  </div>
                                  <span className="text-slate-400 flex-shrink-0">
                                    ({source.chunk_index})
                                  </span>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                      
                      {message.type === "user" && (
                        <div className="flex-shrink-0">
                          <div className="w-8 h-8 sm:w-10 sm:h-10 bg-gradient-to-r from-green-500 to-teal-500 rounded-full flex items-center justify-center">
                            <User className="h-4 w-4 sm:h-5 sm:w-5 text-white" />
                          </div>
                        </div>
                      )}
                    </div>
                  ))}
                  
                  {isLoading && (
                    <div className="flex gap-3 sm:gap-4 justify-start">
                      <div className="flex-shrink-0">
                        <div className="w-8 h-8 sm:w-10 sm:h-10 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full flex items-center justify-center">
                          <Bot className="h-4 w-4 sm:h-5 sm:w-5 text-white" />
                        </div>
                      </div>
                      <div className="bg-slate-800/50 backdrop-blur-xl border border-slate-700/50 rounded-2xl p-3 sm:p-4 shadow-lg">
                        <div className="flex items-center gap-2 text-sm text-slate-300">
                          <div className="flex gap-1">
                            <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce"></div>
                            <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style={{animationDelay: '0.1s'}}></div>
                            <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
                          </div>
                          <span>Procesando...</span>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
            <div ref={messagesEndRef} />
          </ScrollArea>
        </div>

        {/* Input Area - Fixed height */}
        <div className="bg-slate-800/50 backdrop-blur-xl border-t border-slate-700/50 p-4 flex-shrink-0">
          <div className="max-w-4xl mx-auto">
            <div className="relative">
              <div className="flex items-end gap-2 bg-slate-700/50 backdrop-blur-xl rounded-2xl border border-slate-600/50 p-2">
                <div className="flex-1 relative">
                  <textarea
                    ref={textareaRef}
                    value={input}
                    onChange={handleTextareaChange}
                    placeholder="Pregunta sobre tus documentos..."
                    className="w-full bg-transparent text-white placeholder-slate-400 border-none outline-none resize-none max-h-32 min-h-[40px] px-3 py-2 text-sm sm:text-base scrollbar-thin scrollbar-thumb-slate-600 scrollbar-track-transparent"
                    style={{ height: '40px' }}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" && !e.shiftKey) {
                        e.preventDefault();
                        handleSendMessage();
                      }
                    }}
                  />
                </div>
                <Button
                  onClick={handleSendMessage}
                  disabled={!input.trim() || isLoading}
                  className="bg-gradient-to-r from-blue-500 to-purple-500 hover:from-blue-600 hover:to-purple-600 text-white border-none h-10 w-10 p-0 rounded-xl flex-shrink-0"
                >
                  <Send className="h-4 w-4" />
                </Button>
              </div>
            </div>
            
            {documents.length === 0 && (
              <div className="mt-3 p-3 bg-amber-500/10 border border-amber-500/20 rounded-lg">
                <div className="flex items-center gap-2">
                  <AlertTriangle className="h-4 w-4 text-amber-400 flex-shrink-0" />
                  <p className="text-sm text-amber-200">
                    No hay documentos cargados. Sube algunos archivos para comenzar.
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
} 