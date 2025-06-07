"use client"

import type React from "react"

import { useState, useRef, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Textarea } from "@/components/ui/textarea"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"
import { ScrollArea } from "@/components/ui/scroll-area"
import {
  Sparkles,
  Upload,
  Copy,
  History,
  Trash2,
  ImageIcon,
  Video,
  FileText,
  Zap,
  Wand2,
  Sun,
  Moon,
  Lightbulb,
  PenTool,
  Brain,
  MessageSquare,
  CheckCircle,
} from "lucide-react"
import { useToast } from "@/hooks/use-toast"
import { useTheme } from "next-themes"
import Link from "next/link"

interface GeneratedPrompt {
  id: string
  originalIdea: string
  generatedPrompt: string
  model: string
  platform: string
  timestamp: Date
  files?: File[]
  qualityReport?: string
  interpretedKeywords?: string
  structuralFeedback?: string
  variations?: string[]
  ideas?: string[]
}

const LOCAL_MODELS = [
  // --- Modelos Locales (Ligeros, sin API Key) ---
  { value: "gpt2", label: "GPT-2", description: "Modelo base de 124M de parámetros. Rápido y versátil." },
  { value: "distilgpt2", label: "DistilGPT-2", description: "Versión más ligera (82M) de GPT-2, ideal para pruebas rápidas." },
  { value: "google-t5/t5-small", label: "T5-Small", description: "Modelo de Google (60M) para tareas de texto-a-texto." },
  { value: "EleutherAI/gpt-neo-125M", label: "GPT-Neo 125M", description: "Alternativa a GPT-2 entrenada por EleutherAI." },
]

const PLATFORMS = [
  { value: "chatgpt", label: "ChatGPT", icon: "🤖", color: "from-green-400 to-emerald-600" },
  { value: "cursor", label: "Cursor", icon: "⚡", color: "from-blue-400 to-cyan-600" },
  { value: "v0", label: "v0", icon: "✨", color: "from-purple-400 to-pink-600" },
  { value: "sora", label: "Sora", icon: "🎬", color: "from-red-400 to-orange-600" },
  { value: "claude", label: "Claude AI", icon: "🧠", color: "from-indigo-400 to-purple-600" },
  { value: "gemini", label: "Gemini", icon: "💎", color: "from-yellow-400 to-amber-600" },
  { value: "firefly", label: "Adobe Firefly", icon: "🔥", color: "from-pink-400 to-rose-600" },
]

export default function PromptGenPage() {
  const [idea, setIdea] = useState("")
  const [selectedModel, setSelectedModel] = useState("gpt2")
  const [selectedPlatform, setSelectedPlatform] = useState("chatgpt")
  const [generatedPrompt, setGeneratedPrompt] = useState("")
  const [isGenerating, setIsGenerating] = useState(false)
  const [history, setHistory] = useState<GeneratedPrompt[]>([])
  const [uploadedFiles, setUploadedFiles] = useState<File[]>([])
  const fileInputRef = useRef<HTMLInputElement>(null)
  const { toast } = useToast()
  const { theme, setTheme } = useTheme()
  const [mounted, setMounted] = useState(false)
  const [isSidebarMinimized, setIsSidebarMinimized] = useState(false)

  const messagesEndRef = useRef<null | HTMLDivElement>(null);

  const [thinkingSteps, setThinkingSteps] = useState<string[]>([])
  const [promptQuality, setPromptQuality] = useState<string | null>(null)
  const [interpretedKeywords, setInterpretedKeywords] = useState<string | null>(null)
  const [structuralFeedback, setStructuralFeedback] = useState<string | null>(null)
  const [generatedVariations, setGeneratedVariations] = useState<string[]>([])
  const [generatedIdeas, setGeneratedIdeas] = useState<string[] | null>(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [currentGeneratedItem, setCurrentGeneratedItem] = useState<GeneratedPrompt | null>(null)

  // Efecto para manejar la hidratación
  useEffect(() => {
    setMounted(true)
  }, [])

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files || [])
    setUploadedFiles((prev) => [...prev, ...files])
    toast({
      title: "Archivos subidos",
      description: `${files.length} archivo(s) añadido(s) exitosamente`,
    })
  }

  const removeFile = (index: number) => {
    setUploadedFiles((prev) => prev.filter((_, i) => i !== index))
  }

  const API_BASE_URL = "http://localhost:5000/api"

  const handleGenerateAndAnalyze = async () => {
    if (!idea.trim()) {
      toast({
        title: "Idea requerida",
        description: "Por favor, introduce tu idea o prompt.",
        variant: "destructive",
      })
      return
    }

    const tempId = Date.now().toString()
    const platformData = PLATFORMS.find((p) => p.value === selectedPlatform)
    const modelData = LOCAL_MODELS.find((m) => m.value === selectedModel)
    
    const userMessageItem: GeneratedPrompt = {
      id: tempId,
      originalIdea: idea,
      generatedPrompt: "", // Inicialmente vacío
      model: modelData?.label || selectedModel,
      platform: platformData?.label || selectedPlatform,
      timestamp: new Date(),
      files: uploadedFiles.length > 0 ? [...uploadedFiles] : undefined,
    }

    setHistory((prev) => [...prev, userMessageItem])
    setCurrentGeneratedItem(userMessageItem)
    setIdea("")
    setUploadedFiles([])

    setIsGenerating(true)
    setIsAnalyzing(true)
    setThinkingSteps([])
    setPromptQuality(null)
    setInterpretedKeywords(null)
    setStructuralFeedback(null)
    setGeneratedVariations([])
    setGeneratedIdeas(null)
    setGeneratedPrompt("")
    // No establecer setCurrentGeneratedItem a null aquí

    const addThinkingStep = (step: string) => {
      setThinkingSteps((prev) => [...prev, step])
    }

    // Variables para almacenar los resultados directos de las APIs
    let qualityDataResponse: any = null;
    let feedbackDataResponse: any = null;
    let variationsDataResponse: any = null;
    let ideasDataResponse: any = null;

    addThinkingStep("Analizando la calidad del prompt inicial...")
    try {
      const qualityResponse = await fetch(`${API_BASE_URL}/analyze_quality`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: idea }),
      })
      if (!qualityResponse.ok) throw new Error("Error al analizar la calidad")
      qualityDataResponse = await qualityResponse.json(); // Guardar respuesta directa
      addThinkingStep("Calidad analizada. Palabras clave interpretadas: " + (qualityDataResponse?.interpreted_keywords || 'N/A'))
    } catch (error) {
      console.error("Error en análisis de calidad:", error)
      toast({ title: "Error", description: "Fallo al analizar la calidad del prompt.", variant: "destructive" })
      addThinkingStep("Error al analizar la calidad.")
    }

    addThinkingStep("Obteniendo feedback estructural...")
    try {
      const feedbackResponse = await fetch(`${API_BASE_URL}/get_feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: idea, model_name: selectedModel }),
      })
      if (!feedbackResponse.ok) throw new Error("Error al obtener feedback")
      feedbackDataResponse = await feedbackResponse.json(); // Guardar respuesta directa
      addThinkingStep("Feedback estructural recibido.")
    } catch (error) {
      console.error("Error en feedback estructural:", error)
      toast({ title: "Error", description: "Fallo al obtener feedback estructural.", variant: "destructive" })
      addThinkingStep("Error al obtener feedback estructural.")
    }

    addThinkingStep("Generando variaciones del prompt...")
    try {
      const variationsResponse = await fetch(`${API_BASE_URL}/generate_variations`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: idea, model_name: selectedModel, num_variations: 3 }),
      })
      if (!variationsResponse.ok) throw new Error("Error al generar variaciones")
      variationsDataResponse = await variationsResponse.json(); // Guardar respuesta directa
      if (variationsDataResponse.variations && variationsDataResponse.variations.length > 0) {
        setGeneratedPrompt(variationsDataResponse.variations[0])
        addThinkingStep("Variaciones generadas. Mostrando la primera como prompt mejorado.")
      } else {
        addThinkingStep("No se pudieron generar variaciones claras.")
        setGeneratedPrompt(idea)
      }
    } catch (error) {
      console.error("Error en generación de variaciones:", error)
      toast({ title: "Error", description: "Fallo al generar variaciones del prompt.", variant: "destructive" })
      addThinkingStep("Error al generar variaciones.")
    }

    addThinkingStep("Generando ideas relacionadas...")
    try {
      const ideasResponse = await fetch(`${API_BASE_URL}/generate_ideas`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: idea, model_name: selectedModel, num_ideas: 3 }),
      })
      if (!ideasResponse.ok) throw new Error("Error al generar ideas")
      ideasDataResponse = await ideasResponse.json(); // Guardar respuesta directa
      addThinkingStep("Ideas generadas.")
    } catch (error) {
      console.error("Error en generación de ideas:", error)
      toast({ title: "Error", description: "Fallo al generar ideas.", variant: "destructive" })
      addThinkingStep("Error al generar ideas.")
    }
    
    addThinkingStep("Proceso completado.")
    setIsGenerating(false)
    setIsAnalyzing(false)

    // Usar las respuestas directas de la API para construir newItem
    const currentGeneratedPromptText = (variationsDataResponse?.variations && variationsDataResponse.variations.length > 0) 
                                      ? variationsDataResponse.variations[0] 
                                      : idea;

    const updatedItemData = {
      generatedPrompt: currentGeneratedPromptText,
      qualityReport: qualityDataResponse?.quality_report ?? undefined,
      interpretedKeywords: qualityDataResponse?.interpreted_keywords ?? undefined,
      structuralFeedback: feedbackDataResponse?.feedback ?? undefined,
      variations: variationsDataResponse?.variations ?? undefined,
      ideas: ideasDataResponse?.ideas ?? undefined,
    }

    setHistory((prev) => 
      prev.map((item) => (item.id === tempId ? { ...item, ...updatedItemData } : item))
    )
    
    setCurrentGeneratedItem((prev) => (prev && prev.id === tempId ? { ...prev, ...updatedItemData } : prev))

    toast({
      title: "¡Proceso completado!",
      description: "Se ha analizado y mejorado el prompt.",
    })
  }

  useEffect(() => {
    // Comentado para evitar el scroll automático inicial
    // messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [history]);

  const copyToClipboard = (text: string) => {
    if (!text?.trim()) {
      toast({
        title: "Nada que copiar",
        description: "El prompt mejorado aún no está disponible.",
        variant: "destructive",
      })
      return
    }
    navigator.clipboard.writeText(text)
    toast({
      title: "Copiado",
      description: "Prompt mejorado copiado al portapapeles.",
    })
  }

  const clearHistory = () => {
    setHistory([])
    toast({
      title: "Historial borrado",
      description: "Todos los prompts han sido eliminados",
    })
  }

  const getFileIcon = (file: File) => {
    if (file.type.startsWith("image/")) return <ImageIcon className="w-4 h-4" />
    if (file.type.startsWith("video/")) return <Video className="w-4 h-4" />
    return <FileText className="w-4 h-4" />
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 via-white to-gray-100 dark:from-gray-900 dark:via-black dark:to-gray-900 transition-colors duration-300">
      {/* Header */}
      <header className="border-b border-gray-200 dark:border-gray-800 bg-white/80 dark:bg-black/50 backdrop-blur-xl transition-colors duration-300">
        <div className="container mx-auto px-4 py-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="relative">
                <Wand2 className="w-8 h-8 text-cyan-500 dark:text-cyan-400" />
                <div className="absolute inset-0 w-8 h-8 text-cyan-500 dark:text-cyan-400 animate-pulse" />
              </div>
              <div>
                <h1 className="text-xl font-bold bg-gradient-to-r from-cyan-500 via-purple-500 to-pink-500 dark:from-cyan-400 dark:via-purple-400 dark:to-pink-400 bg-clip-text text-transparent">
                  PromptGen
                </h1>
              </div>
            </div>

            <Button
              variant="ghost"
              size="sm"
              onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
              className="h-9 w-9 p-0 text-gray-600 hover:text-gray-900 hover:bg-gray-100 dark:text-gray-400 dark:hover:text-white dark:hover:bg-gray-800/50 transition-colors duration-200"
              title={mounted ? (theme === "dark" ? "Cambiar a modo claro" : "Cambiar a modo oscuro") : "Cambiar tema"}
            >
              {mounted ? (
                theme === "dark" ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />
              ) : (
                <div className="w-5 h-5" /> // Placeholder mientras se carga
              )}
            </Button>
          </div>
        </div>
      </header>

      <div className="flex h-[calc(100vh-76px)]">
        {/* Sidebar - Historial (similar a ChatGPT) */}
        <div
          className={`${isSidebarMinimized ? "w-16" : "w-80"} border-r border-gray-200 dark:border-gray-800 bg-gray-50/80 dark:bg-gray-900/80 overflow-hidden hidden lg:block transition-all duration-300`}
        >
          <div className="p-4 h-full">
            <div className="flex items-center mb-4">
              {!isSidebarMinimized && (
                <h2 className="text-lg font-medium text-gray-900 dark:text-white flex items-center gap-2">
                  <History className="w-4 h-4 text-green-500 dark:text-green-400" />
                  Historial
                </h2>
              )}
              <div className={`flex items-center gap-1 ${!isSidebarMinimized ? "ml-auto" : "w-full justify-center"}`}>
                {!isSidebarMinimized && history.length > 0 && (
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={clearHistory}
                    className="h-8 w-8 p-0 text-gray-600 hover:text-gray-900 dark:text-gray-400 dark:hover:text-white"
                    title="Limpiar historial"
                  >
                    <Trash2 className="w-4 h-4" />
                  </Button>
                )}
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setIsSidebarMinimized(!isSidebarMinimized)}
                  className="h-8 w-8 p-0 text-gray-600 hover:text-gray-900 dark:text-gray-400 dark:hover:text-white"
                  title={isSidebarMinimized ? "Expandir historial" : "Minimizar historial"}
                >
                  {isSidebarMinimized ? (
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  ) : (
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                    </svg>
                  )}
                </Button>
              </div>
            </div>
            {isSidebarMinimized ? (
              <div className="flex flex-col items-center space-y-2">
                <div className="w-8 h-8 rounded-full bg-gray-200 dark:bg-gray-800 flex items-center justify-center">
                  <History className="w-4 h-4 text-green-500 dark:text-green-400" />
                </div>
                <div className="text-xs text-gray-500 dark:text-gray-500 text-center">{history.length}</div>
              </div>
            ) : (
              <ScrollArea className="h-[calc(100vh-150px)]">
                {history.length === 0 ? (
                  <div className="text-center py-8 text-gray-500 dark:text-gray-500">
                    <History className="w-12 h-12 mx-auto mb-4 opacity-50" />
                    <p>No hay prompts generados aún</p>
                  </div>
                ) : (
                  <div className="space-y-3 pr-2">
                    {history.map((item) => (
                      <Card
                        key={item.id}
                        className="border-gray-200 dark:border-gray-700 bg-white/50 dark:bg-gray-800/30"
                      >
                        <CardContent className="p-3">
                          <div className="space-y-2">
                            <div className="flex items-center justify-between">
                              <div className="flex items-center gap-1">
                                <Badge
                                  variant="secondary"
                                  className="bg-purple-100 text-purple-700 dark:bg-purple-900/50 dark:text-purple-300 text-xs"
                                >
                                  {item.platform}
                                </Badge>
                              </div>
                              <Button
                                variant="ghost"
                                size="sm"
                                onClick={() => copyToClipboard(item.generatedPrompt)}
                                disabled={!item.generatedPrompt}
                                className="h-6 w-6 p-0 text-gray-600 hover:text-gray-900 dark:text-gray-400 dark:hover:text-white disabled:opacity-50"
                                title="Copiar prompt mejorado"
                              >
                                <Copy className="w-3 h-3" />
                              </Button>
                            </div>
                            <p className="text-sm text-gray-700 dark:text-gray-300 line-clamp-2">{item.originalIdea}</p>
                            <p className="text-xs text-gray-500 dark:text-gray-500">
                              {item.timestamp.toLocaleString()}
                            </p>
                          </div>
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                )}
              </ScrollArea>
            )}
          </div>
        </div>

        {/* Área principal de contenido */}
        <div className="flex-1 overflow-auto">
          <div className="h-full flex flex-col">
            {/* Área de conversación */}
            <div className="flex-1 overflow-auto px-4 py-8">
              <div className="max-w-4xl mx-auto space-y-6">
                {/* Mostrar historial de conversación */}
                {history.map((item) => (
                  <div key={item.id} className="space-y-4">
                    {/* Mensaje del usuario */}
                    <div className="flex justify-end">
                      <div className="max-w-[80%] bg-gradient-to-r from-cyan-500 to-purple-500 text-white rounded-2xl px-4 py-3">
                        <p className="text-sm">{item.originalIdea}</p>
                        {item.files && item.files.length > 0 && (
                          <div className="flex flex-wrap gap-1 mt-2">
                            {item.files.map((file, index) => (
                              <Badge key={index} variant="secondary" className="bg-white/20 text-white text-xs">
                                {getFileIcon(file)}
                                <span className="ml-1">{file.name}</span>
                              </Badge>
                            ))}
                          </div>
                        )}
                        <div className="flex items-center gap-1 mt-2 text-xs opacity-75">
                          <span>{item.model}</span>
                          <span>•</span>
                          <span>{item.platform}</span>
                        </div>
                      </div>
                    </div>

                    {/* Respuesta de la IA */}
                    {item.generatedPrompt && (
                    <div className="flex justify-start">
                      <div className="max-w-[80%] bg-gray-100 dark:bg-gray-800/50 border border-gray-200 dark:border-gray-700 text-gray-900 dark:text-white rounded-2xl px-4 py-3 space-y-3">
                        {/* Thinking Steps - Solo para la generación actual si es el último item del historial */}
                        { currentGeneratedItem && item.id === currentGeneratedItem.id && thinkingSteps.length > 0 && (
                          <Card className="bg-gray-50 dark:bg-gray-700/50 border-gray-200 dark:border-gray-600">
                            <CardHeader className="p-3">
                              <CardTitle className="text-sm font-medium flex items-center gap-2 text-blue-600 dark:text-blue-400">
                                <Brain className="w-4 h-4" />
                                Pensando...
                              </CardTitle>
                            </CardHeader>
                            <CardContent className="p-3 text-xs">
                              <ul className="space-y-1">
                                {thinkingSteps.map((step, index) => (
                                  <li key={index} className="flex items-center gap-2">
                                    {index === thinkingSteps.length - 1 && (isAnalyzing || isGenerating) ? (
                                        <div className="w-3 h-3 border-2 border-gray-300 dark:border-gray-400 border-t-blue-500 dark:border-t-blue-400 rounded-full animate-spin" />
                                    ) : (
                                        <CheckCircle className="w-3 h-3 text-green-500" />
                                    )}
                                    <span>{step}</span>
                                  </li>
                                ))}
                              </ul>
                            </CardContent>
                          </Card>
                        )}

                        {item.qualityReport && (
                           <Card className="bg-yellow-50 dark:bg-yellow-900/30 border-yellow-200 dark:border-yellow-700">
                            <CardHeader className="p-3">
                              <CardTitle className="text-sm font-medium flex items-center gap-2 text-yellow-700 dark:text-yellow-400">
                                <Sparkles className="w-4 h-4" />
                                Análisis de Calidad del Prompt
                              </CardTitle>
                            </CardHeader>
                            <CardContent className="p-3 text-xs space-y-1">
                              <p className="whitespace-pre-wrap">{item.qualityReport}</p>
                              {(item.interpretedKeywords) && (
                                <p><strong>Palabras Clave:</strong> {item.interpretedKeywords}</p>
                              )}
                            </CardContent>
                          </Card>
                        )}

                        {item.structuralFeedback && (
                          <Card className="bg-green-50 dark:bg-green-900/30 border-green-200 dark:border-green-700">
                            <CardHeader className="p-3">
                              <CardTitle className="text-sm font-medium flex items-center gap-2 text-green-700 dark:text-green-500">
                                <MessageSquare className="w-4 h-4" />
                                Feedback Estructural
                              </CardTitle>
                            </CardHeader>
                            <CardContent className="p-3 text-xs whitespace-pre-wrap">
                              {item.structuralFeedback}
                            </CardContent>
                          </Card>
                        )}
                        
                        <div className="flex items-center gap-2 mb-1 mt-2">
                          <Zap className="w-4 h-4 text-purple-500 dark:text-purple-400" />
                          <span className="text-sm font-medium text-purple-600 dark:text-purple-400">
                            Prompt Mejorado/Sugerido
                          </span>
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => copyToClipboard(item.generatedPrompt)}
                            className="h-6 w-6 p-0 text-gray-600 hover:text-gray-900 dark:text-gray-400 dark:hover:text-white ml-auto"
                          >
                            <Copy className="w-3 h-3" />
                          </Button>
                        </div>
                        <p className="text-gray-800 dark:text-gray-200 leading-relaxed whitespace-pre-wrap text-sm">
                          {item.generatedPrompt}
                        </p>

                        {item.variations && item.variations.length > 0 && (
                          <Card className="bg-indigo-50 dark:bg-indigo-900/30 border-indigo-200 dark:border-indigo-700">
                            <CardHeader className="p-3">
                              <CardTitle className="text-sm font-medium flex items-center gap-2 text-indigo-700 dark:text-indigo-400">
                                <PenTool className="w-4 h-4" />
                                Otras Variaciones Sugeridas
                              </CardTitle>
                            </CardHeader>
                            <CardContent className="p-3 text-xs space-y-1">
                              {item.variations.map((variation, index) => (
                                <div key={index} className="flex items-start gap-2 p-1.5 rounded hover:bg-indigo-100 dark:hover:bg-indigo-800/50">
                                  <p className="flex-grow whitespace-pre-wrap">- {variation}</p>
                                  <Button variant="ghost" size="icon" className="h-5 w-5 p-0" onClick={() => copyToClipboard(variation)}>
                                    <Copy className="w-2.5 h-2.5" />
                                  </Button>
                                </div>
                              ))}
                            </CardContent>
                          </Card>
                        )}

                        {item.ideas && Array.isArray(item.ideas) && item.ideas.length > 0 && (
                          <Card className="bg-pink-50 dark:bg-pink-900/30 border-pink-200 dark:border-pink-700">
                            <CardHeader className="p-3">
                              <CardTitle className="text-sm font-medium flex items-center gap-2 text-pink-700 dark:text-pink-400">
                                <Lightbulb className="w-4 h-4" />
                                Ideas Generadas
                              </CardTitle>
                            </CardHeader>
                            <CardContent className="p-3 text-xs space-y-1">
                              <ul className="list-disc list-inside">
                                {item.ideas.map((idea, index) => (
                                  <li key={index}>{idea}</li>
                                ))}
                              </ul>
                            </CardContent>
                          </Card>
                        )}

                        <div className="flex items-center gap-2 mt-3">
                          <Badge
                            variant="secondary"
                            className="bg-purple-100 text-purple-700 dark:bg-purple-900/50 dark:text-purple-300 text-xs"
                          >
                            {item.platform}
                          </Badge>
                          <Badge
                            variant="secondary"
                            className="bg-blue-100 text-blue-700 dark:bg-blue-900/50 dark:text-blue-300 text-xs"
                          >
                            {item.model}
                          </Badge>
                        </div>
                      </div>
                    </div>
                    )}
                  </div>
                ))}

                {/* Mostrar mensaje de carga si es una nueva generación y no hay historial O si es el primer item del historial y está generando */}
                {(isGenerating || isAnalyzing) && history.length === 0 && (
                  <div className="flex justify-start">
                    <div className="max-w-[80%] bg-gray-100 dark:bg-gray-800/50 border border-gray-200 dark:border-gray-700 text-gray-900 dark:text-white rounded-2xl px-4 py-3">
                      <div className="flex items-center gap-2">
                        <div className="w-4 h-4 border-2 border-gray-300 dark:border-gray-400 border-t-cyan-500 dark:border-t-cyan-400 rounded-full animate-spin" />
                        <span className="text-sm text-gray-700 dark:text-gray-300">
                          {isAnalyzing ? "Analizando y generando..." : "Procesando..."}
                        </span>
                      </div>
                    </div>
                  </div>
                )}

                {/* Mensaje de bienvenida si no hay historial y no se está generando nada */}
                {history.length === 0 && !isGenerating && !isAnalyzing && (
                  <div className="text-center py-16">
                    <div className="relative mb-6">
                      <Wand2 className="w-16 h-16 mx-auto text-cyan-500 dark:text-cyan-400" />
                      <div className="absolute inset-0 w-16 h-16 mx-auto text-cyan-500 dark:text-cyan-400 animate-pulse" />
                    </div>
                    <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">¡Bienvenido a PromptGen!</h2>
                    <p className="text-gray-600 dark:text-gray-400 max-w-md mx-auto">
                      Describe tu idea y te ayudaré a crear el prompt perfecto para cualquier plataforma de IA.
                    </p>
                  </div>
                )}
                {/* Elemento vacío para el auto-scroll */}
                <div ref={messagesEndRef} />
              </div>
            </div>

            {/* Área de entrada fija en la parte inferior */}
            <div className="border-t border-gray-200 dark:border-gray-800 bg-white/50 dark:bg-gray-900/50 backdrop-blur-xl">
              <div className="max-w-4xl mx-auto px-4 py-4">
                <Card className="border-gray-200 dark:border-gray-700 bg-white/80 dark:bg-gray-800/50">
                  <CardContent className="p-4">
                    {/* Mostrar archivos adjuntos si existen */}
                    {uploadedFiles.length > 0 && (
                      <div className="bg-gray-100 dark:bg-gray-700/30 rounded-lg p-3 border border-gray-200 dark:border-gray-600 mb-3">
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                            Archivos adjuntos
                          </span>
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => setUploadedFiles([])}
                            className="h-6 w-6 p-0 text-gray-600 hover:text-red-600 dark:text-gray-400 dark:hover:text-red-400"
                          >
                            <Trash2 className="w-3 h-3" />
                          </Button>
                        </div>
                        <div className="flex flex-wrap gap-2">
                          {uploadedFiles.map((file, index) => (
                            <Badge
                              key={index}
                              variant="secondary"
                              className="bg-gray-200 text-gray-700 dark:bg-gray-700 dark:text-gray-300 flex items-center gap-1"
                            >
                              {getFileIcon(file)}
                              <span className="max-w-[100px] truncate">{file.name}</span>
                              <button
                                onClick={() => removeFile(index)}
                                className="ml-1 hover:text-red-500 dark:hover:text-red-400"
                              >
                                ×
                              </button>
                            </Badge>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Área de entrada con botones integrados */}
                    <div className="relative">
                      {/* Minimalist selectors bar */}
                      <div className="flex items-center gap-2 mb-3">
                        <Select value={selectedModel} onValueChange={setSelectedModel}>
                          <SelectTrigger className="w-auto h-8 px-3 bg-gray-100 dark:bg-gray-700/50 border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300 text-xs">
                            <SelectValue placeholder="Modelo" />
                          </SelectTrigger>
                          <SelectContent className="bg-white dark:bg-gray-800 border-gray-200 dark:border-gray-700">
                            {LOCAL_MODELS.map((model) => (
                              <SelectItem
                                key={model.value}
                                value={model.value}
                                className="text-gray-900 dark:text-white text-xs"
                              >
                                <div className="flex flex-col items-start">
                                  <span>{model.label}</span>
                                  <span className="text-gray-500 dark:text-gray-400 text-xs">{model.description}</span>
                                </div>
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>

                        <Select value={selectedPlatform} onValueChange={setSelectedPlatform}>
                          <SelectTrigger className="w-auto h-8 px-3 bg-gray-100 dark:bg-gray-700/50 border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300 text-xs">
                            <SelectValue placeholder="Plataforma" />
                          </SelectTrigger>
                          <SelectContent className="bg-white dark:bg-gray-800 border-gray-200 dark:border-gray-700">
                            {PLATFORMS.map((platform) => (
                              <SelectItem
                                key={platform.value}
                                value={platform.value}
                                className="text-gray-900 dark:text-white text-xs"
                              >
                                <div className="flex items-center gap-2">
                                  <span>{platform.icon}</span>
                                  <span>{platform.label}</span>
                                </div>
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>

                      <Textarea
                        placeholder="Describe tu idea o concepto..."
                        value={idea}
                        onChange={(e) => setIdea(e.target.value)}
                        className="min-h-[60px] max-h-[200px] pr-20 bg-gray-100 dark:bg-gray-700/50 border-gray-300 dark:border-gray-600 text-gray-900 dark:text-white placeholder:text-gray-500 dark:placeholder:text-gray-400 focus:border-cyan-500 dark:focus:border-cyan-400 focus:ring-cyan-500/20 dark:focus:ring-cyan-400/20 resize-none"
                        onKeyDown={(e) => {
                          if (e.key === "Enter" && !e.shiftKey) {
                            e.preventDefault()
                            handleGenerateAndAnalyze()
                          }
                        }}
                      />
                      <div className="absolute bottom-2 right-2 flex items-center gap-1">
                        <Button
                          type="button"
                          variant="ghost"
                          onClick={() => fileInputRef.current?.click()}
                          className="h-8 w-8 p-0 text-gray-600 hover:text-gray-900 dark:text-gray-400 dark:hover:text-white rounded-full"
                          title="Adjuntar archivos"
                        >
                          <Upload className="w-4 h-4" />
                        </Button>
                        <Button
                          variant="default"
                          onClick={handleGenerateAndAnalyze}
                          disabled={isGenerating || isAnalyzing || !idea.trim()}
                          className="h-8 px-3 bg-gradient-to-r from-cyan-500 to-purple-600 hover:from-cyan-600 hover:to-purple-700 text-white dark:from-cyan-400 dark:to-purple-500 dark:hover:from-cyan-500 dark:hover:to-purple-600 transition-all duration-200 shadow-md hover:shadow-lg transform hover:scale-105 disabled:opacity-60 disabled:transform-none disabled:shadow-none"
                        >
                          {isGenerating || isAnalyzing ? (
                            <div className="w-4 h-4 border-2 border-white/50 border-t-white rounded-full animate-spin mr-2" />
                          ) : (
                            <Sparkles className="w-4 h-4 mr-2" />
                          )}
                          {/*{isAnalyzing ? "Analizando..." : (isGenerating ? "Generando..." : "Mejorar")}*/}

                        </Button>
                      </div>
                      <input
                        ref={fileInputRef}
                        type="file"
                        multiple
                        accept="image/*,video/*,.pdf,.doc,.docx,.txt"
                        onChange={handleFileUpload}
                        className="hidden"
                      />
                    </div>
                  </CardContent>
                </Card>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Footer */}
     {/* <footer className="border-t border-gray-200 dark:border-gray-800 bg-white/50 dark:bg-black/50 backdrop-blur-xl mt-16">
        <div className="container mx-auto px-4 py-8">
          <div className="flex flex-col md:flex-row justify-between items-center gap-4">
            <div className="flex items-center gap-3">
              <Wand2 className="w-6 h-6 text-cyan-500 dark:text-cyan-400" />
              <div>
                <p className="text-gray-900 dark:text-white font-medium">PromptGen</p>
                <p className="text-gray-600 dark:text-gray-400 text-sm">Potenciado por IA avanzada</p>
              </div>
            </div>

            <div className="flex items-center gap-6 text-sm">
              <Link
                href="/privacy-policy"
                className="text-gray-600 hover:text-cyan-500 dark:text-gray-400 dark:hover:text-cyan-400 transition-colors"
              >
                Política de Privacidad
              </Link>
              <Separator orientation="vertical" className="h-4 bg-gray-300 dark:bg-gray-700" />
              <p className="text-gray-500 dark:text-gray-500">© 2025 PromptGen. Todos los derechos reservados.</p>
            </div>
          </div>
        </div>
      </footer> */}
    </div>
  )
}
