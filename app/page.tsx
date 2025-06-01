"use client"

import type React from "react"

import { useState, useRef } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
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
} from "lucide-react"
import { useToast } from "@/hooks/use-toast"
import { useTheme } from "next-themes"

interface GeneratedPrompt {
  id: string
  originalIdea: string
  generatedPrompt: string
  model: string
  platform: string
  timestamp: Date
  files?: File[]
}

const LLM_MODELS = [
  { value: "gpt-4o", label: "GPT-4o", description: "Más avanzado y multimodal" },
  { value: "claude-3.5-sonnet", label: "Claude 3.5 Sonnet", description: "Excelente para análisis y código" },
  { value: "gemini-pro", label: "Gemini Pro", description: "Potente modelo de Google" },
  { value: "llama-3.1", label: "Llama 3.1", description: "Modelo open source avanzado" },
  { value: "grok-beta", label: "Grok Beta", description: "Modelo de xAI con humor" },
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
  const [selectedModel, setSelectedModel] = useState("")
  const [selectedPlatform, setSelectedPlatform] = useState("")
  const [generatedPrompt, setGeneratedPrompt] = useState("")
  const [isGenerating, setIsGenerating] = useState(false)
  const [history, setHistory] = useState<GeneratedPrompt[]>([])
  const [uploadedFiles, setUploadedFiles] = useState<File[]>([])
  const fileInputRef = useRef<HTMLInputElement>(null)
  const { toast } = useToast()
  const [isSidebarMinimized, setIsSidebarMinimized] = useState(false)
  const { theme, setTheme } = useTheme()

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

  const generatePrompt = async () => {
    if (!idea.trim() || !selectedModel || !selectedPlatform) {
      toast({
        title: "Campos requeridos",
        description: "Por favor completa todos los campos requeridos",
        variant: "destructive",
      })
      return
    }

    setIsGenerating(true)

    // Simular generación de prompt (aquí integrarías con AI SDK)
    await new Promise((resolve) => setTimeout(resolve, 2000))

    const platformData = PLATFORMS.find((p) => p.value === selectedPlatform)
    const modelData = LLM_MODELS.find((m) => m.value === selectedModel)

    let prompt = ""

    // Generar prompt específico según la plataforma
    switch (selectedPlatform) {
      case "chatgpt":
        prompt = `Actúa como un experto en ${idea}. Proporciona una respuesta detallada y estructurada que incluya ejemplos prácticos, mejores prácticas y consideraciones importantes. Organiza tu respuesta de manera clara y profesional.`
        break
      case "cursor":
        prompt = `Como desarrollador experto, ayúdame a implementar ${idea}. Incluye código limpio, comentarios explicativos, mejores prácticas de desarrollo y considera la escalabilidad y mantenibilidad del código.`
        break
      case "v0":
        prompt = `Crea una interfaz moderna y responsive para ${idea}. Utiliza componentes de shadcn/ui, Tailwind CSS, y sigue las mejores prácticas de UX/UI. El diseño debe ser accesible y visualmente atractivo.`
        break
      case "sora":
        prompt = `Genera un video que muestre ${idea}. Descripción visual detallada: estilo cinematográfico, iluminación profesional, movimientos de cámara fluidos, duración de 30-60 segundos, calidad 4K.`
        break
      case "claude":
        prompt = `Analiza profundamente ${idea} desde múltiples perspectivas. Proporciona un análisis estructurado, considera pros y contras, implicaciones a largo plazo y recomendaciones basadas en evidencia.`
        break
      case "gemini":
        prompt = `Explora ${idea} de manera integral. Incluye contexto histórico, estado actual, tendencias futuras y conexiones interdisciplinarias. Presenta la información de forma clara y bien organizada.`
        break
      case "firefly":
        prompt = `Crea una imagen que represente ${idea}. Estilo: profesional, alta calidad, composición equilibrada, colores vibrantes, iluminación dramática, resolución 4K, formato 16:9.`
        break
    }

    setGeneratedPrompt(prompt)

    // Agregar al historial
    const newPrompt: GeneratedPrompt = {
      id: Date.now().toString(),
      originalIdea: idea,
      generatedPrompt: prompt,
      model: modelData?.label || selectedModel,
      platform: platformData?.label || selectedPlatform,
      timestamp: new Date(),
      files: uploadedFiles.length > 0 ? [...uploadedFiles] : undefined,
    }

    setHistory((prev) => [newPrompt, ...prev])
    setIsGenerating(false)

    toast({
      title: "¡Prompt generado!",
      description: `Optimizado para ${platformData?.label} usando ${modelData?.label}`,
    })
  }

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text)
    toast({
      title: "Copiado",
      description: "Prompt copiado al portapapeles",
    })
  }

  const clearHistory = () => {
    setHistory([])
    toast({
      title: "Historial limpiado",
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
              title={theme === "dark" ? "Cambiar a modo claro" : "Cambiar a modo oscuro"}
            >
              {theme === "dark" ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
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
            <div className="flex items-center justify-between mb-4">
              {!isSidebarMinimized && (
                <h2 className="text-lg font-medium text-gray-900 dark:text-white flex items-center gap-2">
                  <History className="w-4 h-4 text-green-500 dark:text-green-400" />
                  Historial
                </h2>
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
              {!isSidebarMinimized && history.length > 0 && (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={clearHistory}
                  className="h-7 w-7 p-0 text-gray-600 hover:text-gray-900 dark:text-gray-400 dark:hover:text-white"
                >
                  <Trash2 className="w-4 h-4" />
                </Button>
              )}
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
                                className="h-6 w-6 p-0 text-gray-600 hover:text-gray-900 dark:text-gray-400 dark:hover:text-white"
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
                    <div className="flex justify-start">
                      <div className="max-w-[80%] bg-gray-100 dark:bg-gray-800/50 border border-gray-200 dark:border-gray-700 text-gray-900 dark:text-white rounded-2xl px-4 py-3">
                        <div className="flex items-center gap-2 mb-2">
                          <Zap className="w-4 h-4 text-yellow-500 dark:text-yellow-400" />
                          <span className="text-sm font-medium text-yellow-600 dark:text-yellow-400">
                            Prompt Generado
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
                  </div>
                ))}

                {/* Mostrar mensaje de carga si está generando */}
                {isGenerating && (
                  <div className="flex justify-start">
                    <div className="max-w-[80%] bg-gray-100 dark:bg-gray-800/50 border border-gray-200 dark:border-gray-700 text-gray-900 dark:text-white rounded-2xl px-4 py-3">
                      <div className="flex items-center gap-2">
                        <div className="w-4 h-4 border-2 border-gray-300 dark:border-gray-400 border-t-cyan-500 dark:border-t-cyan-400 rounded-full animate-spin" />
                        <span className="text-sm text-gray-700 dark:text-gray-300">Generando prompt...</span>
                      </div>
                    </div>
                  </div>
                )}

                {/* Mensaje de bienvenida si no hay historial */}
                {history.length === 0 && !isGenerating && (
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
                            {LLM_MODELS.map((model) => (
                              <SelectItem
                                key={model.value}
                                value={model.value}
                                className="text-gray-900 dark:text-white text-xs"
                              >
                                {model.label}
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
                            generatePrompt()
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
                          onClick={generatePrompt}
                          disabled={isGenerating || !idea.trim() || !selectedModel || !selectedPlatform}
                          className="h-8 w-8 p-0 bg-gradient-to-r from-cyan-500 to-purple-500 hover:from-cyan-600 hover:to-purple-600 text-white rounded-full disabled:opacity-50 disabled:cursor-not-allowed"
                          title="Generar prompt (Enter)"
                        >
                          {isGenerating ? (
                            <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                          ) : (
                            <Sparkles className="w-4 h-4" />
                          )}
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
      <footer className="border-t border-gray-200 dark:border-gray-800 bg-white/50 dark:bg-black/50 backdrop-blur-xl mt-16">
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
              <a
                href="/privacy-policy"
                className="text-gray-600 hover:text-cyan-500 dark:text-gray-400 dark:hover:text-cyan-400 transition-colors"
              >
                Política de Privacidad
              </a>
              <Separator orientation="vertical" className="h-4 bg-gray-300 dark:bg-gray-700" />
              <p className="text-gray-500 dark:text-gray-500">© 2025 PromptGen. Todos los derechos reservados.</p>
            </div>
          </div>
        </div>
      </footer>
    </div>
  )
}
