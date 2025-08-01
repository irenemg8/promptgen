import type React from "react"
import type { Metadata, Viewport } from "next"
import { Inter } from "next/font/google"
import "./globals.css"
import { Toaster } from "@/components/ui/toaster"
import { ThemeProvider } from "@/components/theme-provider"

const inter = Inter({ subsets: ["latin"] })

export const metadata: Metadata = {
  title: "PromptGen - Generador Inteligente de Prompts para IA",
  description:
    "Transforma tus ideas en prompts optimizados para ChatGPT, Claude, v0, Sora y más plataformas de IA. Diseño moderno con tendencias 2025.",
  keywords: "prompt generator, AI, ChatGPT, Claude, v0, Sora, Gemini, Adobe Firefly, artificial intelligence",
  authors: [{ name: "PromptGen Team" }],
  generator: 'v0.dev'
}

export const viewport: Viewport = {
  width: 'device-width',
  initialScale: 1,
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="es" suppressHydrationWarning>
      <body className={inter.className}>
        <ThemeProvider
          attribute="class"
          defaultTheme="dark"
          enableSystem
          disableTransitionOnChange
          storageKey="theme"
        >
          {children}
          <Toaster />
        </ThemeProvider>
      </body>
    </html>
  )
}
