import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Shield, ArrowLeft } from "lucide-react"
import { Button } from "@/components/ui/button"
import Link from "next/link"

export default function PrivacyPolicyPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-black to-gray-900">
      {/* Header */}
      <header className="border-b border-gray-800 bg-black/50 backdrop-blur-xl">
        <div className="container mx-auto px-4 py-6">
          <div className="flex items-center gap-4">
            <Link href="/">
              <Button variant="ghost" size="sm" className="text-gray-400 hover:text-white">
                <ArrowLeft className="w-4 h-4 mr-2" />
                Volver
              </Button>
            </Link>
            <div className="flex items-center gap-3">
              <Shield className="w-8 h-8 text-cyan-400" />
              <div>
                <h1 className="text-3xl font-bold bg-gradient-to-r from-cyan-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">
                  Política de Privacidad
                </h1>
                <p className="text-gray-400 text-sm">PromptGen - Protección de datos</p>
              </div>
            </div>
          </div>
        </div>
      </header>

      <div className="container mx-auto px-4 py-8 max-w-4xl">
        <Card className="border-gray-800 bg-gray-900/50 backdrop-blur-xl shadow-2xl">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-white">
              <Shield className="w-5 h-5 text-green-400" />
              Política de Privacidad de Datos
            </CardTitle>
            <p className="text-gray-400">Última actualización: 1 de enero de 2025</p>
          </CardHeader>
          <CardContent className="space-y-6 text-gray-300">
            <section>
              <h2 className="text-xl font-semibold text-white mb-3">1. Información que Recopilamos</h2>
              <div className="space-y-2 text-sm">
                <p>En PromptGen, recopilamos la siguiente información:</p>
                <ul className="list-disc list-inside space-y-1 ml-4">
                  <li>Ideas y conceptos que ingresas para generar prompts</li>
                  <li>Archivos que subes (imágenes, videos, documentos)</li>
                  <li>Historial de prompts generados</li>
                  <li>Preferencias de modelos y plataformas seleccionadas</li>
                  <li>Información técnica básica (dirección IP, navegador, dispositivo)</li>
                </ul>
              </div>
            </section>

            <section>
              <h2 className="text-xl font-semibold text-white mb-3">2. Cómo Utilizamos tu Información</h2>
              <div className="space-y-2 text-sm">
                <p>Utilizamos la información recopilada para:</p>
                <ul className="list-disc list-inside space-y-1 ml-4">
                  <li>Generar prompts personalizados y optimizados</li>
                  <li>Mantener tu historial de prompts generados</li>
                  <li>Mejorar nuestros algoritmos de generación</li>
                  <li>Proporcionar soporte técnico cuando sea necesario</li>
                  <li>Analizar el uso de la plataforma para mejoras</li>
                </ul>
              </div>
            </section>

            <section>
              <h2 className="text-xl font-semibold text-white mb-3">3. Almacenamiento y Seguridad</h2>
              <div className="space-y-2 text-sm">
                <p>Nos comprometemos a proteger tu información:</p>
                <ul className="list-disc list-inside space-y-1 ml-4">
                  <li>Los datos se almacenan de forma segura en servidores cifrados</li>
                  <li>Utilizamos protocolos HTTPS para todas las comunicaciones</li>
                  <li>El acceso a los datos está restringido al personal autorizado</li>
                  <li>Realizamos copias de seguridad regulares para prevenir pérdida de datos</li>
                  <li>Los archivos subidos se procesan y almacenan temporalmente</li>
                </ul>
              </div>
            </section>

            <section>
              <h2 className="text-xl font-semibold text-white mb-3">4. Compartir Información</h2>
              <div className="space-y-2 text-sm">
                <p>No vendemos, alquilamos ni compartimos tu información personal con terceros, excepto:</p>
                <ul className="list-disc list-inside space-y-1 ml-4">
                  <li>Cuando sea requerido por ley o autoridades competentes</li>
                  <li>Para proteger nuestros derechos legales o la seguridad de usuarios</li>
                  <li>
                    Con proveedores de servicios que nos ayudan a operar la plataforma (bajo estrictos acuerdos de
                    confidencialidad)
                  </li>
                  <li>En caso de fusión, adquisición o venta de activos (con notificación previa)</li>
                </ul>
              </div>
            </section>

            <section>
              <h2 className="text-xl font-semibold text-white mb-3">5. Tus Derechos</h2>
              <div className="space-y-2 text-sm">
                <p>Tienes derecho a:</p>
                <ul className="list-disc list-inside space-y-1 ml-4">
                  <li>Acceder a tu información personal almacenada</li>
                  <li>Solicitar la corrección de datos inexactos</li>
                  <li>Eliminar tu cuenta y datos asociados</li>
                  <li>Exportar tu historial de prompts</li>
                  <li>Limitar el procesamiento de tus datos</li>
                  <li>Retirar el consentimiento en cualquier momento</li>
                </ul>
              </div>
            </section>

            <section>
              <h2 className="text-xl font-semibold text-white mb-3">6. Cookies y Tecnologías Similares</h2>
              <div className="space-y-2 text-sm">
                <p>Utilizamos cookies y tecnologías similares para:</p>
                <ul className="list-disc list-inside space-y-1 ml-4">
                  <li>Mantener tu sesión activa</li>
                  <li>Recordar tus preferencias de configuración</li>
                  <li>Analizar el uso de la plataforma</li>
                  <li>Mejorar la experiencia del usuario</li>
                </ul>
                <p className="mt-2">Puedes gestionar las cookies desde la configuración de tu navegador.</p>
              </div>
            </section>

            <section>
              <h2 className="text-xl font-semibold text-white mb-3">7. Retención de Datos</h2>
              <div className="space-y-2 text-sm">
                <p>Conservamos tu información durante:</p>
                <ul className="list-disc list-inside space-y-1 ml-4">
                  <li>El tiempo que mantengas tu cuenta activa</li>
                  <li>Hasta 30 días después de la eliminación de la cuenta (para recuperación)</li>
                  <li>El tiempo requerido por obligaciones legales</li>
                  <li>Los archivos subidos se eliminan automáticamente después de 7 días</li>
                </ul>
              </div>
            </section>

            <section>
              <h2 className="text-xl font-semibold text-white mb-3">8. Menores de Edad</h2>
              <div className="space-y-2 text-sm">
                <p>
                  PromptGen no está dirigido a menores de 13 años. No recopilamos conscientemente información personal
                  de menores de 13 años. Si descubrimos que hemos recopilado información de un menor, la eliminaremos
                  inmediatamente.
                </p>
              </div>
            </section>

            <section>
              <h2 className="text-xl font-semibold text-white mb-3">9. Cambios en esta Política</h2>
              <div className="space-y-2 text-sm">
                <p>
                  Podemos actualizar esta política de privacidad ocasionalmente. Te notificaremos sobre cambios
                  significativos mediante:
                </p>
                <ul className="list-disc list-inside space-y-1 ml-4">
                  <li>Notificación en la plataforma</li>
                  <li>Email (si proporcionaste uno)</li>
                  <li>Actualización de la fecha de "última actualización"</li>
                </ul>
              </div>
            </section>

            <section>
              <h2 className="text-xl font-semibold text-white mb-3">10. Contacto</h2>
              <div className="space-y-2 text-sm">
                <p>
                  Si tienes preguntas sobre esta política de privacidad o el manejo de tus datos, puedes contactarnos:
                </p>
                <ul className="list-disc list-inside space-y-1 ml-4">
                  <li>Email: privacy@promptgen.com</li>
                  <li>Dirección: [Tu dirección de empresa]</li>
                  <li>Teléfono: [Tu número de contacto]</li>
                </ul>
              </div>
            </section>

            <div className="bg-gray-800/50 rounded-lg p-4 border border-gray-700 mt-8">
              <p className="text-sm text-gray-400">
                <strong className="text-white">Nota importante:</strong> Esta política de privacidad se rige por las
                leyes de protección de datos aplicables, incluyendo GDPR (Europa) y CCPA (California). Nos comprometemos
                a cumplir con todas las regulaciones de privacidad relevantes.
              </p>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
