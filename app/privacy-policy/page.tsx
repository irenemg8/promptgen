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
            <p className="text-gray-400">Última actualización: 15 de julio de 2023</p>
          </CardHeader>
          <CardContent className="space-y-6 text-gray-300">
            <section className="bg-green-900/20 border border-green-700/30 rounded-lg p-4">
              <h2 className="text-xl font-semibold text-green-400 mb-3">Compromiso de No Almacenamiento de Datos</h2>
              <div className="space-y-2 text-sm">
                <p className="font-medium">
                  En PromptGen, nos comprometemos firmemente a proteger tu privacidad. <span className="text-white font-bold">No almacenamos, guardamos ni retenemos ningún dato</span> que ingreses o generes en nuestra plataforma.
                </p>
                <p>
                  Toda la información procesada en PromptGen es temporal y se elimina completamente cuando cierras la aplicación o refrescas la página. No mantenemos bases de datos de usuarios ni registros de actividad.
                </p>
              </div>
            </section>

            <section>
              <h2 className="text-xl font-semibold text-white mb-3">1. Información que NO Recopilamos</h2>
              <div className="space-y-2 text-sm">
                <p>PromptGen opera bajo un principio de privacidad completa:</p>
                <ul className="list-disc list-inside space-y-1 ml-4">
                  <li>No guardamos las ideas o conceptos que ingresas para generar prompts</li>
                  <li>No almacenamos archivos que se suben a la plataforma</li>
                  <li>No conservamos historial de prompts generados en nuestros servidores</li>
                  <li>No creamos perfiles de usuario ni guardamos preferencias</li>
                  <li>No recopilamos información personal identificable</li>
                </ul>
              </div>
            </section>

            <section>
              <h2 className="text-xl font-semibold text-white mb-3">2. Procesamiento Local y Temporal</h2>
              <div className="space-y-2 text-sm">
                <p>Todo el procesamiento en PromptGen funciona de la siguiente manera:</p>
                <ul className="list-disc list-inside space-y-1 ml-4">
                  <li>Los datos ingresados se procesan temporalmente solo durante la sesión activa</li>
                  <li>El historial de prompts se guarda únicamente en la memoria local de tu navegador</li>
                  <li>Toda la información se elimina al cerrar la aplicación o refrescar la página</li>
                  <li>Los archivos nunca se cargan a servidores externos</li>
                  <li>No utilizamos servicios de análisis que rastreen tu actividad</li>
                </ul>
              </div>
            </section>

            <section>
              <h2 className="text-xl font-semibold text-white mb-3">3. Seguridad de Datos</h2>
              <div className="space-y-2 text-sm">
                <p>Aunque no almacenamos datos, garantizamos la seguridad durante el uso:</p>
                <ul className="list-disc list-inside space-y-1 ml-4">
                  <li>Conexiones cifradas mediante HTTPS para todas las comunicaciones</li>
                  <li>Procesamiento de prompts en tiempo real sin persistencia de datos</li>
                  <li>Sin acceso de terceros a tu información durante la sesión</li>
                  <li>Protocolo de borrado automático de datos temporales</li>
                </ul>
              </div>
            </section>

            <section>
              <h2 className="text-xl font-semibold text-white mb-3">4. Datos del Navegador</h2>
              <div className="space-y-2 text-sm">
                <p>La única información que puede permanecer en tu dispositivo:</p>
                <ul className="list-disc list-inside space-y-1 ml-4">
                  <li>Preferencias de tema (claro/oscuro) guardadas en localStorage</li>
                  <li>Datos de sesión temporal mientras la aplicación está abierta</li>
                  <li>Caché temporal del navegador para mejorar el rendimiento</li>
                </ul>
                <p className="mt-2">Puedes eliminar estos datos en cualquier momento limpiando la caché y el almacenamiento local de tu navegador.</p>
              </div>
            </section>

            <section>
              <h2 className="text-xl font-semibold text-white mb-3">5. Sin Cookies de Seguimiento</h2>
              <div className="space-y-2 text-sm">
                <p>
                  PromptGen no utiliza cookies de seguimiento ni tecnologías similares para monitorear tu actividad. No analizamos patrones de uso ni creamos perfiles de comportamiento.
                </p>
              </div>
            </section>

            <section>
              <h2 className="text-xl font-semibold text-white mb-3">6. Sin Compartición de Datos</h2>
              <div className="space-y-2 text-sm">
                <p>
                  No compartimos ninguna información con terceros por una simple razón: no recopilamos ni almacenamos datos que se puedan compartir.
                </p>
              </div>
            </section>

            <section>
              <h2 className="text-xl font-semibold text-white mb-3">7. Accesibilidad Local</h2>
              <div className="space-y-2 text-sm">
                <p>
                  Si deseas conservar los prompts generados, deberás copiarlos manualmente, ya que no se guardan automáticamente después de cerrar la aplicación.
                </p>
              </div>
            </section>

            <section>
              <h2 className="text-xl font-semibold text-white mb-3">8. Cambios en esta Política</h2>
              <div className="space-y-2 text-sm">
                <p>
                  Cualquier cambio futuro en esta política respetará siempre nuestro compromiso de no almacenar datos. Las actualizaciones se publicarán en esta página.
                </p>
              </div>
            </section>

            <section>
              <h2 className="text-xl font-semibold text-white mb-3">9. Contacto</h2>
              <div className="space-y-2 text-sm">
                <p>
                  Si tienes preguntas sobre esta política de privacidad, puedes contactarnos:
                </p>
                <ul className="list-disc list-inside space-y-1 ml-4">
                  <li>Email: privacy@promptgen.com</li>
                </ul>
              </div>
            </section>

            <div className="bg-blue-900/20 rounded-lg p-4 border border-blue-700/30 mt-8">
              <p className="text-sm text-gray-400">
                <strong className="text-white">Resumen:</strong> PromptGen es una herramienta que respeta completamente tu privacidad. No almacenamos, compartimos ni analizamos ninguna información que proporciones o generes. Todos los datos son temporales y se eliminan automáticamente al finalizar tu sesión.
              </p>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
