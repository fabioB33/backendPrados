from fastapi import FastAPI, APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timezone
from openai import AsyncOpenAI
import aiofiles
import json
import io
import base64
from elevenlabs import ElevenLabs, Voice, VoiceSettings
import aiosqlite

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# LLM Configuration
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
HEYGEN_API_KEY = os.environ.get('HEYGEN_API_KEY', '')
ELEVENLABS_API_KEY = os.environ.get('ELEVENLABS_API_KEY', '')

# Initialize OpenAI client
openai_client = None
if OPENAI_API_KEY:
    try:
        openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        logger.info("✅ OpenAI client initialized")
    except Exception as e:
        logger.error(f"❌ Error initializing OpenAI: {e}")

# Initialize ElevenLabs client
elevenlabs_client = None
if ELEVENLABS_API_KEY:
    try:
        elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
        logger.info("✅ ElevenLabs client initialized")
    except Exception as e:
        logger.error(f"❌ Error initializing ElevenLabs: {e}")

app = FastAPI()

# CORS Configuration - Configuración directa y robusta
# Dominios permitidos (hardcodeados - estos son los que realmente funcionan)
ALLOWED_ORIGINS = [
    "https://legbotdev.pradosdeparaiso.com.pe",  # Dominio real (con 'g')
    "https://www.legbotdev.pradosdeparaiso.com.pe",
    "http://localhost:3000",
    "http://localhost:3001",
]

# Obtener de variable de entorno si existe, pero PRIORIZAR los hardcodeados
cors_origins_str = os.environ.get('CORS_ORIGINS', '')
if cors_origins_str and cors_origins_str != '*':
    # Parsear variable de entorno
    env_origins = [origin.strip() for origin in cors_origins_str.split(',') if origin.strip()]
    # Combinar pero los hardcodeados tienen prioridad
    cors_origins = ALLOWED_ORIGINS + [origin for origin in env_origins if origin not in ALLOWED_ORIGINS]
    cors_origins = list(set(cors_origins))  # Eliminar duplicados
else:
    cors_origins = ALLOWED_ORIGINS

logger.info(f"🌐 CORS Origins configurados: {cors_origins}")

# Agregar CORS middleware INMEDIATAMENTE después de crear la app
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

api_router = APIRouter(prefix="/api")

# Configuración de base de datos SQLite
DB_PATH = ROOT_DIR / "conversations.db"
MAX_HISTORY_MESSAGES = 20

# Inicializar base de datos al iniciar
async def init_db():
    """Inicializa la base de datos SQLite para almacenar conversaciones"""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_session_created 
            ON conversations(session_id, created_at)
        """)
        await db.commit()
        logger.info(f"✅ Base de datos inicializada: {DB_PATH}")

# Ejecutar inicialización al arrancar
@app.on_event("startup")
async def startup_event():
    await init_db()

# Helper functions para manejo de conversaciones con persistencia SQLite
def get_or_create_session(session_id: Optional[str] = None) -> str:
    """Obtiene un session_id existente o crea uno nuevo"""
    if not session_id:
        session_id = str(uuid.uuid4())
        logger.info(f"🆕 Nueva sesión creada: {session_id}")
    return session_id

async def add_to_history(session_id: str, user_message: str, ai_response: str):
    """Agrega mensajes al historial de la conversación en SQLite"""
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            # Insertar mensaje del usuario
            await db.execute(
                "INSERT INTO conversations (session_id, role, content) VALUES (?, ?, ?)",
                (session_id, "user", user_message)
            )
            # Insertar respuesta del asistente
            await db.execute(
                "INSERT INTO conversations (session_id, role, content) VALUES (?, ?, ?)",
                (session_id, "assistant", ai_response)
            )
            await db.commit()
            
            # Limpiar mensajes antiguos (mantener solo los últimos N)
            await db.execute("""
                DELETE FROM conversations 
                WHERE session_id = ? 
                AND id NOT IN (
                    SELECT id FROM conversations 
                    WHERE session_id = ? 
                    ORDER BY created_at DESC 
                    LIMIT ?
                )
            """, (session_id, session_id, MAX_HISTORY_MESSAGES))
            await db.commit()
            
            logger.info(f"💾 Historial guardado en BD para sesión {session_id[:8]}...")
    except Exception as e:
        logger.error(f"❌ Error guardando historial: {str(e)}")
        # No lanzar error para no interrumpir la respuesta al usuario

async def get_conversation_history(session_id: str) -> List[Dict[str, str]]:
    """Obtiene el historial de conversación para una sesión desde SQLite"""
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("""
                SELECT role, content 
                FROM conversations 
                WHERE session_id = ? 
                ORDER BY created_at ASC 
                LIMIT ?
            """, (session_id, MAX_HISTORY_MESSAGES)) as cursor:
                rows = await cursor.fetchall()
                history = [{"role": row["role"], "content": row["content"]} for row in rows]
                logger.info(f"📚 Historial cargado: {len(history)} mensajes para sesión {session_id[:8]}...")
                return history
    except Exception as e:
        logger.error(f"❌ Error cargando historial: {str(e)}")
        return []

async def clear_conversation(session_id: str):
    """Limpia el historial de una conversación en SQLite"""
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute("DELETE FROM conversations WHERE session_id = ?", (session_id,))
            await db.commit()
            logger.info(f"🗑️ Historial limpiado para sesión {session_id[:8]}...")
    except Exception as e:
        logger.error(f"❌ Error limpiando historial: {str(e)}")
        raise

# Información legal precargada - BASE DE CONOCIMIENTOS CORREGIDA (VERSIÓN FINAL)
LEGAL_INFO = """
BASE DE CONOCIMIENTOS CORREGIDA (VERSIÓN FINAL)

1. ¿Qué es Prados del Paraíso? Prados de Paraíso es una marca comercial utilizada por Desarrolladora Santa María del Norte S.A.C., para desarrollar proyectos inmobiliarios con un enfoque ecológico y sostenible. Esta marca busca ofrecer una visión innovadora en el sector inmobiliario, creando proyectos que combinan eficiencia ambiental, un diseño funcional y una buena calidad de vida. Responde a la demanda actual de estilos de vida responsables y un desarrollo inmobiliario consciente.

2. ¿Qué proyectos tiene Prados del Paraíso? Actualmente, la marca Prados de Paraíso cuenta con dos proyectos. Uno exitosamente entregado, denominado "Prados de Paraíso – Casa Huerto Ecológico"; y el segundo proyecto: "Prados de Paraíso Villa Eco - Sostenible", el cual se encuentra en desarrollo. Ambos proyectos están respaldados por una sólida trayectoria en el mercado inmobiliario y buscan ofrecer oportunidades de inversión segura con visión de futuro.

3. ¿Dónde se ubica el proyecto Villa Eco- Sostenible? El proyecto Villa Eco-Sostenible se encuentra ubicado a la altura del 137.25 Km de la Carretera Panamericana Norte, distrito de Santa María, Provincia de Huaura y Departamento de Lima.

4. ¿Quién desarrolla el proyecto? El proyecto es promovido por Desarrolladora Santa María del Norte S.A.C., una empresa con experiencia en el mercado inmobiliario. Además, cuenta con el respaldo y asesoramiento legal de DS CASAHIERRO ABOGADOS y tiene un convenio con la NOTARIA TAMBINI para garantizar la transparencia y seguridad jurídica en los procesos.

5. ¿La empresa es formal? Sí, la empresa es formal y cuenta con el respaldo de la marca Prados de Paraíso, el cual tiene una una trayectoria sólida en el desarrollo de proyectos inmobiliarios. Además, se encuentra inscrita en la Partida Electrónica N° 15437655 del Registro de Personas Jurídicas de Lima.

6. ¿Desde cuándo existe el proyecto? El proyecto "Villa Eco-Sostenible" inicia en octubre del 2023.

7. ¿Qué es exactamente lo que ofrecen? Prados del Paraíso ofrece transferencia de posesión de lotes, lo que permite a los adquirentes disfrutar y el uso efectivo del lote. Es importante que sepas que la condición legal del predio es la posesión, no la propiedad titulada. Nuestra empresa tiene una posesión del terreno desde 1998, respaldada por escrituras públicas y reconocida por la Municipalidad de Santa María a través de cartillas municipales PR y HR. Al adquirir un lote con nosotros, formalizamos esta transferencia mediante un Contrato de Transferencia de Posesión, lo que te otorga el derecho de uso y disfrute del lote. En resumen, no solo adquieres un lote, sino una oportunidad de inversión segura y con visión de futuro, con el respaldo de una comunidad de más de 800 clientes satisfechos.

8. ¿Es lo mismo transferencia de posesión que comprar un terreno? No, no es exactamente lo mismo, aunque en la práctica ambos te permiten usar el terreno. Aquí te explico la diferencia clave de manera sencilla: Comprar la Propiedad (Título de Propiedad): Significa que te conviertes en el dueño legal absoluto y tu nombre aparece inscrito en los Registros Públicos (SUNARP). La Transferencia de Posesión (lo que ofrecemos en Prados de Paraíso): Significa que adquieres el uso, disfrute y control del lote. Tienes un respaldo mediante el Contrato de Transferencia de Posesión y Escritura Pública.

9. ¿Qué diferencia hay entre posesión y propiedad? La propiedad es el derecho real pleno que se ejerce sobre un bien y que faculta a su titular a usar, disfrutar, disponer y reivindicar dicho bien, otorgándole la titularidad legal como propietario, conforme al marco normativo vigente. Características: Como propietario, tienes el derecho jurídico absoluto. Registro: La propiedad es lo que se inscribe formalmente en los Registros Públicos (SUNARP). Cómo se adquiere: Generalmente mediante un contrato de compraventa de bien inmueble (o de acciones y derechos) que se eleva a Escritura Pública y luego se inscribe.
La Posesión ¿Qué es? Es el poder de hecho que ejerces sobre el bien. Significa que usas y disfrutas del lote físicamente (lo ocupas, lo cercas, construyes, etc.), independientemente de si eres el titular registral o no. Respaldo Legal: Es un derecho real reconocido por el Código Civil (artículo 896). Se transfiere a través de un Contrato de Transferencia de Posesión, el cual también puede elevarse a Escritura Pública para darle mayor seguridad jurídica. En resumen: Mientras que la propiedad es el "título" legal inscrito, la posesión es el "uso y control físico" del terreno.

10. ¿Puedo construir en el lote? Sí, puedes construir en el lote, sujeto a las normativas locales y el contrato de posesión.

11. ¿La escritura me hace propietario? No, la escritura pública de transferencia de posesión no le hace propietario en el sentido registral. Es una distinción muy importante que debemos aclarar. La escritura pública en el contexto de Prados de Paraíso formaliza la transferencia de la posesión. Sin embargo, la propiedad es un derecho distinto que otorga la titularidad del bien y es susceptible de inscripción en Registros Públicos (SUNARP). En resumen: La Escritura Pública de Transferencia de Posesión le otorga un respaldo sobre su posesión. Para ser propietario y que su nombre aparezca en Registros Públicos, se requiere un proceso adicional de saneamiento.

12. ¿La empresa responde por el lote? La empresa responde por el lote en el sentido de que garantiza la transferencia de la posesión del predio. Nuestra empresa, Desarrolladora Santa María del Norte S.A.C., formaliza esta transferencia mediante un Contrato de Transferencia de Posesión, el cual se eleva a Escritura Pública, a solicitud del cliente. Este contrato otorga el derecho de uso y disfrute del lote asignado.

13. ¿Qué planos entregarán a la firma del contrato de transferencia de posesión? Se te proporcionará plano de ubicación, también memoria descriptiva y planos perimétricos.

14. ¿Cómo se respalda legalmente la posesión o qué documentos se entregan a los clientes? La empresa ejerce posesión sobre el proyecto. Esto significa que, aunque no tienen una partida registral a su nombre como propietarios directos en este momento, sí cuentan con documentos jurídicos que acreditan y respaldan el derecho de posesión sobre el terreno. Esta posesión se considera de buena fe, actuando con transparencia y lealtad. Los documentos que respaldan la posesión de la empresa desde 1998 incluyen:
    • Escrituras Públicas: Estos son documentos elaborados por un notario que dan fe de los actos jurídicos celebrados, en este caso, las transferencias de posesión a lo largo del tiempo.
    • Cartillas municipales (PR y HR): La Municipalidad de Santa María reconoce la posesión de la empresa de manera indirecta a través de la emisión de estas cartillas, que les permiten cumplir con sus obligaciones tributarias.
Ahora, en cuanto a los documentos que se entregan a los clientes por la adquisición de los lotes, incluye:
    • Contrato de transferencia de posesión: Este es el documento fundamental que formaliza la adquisición de la posesión del terreno por parte del cliente.
    • Pagos de tributos municipales (PR y HR): Estos documentos demuestran el cumplimiento de las obligaciones tributarias relacionadas con el terreno.

15. ¿Cuál es el estado legal del proyecto y el proceso de adquisición de lote? a) Estado Legal del Proyecto La condición actual del proyecto es de posesión, que se encuentra respaldada en lo siguiente:
    • Respaldo Documental: Aunque no contamos con una partida registral a nombre de la empresa, nuestra posesión está respaldada por Escrituras Públicas que datan desde 1998.
    • Reconocimiento Municipal: La Municipalidad de Santa María reconoce nuestra posesión indirectamente mediante la emisión de cartillas municipales de Predio Rústico (PR) y Hoja Resumen (HR), lo que nos permite cumplir con obligaciones tributarias.
b) Proceso de Adquisición: El proceso para adquirir un lote con nosotros se basa en la transferencia de esta posesión. Los pasos son:
    • Firma del Contrato: Se firma un Contrato de Transferencia de Posesión. Este es el documento legal que formaliza que nosotros te cedemos los derechos sobre el lote asignado.
    • Trámite Notarial (Escritura Pública): Para mayor seguridad jurídica, este contrato se eleva a Escritura Pública ante notario. Esto le da plena fuerza legal al acto y fecha cierta al documento.
    • Entrega: Una vez completado el proceso y los pagos correspondientes, se te hace entrega física del lote para que puedas ejercer tu derecho de posesión (uso y disfrute).

16. ¿Qué documentos entrega la empresa al transferir la posesión? Para formalizar la transferencia y brindarte seguridad jurídica sobre tu lote en Prados de Paraíso, la empresa te entregará la siguiente documentación:
    • Contrato de Transferencia de Posesión: Este es el documento principal mediante el cual obtienes el derecho de uso y disfrute del lote asignado. Es importante mencionar que, a solicitud del cliente, el contrato se eleva a Escritura Pública ante notario para certificar la autenticidad de las firmas y darle mayor formalidad al acto.
    • Escrituras Públicas: Se te facilitarán las escrituras que respaldan la posesión legítima del predio por parte de la empresa desde el año 1998.
    • Cartillas Municipales: Se entregarán los documentos de PR (Predio Rústico) y HR (Hoja Resumen), que demuestran el cumplimiento de obligaciones municipales.

17. ¿Qué significa una transferencia de posesión? Una transferencia de posesión significa que se te otorga el uso y disfrute del predio. En el caso de Prados de Paraíso, la empresa transfiere la posesión del lote. Este derecho se formaliza a través de un Contrato de Transferencia de Posesión, el cual se eleva a Escritura Pública ante un notario.

18. ¿Qué derechos tengo como poseedor? Como poseedor, usted tiene el derecho de disponer y disfrutar del bien como si fuera suyo, ejerciendo un poder de hecho. Esto significa que puede usar y controlar el lote, incluso si aún no es el propietario. Esto significa que puedes usar el lote, construir, cultivarlo o darle el uso que desees, siempre dentro de los límites legales y contractuales.

19. ¿Puedo perder mi lote? Entendiendo perfectamente su preocupación, es una pregunta muy importante. Quiero darle tranquilidad: nuestra empresa mantiene una posesión, respaldada por documentos legales sólidos como escrituras públicas que datan desde mil novecientos noventa y ocho, además de ejercer una posesión efectiva. Es importante aclarar que actuamos de buena fe y tenemos el reconocimiento de la Municipalidad a través de los pagos de tributos (Predio Rústico y Hoja Resumen). Al suscribir su contrato de transferencia de posesión, usted adquiere por tracto sucesivo, el derecho posesorio y de posesión que la empresa tiene, desde 1998, y no sería posible que de forma legal usted pueda perder su lote.

20. ¿Direfor, siendo el legítimo propietario, me puede quitar mi lote? Entiendo tu preocupación, es una pregunta muy importante. Mira, es cierto que el predio donde se desarrolla Prados de Paraíso figura a nombre de DIREFOR en los Registros Públicos. No obstante, nuestra empresa, Desarrolladora Santa María del Norte S.A.C., mantiene la posesión del predio desde el año 1998. Es decir, con anterioridad la ley 29618, que habla sobre la imprescriptibilidad de los predios del Estado. Esto significa que, aunque no tenemos un título de propiedad registrado a nuestro nombre, ejercemos la posesión del terreno, con el respaldo de escrituras públicas y cartillas municipales. Por lo tanto, la presencia de DIREFOR como titular registral no implica que seamos invasores ni representa un riesgo para tu posesión. Nosotros te garantizamos la entrega de la posesión de tu lote mediante un Contrato de Transferencia de Posesión, lo que te otorga el uso y disfrute.

21. Si llevo un proceso de saneamiento vía prescripción adquisitiva de dominio, y pierdo el proceso, ¿me pueden quitar mi lote o mi posesión? El procedimiento de prescripción adquisitiva de dominio tiene como finalidad que el poseedor adquiera la propiedad del bien, siempre que cumpla con los requisitos legales establecidos. Si dicho proceso no resulta favorable, ello significa únicamente que no se ha logrado acreditar, en ese momento y por esa vía, el derecho de propiedad sobre el lote. No obstante, la improcedencia o rechazo del proceso de prescripción no implica automáticamente la pérdida de la posesión. Usted adquirió la posesión del lote mediante un Contrato de Transferencia de Posesión, lo que le otorga el derecho de uso y disfrute, mientras dicha posesión no sea cuestionada o despojada por una resolución judicial firme. Asimismo, el proceso de prescripción adquisitiva no tiene por objeto desalojar al poseedor, sino evaluar si se cumplen los requisitos para adquirir la propiedad. Por ello, perder dicho proceso no habilita por sí solo a un tercero a quitarle el lote, ni extingue su derecho posesorio. En consecuencia, aun cuando la prescripción adquisitiva no prospere, usted mantiene su posesión, siempre que continúe ejerciéndola conforme a ley y cumpla con las obligaciones contractuales asumidas.

22. ¿La empresa participa en el proceso de formalización o saneamiento? Gracias por tu consulta, es muy importante aclararlo. La empresa no realiza directamente el trámite de formalización o saneamiento del título de propiedad, ya que este es un proceso personal que corresponde a cada cliente. Lo que sí hacemos es garantizar la entrega de la posesión del lote, que se formaliza mediante un Contrato de Transferencia de Posesión. Esto te permite usar y disfrutar tu lote con tranquilidad. Una vez que el proyecto ha sido entregado y el lote se encuentra totalmente cancelado, el cliente puede iniciar, de manera independiente, el proceso de formalización para obtener su título de propiedad, asumiendo los costos del trámite. Como parte de nuestro acompañamiento, la empresa te brinda todo el respaldo documentario necesario, como:
    • Escrituras públicas que acreditan la posesión desde 1998.
    • Documentación municipal (Predio Rústico y Hoja Resumen). Con esta documentación, podrás evaluar, junto con un abogado de tu confianza, la vía de formalización más adecuada para tu caso.

23. ¿Existe el riesgo de que DIREFOR inicie una demanda de reivindicación o desalojo? Entendemos perfectamente su preocupación; es una consulta razonable al evaluar una inversión de este tipo. En el proyecto Prados de Paraíso, la seguridad jurídica se sustenta en que la empresa ejerce una posesión desde el año 1998, es decir, con anterioridad a la inscripción registral a favor del Estado. Si bien la empresa no cuenta con una partida registral de propiedad a su nombre, sí ejerce y administra el terreno de manera efectiva y documentada. Esta posesión se encuentra respaldada por:
    • Escrituras públicas que acreditan la posesión desde 1998.
    • Documentación municipal (Predio Rústico y Hoja Resumen), que evidencia el cumplimiento de obligaciones tributarias y el reconocimiento fáctico de la posesión por parte de la Municipalidad de Santa María.
Es importante precisar que una eventual demanda de reivindicación o desalojo no prospera automáticamente cuando existe una posesión antigua, pública y ejercida de buena fe, como en este caso. Adicionalmente, la empresa actúa con transparencia y respaldo legal permanente, contando con la asesoría especializada de DS Casa Hierro Abogados, así como con un convenio con la Notaría Tambini para la correcta formalización de los contratos. Asimismo, se mantiene una relación armónica con las asociaciones vecinales colindantes, lo que contribuye a un entorno estable y ordenado. Si bien, en términos generales, ninguna situación jurídica puede calificarse como de riesgo cero, la solidez de la posesión, el sustento documental y el acompañamiento legal existente reducen significativamente la probabilidad de acciones de reivindicación o desalojo.

24. ¿La posesión que ustedes transfieren me permite defenderme legalmente frente a terceros o solo frente a la empresa? La posesión que nosotros transferimos está protegida por el Código Civil Peruano. Esto significa que no solo te permite defenderse legalmente frente a nuestra empresa, sino también frente a terceros. El Código Civil reconoce a la posesión como un derecho real y te otorga la facultad de usar y disfrutar del bien como si fuera tuyo. Al adquirir la posesión mediante un Contrato de Transferencia de Posesión, elevado a Escritura Pública, obtienes un respaldo sólido. Además, un aspecto clave es la "suma de plazos posesorios", regulada en el artículo 898 del Código Civil. Este mecanismo te permite sumar tu tiempo de posesión al tiempo que nuestra empresa ha poseído el terreno desde 1998.

25. ¿Por qué la empresa no sanea primero el terreno y después lo vende? Prados del Paraíso se desarrolla sobre un predio cuya condición legal actual es la posesión, no la propiedad. Esto significa que la empresa ejerce el uso y disfrute del inmueble, situación que se encuentra formalizada y respaldada documentalmente, incluyendo escrituras públicas que acreditan la continuidad posesoria desde 1998, así como documentación municipal emitida por la Municipalidad de Santa María, que reconoce indirectamente dicha posesión. La gerencia de la empresa ha adoptado una decisión estratégica de estructurar el proyecto bajo el modelo de transferencia de posesión, priorizando una alternativa clara, transparente y comercialmente viable para los interesados, sin ofrecer ni prometer procesos de titulación o saneamiento registral. Es importante tener en cuenta que la posesión puede ser válidamente transferida. Por ello, la empresa garantiza la entrega de la posesión mediante un Contrato de Transferencia de Posesión, el cual se formaliza una vez que el adquirente ha cumplido con el pago total del valor del lote. A partir de ese momento, el adquirente, en su calidad de poseedor, puede evaluar de manera independiente si desea iniciar algún procedimiento de saneamiento o formalización de la titularidad, asumiendo directamente los costos, trámites y decisiones que ello implique. Con el fin de facilitar cualquier evaluación futura, la empresa pone a disposición del adquirente toda la documentación existente, incluyendo las escrituras públicas y las constancias municipales vinculadas a la posesión.

26. ¿Existe hoy algún juicio, denuncia o problema legal activo sobre este terreno? Basándome en la información legal disponible sobre el proyecto Prados de Paraíso, puedo confirmarte que no existe ningún juicio, denuncia o problema legal activo sobre el terreno. Aunque la partida registral figura a nombre de DIREFOR (una entidad del Estado) debido a la Ley N° 29618 (que pasó terrenos sin dueño registrado al Estado en 2010), esto no implica que seamos invasores ni que haya un conflicto. Nuestra posesión está respaldada por escrituras públicas que datan desde 1998. En resumen, el proyecto se desarrolla en un marco de transparencia, sin litigios que pongan en riesgo tu adquisición de la posesión.

27. Si yo compro hoy el lote y mañana hay un problema legal con el terreno, ¿qué respaldo real tengo como adquiriente? Lo primero que debes saber es que la condición legal del predio que adquieres es la POSESIÓN, no la PROPIEDAD. Esto significa que nuestra empresa te garantiza la entrega de la posesión del lote, lo que te otorga el derecho de uso y disfrute del mismo. Esta transferencia se formaliza mediante un Contrato de Transferencia de Posesión. Nuestra posesión está respaldada por escrituras públicas que datan desde 1998. Además, la Municipalidad de Santa María reconoce nuestra posesión de manera indirecta a través de la emisión de cartillas municipales PR y HR, que nos permiten cumplir con nuestras obligaciones tributarias. En resumen, tu respaldo como adquirente se basa en:
    • El Contrato de Transferencia de Posesión, que te otorga el derecho de uso y disfrute.
    • La posesión documentada de nuestra empresa, respaldada por escrituras públicas y reconocimiento municipal.

28. ¿Qué riesgos existen al adquirir el lote por transferencia de posesión? Al adquirir un lote mediante transferencia de posesión, el riesgo principal a considerar es que no se adquiere la propiedad, sino únicamente el derecho de uso y disfrute del terreno. Esto implica que:
    • La obtención del título de propiedad no es automática y dependerá de que el adquirente inicie, evalúe y asuma un proceso de saneamiento de manera personal.
    • La empresa no garantiza la titulación, sino la entrega de una posesión documentada y formalizada mediante contrato.

29. ¿La empresa garantiza que no habrá problemas legales en el futuro? La empresa no puede garantizar escenarios futuros ajenos a su control. Lo que sí garantiza, de manera expresa y contractual, es la entrega de la posesión del lote en la condición legal informada. En la actualidad, la empresa ejerce una posesión que se encuentra debidamente respaldada por escrituras públicas que acreditan su ejercicio posesorio, así como por documentación municipal correspondiente al predio matriz del proyecto, como las cartillas municipales. Esta posesión es la que se transfiere al adquirente mediante un Contrato de Transferencia de Posesión.

30. ¿Qué obligaciones asume el adquirente? Al adquirir un lote en Prados del Paraíso mediante transferencia de posesión, el ADQUIRENTE asume las siguientes obligaciones principales:
    • Pagar el precio pactado por la transferencia de posesión, ya sea al contado o conforme al cronograma de pagos establecido en el contrato.
    • Cumplir con las condiciones contractuales para la entrega de la posesión, incluyendo la cancelación total del valor del lote.
    • Asumir de manera los trámites notariales y administrativos que origine la Escritura Pública del Contrato de Transferencia.
    • Cumplir con el reglamento interno del proyecto y con las disposiciones aplicables sobre uso del lote.

31. ¿Se paga algún impuesto o tributo por la transferencia de posesión? El ADQUIRENTE puede asumir obligaciones tributarias municipales, como el impuesto predial, una vez que la posesión le sea entregada, conforme a la normativa municipal aplicable. Estos tributos se gestionan sobre el predio matriz del proyecto, mientras no exista individualización administrativa por lote.

32. ¿El contrato contempla cláusulas de saneamiento posesorio? El contrato de Transferencia de Posesión está estructurado para regular y garantizar la entrega de la posesión, no para ejecutar un saneamiento de la propiedad.

33. ¿La empresa ha evaluado iniciar el proceso prescripción adquisitiva del proyecto? La decisión de iniciar un proceso de prescripción adquisitiva corresponde a una estratégica, que considera factores legales, técnicos y comerciales. Actualmente, la empresa no ofrece el inicio de un proceso de prescripción adquisitiva como parte del proyecto, ya que su actividad consiste en la transferencia de posesión, no en la comercialización de propiedad saneada.

34. ¿La transferencia de posesión podría considerarse simulación de compraventa? Es una excelente pregunta que aborda un punto legal importante. La transferencia de posesión y la compraventa de propiedad son actos jurídicos distintos. Una transferencia de posesión, como la que se realiza en Prados de Paraíso, implica ceder el derecho de ejercer el poder de hecho sobre un bien, es decir, usarlo y disfrutarlo, lo cual se formaliza mediante un contrato de transferencia de posesión elevado a escritura pública. Por otro lado, una compraventa de propiedad implica la transferencia del derecho de propiedad, que es el derecho de ser dueño legalmente del bien, lo cual se inscribe en Registros Públicos. La simulación ocurre cuando las partes aparentan celebrar un acto jurídico, pero en realidad tienen la intención de realizar otro, o ninguna en absoluto, con el fin de engañar a terceros o evadir la ley. En el caso de Prados de Paraíso, la transferencia de posesión es un acto transparente, respaldado por asesoramiento legal especializado y notaría, que busca otorgar al adquiriente un derecho real sobre el bien.

35. ¿Cómo se gestiona la eventual formalización futura de la posesión? La formalización futura de la posesión se gestiona mediante un proceso de saneamiento físico-legal, el cual permite que el poseedor evalúe la posibilidad de acceder al derecho de propiedad y, de ser el caso, su inscripción en Registros Públicos. Este proceso no forma parte del servicio ofrecido por la empresa y debe ser evaluado y asumido de manera personal por el adquirente, una vez que haya recibido la posesión del lote y cumplido con las condiciones contractuales. Existen distintas vías legales que pueden ser analizadas por el adquirente con asesoría especializada, entre ellas:
    • La prescripción adquisitiva de dominio, que se tramita vía judicial.
    • Otras vías administrativas, cuando resulten legalmente aplicables según la naturaleza del predio y la normativa vigente. La empresa le brinda el respaldo documental para que pueda iniciar ese proceso de saneamiento.

36. ¿Qué obligaciones mantiene la empresa luego de la transferencia? Luego de la transferencia de la posesión del lote, las obligaciones de la empresa se mantienen únicamente dentro del marco de lo expresamente establecido en el contrato. Principalmente, la empresa se obliga a:
    • Entregar la posesión del lote en la condición legal informada.
    • Proporcionar la documentación posesoría que sustenta la transferencia realizada.
    • Cumplir con las obligaciones contractuales pendientes, de corresponder.

37. ¿La empresa mantiene la administración sobre áreas recreativas? La empresa asume la gestión inicial necesaria para la organización del proyecto; sin embargo, la administración y mantenimiento de las áreas recreativas, puede ser asumida posteriormente por una asociación, conforme a lo previsto en el reglamento interno y a la consolidación del proyecto.

38. ¿Existen contingencias penales asociadas al modelo de negocio? De acuerdo con la naturaleza del proyecto y lo establecido contractualmente, no existen contingencias penales inherentes al modelo de negocio de Prados de Paraíso. El proyecto se basa en la transferencia de posesión, una figura reconocida por el ordenamiento jurídico, respaldada por documentación posesoria y formalizada mediante Escrituras Públicas.

39. ¿Qué respaldo real tiene el cliente si surge un conflicto? En caso de surgir un conflicto, el respaldo real para el cliente se fundamenta en varios pilares. Primero, la seguridad jurídica de la empresa se basa en una posesión respaldada por escrituras públicas desde 1998 y documentación formal. Además, la empresa cuenta con asesoramiento legal especializado de DS CASAHIERRO ABOGADOS y un convenio con la NOTARIA TAMBINI, lo que añade un nivel de formalidad y legalidad a las transacciones. Finalmente, la empresa se compromete a entregar toda la documentación necesaria para su proceso de saneamiento.

40. ¿Qué es DIREFOR y por qué figura como propietario? DIREFOR es la Dirección de Formalización de la Propiedad Rural, una entidad del Estado. Figura como titular registral del predio matriz como consecuencia de un cambio normativo. Con la entrada en vigencia de la Ley N.° 29618 (año 2010), los terrenos que no contaban con propiedad inscrita pasaron a ser registrados a nombre del Estado, razón por la cual actualmente DIREFOR aparece como propietario en Registros Públicos. Es importante precisar que esta inscripción no desconoce ni invalida la posesión existente. La empresa ejerce una posesión desde 1998, debidamente documentada.

41. ¿Es legal transferir la posesión de un terreno del Estado? La legislación peruana reconoce la posesión como una situación jurídica protegida, distinta y diferente al derecho de propiedad. En ese sentido, lo que se transfiere en Prados de Paraíso es la posesión, no la propiedad del terreno. La empresa ejerce una posesión anterior a la inscripción estatal, debidamente documentada, y transfiere esa situación posesoria mediante un Contrato de Transferencia de Posesión.

42. ¿Qué sucede si se revierte la posesión a favor del Estado? No existe, a la fecha, ningún procedimiento administrativo o judicial que busque revertir la posesión del predio a favor del Estado. Si bien DIREFOR figura como titular registral del predio matriz por mandato de la Ley N.° 29618, ello no implica automáticamente la pérdida de la posesión existente, la cual se ejerce desde 1998 y se encuentra documentada.

43. ¿La municipalidad reconoce oficialmente el proyecto? La Municipalidad de Santa María reconoce nuestra posesión de manera indirecta, a través de la emisión de cartillas municipales: PR (Predio Rústico) y HR (Hoja Resumen). Estos documentos son importantes porque:
    • Permiten cumplir con las obligaciones tributarias.
    • Demuestran que la municipalidad tiene registro de nuestra actividad y posesión sobre el predio.

44. ¿Cómo impacta la ley que prohíbe la prescripción adquisitiva de inmuebles contra el Estado? La Ley N.º 29618 (2010) establece que los bienes inmuebles de dominio privado estatal no pueden ser adquiridos por particulares mediante prescripción adquisitiva, es decir, ya no es posible reclamar la propiedad de terrenos estatales solo por haberlos poseído durante mucho tiempo.
En el caso de Prados de Paraíso, el predio original pertenecía a DIREFOR, pero la empresa cuenta con 27 años de posesión, lo que significa que:
- Su posesión se inició antes de que la ley entrara en vigor, por lo que se mantiene la legitimidad de la posesión transferida a los clientes.
- La ley sólo impide nuevas adquisiciones por prescripción sobre bienes estatales a partir de 2010; no afecta la posesión histórica que ya existía.
En pocas palabras, la ley protege al Estado frente a nuevas reclamaciones de prescripción, pero no invalida la posesión ya existente, que es la que la empresa transfiere a los adquirentes.

45. ¿La empresa acompaña judicialmente al cliente si hay alguna contingencia legal? En caso de que enfrentes una contingencia legal o decidas iniciar un proceso de formalización de tu lote, la gestión y representación legal corresponde al cliente. La empresa proporciona toda la documentación probatoria disponible para respaldar tu caso y facilitar tu defensa, pero la representación ante un juez debe ser realizada por tu propio abogado.

46. ¿La empresa indemnizará en caso de pérdida de posesión? La empresa no menciona una política de indemnización específica en caso de pérdida de posesión, sino que se enfoca en asegurar la posesión que se transfiere y en proporcionar toda la documentación necesaria para que el cliente, si lo desea, pueda iniciar su propio proceso de saneamiento y obtener el título de propiedad.

47. ¿Se puede individualizar la posesión por cada lote? ¡Claro que sí! Cuando firmas el Contrato de Transferencia de Posesión, este documento delimita y asigna el derecho de uso y disfrute exclusivo sobre un lote determinado dentro del proyecto. Es decir, tú tienes el control físico y el derecho a usar y disfrutar ese espacio concreto, cercarlo o construir en él.

48. ¿El adquirente podría ser demandado directamente ante un posible proceso judicial iniciado por el Estado? Como adquirente de la posesión, usted sería la parte directamente involucrada en cualquier proceso judicial que el Estado pudiera iniciar. No obstante, la posesión que recibes está respaldada por documentación histórica y escrituras públicas desde 1998, lo que garantiza la posesión sobre el lote. Esto te permite usar y disfrutar tu inversión con total tranquilidad y confianza.

49. ¿Qué pasa si el proyecto no logra consolidarse? Entendemos que esta es una preocupación importante para cualquier inversión. La garantía principal de Prados de Paraíso es la antigüedad de la posesión que se transfiere a nuestros clientes, respaldada por escrituras públicas desde 1998. Si el proyecto no se consolida completamente, por ejemplo, en cuanto a infraestructura o desarrollo planificado, usted seguirá manteniendo la posesión de su lote, con pleno ejercicio de uso y disfrute sobre ese espacio.

50. ¿El contrato me protege frente a cualquier contingencia legal? El contrato está diseñado principalmente para regular la transferencia de la posesión y las obligaciones de pago, garantizándote que recibes la posesión de tu lote respaldada por documentos históricos. Si bien el contrato brinda seguridad sobre la posesión física y la documentación que acredita tu derecho de ocupación, no cubre situaciones externas, como litigios con terceros o el Estado que puedan surgir en el futuro.

51. ¿La empresa responde económicamente frente a la pérdida de la posesión del proyecto? La empresa no asume responsabilidad económica por la pérdida de la posesión causada por hechos externos o ajenos al incumplimiento del comprador. En caso de que el adquirente incumpla el contrato, la empresa tiene derecho a aplicar penalidades. En resumen: La empresa respalda la posesión pero no indemniza económicamente al comprador por causas externas a su incumplimiento contractual.

52. ¿Las cartillas PR y HR están a nombre de mi lote específico? Las cartillas municipales PR (Predio Rústico) y HR (Hoja Resumen) son emitidas por la Municipalidad de Santa María a nombre de la empresa, ya que reconocen la posesión del predio en su conjunto y permiten cumplir con las obligaciones tributarias correspondientes. No obstante, estas cartillas no están individualizadas a tu lote específico, pero la empresa te proporcionará estas documentaciones como respaldo de tu posesión dentro del proyecto.

53. ¿Mi lote tendrá su propia cartilla municipal? Las cartillas municipales PR y HR se emiten a nombre de la empresa para el predio en su totalidad, no para cada lote individualmente desde el inicio. La empresa se compromete a realizar el procedimiento de Individualización Administrativa ante la Municipalidad Distrital para tu lote. Esto significa que se harán los esfuerzos para que tu lote tenga su propia documentación municipal, como las Declaraciones Juradas.

54. ¿La empresa tiene Libro de Reclamaciones? Reclamaciones en formato físico, disponible en nuestras oficinas ubicadas en Calle Libertadores 155, Oficina 302, distrito de San Isidro. Asimismo, ponemos a disposición de nuestros clientes el formato virtual, accesible a través de nuestra página web: https://pradosdeparaiso.com.pe/.

55. ¿Qué pasa si no estoy conforme con la respuesta de la empresa? En caso de que no estés conforme con la respuesta inicial brindada por la empresa, siempre existe la posibilidad de continuar el diálogo a través de nuestros canales internos, con el objetivo de evaluar nuevamente el caso y buscar una solución adecuada. La empresa prioriza la atención y resolución directa de los reclamos, por lo que puede solicitarse una revisión adicional, una reunión de aclaración o la intervención de un área especializada, antes de acudir a instancias externas. Solo si, luego de agotar estas vías internas, el reclamo no resulta satisfactorio, el consumidor mantiene su derecho de recurrir a los organismos de protección al consumidor, conforme a la normativa vigente.

56. ¿Cuáles son los plazos de atención de un reclamo? Conforme al Reglamento de Libro de Reclamaciones y su modificatoria, el plazo máximo para que un proveedor atienda un reclamo y brinde una respuesta es de 15 días hábiles improrrogables.

57. ¿La empresa se responsabiliza por daños externos? La empresa no asume responsabilidad por daños ocasionados por factores externos que estén fuera de su control, tales como desastres naturales, actos de terceros, decisiones de autoridades o cualquier otro evento fortuito o de fuerza mayor. La responsabilidad de la empresa se limita a cumplir con las obligaciones expresamente asumidas en el contrato, principalmente la entrega de la posesión del lote y la documentación correspondiente.

58. Si la empresa Desarrolladora Santa Mariaa del Norte S.A.C deja de pagar la deuda pendiente con el señor Manuel Ampuero por la transferencia de posesión, ¿Eso podría hacer que yo pierda mi lote o mi derecho de posesión?
Desde la suscripción de la Escritura Pública por la que el señor Manuel Ampuero transfirió la posesión a favor de Desarrolladora Santa María del Norte (en adelante, la "empresa"), adquirió válidamente la posesión efectiva del terreno.
En consecuencia, desde esa fecha la empresa ostenta la calidad de poseedora,, con plena facultad para transferir dicha posesión a terceros. Esta condición posesoria no se ve afectada por las obligaciones internas o relaciones económicas que puedan existir entre las partes que intervinieron en la transferencia original.
Así, aun en el supuesto de que la empresa incumpliera algún pago u obligación económica pendiente frente al señor Ampuero, ello no genera la pérdida, restitución ni afectación de la posesión ya transferida. La posesión se mantiene firme, pues fue otorgada formalmente mediante escritura pública y recae sobre la empresa como persona jurídica.
Por tanto, cualquier relación económica entre las partes originales es independiente y no incide en la situación posesoria del predio, ni en la validez de la posesión que posteriormente se transfiera a los futuros posesionarios.
En consecuencia, se reafirma que no existe riesgo alguno para el cliente respecto a la estabilidad, continuidad o validez de la posesión que adquirirá.
"""

# Helper functions
async def get_ai_response(system_prompt: str, user_message: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
    """Generate AI response using OpenAI - Optimized for speed with conversation memory"""
    if not openai_client:
        raise HTTPException(status_code=503, detail="OpenAI not configured")
    
    try:
        # Construir lista de mensajes con historial
        messages = [{"role": "system", "content": system_prompt}]
        
        # Agregar historial de conversación si existe
        if conversation_history:
            messages.extend(conversation_history)
            logger.info(f"📚 Usando historial: {len(conversation_history)} mensajes previos")
        
        # Agregar el mensaje actual del usuario
        messages.append({"role": "user", "content": user_message})
        
        response = await openai_client.chat.completions.create(
            model="gpt-4o-mini",  # Más rápido que gpt-4o, mantiene buena calidad
            messages=messages,
            temperature=0.65,  # Balance entre naturalidad y velocidad
            max_tokens=600,  # Respuestas desarrolladas pero optimizadas para velocidad
            timeout=15.0  # Timeout más corto para evitar esperas largas
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error generating AI response: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating AI response: {str(e)}")

# Routes
@api_router.get("/")
async def root():
    return {"message": "Prados de Paraíso Legal Hub API"}

@api_router.get("/health")
async def health_check():
    """Health check endpoint"""
    cors_origins = os.environ.get('CORS_ORIGINS', '*')
    return {
        "status": "ok",
        "cors_origins": cors_origins,
        "cors_origins_parsed": [origin.strip() for origin in cors_origins.split(',') if origin.strip()],
        "backend_url": "backendprados.onrender.com",
        "openai_configured": bool(OPENAI_API_KEY),
        "elevenlabs_configured": bool(ELEVENLABS_API_KEY)
    }

@api_router.options("/{full_path:path}")
async def options_handler(full_path: str):
    """Handle OPTIONS requests for CORS preflight"""
    from fastapi.responses import Response
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS, HEAD, PATCH",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Max-Age": "3600",
        }
    )

# Text-to-Speech with ElevenLabs
@api_router.post("/tts")
async def text_to_speech(request: dict):
    """Convert text to speech using ElevenLabs"""
    try:
        if not elevenlabs_client:
            raise HTTPException(status_code=503, detail="ElevenLabs not configured")
        
        text = request.get('text', '')
        if not text:
            raise HTTPException(status_code=400, detail="Text is required")
        
        # Generate audio using ElevenLabs
        # Using Spanish voice - Rachel (multilingual)
        audio_generator = elevenlabs_client.text_to_speech.convert(
            text=text,
            voice_id="21m00Tcm4TlvDq8ikWAM",  # Rachel voice
            model_id="eleven_multilingual_v2",
            voice_settings=VoiceSettings(
                stability=0.5,
                similarity_boost=0.75,
                style=0.0,
                use_speaker_boost=True
            )
        )
        
        # Collect audio bytes
        audio_bytes = b""
        for chunk in audio_generator:
            audio_bytes += chunk
        
        # Return base64 encoded audio
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        return {
            "audio": audio_base64,
            "format": "mp3"
        }
        
    except Exception as e:
        logger.error(f"Error in TTS: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Voice Chat Endpoint (Push-to-Talk)
@api_router.post("/voice-chat")
async def voice_chat(audio: UploadFile = File(...), session_id: Optional[str] = Form(None)):
    """
    Complete voice chat flow with conversation memory:
    1. Transcribe audio using ElevenLabs STT
    2. Get AI response using LLM with conversation history
    3. Convert response to speech using ElevenLabs TTS
    4. Return session_id for maintaining conversation context
    """
    try:
        if not elevenlabs_client:
            raise HTTPException(status_code=503, detail="ElevenLabs not configured")
        
        if not OPENAI_API_KEY:
            raise HTTPException(status_code=503, detail="OpenAI not configured")
        
        # Obtener o crear session_id
        session_id = get_or_create_session(session_id)
        
        # Step 1: Transcribe audio to text using ElevenLabs STT
        logger.info(f"📝 Transcribing audio (sesión {session_id[:8]}...)...")
        audio_content = await audio.read()
        
        transcription_response = elevenlabs_client.speech_to_text.convert(
            file=io.BytesIO(audio_content),
            model_id="scribe_v1"
        )
        
        # Extract transcribed text
        transcribed_text = transcription_response.text if hasattr(transcription_response, 'text') else str(transcription_response)
        logger.info(f"✅ Transcribed: {transcribed_text}")
        
        if not transcribed_text or len(transcribed_text.strip()) == 0:
            raise HTTPException(status_code=400, detail="No se pudo transcribir el audio. Intenta hablar más claro.")
        
        # Obtener historial de conversación
        history = get_conversation_history(session_id)
        
        # Step 2: Get AI response con historial
        logger.info("🤖 Generating AI response...")
        system_prompt = f"""Eres un asistente legal experto en Prados de Paraíso. Responde preguntas sobre condiciones legales, propiedad, posesión y saneamiento.

Información legal disponible:
{LEGAL_INFO}

IMPORTANTE: Responde de forma amigable, profesional y desarrollada. Explica los conceptos de manera clara y completa, como si estuvieras conversando con un cliente. Sé empático y comprensivo. Desarrolla tus respuestas de forma concisa pero completa (4-6 frases), usando ejemplos cuando sea útil. Mantén la información precisa y basada en la información legal disponible. Si no tienes la información específica, indica amablemente que el usuario debe consultar con el equipo legal.
Recuerda el contexto de la conversación anterior para dar respuestas coherentes y naturales."""
        
        ai_response = await get_ai_response(system_prompt, transcribed_text, history)
        logger.info(f"✅ AI Response: {ai_response[:100]}...")
        
        # Guardar en historial
        await add_to_history(session_id, transcribed_text, ai_response)
        
        # Step 3: Convert AI response to speech
        logger.info("🔊 Converting response to speech...")
        audio_generator = elevenlabs_client.text_to_speech.convert(
            text=ai_response,
            voice_id="21m00Tcm4TlvDq8ikWAM",  # Rachel voice (multilingual)
            model_id="eleven_multilingual_v2",
            voice_settings=VoiceSettings(
                stability=0.5,
                similarity_boost=0.75,
                style=0.0,
                use_speaker_boost=True
            )
        )
        
        # Collect audio bytes
        audio_bytes = b""
        for chunk in audio_generator:
            audio_bytes += chunk
        
        # Return base64 encoded audio
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        logger.info("✅ Voice chat completed successfully")
        
        return {
            "session_id": session_id,  # Devolver session_id para mantener contexto
            "transcribed_text": transcribed_text,
            "ai_response": ai_response,
            "audio_url": f"data:audio/mpeg;base64,{audio_base64}",
            "format": "mp3"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error in voice chat: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error procesando consulta de voz: {str(e)}")

# Text Chat Endpoint (alternative to voice)
@api_router.post("/text-chat")
async def text_chat(request: dict):
    """
    Text-based chat flow (alternative to voice) with conversation memory:
    1. Get user text input and session_id (optional)
    2. Get AI response using LLM with conversation history
    3. Convert response to speech using ElevenLabs TTS (optional)
    4. Return session_id for maintaining conversation context
    """
    try:
        if not OPENAI_API_KEY:
            raise HTTPException(status_code=503, detail="OpenAI not configured")
        
        text = request.get('text', '').strip()
        if not text:
            raise HTTPException(status_code=400, detail="Text is required")
        
        # Obtener o crear session_id
        session_id = get_or_create_session(request.get('session_id'))
        logger.info(f"💬 Text chat request (sesión {session_id[:8]}...): {text}")
        
        # Obtener historial de conversación
        history = await get_conversation_history(session_id)
        
        # Get AI response con historial
        system_prompt = f"""Eres un asistente legal experto en Prados de Paraíso. Responde preguntas sobre condiciones legales, propiedad, posesión y saneamiento.

Información legal disponible:
{LEGAL_INFO}

IMPORTANTE: Responde de forma amigable, profesional y desarrollada. Explica los conceptos de manera clara y completa, como si estuvieras conversando con un cliente. Sé empático y comprensivo. Desarrolla tus respuestas de forma concisa pero completa (4-6 frases), usando ejemplos cuando sea útil. Mantén la información precisa y basada en la información legal disponible. Si no tienes la información específica, indica amablemente que el usuario debe consultar con el equipo legal.
Recuerda el contexto de la conversación anterior para dar respuestas coherentes y naturales."""
        
        ai_response = await get_ai_response(system_prompt, text, history)
        logger.info(f"✅ AI Response generated")
        
        # Guardar en historial
        await add_to_history(session_id, text, ai_response)
        
        # Optionally convert to speech if ElevenLabs is available
        audio_url = None
        if elevenlabs_client:
            try:
                # Use Dr. Prados voice (same as agent)
                agent_voice_id = "5kMbtRSEKIkRZSdXxrZg"
                
                logger.info(f"🔊 Converting response to speech with Dr. Prados voice...")
                audio_generator = elevenlabs_client.text_to_speech.convert(
                    text=ai_response,
                    voice_id=agent_voice_id,
                    model_id="eleven_multilingual_v2",
                    voice_settings=VoiceSettings(
                        stability=0.5,
                        similarity_boost=0.75,
                        style=0.0,
                        use_speaker_boost=True
                    )
                )
                
                audio_bytes = b""
                for chunk in audio_generator:
                    audio_bytes += chunk
                
                audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
                audio_url = f"data:audio/mpeg;base64,{audio_base64}"
                logger.info("✅ Audio generated")
            except Exception as e:
                logger.warning(f"⚠️ Could not generate audio: {str(e)}")
        
        return {
            "session_id": session_id,  # Devolver session_id para mantener contexto
            "user_text": text,
            "ai_response": ai_response,
            "audio_url": audio_url,
            "format": "mp3" if audio_url else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        logger.error(f"❌ Error in text chat: {str(e)}")
        logger.error(f"📋 Traceback: {error_trace}")
        raise HTTPException(status_code=500, detail=f"Error procesando consulta: {str(e)}")

# Voice Agent Endpoint (using ElevenLabs Agent's voice and knowledge)
@api_router.post("/voice-agent")
async def voice_agent(audio: UploadFile = File(...), agent_id: str = Form(...), session_id: Optional[str] = Form(None)):
    """
    Send audio to get response using the ElevenLabs Agent's configured voice with conversation memory.
    This endpoint:
    1. Transcribes user audio (STT)
    2. Gets agent configuration (voice, personality)
    3. Generates response using agent's knowledge base context with conversation history
    4. Converts to speech using agent's voice (TTS)
    5. Returns session_id for maintaining conversation context
    """
    try:
        if not elevenlabs_client:
            raise HTTPException(status_code=503, detail="ElevenLabs not configured")
        
        if not OPENAI_API_KEY:
            raise HTTPException(status_code=503, detail="OpenAI not configured")
        
        # Obtener o crear session_id
        session_id = get_or_create_session(session_id)
        logger.info(f"🎙️ Processing voice with agent: {agent_id} (sesión {session_id[:8]}...)")
        
        # Step 1: Transcribe audio
        audio_content = await audio.read()
        
        # Verificar que el audio no esté vacío
        if len(audio_content) < 1000:  # Mínimo ~1KB
            raise HTTPException(status_code=400, detail="El audio es demasiado corto. Por favor, graba al menos 1 segundo de audio.")
        
        try:
            transcription_response = elevenlabs_client.speech_to_text.convert(
                file=io.BytesIO(audio_content),
                model_id="scribe_v1"
            )
            
            transcribed_text = transcription_response.text if hasattr(transcription_response, 'text') else str(transcription_response)
            logger.info(f"✅ Transcribed: {transcribed_text}")
            
            if not transcribed_text or len(transcribed_text.strip()) == 0:
                raise HTTPException(status_code=400, detail="No se pudo transcribir el audio. Intenta hablar más claro o grabar nuevamente.")
        except Exception as e:
            error_msg = str(e)
            if "audio_too_short" in error_msg.lower() or "too short" in error_msg.lower():
                raise HTTPException(status_code=400, detail="El audio es demasiado corto. Por favor, graba al menos 1-2 segundos de audio.")
            elif "400" in error_msg or "Bad Request" in error_msg:
                raise HTTPException(status_code=400, detail=f"Error al procesar el audio: {error_msg}")
            else:
                logger.error(f"Error en transcripción: {error_msg}")
                raise HTTPException(status_code=500, detail="Error al transcribir el audio. Intenta nuevamente.")
        
        # Step 2: Get agent details to use the correct voice
        agent_voice_id = "5kMbtRSEKIkRZSdXxrZg"  # Dr. Prados voice (from agent config)
        agent_name = "Doctor Prados de Paraiso"
        
        try:
            # Try to get agent details to confirm voice
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"https://api.elevenlabs.io/v1/convai/agents/{agent_id}",
                    headers={"xi-api-key": ELEVENLABS_API_KEY},
                    timeout=5.0
                )
                if response.status_code == 200:
                    agent_data = response.json()
                    if 'conversation_config' in agent_data:
                        tts_config = agent_data.get('conversation_config', {}).get('tts', {})
                        if 'voice_id' in tts_config:
                            agent_voice_id = tts_config['voice_id']
                            logger.info(f"✅ Using agent voice: {agent_voice_id}")
                        agent_name = agent_data.get('name', agent_name)
        except Exception as e:
            logger.warning(f"⚠️ Could not fetch agent details: {str(e)}, using default Dr. Prados voice")
        
        # Obtener historial de conversación
        history = await get_conversation_history(session_id)
        
        # Step 3: Generate AI response using the knowledge base context con historial
        system_prompt = f"""Eres {agent_name}, un asistente legal experto especializado en Prados de Paraíso. Responde preguntas sobre condiciones legales, propiedad, posesión y saneamiento.

Información legal disponible:
{LEGAL_INFO}

IMPORTANTE: Responde de forma amigable, cálida y desarrollada como lo haría el Dr. Prados. Sé conversacional y empático. Explica los conceptos de manera clara y completa de forma concisa pero desarrollada (4-6 frases). Usa un tono cercano y profesional, como si estuvieras hablando con un amigo o colega. Desarrolla tus respuestas con suficiente detalle para que sean útiles y comprensibles, pero de forma eficiente.
Recuerda el contexto de la conversación anterior para dar respuestas coherentes y naturales."""
        
        ai_response = await get_ai_response(system_prompt, transcribed_text, history)
        logger.info(f"✅ AI Response generated")
        
        # Guardar en historial
        await add_to_history(session_id, transcribed_text, ai_response)
        
        # Step 4: Convert to speech using agent's voice
        audio_generator = elevenlabs_client.text_to_speech.convert(
            text=ai_response,
            voice_id=agent_voice_id,
            model_id="eleven_multilingual_v2",
            voice_settings=VoiceSettings(
                stability=0.5,
                similarity_boost=0.75,
                style=0.0,
                use_speaker_boost=True
            )
        )
        
        audio_bytes = b""
        for chunk in audio_generator:
            audio_bytes += chunk
        
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        logger.info("✅ Voice agent response completed")
        
        return {
            "session_id": session_id,  # Devolver session_id para mantener contexto
            "transcribed_text": transcribed_text,
            "agent_response": ai_response,
            "audio_url": f"data:audio/mpeg;base64,{audio_base64}",
            "format": "mp3",
            "voice_used": agent_voice_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        logger.error(f"❌ Error in voice agent: {str(e)}")
        logger.error(f"📋 Traceback: {error_trace}")
        raise HTTPException(status_code=500, detail=f"Error procesando consulta: {str(e)}")

# Endpoint para limpiar historial de conversación
@api_router.post("/clear-conversation")
async def clear_conversation_endpoint(request: dict):
    """
    Limpia el historial de una conversación específica
    """
    try:
        session_id = request.get('session_id')
        if not session_id:
            raise HTTPException(status_code=400, detail="session_id es requerido")
        
        await clear_conversation(session_id)
        logger.info(f"🗑️ Conversación limpiada para sesión {session_id[:8]}...")
        
        return {
            "message": "Conversación limpiada exitosamente",
            "session_id": session_id
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error limpiando conversación: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error limpiando conversación: {str(e)}")


app.include_router(api_router)