from fastapi import FastAPI, APIRouter, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect, Form
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timezone
import asyncio
from openai import AsyncOpenAI
import aiofiles
import json
import io
import base64
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from elevenlabs import ElevenLabs, Voice, VoiceSettings

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# LLM Configuration (OpenAI)
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')
HEYGEN_API_KEY = os.environ.get('HEYGEN_API_KEY', '')
ELEVENLABS_API_KEY = os.environ.get('ELEVENLABS_API_KEY', '')

# Initialize ElevenLabs client
elevenlabs_client = None
if ELEVENLABS_API_KEY:
    try:
        elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
        logger.info("✅ ElevenLabs client initialized")
    except Exception as e:
        logger.error(f"❌ Error initializing ElevenLabs: {e}")

# Import custom services
from services.sqlite_knowledge import SQLiteKnowledgeBase
from services.heygen_service import LiveAvatarService as LiveAvatarAPIService

# Initialize SQLite Knowledge Base (reemplaza MongoDB)
sqlite_kb = SQLiteKnowledgeBase()
liveavatar_service = LiveAvatarAPIService()

logger.info(f"✅ SQLite Knowledge Base initialized ({sqlite_kb.count_documents()} documents)")
logger.info("✅ LiveAvatar Service initialized")


# HeyGen Configuration
HEYGEN_API_KEY = os.environ.get('HEYGEN_API_KEY', '')
HEYGEN_AVATAR_ID = os.environ.get('HEYGEN_AVATAR_ID', '')

app = FastAPI()

# ⚠️ CONFIGURACIÓN CRÍTICA DE CORS
origins = [
    "https://legbotdev.pradosdeparaiso.com.pe",  # Dominio de producción
    "http://localhost:3000",  # Para desarrollo local
    "http://localhost:3001",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_router = APIRouter(prefix="/api")

# Información legal precargada (debe coincidir con base de conocimientos de ElevenLabs)
LEGAL_INFO = """
Eres un asistente legal muy útil. Utiliza la información que tienes en tu base de conocimientos para contestar las preguntas.

Utiliza el archivo documentacion/CorregidoLegal.docx para contestar todas las preguntas del usuario si no encontrás las respuestas seguí con el resto.

Base de Conocimientos - Prados de Paraíso:

[AQUÍ debes agregar TODO el contenido de tu archivo CorregidoLegal.docx]

1. CONDICIÓN LEGAL DEL PROYECTO:
- 50% del terreno: Propiedad adquirida mediante compraventa de acciones y derechos
- 50% restante: Terreno bajo condición de posesión legítima y mediata

2. DIFERENCIA ENTRE PROPIEDAD Y POSESIÓN:
- Propiedad: Derecho que otorga titularidad legal inscribible en Registros Públicos
- Posesión: Ejercicio de hecho de poderes inherentes a la propiedad

3. PREGUNTAS FRECUENTES:

Q1: Cuándo entregan el título de propiedad?
R: La condición legal es la POSESIÓN. Se entrega contrato de transferencia de posesión. Para obtener título de propiedad, el cliente debe gestionar saneamiento tras completar pago.

Q2: En qué estado se encuentra el lote?
R: Posesión legítima, mediata y de buena fe, respaldada por escrituras públicas desde 1998.

Q3: Tenemos partida registral?
R: No hay partida registral a nombre de la desarrolladora. El predio figura a nombre de DIREFOR (entidad estatal). Esto no representa riesgo ya que poseemos legítimamente desde 1998.

Q4: Tipos de posesión?
R: Legítima (mediata e inmediata) e Ilegítima (buena fe, mala fe, precaria). Nuestra situación: Posesión Legítima Mediata y de Buena Fe.

Q5: Por qué no hay partida registral?
R: Decisión estratégica comercial. La posesión es un derecho reconocido y protegido por ley.
'''
- Notaría Tambini
- Casahierro Abogados
"""

async def _generate_llm_response(system_prompt: str, user_text: str, model: str = "gpt-4o") -> str:
    """Generate LLM response using OpenAI API."""
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=503, detail="LLM not configured (OPENAI_API_KEY)")
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text}
        ]
    )
    return response.choices[0].message.content or ""

# Models
class User(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    email: str
    name: str
    role: str = "seller"  # seller, client, admin
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class UserCreate(BaseModel):
    email: str
    name: str
    role: str = "seller"

class Message(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    conversation_id: str
    role: str  # user, assistant
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class MessageCreate(BaseModel):
    conversation_id: str
    content: str

class Conversation(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    user_name: str
    title: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    message_count: int = 0

class ConversationCreate(BaseModel):
    user_id: str
    user_name: str
    title: str = "Nueva Consulta"

class Document(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    filename: str
    content: str
    uploaded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

# Helper function
def prepare_for_mongo(data: dict) -> dict:
    '''Convert datetime objects to ISO strings for MongoDB'''
    for key, value in data.items():
        if isinstance(value, datetime):
            data[key] = value.isoformat()
    return data

# Routes
@api_router.get("/")
async def root():
    return {"message": "Prados de Paraíso Legal Hub API"}

# User routes
@api_router.post("/users", response_model=User)
async def create_user(user: UserCreate):
    user_obj = User(**user.model_dump())
    doc = prepare_for_mongo(user_obj.model_dump())
    await db.users.insert_one(doc)
    return user_obj

@api_router.get("/users", response_model=List[User])
async def get_users():
    users = await db.users.find({}, {"_id": 0}).to_list(1000)
    for user in users:
        if isinstance(user.get('created_at'), str):
            user['created_at'] = datetime.fromisoformat(user['created_at'])
    return users

@api_router.get("/users/{user_id}", response_model=User)
async def get_user(user_id: str):
    user = await db.users.find_one({"id": user_id}, {"_id": 0})
    if not user:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    if isinstance(user.get('created_at'), str):
        user['created_at'] = datetime.fromisoformat(user['created_at'])
    return user

# Conversation routes
@api_router.post("/conversations", response_model=Conversation)
async def create_conversation(conv: ConversationCreate):
    conv_obj = Conversation(**conv.model_dump())
    doc = prepare_for_mongo(conv_obj.model_dump())
    await db.conversations.insert_one(doc)
    return conv_obj

@api_router.get("/conversations/user/{user_id}", response_model=List[Conversation])
async def get_user_conversations(user_id: str):
    conversations = await db.conversations.find(
        {"user_id": user_id}, 
        {"_id": 0}
    ).sort("updated_at", -1).to_list(100)
    
    for conv in conversations:
        for field in ['created_at', 'updated_at']:
            if isinstance(conv.get(field), str):
                conv[field] = datetime.fromisoformat(conv[field])
    return conversations

@api_router.get("/conversations/{conversation_id}", response_model=Conversation)
async def get_conversation(conversation_id: str):
    conv = await db.conversations.find_one({"id": conversation_id}, {"_id": 0})
    if not conv:
        raise HTTPException(status_code=404, detail="Conversación no encontrada")
    for field in ['created_at', 'updated_at']:
        if isinstance(conv.get(field), str):
            conv[field] = datetime.fromisoformat(conv[field])
    return conv

# Message routes
@api_router.post("/messages", response_model=Message)
async def create_message_endpoint(msg: MessageCreate):
    try:
        # Create user message
        user_msg = Message(
            conversation_id=msg.conversation_id,
            role="user",
            content=msg.content
        )
        doc = prepare_for_mongo(user_msg.model_dump())
        await db.messages.insert_one(doc)
        
        # Get conversation context
        messages = await db.messages.find(
            {"conversation_id": msg.conversation_id},
            {"_id": 0}
        ).sort("timestamp", 1).to_list(50)
        
        # Generate AI response
        system_prompt = f'''Eres un asistente legal experto en Prados de Paraíso. 
Tu trabajo es responder preguntas sobre condiciones legales, propiedad, posesión y saneamiento.

Información legal disponible:
{LEGAL_INFO}

Responde de manera profesional, clara y precisa. Si no tienes información específica, 
indica que el usuario debe consultar con el equipo legal.'''
        
        ai_response = await _generate_llm_response(system_prompt, msg.content, model="gpt-4o")
        
        # Create assistant message
        assistant_msg = Message(
            conversation_id=msg.conversation_id,
            role="assistant",
            content=ai_response
        )
        doc = prepare_for_mongo(assistant_msg.model_dump())
        await db.messages.insert_one(doc)
        
        # Update conversation
        await db.conversations.update_one(
            {"id": msg.conversation_id},
            {
                "$set": {"updated_at": datetime.now(timezone.utc).isoformat()},
                "$inc": {"message_count": 2}
            }
        )
        
        return assistant_msg
    except Exception as e:
        logger.error(f"Error creating message: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/messages/{conversation_id}", response_model=List[Message])
async def get_messages(conversation_id: str):
    messages = await db.messages.find(
        {"conversation_id": conversation_id},
        {"_id": 0}
    ).sort("timestamp", 1).to_list(1000)
    
    for msg in messages:
        if isinstance(msg.get('timestamp'), str):
            msg['timestamp'] = datetime.fromisoformat(msg['timestamp'])
    return messages

# Document routes
@api_router.post("/documents")
async def upload_document(
    file: UploadFile = File(...),
    user_id: str = Form(...)
):
    try:
        content = await file.read()
        content_str = content.decode('utf-8', errors='ignore')
        
        doc = Document(
            user_id=user_id,
            filename=file.filename,
            content=content_str[:10000]  # Limit size
        )
        doc_dict = prepare_for_mongo(doc.model_dump())
        await db.documents.insert_one(doc_dict)
        
        return {"success": True, "document_id": doc.id, "filename": file.filename}
    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/documents/user/{user_id}")
async def get_user_documents(user_id: str):
    docs = await db.documents.find(
        {"user_id": user_id},
        {"_id": 0, "content": 0}
    ).sort("uploaded_at", -1).to_list(100)
    
    for doc in docs:
        if isinstance(doc.get('uploaded_at'), str):
            doc['uploaded_at'] = datetime.fromisoformat(doc['uploaded_at'])
    return docs

# Analytics routes
@api_router.get("/analytics/overview")
async def get_analytics():
    try:
        total_users = await db.users.count_documents({})
        total_conversations = await db.conversations.count_documents({})
        total_messages = await db.messages.count_documents({})
        total_documents = await db.documents.count_documents({})
        
        # Get recent activity
        recent_convs = await db.conversations.find(
            {},
            {"_id": 0}
        ).sort("updated_at", -1).limit(10).to_list(10)
        
        return {
            "total_users": total_users,
            "total_conversations": total_conversations,
            "total_messages": total_messages,
            "total_documents": total_documents,
            "recent_activity": recent_convs
        }
    except Exception as e:
        logger.error(f"Error getting analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Export conversation to PDF
@api_router.get("/conversations/{conversation_id}/export")
async def export_conversation(conversation_id: str):
    try:
        # Get conversation
        conv = await db.conversations.find_one({"id": conversation_id}, {"_id": 0})
        if not conv:
            raise HTTPException(status_code=404, detail="Conversación no encontrada")
        
        # Get messages
        messages = await db.messages.find(
            {"conversation_id": conversation_id},
            {"_id": 0}
        ).sort("timestamp", 1).to_list(1000)
        
        # Create PDF
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title = Paragraph(f"<b>{conv.get('title', 'Conversación')}</b>", styles['Title'])
        story.append(title)
        story.append(Spacer(1, 12))
        
        # Messages
        for msg in messages:
            role = "Usuario" if msg['role'] == 'user' else "Asistente"
            timestamp = msg.get('timestamp', '')
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp)
            
            msg_text = f"<b>{role}</b> - {timestamp.strftime('%Y-%m-%d %H:%M')}<br/>{msg['content']}"
            p = Paragraph(msg_text, styles['Normal'])
            story.append(p)
            story.append(Spacer(1, 12))
        
        doc.build(story)
        buffer.seek(0)
        
        return StreamingResponse(
            buffer,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=conversacion_{conversation_id}.pdf"}
        )
    except Exception as e:
        logger.error(f"Error exporting conversation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Search conversations
@api_router.get("/search")
async def search_conversations(q: str, user_id: Optional[str] = None):
    try:
        # Search in messages
        query = {"content": {"$regex": q, "$options": "i"}}
        messages = await db.messages.find(query, {"_id": 0}).limit(50).to_list(50)
        
        # Get unique conversation IDs
        conv_ids = list(set(msg['conversation_id'] for msg in messages))
        
        # Get conversations
        conv_query = {"id": {"$in": conv_ids}}
        if user_id:
            conv_query["user_id"] = user_id
        
        conversations = await db.conversations.find(
            conv_query,
            {"_id": 0}
        ).to_list(50)
        
        return {
            "conversations": conversations,
            "message_matches": len(messages)
        }
    except Exception as e:
        logger.error(f"Error searching: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Text-to-Speech with ElevenLabs
@api_router.post("/tts")
async def text_to_speech(request: dict):
    '''Convert text to speech using ElevenLabs'''
    try:
        if not elevenlabs_client:
            raise HTTPException(status_code=503, detail="ElevenLabs not configured")
        
        text = request.get('text', '')
        if not text:
            raise HTTPException(status_code=400, detail="Text is required")
        
        # Generate audio using ElevenLabs with streaming
        # Using Lina - Warm Latin American female voice (Colombian accent, works well for Peruvian Spanish)
        audio_stream = elevenlabs_client.text_to_speech.stream(
            text=text,
            voice_id="VmejBeYhbrcTPwDniox7",  # Lina - Latin American female
            model_id="eleven_multilingual_v2",
            voice_settings=VoiceSettings(
                stability=0.6,
                similarity_boost=0.8,
                style=0.0,
                use_speaker_boost=True
            )
        )
        
        # Collect audio bytes from stream
        audio_bytes = b""
        for chunk in audio_stream:
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
async def voice_chat(audio: UploadFile = File(...)):
    '''
    Complete voice chat flow:
    1. Transcribe audio using ElevenLabs STT
    2. Get AI response using LLM
    3. Convert response to speech using ElevenLabs TTS
    '''
    try:
        if not elevenlabs_client:
            raise HTTPException(status_code=503, detail="ElevenLabs not configured")
        
        if not OPENAI_API_KEY:
            raise HTTPException(status_code=503, detail="LLM not configured")
        
        # Step 1: Transcribe audio to text using ElevenLabs STT
        logger.info("📝 Transcribing audio...")
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
        
        # Step 2: Get AI response
        logger.info("🤖 Generating AI response...")
        system_prompt = f'''Eres un asistente legal experto en Prados de Paraíso. 
Tu trabajo es responder preguntas sobre condiciones legales, propiedad, posesión y saneamiento.

Información legal disponible:
{LEGAL_INFO}

Responde de manera profesional, clara, concisa y precisa. Mantén las respuestas breves (máximo 3-4 frases) 
ya que serán convertidas a voz. Si no tienes información específica, indica que el usuario debe consultar 
con el equipo legal.'''
        
        ai_response = await _generate_llm_response(system_prompt, transcribed_text, model="gpt-4o")
        logger.info(f"✅ AI Response: {ai_response[:100]}...")
        
        # Step 3: Convert AI response to speech
        logger.info("🔊 Converting response to speech...")
        audio_stream = elevenlabs_client.text_to_speech.stream(
            text=ai_response,
            voice_id="VmejBeYhbrcTPwDniox7",  # Lina - Latin American female
            model_id="eleven_multilingual_v2",
            voice_settings=VoiceSettings(
                stability=0.6,
                similarity_boost=0.8,
                style=0.0,
                use_speaker_boost=True
            )
        )
        
        # Collect audio bytes from stream
        audio_bytes = b""
        for chunk in audio_stream:
            audio_bytes += chunk
        
        # Return base64 encoded audio
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        logger.info("✅ Voice chat completed successfully")
        
        return {
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
    '''
    Text-based chat flow (alternative to voice):
    1. Get user text input
    2. Get AI response using LLM
    3. Convert response to speech using ElevenLabs TTS (optional)
    '''
    try:
        if not OPENAI_API_KEY:
            raise HTTPException(status_code=503, detail="LLM not configured")
        
        text = request.get('text', '').strip()
        if not text:
            raise HTTPException(status_code=400, detail="Text is required")
        
        logger.info(f"💬 Text chat request: {text}")
        
        # Get AI response
        system_prompt = f'''Eres un asistente legal experto en Prados de Paraíso. 
Tu trabajo es responder preguntas sobre condiciones legales, propiedad, posesión y saneamiento.

Información legal disponible:
{LEGAL_INFO}

Responde de manera profesional, clara y precisa. Si no tienes información específica, 
indica que el usuario debe consultar con el equipo legal.'''
        
        ai_response = await _generate_llm_response(system_prompt, text, model="gpt-4o")
        logger.info(f"✅ AI Response generated")
        
        # Optionally convert to speech if ElevenLabs is available
        audio_url = None
        if elevenlabs_client:
            try:
                # Use Lina - Warm Latin American female voice (works well for Peruvian Spanish)
                agent_voice_id = "VmejBeYhbrcTPwDniox7"  # Lina voice
                
                logger.info(f"🔊 Converting response to speech with Lina (streaming mode)...")
                audio_stream = elevenlabs_client.text_to_speech.stream(
                    text=ai_response,
                    voice_id=agent_voice_id,
                    model_id="eleven_multilingual_v2",
                    voice_settings=VoiceSettings(
                        stability=0.6,
                        similarity_boost=0.8,
                        style=0.0,
                        use_speaker_boost=True
                    )
                )
                
                audio_bytes = b""
                for chunk in audio_stream:
                    audio_bytes += chunk
                
                audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
                audio_url = f"data:audio/mpeg;base64,{audio_base64}"
                logger.info("Audio generated")
            except Exception as e:
                logger.warning(f"Could not generate audio: {str(e)}")
        
        return {
            "user_text": text,
            "ai_response": ai_response,
            "audio_url": audio_url,
            "format": "mp3" if audio_url else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in text chat: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error procesando consulta: {str(e)}")


# HeyGen Streaming Avatar Endpoints
@api_router.post("/heygen/streaming-token")
async def create_heygen_streaming_token():
    """
    Generate a session token for HeyGen Streaming Avatar.
    This token is used by the frontend SDK to establish a WebRTC connection.
    """
    try:
        if not HEYGEN_API_KEY:
            raise HTTPException(status_code=503, detail="HeyGen API key not configured")
        
        import httpx
        
        logger.info("🎬 Creating HeyGen streaming session token...")
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.heygen.com/v1/streaming.create_token",
                headers={
                    "x-api-key": HEYGEN_API_KEY,
                    "Content-Type": "application/json"
                },
                timeout=10.0
            )
            
            if response.status_code != 200:
                logger.error(f"❌ HeyGen token creation failed: {response.status_code} - {response.text}")
                raise HTTPException(
                    status_code=response.status_code, 
                    detail=f"HeyGen API error: {response.text}"
                )
            
            data = response.json()
            token = data.get("data", {}).get("token")
            
            if not token:
                logger.error(f"❌ No token in response: {data}")
                raise HTTPException(status_code=500, detail="No token returned from HeyGen")
            
            logger.info("✅ HeyGen streaming token created successfully")
            
            return {
                "token": token,
                "avatar_id": HEYGEN_AVATAR_ID
            }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error creating HeyGen token: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating streaming token: {str(e)}")


# Voice Agent Endpoint (using ElevenLabs Agent voice and knowledge)
@api_router.post("/voice-agent")
async def voice_agent(audio: UploadFile = File(...), agent_id: str = Form(...)):
    '''
    Send audio to get response using the ElevenLabs Agent configured voice.
    This endpoint:
    1. Transcribes user audio (STT)
    2. Gets agent configuration (voice, personality)
    3. Generates response using agent knowledge base context
    4. Converts to speech using agent voice (TTS)
    '''
    try:
        if not elevenlabs_client:
            raise HTTPException(status_code=503, detail="ElevenLabs not configured")
        
        if not OPENAI_API_KEY:
            raise HTTPException(status_code=503, detail="LLM not configured")
        
        logger.info(f"🎙️ Processing voice with agent: {agent_id}")
        
        # Step 1: Transcribe audio
        audio_content = await audio.read()
        transcription_response = elevenlabs_client.speech_to_text.convert(
            file=io.BytesIO(audio_content),
            model_id="scribe_v1"
        )
        
        transcribed_text = transcription_response.text if hasattr(transcription_response, 'text') else str(transcription_response)
        logger.info(f"✅ Transcribed: {transcribed_text}")
        
        if not transcribed_text or len(transcribed_text.strip()) == 0:
            raise HTTPException(status_code=400, detail="No se pudo transcribir el audio.")
        
        # Step 2: Get agent details to use the correct voice
        agent_voice_id = "VmejBeYhbrcTPwDniox7"  # Lina - Latin American female voice
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
        
        # Step 3: Generate AI response using the knowledge base context
        system_prompt = f'''Eres {agent_name}, un asistente legal experto especializado en Prados de Paraíso.
Tu trabajo es responder preguntas sobre condiciones legales, propiedad, posesión y saneamiento del proyecto.

Información legal disponible:
{LEGAL_INFO}

Responde de manera profesional, clara, concisa y amigable como lo haría el Dr. Prados.
Mantén las respuestas breves (máximo 3-4 frases) ya que serán convertidas a voz.'''
        
        ai_response = await _generate_llm_response(system_prompt, transcribed_text, model="gpt-4o")
        logger.info(f"✅ AI Response generated")
        
        # Step 4: Convert to speech using agent's voice
        audio_stream = elevenlabs_client.text_to_speech.stream(
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
        for chunk in audio_stream:
            audio_bytes += chunk
        
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        logger.info("✅ Voice agent response completed")
        
        return {
            "transcribed_text": transcribed_text,
            "agent_response": ai_response,
            "audio_url": f"data:audio/mpeg;base64,{audio_base64}",
            "format": "mp3",
            "voice_used": agent_voice_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error in voice agent: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error procesando consulta: {str(e)}")


# WebSocket for real-time chat
@api_router.websocket("/ws/chat/{conversation_id}")
async def websocket_chat(websocket: WebSocket, conversation_id: str):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            # Create user message
            user_msg = Message(
                conversation_id=conversation_id,
                role="user",
                content=message_data['content']
            )
            doc = prepare_for_mongo(user_msg.model_dump())
            await db.messages.insert_one(doc)
            
            # Send user message confirmation
            await websocket.send_json(user_msg.model_dump(mode='json'))
            
            # Generate AI response
            system_prompt = f'''Eres un asistente legal experto en Prados de Paraíso.

Información legal:
{LEGAL_INFO}

Responde de manera profesional y clara.'''
            
            ai_response = await _generate_llm_response(system_prompt, message_data['content'], model="gpt-4o")
            
            # Create assistant message
            assistant_msg = Message(
                conversation_id=conversation_id,
                role="assistant",
                content=ai_response
            )
            doc = prepare_for_mongo(assistant_msg.model_dump())
            await db.messages.insert_one(doc)
            
            # Send assistant message
            await websocket.send_json(assistant_msg.model_dump(mode='json'))
            
            # Update conversation
            await db.conversations.update_one(
                {"id": conversation_id},
                {
                    "$set": {"updated_at": datetime.now(timezone.utc).isoformat()},
                    "$inc": {"message_count": 2}
                }
            )
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for conversation {conversation_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        await websocket.close()

# ============================================================================
# LIVE AVATAR ENDPOINTS
# ============================================================================

class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    message: str
    conversation_id: Optional[str] = None

@api_router.post("/liveavatar/session-token")
async def get_liveavatar_token():
    """
    Generate LiveAvatar session token for frontend
    """
    try:
        if not liveavatar_service:
            raise HTTPException(status_code=503, detail="LiveAvatar service not initialized")
        
        token_data = await liveavatar_service.create_session_token()
        
        if not token_data:
            raise HTTPException(status_code=500, detail="Failed to create LiveAvatar token")
        
        return token_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating LiveAvatar token: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@api_router.get("/liveavatar/config")
async def get_liveavatar_config():
    """
    Get LiveAvatar configuration
    """
    if not liveavatar_service:
        raise HTTPException(status_code=503, detail="LiveAvatar service not initialized")
    
    return liveavatar_service.get_avatar_config()

# ============================================================================
# LIVEAVATAR API ENDPOINTS (FULL Mode con LiveKit)
# ============================================================================

class LiveAvatarSessionRequest(BaseModel):
    """Request model for creating LiveAvatar session"""
    include_context: bool = True

@api_router.post("/liveavatar/create-session")
async def create_liveavatar_session(request: LiveAvatarSessionRequest):
    """
    Crea una sesión de LiveAvatar en FULL Mode con LiveKit WebRTC
    Cierra automáticamente sesiones anteriores para evitar límite de concurrencia.
    
    Returns:
        Session token, room name y configuración de LiveKit
    """
    try:
        if not liveavatar_service:
            raise HTTPException(status_code=503, detail="LiveAvatar service not initialized")
        
        # Try to close any existing sessions first to avoid concurrent limit
        try:
            logger.info("🔄 Checking for existing sessions...")
            existing_sessions = await liveavatar_service.list_sessions()
            if existing_sessions and len(existing_sessions) > 0:
                logger.warning(f"⚠️ Found {len(existing_sessions)} existing session(s), closing them...")
                for session in existing_sessions:
                    session_id = session.get('session_id')
                    if session_id:
                        await liveavatar_service.close_session(session_id)
                        logger.info(f"✅ Closed session: {session_id}")
                # Wait a moment for cleanup
                await asyncio.sleep(1)
        except Exception as cleanup_error:
            logger.error(f"Error cleaning up sessions: {str(cleanup_error)}")
            # Continue anyway
        
        # System prompt para Marianne (Asistente Legal)
        context = None
        if request.include_context:
            context = '''Eres Marianne, la Asistente Legal IA de Prados de Paraíso.

Tu rol es proporcionar información clara, precisa y profesional sobre temas de 
propiedad, posesión legítima y saneamiento legal en Perú, basándote EXCLUSIVAMENTE 
en la base de conocimientos proporcionada.

Características de tu personalidad:
- Profesional pero cercana y amable
- Paciente y didáctica al explicar conceptos legales complejos
- Hablas con acento argentino neutral
- Usas lenguaje accesible sin perder precisión técnica

IMPORTANTE:
- Solo responde con información que esté en la base de conocimientos
- Si no tienes información sobre algo, admítelo claramente
- Sé concisa pero completa en tus respuestas
- Responde siempre en español
'''
        
        session_data = await liveavatar_service.create_session_token(context=context)
        
        return {
            "success": True,
            "session": session_data
        }
        
    except Exception as e:
        logger.error(f"Error creating LiveAvatar session: {str(e)}")
        error_msg = str(e)
        
        # Provide user-friendly error messages
        if "Concurrent limit" in error_msg:
            error_msg = "Límite de sesiones alcanzado. Por favor recarga la página en unos segundos."
        elif "avatar not found" in error_msg or "Avatar not found" in error_msg:
            error_msg = "Avatar no disponible. Contacta al administrador para configurar un Avatar ID válido."
        
        raise HTTPException(status_code=500, detail=error_msg)

class LiveAvatarContextRequest(BaseModel):
    """Request model for sending knowledge context"""
    room_name: str
    context: str

@api_router.post("/liveavatar/send-context")
async def send_liveavatar_context(request: LiveAvatarContextRequest):
    """
    Envía contexto de la base de conocimientos a la sesión activa
    """
    try:
        if not liveavatar_service:
            raise HTTPException(status_code=503, detail="LiveAvatar service not initialized")
        
        result = await liveavatar_service.send_knowledge_context(
            room_name=request.room_name,
            context=request.context
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error sending context to LiveAvatar: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.delete("/liveavatar/close-session/{room_name}")
async def close_liveavatar_session(room_name: str):
    """
    Cierra una sesión activa de LiveAvatar
    """
    try:
        if not liveavatar_service:
            raise HTTPException(status_code=503, detail="LiveAvatar service not initialized")
        
        success = await liveavatar_service.close_session(room_name)
        
        return {"success": success}
        
    except Exception as e:
        logger.error(f"Error closing LiveAvatar session: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# CHAT WITH SEMANTIC SEARCH + LIVEAVATAR
# ============================================================================

@api_router.post("/chat")
async def chat_with_knowledge_base(request: ChatRequest):
    """
    Chat endpoint with semantic search in SQLite knowledge base
    
    Flow:
    1. Receive user message
    2. Search relevant documents in SQLite (semantic search)
    3. Generate response with LLM using retrieved context
    4. Return response for LiveAvatar to speak
    """
    try:
        user_message = request.message.strip()
        if not user_message:
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        logger.info(f"💬 Chat request: {user_message[:100]}...")
        
        # Step 1: Semantic search in SQLite knowledge base
        relevant_docs = sqlite_kb.search(query=user_message, top_k=3)
        
        # Step 2: Build context from retrieved documents
        context_parts = []
        for i, doc in enumerate(relevant_docs, 1):
            if doc['score'] > 0.4:  # Solo incluir documentos relevantes
                context_parts.append(
                    f"[Documento {i}] {doc['titulo']}\n{doc['contenido']}"
                )
        
        context = "\n\n".join(context_parts) if context_parts else "No se encontró información específica en la base de conocimientos."
        
        # Step 3: Generate response with LLM
        session_id = request.conversation_id or str(uuid.uuid4())
        system_prompt = f'''Eres Marianne, asistente legal experta de Prados de Paraíso, especializada en
temas de propiedad, posesión legítima y saneamiento legal en Perú.

Tu rol es proporcionar información clara, precisa y profesional basada EXCLUSIVAMENTE en la 
base de conocimientos proporcionada.

IMPORTANTE:
- Solo responde con información que esté en la base de conocimientos
- Si no tienes información sobre algo, admítelo claramente
- Sé concisa pero completa en tus respuestas
- Usa lenguaje profesional pero accesible y amigable
- Responde en español con acento argentino neutral

BASE DE CONOCIMIENTOS:
{context}
'''
        ai_response = await _generate_llm_response(system_prompt, user_message, model="gpt-4o-mini")
        
        logger.info(f"✅ Generated response: {ai_response[:100]}...")
        
        # Step 4: Return response
        return {
            "message": user_message,
            "response": ai_response,
            "context_used": len(relevant_docs),
            "sources": [
                {
                    "title": doc["titulo"],
                    "relevance": round(doc["score"], 2)
                }
                for doc in relevant_docs
            ],
            "conversation_id": session_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")

# ============================================================================
# END LIVE AVATAR ENDPOINTS
# ============================================================================

# Include all API routes
app.include_router(api_router)

@app.on_event("startup")
async def startup_db_client():
    logger.info("✅ Application started successfully")

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()