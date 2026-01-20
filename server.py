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
from emergentintegrations.llm.chat import LlmChat, UserMessage
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
mongo_url = os.environ.get('MONGO_URL')
if not mongo_url:
    raise ValueError("MONGO_URL environment variable is required")

db_name = os.environ.get('DB_NAME', 'prados_legal')
client = AsyncIOMotorClient(mongo_url)
db = client[db_name]

# LLM Configuration
EMERGENT_LLM_KEY = os.environ.get('EMERGENT_LLM_KEY')
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

app = FastAPI()
api_router = APIRouter(prefix="/api")

# Información legal precargada
LEGAL_INFO = """
Prados de Paraíso - Información Legal Completa:

1. CONDICIÓN LEGAL DEL PROYECTO:
- 50% del terreno: Propiedad adquirida mediante compraventa de acciones y derechos
- 50% restante: Terreno bajo condición de posesión legítima y mediata

2. DIFERENCIA ENTRE PROPIEDAD Y POSESIÓN:
- Propiedad: Derecho que otorga titularidad legal inscribible en Registros Públicos
- Posesión: Ejercicio de hecho de poderes inherentes a la propiedad

3. PREGUNTAS FRECUENTES:

Q1: ¿Cuándo entregan el título de propiedad?
R: La condición legal es la POSESIÓN. Se entrega contrato de transferencia de posesión. Para obtener título de propiedad, el cliente debe gestionar saneamiento tras completar pago.

Q2: ¿En qué estado se encuentra el lote?
R: Posesión legítima, mediata y de buena fe, respaldada por escrituras públicas desde 1998.

Q3: ¿Tenemos partida registral?
R: No hay partida registral a nombre de la desarrolladora. El predio figura a nombre de DIREFOR (entidad estatal). Esto no representa riesgo ya que poseemos legítimamente desde 1998.

Q4: ¿Tipos de posesión?
R: Legítima (mediata e inmediata) e Ilegítima (buena fe, mala fe, precaria). Nuestra situación: Posesión Legítima Mediata y de Buena Fe.

Q5: ¿Por qué no hay partida registral?
R: Decisión estratégica comercial. La posesión es un derecho reconocido y protegido por ley.

Q6: ¿Procedimiento para sacar partida registral?
R: Vía prescripción adquisitiva de dominio. Requiere asesoría legal especializada. Costos asumidos por el adquirente.

Q7: ¿Garantía al comprar?
R: Marca con trayectoria, posesión legítima respaldada por escrituras públicas, asesoramiento legal especializado (DS Casa Hierro Abogados), convenio con Notaría Tambini, y más de 500 clientes satisfechos.

4. SANEAMIENTO FÍSICO LEGAL:
- Proceso de regularización para acceso a Registros Públicos
- Vía: Prescripción Adquisitiva de Dominio (Usucapión)
- Requisitos: Posesión continua, pacífica y pública

5. RESPALDO LEGAL:
- Notaría Tambini
- Casahierro Abogados
"""

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
    """Convert datetime objects to ISO strings for MongoDB"""
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
        system_prompt = f"""Eres un asistente legal experto en Prados de Paraíso. 
Tu trabajo es responder preguntas sobre condiciones legales, propiedad, posesión y saneamiento.

Información legal disponible:
{LEGAL_INFO}

Responde de manera profesional, clara y precisa. Si no tienes información específica, 
indica que el usuario debe consultar con el equipo legal."""
        
        chat = LlmChat(
            api_key=EMERGENT_LLM_KEY,
            session_id=msg.conversation_id,
            system_message=system_prompt
        ).with_model("openai", "gpt-4o")
        
        user_message = UserMessage(text=msg.content)
        ai_response = await chat.send_message(user_message)
        
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
async def voice_chat(audio: UploadFile = File(...)):
    """
    Complete voice chat flow:
    1. Transcribe audio using ElevenLabs STT
    2. Get AI response using LLM
    3. Convert response to speech using ElevenLabs TTS
    """
    try:
        if not elevenlabs_client:
            raise HTTPException(status_code=503, detail="ElevenLabs not configured")
        
        if not EMERGENT_LLM_KEY:
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
        system_prompt = f"""Eres un asistente legal experto en Prados de Paraíso. 
Tu trabajo es responder preguntas sobre condiciones legales, propiedad, posesión y saneamiento.

Información legal disponible:
{LEGAL_INFO}

Responde de manera profesional, clara, concisa y precisa. Mantén las respuestas breves (máximo 3-4 frases) 
ya que serán convertidas a voz. Si no tienes información específica, indica que el usuario debe consultar 
con el equipo legal."""
        
        chat = LlmChat(
            api_key=EMERGENT_LLM_KEY,
            session_id="voice_chat_" + str(uuid.uuid4()),
            system_message=system_prompt
        ).with_model("openai", "gpt-4o")
        
        user_message = UserMessage(text=transcribed_text)
        ai_response = await chat.send_message(user_message)
        logger.info(f"✅ AI Response: {ai_response[:100]}...")
        
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
    Text-based chat flow (alternative to voice):
    1. Get user text input
    2. Get AI response using LLM
    3. Convert response to speech using ElevenLabs TTS (optional)
    """
    try:
        if not EMERGENT_LLM_KEY:
            raise HTTPException(status_code=503, detail="LLM not configured")
        
        text = request.get('text', '').strip()
        if not text:
            raise HTTPException(status_code=400, detail="Text is required")
        
        logger.info(f"💬 Text chat request: {text}")
        
        # Get AI response
        system_prompt = f"""Eres un asistente legal experto en Prados de Paraíso. 
Tu trabajo es responder preguntas sobre condiciones legales, propiedad, posesión y saneamiento.

Información legal disponible:
{LEGAL_INFO}

Responde de manera profesional, clara y precisa. Si no tienes información específica, 
indica que el usuario debe consultar con el equipo legal."""
        
        chat = LlmChat(
            api_key=EMERGENT_LLM_KEY,
            session_id="text_chat_" + str(uuid.uuid4()),
            system_message=system_prompt
        ).with_model("openai", "gpt-4o")
        
        user_message = UserMessage(text=text)
        ai_response = await chat.send_message(user_message)
        logger.info(f"✅ AI Response generated")
        
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
            "user_text": text,
            "ai_response": ai_response,
            "audio_url": audio_url,
            "format": "mp3" if audio_url else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error in text chat: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error procesando consulta: {str(e)}")

# Voice Agent Endpoint (using ElevenLabs Agent's voice and knowledge)
@api_router.post("/voice-agent")
async def voice_agent(audio: UploadFile = File(...), agent_id: str = Form(...)):
    """
    Send audio to get response using the ElevenLabs Agent's configured voice.
    This endpoint:
    1. Transcribes user audio (STT)
    2. Gets agent configuration (voice, personality)
    3. Generates response using agent's knowledge base context
    4. Converts to speech using agent's voice (TTS)
    """
    try:
        if not elevenlabs_client:
            raise HTTPException(status_code=503, detail="ElevenLabs not configured")
        
        if not EMERGENT_LLM_KEY:
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
        
        # Step 3: Generate AI response using the knowledge base context
        system_prompt = f"""Eres {agent_name}, un asistente legal experto especializado en Prados de Paraíso.
Tu trabajo es responder preguntas sobre condiciones legales, propiedad, posesión y saneamiento del proyecto.

Información legal disponible:
{LEGAL_INFO}

Responde de manera profesional, clara, concisa y amigable como lo haría el Dr. Prados.
Mantén las respuestas breves (máximo 3-4 frases) ya que serán convertidas a voz."""
        
        chat = LlmChat(
            api_key=EMERGENT_LLM_KEY,
            session_id=f"agent_{agent_id}_{uuid.uuid4()}",
            system_message=system_prompt
        ).with_model("openai", "gpt-4o")
        
        user_message = UserMessage(text=transcribed_text)
        ai_response = await chat.send_message(user_message)
        logger.info(f"✅ AI Response generated")
        
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
            system_prompt = f"""Eres un asistente legal experto en Prados de Paraíso.

Información legal:
{LEGAL_INFO}

Responde de manera profesional y clara."""
            
            chat = LlmChat(
                api_key=EMERGENT_LLM_KEY,
                session_id=conversation_id,
                system_message=system_prompt
            ).with_model("openai", "gpt-4o")
            
            user_message = UserMessage(text=message_data['content'])
            ai_response = await chat.send_message(user_message)
            
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

app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()