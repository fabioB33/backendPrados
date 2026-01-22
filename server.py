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

# Helper functions
async def get_ai_response(system_prompt: str, user_message: str) -> str:
    """Generate AI response using OpenAI"""
    if not openai_client:
        raise HTTPException(status_code=503, detail="OpenAI not configured")
    
    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.7,
            max_tokens=1000
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
    return {"message": "OK"}

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
        
        if not OPENAI_API_KEY:
            raise HTTPException(status_code=503, detail="OpenAI not configured")
        
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
        
        ai_response = await get_ai_response(system_prompt, transcribed_text)
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
        if not OPENAI_API_KEY:
            raise HTTPException(status_code=503, detail="OpenAI not configured")
        
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
        
        ai_response = await get_ai_response(system_prompt, text)
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
        import traceback
        error_trace = traceback.format_exc()
        logger.error(f"❌ Error in text chat: {str(e)}")
        logger.error(f"📋 Traceback: {error_trace}")
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
        
        if not OPENAI_API_KEY:
            raise HTTPException(status_code=503, detail="OpenAI not configured")
        
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
        
        ai_response = await get_ai_response(system_prompt, transcribed_text)
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
        import traceback
        error_trace = traceback.format_exc()
        logger.error(f"❌ Error in voice agent: {str(e)}")
        logger.error(f"📋 Traceback: {error_trace}")
        raise HTTPException(status_code=500, detail=f"Error procesando consulta: {str(e)}")


app.include_router(api_router)