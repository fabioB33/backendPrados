"""
LiveAvatar Service
Maneja la creación de sesiones de LiveAvatar con REST API directa
"""
import os
import httpx
import logging
from typing import Dict, Optional, List

logger = logging.getLogger(__name__)

class LiveAvatarService:
    """Service para interactuar con LiveAvatar API"""
    
    def __init__(self):
        # Use HEYGEN_API_KEY and HEYGEN_AVATAR_ID from .env
        self.api_key = os.getenv('HEYGEN_API_KEY')
        # Use configured avatar_id or fallback to a public avatar for testing
        configured_avatar = os.getenv('HEYGEN_AVATAR_ID')
        # Use Wayne_20240711 as fallback (public interactive avatar)
        self.avatar_id = configured_avatar if configured_avatar else "Wayne_20240711"
        self.base_url = "https://api.heygen.com/v1"
        
        if not self.api_key:
            logger.warning("⚠️ HeyGen API key not configured")
        else:
            logger.info(f"✅ HeyGen Service initialized (Avatar: {self.avatar_id})")
        
        if not self.api_key:
            logger.warning("⚠️ LiveAvatar API key not configured")
        else:
            logger.info(f"✅ LiveAvatar Service initialized (Avatar: {self.avatar_id})")
    
    async def create_session_token(self, context: Optional[str] = None) -> Dict:
        """
        Crea una nueva sesión de streaming de HeyGen con managed LiveKit credentials
        
        Args:
            context: System prompt / personalidad del avatar (knowledge)
            
        Returns:
            Dict con url (LiveKit WebSocket URL), access_token y session_id
        """
        try:
            if not self.api_key:
                raise ValueError("HeyGen API key not configured")
            
            headers = {
                "x-api-key": self.api_key,
                "Content-Type": "application/json"
            }
            
            # Payload según documentación de HeyGen streaming.new
            payload = {
                "quality": "high",
                "avatar_id": self.avatar_id  # Use avatar_id
            }
            
            # Don't include voice settings - HeyGen will use default voice for avatar
            
            # Si se proporciona context, agregarlo como knowledge
            if context:
                payload["knowledge_base"] = context
            
            logger.info(f"🔄 Creating HeyGen streaming session for avatar: {self.avatar_id}")
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Endpoint correcto según documentación de HeyGen
                response = await client.post(
                    f"{self.base_url}/streaming.new",
                    headers=headers,
                    json=payload
                )
                
                if response.status_code != 200:
                    error_detail = response.text
                    logger.error(f"HeyGen API error ({response.status_code}): {error_detail}")
                    raise Exception(f"HeyGen API returned {response.status_code}: {error_detail}")
                
                result = response.json()
                
                logger.info(f"📦 HeyGen API Response keys: {result.keys()}")
                
                # Expected response format according to HeyGen docs:
                # {
                #   "code": 100,
                #   "data": {
                #     "session_id": "...",
                #     "url": "wss://...",  (LiveKit WebSocket URL)
                #     "access_token": "...",  (Client token for LiveKit)
                #     "livekit_agent_token": "..."  (Agent token)
                #   }
                # }
                
                # Check if response has 'data' key (successful response)
                if "data" in result and result.get("data"):
                    data = result.get("data", {})
                    logger.info(f"✅ HeyGen session created successfully")
                    logger.info(f"Session ID: {data.get('session_id', 'N/A')}")
                    logger.info(f"LiveKit URL: {data.get('url', 'N/A')}")
                    
                    return {
                        "session_token": data.get("access_token"),  # Client token for LiveKit
                        "session_id": data.get("session_id"),
                        "url": data.get("url"),  # LiveKit WebSocket URL
                        "avatar_id": self.avatar_id
                    }
                else:
                    raise Exception(f"HeyGen API error: {result.get('message', 'Unknown error')}")
                
        except Exception as e:
            logger.error(f"Error creating HeyGen session: {str(e)}")
            raise
    
    async def list_sessions(self) -> List[Dict]:
        """
        Lista todas las sesiones activas de streaming
        
        Returns:
            List de sesiones activas
        """
        try:
            if not self.api_key:
                return []
            
            headers = {
                "x-api-key": self.api_key
            }
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{self.base_url}/streaming.list",
                    headers=headers
                )
                
                if response.status_code == 200:
                    result = response.json()
                    sessions = result.get("data", {}).get("sessions", [])
                    return sessions
                
                return []
                
        except Exception as e:
            logger.error(f"Error listing sessions: {str(e)}")
            return []
    
    async def close_session(self, session_id: str) -> bool:
        """
        Cierra una sesión de streaming activa
        
        Args:
            session_id: ID de la sesión a cerrar
            
        Returns:
            True si se cerró exitosamente
        """
        try:
            if not self.api_key:
                return False
            
            headers = {
                "x-api-key": self.api_key,
                "Content-Type": "application/json"
            }
            
            payload = {
                "session_id": session_id
            }
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"{self.base_url}/streaming.stop",
                    headers=headers,
                    json=payload
                )
                
                if response.status_code == 200:
                    logger.info(f"✅ Session {session_id} closed successfully")
                    return True
                else:
                    logger.error(f"Failed to close session {session_id}: {response.status_code}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error closing session: {str(e)}")
            return False
        """
        Envía contexto de la base de conocimientos a la sesión activa
        
        Args:
            room_name: Nombre de la sala de LiveKit
            context: Contexto legal de la base de conocimientos
            
        Returns:
            Dict con respuesta de la API
        """
        try:
            headers = {
                "X-API-KEY": self.api_key,
                "Content-Type": "application/json"
            }
            
            payload = {
                "room_name": room_name,
                "event": "knowledge.update",
                "data": {
                    "context": context
                }
            }
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.base_url}/sessions/event",
                    headers=headers,
                    json=payload
                )
                
                if response.status_code != 200:
                    error_detail = response.text
                    logger.error(f"LiveAvatar send context error: {error_detail}")
                    raise Exception(f"Failed to send context: {error_detail}")
                
                return response.json()
                
        except Exception as e:
            logger.error(f"Error sending knowledge context: {str(e)}")
            raise
    
    async def close_session(self, room_name: str) -> bool:
        """
        Cierra una sesión activa
        
        Args:
            room_name: Nombre de la sala a cerrar
            
        Returns:
            bool indicando éxito
        """
        try:
            headers = {
                "X-API-KEY": self.api_key,
                "Content-Type": "application/json"
            }
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.delete(
                    f"{self.base_url}/sessions/{room_name}",
                    headers=headers
                )
                
                return response.status_code == 200
                
        except Exception as e:
            logger.error(f"Error closing session: {str(e)}")
            return False
