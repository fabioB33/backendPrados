"""
LiveAvatar Service
Maneja la interacción con la API de HeyGen LiveAvatar
"""
import os
import httpx
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)

class LiveAvatarService:
    """
    Servicio para interactuar con LiveAvatar API
    """
    
    def __init__(self):
        self.api_key = os.getenv("LIVEAVATAR_API_KEY")
        self.avatar_id = os.getenv("LIVEAVATAR_AVATAR_ID")
        self.base_url = "https://api.heygen.com/v1"
        
        if not self.api_key:
            logger.warning("⚠️ LIVEAVATAR_API_KEY not configured")
        if not self.avatar_id:
            logger.warning("⚠️ LIVEAVATAR_AVATAR_ID not configured")
        
        logger.info(f"✅ LiveAvatar Service initialized (Avatar ID: {self.avatar_id})")
    
    async def create_session_token(self) -> Optional[Dict]:
        """
        Crea un token de sesión para LiveAvatar
        
        Returns:
            Dict con token y información de sesión o None si hay error
        """
        if not self.api_key:
            logger.error("LiveAvatar API key not configured")
            return None
        
        try:
            async with httpx.AsyncClient() as client:
                headers = {
                    "x-api-key": self.api_key,
                    "Content-Type": "application/json"
                }
                
                response = await client.post(
                    f"{self.base_url}/streaming.create_token",
                    headers=headers,
                    timeout=10.0
                )
                
                if response.status_code != 200:
                    logger.error(f"LiveAvatar token error: {response.status_code} - {response.text}")
                    return None
                
                data = response.json()
                token_data = data.get("data", {})
                
                logger.info("✅ LiveAvatar session token created")
                return {
                    "token": token_data.get("token"),
                    "avatar_id": self.avatar_id,
                    "expires_at": token_data.get("expires_at")
                }
                
        except httpx.TimeoutException:
            logger.error("LiveAvatar API timeout")
            return None
        except Exception as e:
            logger.error(f"Error creating LiveAvatar token: {str(e)}")
            return None
    
    def get_avatar_config(self) -> Dict:
        """
        Retorna la configuración del avatar para el cliente
        """
        return {
            "avatar_id": self.avatar_id,
            "avatar_name": "Marianne - IA",
            "language": "es-ES",
            "voice_settings": {
                "rate": 1.0,
                "emotion": "friendly"
            }
        }
