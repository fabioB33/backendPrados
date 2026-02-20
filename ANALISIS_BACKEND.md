# Análisis en profundidad del backend - Prados de Paraíso

## Estructura del proyecto

| Ruta | Descripción |
|------|-------------|
| `server.py` | FastAPI: rutas API, MongoDB, LLM, ElevenLabs, SQLite KB |
| `services/sqlite_knowledge.py` | Base de conocimiento legal en SQLite + embeddings (SentenceTransformer) |
| `services/heygen_service.py` | Sesiones HeyGen LiveAvatar (streaming) |
| `services/liveavatar.py` | Alternativa LiveAvatar (LIVEAVATAR_API_KEY) |
| `services/knowledge_base.py` | (revisar uso) |
| `load_documents.py` | Carga documentos legales en SQLite |
| `migrate_to_sqlite.py` | Migración MongoDB → SQLite |
| `test_backend.py` | Script de prueba (imports, rutas, config) |
| `tests/test_api.py` | Tests pytest de la API |
| `tests/conftest.py` | Fixtures pytest |
| `Dockerfile` | Imagen Python 3.11, uvicorn puerto 8000 |
| `.env.example` / `env.example` | Ejemplos de variables de entorno |

---

## Stack y dependencias

- **Framework:** FastAPI, Uvicorn
- **Bases de datos:** MongoDB (Motor), SQLite (base de conocimiento legal)
- **LLM:** `emergentintegrations.llm.chat` (LlmChat, UserMessage) con modelo `openai` / `gpt-4o`
- **Voz:** ElevenLabs (TTS/STT)
- **Avatar:** HeyGen (HEYGEN_API_KEY, HEYGEN_AVATAR_ID) vía `heygen_service.py`
- **PDF:** reportlab
- **requirements.txt:** ~140 paquetes; incluye `openai`, `sentence-transformers`, `torch`, `pymongo`, `motor`, `elevenlabs`, etc. No aparece `emergentintegrations` en el listado leído (puede estar como dependencia de otro paquete o faltar).

---

## Errores y problemas detectados

### 1. **MONGO_URL y DB_NAME obligatorios** (crítico)

En `server.py` líneas 31-33:

```python
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]
```

Si `MONGO_URL` o `DB_NAME` no están definidos, la aplicación lanza `KeyError` al arrancar. En entornos tipo Render/Railway sin MongoDB configurado, el servicio no inicia.

**Recomendación:** Usar `os.environ.get('MONGO_URL', 'mongodb://localhost:27017')` y `os.environ.get('DB_NAME', 'prados_legal_hub')` (o similar) y documentar en `.env.example`.

---

### 2. **LLM: emergentintegrations vs .env.example** (crítico)

- El código usa **`emergentintegrations.llm.chat`** y **`EMERGENT_LLM_KEY`** (líneas 14, 35, 266, 511, 607, 724).
- `.env.example` solo documenta **`OPENAI_API_KEY`**.
- No hay `OPENAI_API_KEY` en `server.py`; sí `EMERGENT_LLM_KEY`. Quien clone el repo y use solo `.env.example` no tendrá la clave que espera el código.
- Si el proyecto debe usar OpenAI directo, habría que sustituir `emergentintegrations` por `openai` y usar `OPENAI_API_KEY` (como en el resumen de conversación anterior).

**Recomendación:** Unificar: o bien documentar `EMERGENT_LLM_KEY` en `.env.example` y asegurar que `emergentintegrations` esté en `requirements.txt`, o bien migrar a `openai` + `OPENAI_API_KEY` y actualizar `.env.example` y documentación.

---

### 3. **Test backend importa variable inexistente** (medio)

En `test_backend.py` líneas 59-62:

```python
from server import OPENAI_API_KEY, ELEVENLABS_API_KEY
```

En `server.py` no existe `OPENAI_API_KEY`, solo `EMERGENT_LLM_KEY`. Al ejecutar `test_config()` se produce `AttributeError`.

**Recomendación:** Importar `EMERGENT_LLM_KEY` (o la variable que realmente use `server.py`) o definir `OPENAI_API_KEY` en el servidor si se migra a OpenAI.

---

### 4. **SQLite + SentenceTransformer: rendimiento y despliegue** (medio)

En `services/sqlite_knowledge.py`:

- **Ruta por defecto:** `db_path: str = "/app/backend/prados.db"` — válida en Docker/Render si el trabajo es `/app` y el backend está en `/app/backend`; en desarrollo local puede fallar si se ejecuta desde otra ruta.
- **Modelo:** Se carga `SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')` en `__init__`. Es pesado (memoria y tiempo) y puede provocar timeouts o “out of memory” en entornos con límite (p. ej. 512 MB).
- **Dependencias:** `sentence-transformers` arrastra `torch`; `requirements.txt` incluye `torch==2.10.0` — mucho peso para un servicio ligero.

**Recomendación:**  
- Hacer la ruta de SQLite configurable por env (p. ej. `SQLITE_DB_PATH`) con fallback según entorno (local vs `/app`).  
- Opción de búsqueda solo por keywords (sin embeddings) en producción, o cargar el modelo bajo demanda / en un worker separado.  
- Valorar búsqueda por keywords o un embedding más liviano para reducir memoria.

---

### 5. **Dos servicios de avatar (HeyGen)** (bajo)

- **`services/heygen_service.py`:** usa `HEYGEN_API_KEY` y `HEYGEN_AVATAR_ID`; expone `create_session_token` para streaming.
- **`services/liveavatar.py`:** usa `LIVEAVATAR_API_KEY` y `LIVEAVATAR_AVATAR_ID`.

El `server.py` no importa ninguno de los dos en el fragmento revisado; las rutas de “session” para el frontend (Live Avatar) no se han localizado en este análisis. Si el frontend llama a un endpoint tipo `/api/liveavatar/session` o similar, debe existir en `server.py` y usar uno de estos servicios de forma coherente (por ejemplo solo `heygen_service` con `HEYGEN_*`).

**Recomendación:** Confirmar qué endpoint usa el frontend para la sesión del avatar y asegurar que existe en `server.py` y que usa las mismas variables de entorno que se documentan en `.env.example`.

---

### 6. **Endpoint /api/chat vs /api/text-chat** (medio)

El frontend (p. ej. `TextChatInput.js`) puede llamar a **`POST /api/chat`** con cuerpo `{ message }`. En el backend solo se han visto:

- **`POST /api/text-chat`** con cuerpo `{ text }`.

Si el frontend usa `/api/chat` y el backend solo tiene `/api/text-chat`, las peticiones fallan (404 o 422). Si el frontend ya usa `/api/text-chat` con `text`, no hay problema.

**Recomendación:** Comprobar en el frontend la URL y el nombre del campo (`message` vs `text`). Si se usa `/api/chat`, añadir en el backend un endpoint `POST /api/chat` que acepte `{ "message": "..." }` y delegue en la misma lógica que `text-chat` (o unificar en un solo endpoint y actualizar el frontend).

---

### 7. **CORS** (informativo)

En `server.py` se configuran `origins` con dominios concretos y `localhost:3000`/`3001`. Para desarrollo o nuevos dominios hay que añadirlos a `allow_origins`.

---

### 8. **Dockerfile** (informativo)

- Base: `python:3.11-slim`.  
- Puerto: `EXPOSE 8000` y `PORT` por defecto 8000.  
- Comando: `uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000}`.  
- No se instala `emergentintegrations` explícitamente; si es dependencia de otro paquete, debe estar en `requirements.txt`. Si no, el contenedor fallará al importar.

---

### 9. **Archivos .env duplicados** (bajo)

Existen `.env.example` y `env.example`. Conviene dejar uno solo (por ejemplo `.env.example`) y documentar ahí todas las variables necesarias (MONGO_*, EMERGENT_LLM_KEY o OPENAI_API_KEY, ELEVENLABS_*, HEYGEN_*, etc.).

---

### 10. **requirements.txt muy grande** (informativo)

Incluye ML (torch, sentence-transformers, scikit-learn, pandas, etc.), Google GenAI, LiteLLM, Hugging Face, etc. No todo es necesario para el flujo mínimo (API + MongoDB + LLM + ElevenLabs + SQLite). Un entorno mínimo podría reducir dependencias para despliegues más ligeros y rápidos.

---

## Resumen de prioridades

| Prioridad | Problema | Acción sugerida |
|-----------|----------|------------------|
| Crítico   | MONGO_URL/DB_NAME obligatorios | Usar `get` con valores por defecto y documentar |
| Crítico   | LLM: emergentintegrations vs OPENAI y .env | Unificar: OpenAI + OPENAI_API_KEY o documentar EMERGENT_LLM_KEY y dependencias |
| Medio     | test_backend.py importa OPENAI_API_KEY | Usar EMERGENT_LLM_KEY o alinear con el servidor |
| Medio     | SQLite path y SentenceTransformer en producción | Env para ruta, modo sin modelo pesado o búsqueda por keywords |
| Medio     | /api/chat vs /api/text-chat y body message vs text | Alinear endpoint y nombre del campo con el frontend |
| Bajo      | Dos servicios HeyGen (heygen_service vs liveavatar) | Unificar y documentar qué endpoint y env usa el frontend |
| Bajo      | .env.example vs env.example | Un solo archivo de ejemplo con todas las variables |

---

## Variables de entorno a documentar

Según el código actual:

- **MongoDB:** `MONGO_URL`, `DB_NAME`
- **LLM:** `EMERGENT_LLM_KEY` (o `OPENAI_API_KEY` si se migra)
- **Voz:** `ELEVENLABS_API_KEY`
- **Avatar:** `HEYGEN_API_KEY`, `HEYGEN_AVATAR_ID` (y opcionalmente LIVEAVATAR_* si se usa `liveavatar.py`)
- **Servidor:** `HOST`, `PORT` (opcionales)
- **SQLite (si se añade):** `SQLITE_DB_PATH` o equivalente

---

## Rutas API principales (resumen)

- `GET /api/` — Root  
- `POST /api/users`, `GET /api/users`, `GET /api/users/{user_id}`  
- `POST /api/conversations`, `GET /api/conversations/user/{user_id}`, `GET /api/conversations/{id}`  
- `POST /api/messages`, `GET /api/messages/{conversation_id}`  
- `POST /api/documents`, `GET /api/documents/user/{user_id}`  
- `GET /api/analytics/overview`  
- `GET /api/conversations/{id}/export` (PDF)  
- `GET /api/search`  
- `POST /api/tts` — Text-to-Speech (ElevenLabs)  
- `POST /api/voice-chat` — Voz completo (STT + LLM + TTS)  
- `POST /api/text-chat` — Chat por texto (+ TTS opcional)  
- `WebSocket /api/ws/chat/{conversation_id}`  

No se ha localizado en este análisis un endpoint explícito `POST /api/chat` ni `GET/POST /api/liveavatar/session`; conviene verificarlos si el frontend los usa.
