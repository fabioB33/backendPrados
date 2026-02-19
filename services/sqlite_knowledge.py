"""
SQLite Knowledge Base Service
Gestiona el conocimiento legal en base de datos local SQLite.
En Render usa búsqueda por keywords (sin modelos) para evitar exceder 512MB RAM.
Localmente puede usar SentenceTransformer si USE_SEMANTIC_SEARCH=1.
"""
import os
import sqlite3
import json
import logging
import re
from typing import List, Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Solo cargar SentenceTransformer si se usa búsqueda semántica (local)
_USE_SEMANTIC = os.environ.get("USE_SEMANTIC_SEARCH", "").lower() in ("1", "true", "yes")


def _default_sqlite_path() -> str:
    """Ruta por defecto: env SQLITE_DB_PATH, o /tmp en Render, o carpeta backend en local."""
    if os.environ.get("SQLITE_DB_PATH"):
        return os.environ["SQLITE_DB_PATH"]
    # En Render/containers el código suele estar en /app y el filesystem es read-only salvo /tmp
    if Path("/app").exists() and Path(__file__).resolve().parts[:2] == ("/", "app"):
        return "/tmp/prados.db"
    return str(Path(__file__).resolve().parent.parent / "prados.db")


class SQLiteKnowledgeBase:
    def __init__(self, db_path: Optional[str] = None):
        """
        Inicializa la base de conocimiento SQLite
        
        Args:
            db_path: Ruta al archivo de base de datos (opcional; por defecto desde env o automático)
        """
        self.db_path = db_path or _default_sqlite_path()
        self.model = None
        self.conn = None
        self.use_semantic = _USE_SEMANTIC
        
        # Inicializar base de datos
        self._init_database()
        
        # Cargar modelo de embeddings solo si se usa búsqueda semántica (consume ~500MB)
        if self.use_semantic:
            self._load_model()
        else:
            logger.info("✅ Using keyword search (lightweight, no model)")
        
        logger.info(f"✅ SQLite KnowledgeBase initialized at {self.db_path}")
    
    def _init_database(self):
        """Crea la base de datos y tablas si no existen"""
        try:
            db_path = Path(self.db_path)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Crear tabla conocimiento_legal
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conocimiento_legal (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    titulo TEXT NOT NULL,
                    contenido TEXT NOT NULL,
                    embedding TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Crear índice para búsqueda por título
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_titulo 
                ON conocimiento_legal(titulo)
            ''')
            
            conn.commit()
            conn.close()
            
            logger.info("✅ Database tables created/verified")
            
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            raise
    
    def _load_model(self):
        """Carga el modelo de sentence transformers (solo si USE_SEMANTIC_SEARCH=1)"""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            logger.info("✅ Sentence transformer model loaded")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def _get_connection(self):
        """Obtiene una conexión a la base de datos"""
        return sqlite3.connect(self.db_path)
    
    def add_document(self, titulo: str, contenido: str, metadata: Optional[Dict] = None):
        """
        Agrega un documento a la base de conocimiento
        
        Args:
            titulo: Título del documento
            contenido: Contenido del documento
            metadata: Metadatos adicionales
        """
        try:
            # Generar embedding
            embedding = self.model.encode(contenido).tolist()
            embedding_json = json.dumps(embedding)
            
            # Convertir metadata a JSON
            metadata_json = json.dumps(metadata) if metadata else None
            
            # Insertar en base de datos
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO conocimiento_legal (titulo, contenido, embedding, metadata)
                VALUES (?, ?, ?, ?)
            ''', (titulo, contenido, embedding_json, metadata_json))
            
            conn.commit()
            doc_id = cursor.lastrowid
            conn.close()
            
            logger.info(f"✅ Document added: {titulo} (ID: {doc_id})")
            return doc_id
            
        except Exception as e:
            logger.error(f"Error adding document: {str(e)}")
            raise
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Realiza búsqueda en la base de conocimiento.
        Con USE_SEMANTIC_SEARCH=1: búsqueda semántica (requiere ~500MB RAM).
        Por defecto: búsqueda por keywords (ligera, ideal para Render).
        """
        if self.model:
            return self._search_semantic(query, top_k)
        return self._search_keywords(query, top_k)
    
    def _search_keywords(self, query: str, top_k: int) -> List[Dict]:
        """Búsqueda por coincidencia de palabras clave (sin modelo, ligera)."""
        try:
            words = [w.strip().lower() for w in re.split(r'\s+', query) if len(w.strip()) > 2]
            if not words:
                words = [query.lower()]
            
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute('SELECT id, titulo, contenido FROM conocimiento_legal')
            rows = cursor.fetchall()
            conn.close()
            
            if not rows:
                logger.warning("No documents in knowledge base")
                return []
            
            results = []
            for row in rows:
                doc_id, titulo, contenido = row
                text = f"{titulo} {contenido}".lower()
                matches = sum(1 for w in words if w in text)
                score = matches / len(words) if words else 0
                if score > 0:
                    results.append({
                        'id': doc_id,
                        'titulo': titulo,
                        'contenido': contenido,
                        'score': float(score)
                    })
            
            results.sort(key=lambda x: x['score'], reverse=True)
            top_results = results[:top_k]
            logger.info(f"Search (keywords): '{query}' - Found {len(top_results)} documents")
            return top_results
        except Exception as e:
            logger.error(f"Error searching: {str(e)}")
            return []
    
    def _search_semantic(self, query: str, top_k: int) -> List[Dict]:
        """Búsqueda semántica con embeddings (solo si model cargado)."""
        try:
            import numpy as np
            query_embedding = self.model.encode(query)
            
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute('SELECT id, titulo, contenido, embedding FROM conocimiento_legal')
            rows = cursor.fetchall()
            conn.close()
            
            if not rows:
                return []
            
            results = []
            for row in rows:
                doc_id, titulo, contenido, embedding_json = row
                if not embedding_json:
                    continue
                doc_embedding = np.array(json.loads(embedding_json))
                similarity = np.dot(query_embedding, doc_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding) + 1e-9
                )
                results.append({
                    'id': doc_id,
                    'titulo': titulo,
                    'contenido': contenido,
                    'score': float(similarity)
                })
            
            results.sort(key=lambda x: x['score'], reverse=True)
            return results[:top_k]
        except Exception as e:
            logger.error(f"Error in semantic search: {str(e)}")
            return []
    
    def get_all_documents(self) -> List[Dict]:
        """Obtiene todos los documentos de la base de conocimiento"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('SELECT id, titulo, contenido, metadata FROM conocimiento_legal')
            rows = cursor.fetchall()
            conn.close()
            
            documents = []
            for row in rows:
                doc_id, titulo, contenido, metadata_json = row
                metadata = json.loads(metadata_json) if metadata_json else {}
                
                documents.append({
                    'id': doc_id,
                    'titulo': titulo,
                    'contenido': contenido[:200] + '...',  # Preview
                    'metadata': metadata
                })
            
            return documents
            
        except Exception as e:
            logger.error(f"Error getting documents: {str(e)}")
            return []
    
    def count_documents(self) -> int:
        """Cuenta el número de documentos en la base"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM conocimiento_legal')
            count = cursor.fetchone()[0]
            conn.close()
            return count
        except Exception as e:
            logger.error(f"Error counting documents: {str(e)}")
            return 0
    
    def clear_database(self):
        """Limpia todos los documentos de la base (usar con cuidado)"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute('DELETE FROM conocimiento_legal')
            conn.commit()
            conn.close()
            logger.info("✅ Database cleared")
        except Exception as e:
            logger.error(f"Error clearing database: {str(e)}")
            raise
