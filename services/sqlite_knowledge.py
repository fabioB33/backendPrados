"""
SQLite Knowledge Base Service
Gestiona el conocimiento legal en base de datos local SQLite.
En entornos con poca RAM (ej. Render 512MB) usar DISABLE_EMBEDDINGS=1 para
búsqueda por palabras clave sin cargar SentenceTransformer/torch.
"""
import os
import sqlite3
import json
import logging
from typing import List, Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Solo importar numpy/sentence_transformers cuando se use embeddings (ahorra ~400MB+ en Render)
_EMBEDDINGS_DISABLED = os.environ.get("DISABLE_EMBEDDINGS", "").strip().lower() in ("1", "true", "yes")


def _default_sqlite_path() -> str:
    """Ruta por defecto: env SQLITE_DB_PATH, o directorio escribible (/tmp en Render)."""
    path = os.environ.get("SQLITE_DB_PATH", "").strip()
    if path:
        return path
    # Render y otros PaaS: /tmp es el único directorio escribible; usarlo por defecto
    tmp_db = Path("/tmp") / "prados.db"
    if Path("/tmp").exists():
        return str(tmp_db)
    # Fallback local: junto al backend
    root = Path(__file__).resolve().parent.parent
    return str(root / "prados.db")


class SQLiteKnowledgeBase:
    def __init__(self, db_path: Optional[str] = None):
        """
        Inicializa la base de conocimiento SQLite.
        Usa SQLITE_DB_PATH o ruta en /tmp para Render.
        """
        self.db_path = db_path or _default_sqlite_path()
        self.model = None
        self.conn = None
        
        # Inicializar base de datos
        self._init_database()
        
        # Cargar modelo de embeddings solo si no está desactivado (ahorra RAM en Render 512MB)
        if not _EMBEDDINGS_DISABLED:
            self._load_model()
        else:
            logger.info("✅ SQLite KnowledgeBase: embeddings disabled (keyword search only)")
        
        logger.info(f"✅ SQLite KnowledgeBase initialized at {self.db_path}")
    
    def _init_database(self):
        """Crea la base de datos y tablas si no existen"""
        try:
            parent = Path(self.db_path).parent
            parent.mkdir(parents=True, exist_ok=True)
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
        """Carga el modelo de sentence transformers (solo si embeddings no están desactivados)."""
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
        Agrega un documento a la base de conocimiento.
        Si embeddings están desactivados, guarda sin embedding (búsqueda por keywords).
        """
        try:
            embedding_json = None
            if self.model is not None:
                embedding = self.model.encode(contenido).tolist()
                embedding_json = json.dumps(embedding)
            
            metadata_json = json.dumps(metadata) if metadata else None
            
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
        Búsqueda en la base de conocimiento.
        Con embeddings: semántica. Sin embeddings (DISABLE_EMBEDDINGS): por palabras clave.
        """
        try:
            if self.model is None:
                return self._search_keyword(query, top_k)
            
            import numpy as np
            query_embedding = self.model.encode(query)
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute('SELECT id, titulo, contenido, embedding FROM conocimiento_legal WHERE embedding IS NOT NULL')
            rows = cursor.fetchall()
            conn.close()
            
            if not rows:
                logger.warning("No documents with embeddings in knowledge base")
                return self._search_keyword(query, top_k)
            
            results = []
            for row in rows:
                doc_id, titulo, contenido, embedding_json = row
                if not embedding_json:
                    continue
                doc_embedding = np.array(json.loads(embedding_json))
                similarity = np.dot(query_embedding, doc_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding) + 1e-9
                )
                results.append({'id': doc_id, 'titulo': titulo, 'contenido': contenido, 'score': float(similarity)})
            results.sort(key=lambda x: x['score'], reverse=True)
            top_results = results[:top_k]
            logger.info(f"Search query: '{query}' - Found {len(top_results)} relevant documents")
            return top_results
        except Exception as e:
            logger.error(f"Error searching: {str(e)}")
            return []
    
    def _search_keyword(self, query: str, top_k: int) -> List[Dict]:
        """Búsqueda por palabras clave (SQL LIKE). Usado cuando embeddings están desactivados."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            terms = [t.strip() for t in query.split() if t.strip()]
            if not terms:
                cursor.execute("SELECT id, titulo, contenido FROM conocimiento_legal LIMIT ?", (top_k,))
            else:
                placeholders = " OR ".join(["(titulo LIKE ? OR contenido LIKE ?)" for _ in terms])
                params = []
                for t in terms:
                    p = f"%{t}%"
                    params.extend([p, p])
                params.append(top_k)
                cursor.execute(
                    "SELECT id, titulo, contenido FROM conocimiento_legal WHERE " + placeholders + " LIMIT ?",
                    params
                )
            rows = cursor.fetchall()
            conn.close()
            results = [{"id": r[0], "titulo": r[1], "contenido": r[2], "score": 1.0} for r in rows]
            logger.info(f"Keyword search '{query}' - Found {len(results)} documents")
            return results
        except Exception as e:
            logger.error(f"Error in keyword search: {str(e)}")
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
