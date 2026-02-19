"""
SQLite Knowledge Base Service
Gestiona el conocimiento legal en base de datos local SQLite
"""
import sqlite3
import json
import logging
from typing import List, Dict, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

logger = logging.getLogger(__name__)

class SQLiteKnowledgeBase:
    def __init__(self, db_path: str = "/app/backend/prados.db"):
        """
        Inicializa la base de conocimiento SQLite
        
        Args:
            db_path: Ruta al archivo de base de datos
        """
        self.db_path = db_path
        self.model = None
        self.conn = None
        
        # Inicializar base de datos
        self._init_database()
        
        # Cargar modelo de embeddings
        self._load_model()
        
        logger.info(f"✅ SQLite KnowledgeBase initialized at {db_path}")
    
    def _init_database(self):
        """Crea la base de datos y tablas si no existen"""
        try:
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
        """Carga el modelo de sentence transformers"""
        try:
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
        Realiza búsqueda semántica en la base de conocimiento
        
        Args:
            query: Consulta del usuario
            top_k: Número de resultados a retornar
            
        Returns:
            Lista de documentos relevantes con sus scores
        """
        try:
            # Generar embedding de la consulta
            query_embedding = self.model.encode(query)
            
            # Obtener todos los documentos
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('SELECT id, titulo, contenido, embedding FROM conocimiento_legal')
            rows = cursor.fetchall()
            conn.close()
            
            if not rows:
                logger.warning("No documents in knowledge base")
                return []
            
            # Calcular similitud coseno
            results = []
            for row in rows:
                doc_id, titulo, contenido, embedding_json = row
                doc_embedding = np.array(json.loads(embedding_json))
                
                # Similitud coseno
                similarity = np.dot(query_embedding, doc_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
                )
                
                results.append({
                    'id': doc_id,
                    'titulo': titulo,
                    'contenido': contenido,
                    'score': float(similarity)
                })
            
            # Ordenar por score descendente
            results.sort(key=lambda x: x['score'], reverse=True)
            
            # Retornar top_k resultados
            top_results = results[:top_k]
            
            logger.info(f"Search query: '{query}' - Found {len(top_results)} relevant documents")
            
            return top_results
            
        except Exception as e:
            logger.error(f"Error searching: {str(e)}")
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
