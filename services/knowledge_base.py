"""
Knowledge Base Service
Maneja búsqueda semántica en MongoDB para consultas legales
"""
import os
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
import numpy as np
from motor.motor_asyncio import AsyncIOMotorDatabase
import logging

logger = logging.getLogger(__name__)

class KnowledgeBaseService:
    """
    Servicio para búsqueda semántica en base de conocimientos legal
    """
    
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        # Usar modelo multilingüe para español
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        logger.info("✅ KnowledgeBase Service initialized")
    
    async def embed_query(self, query: str) -> List[float]:
        """
        Genera embedding para una consulta
        """
        try:
            embedding = self.model.encode(query, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise
    
    async def semantic_search(
        self,
        query: str,
        limit: int = 3,
        min_score: float = 0.5
    ) -> List[Dict]:
        """
        Realiza búsqueda semántica en la base de conocimientos
        
        Args:
            query: Consulta del usuario
            limit: Número máximo de resultados
            min_score: Score mínimo de similitud (0-1)
        
        Returns:
            Lista de documentos relevantes con sus scores
        """
        try:
            # Generar embedding de la consulta
            query_embedding = await self.embed_query(query)
            
            # Búsqueda en MongoDB con vector search
            # Si MongoDB no tiene índices vectoriales, usamos similitud de coseno manual
            documents = await self._cosine_similarity_search(
                query_embedding,
                limit,
                min_score
            )
            
            logger.info(f"Found {len(documents)} relevant documents for query: {query[:50]}...")
            return documents
            
        except Exception as e:
            logger.error(f"Error in semantic search: {str(e)}")
            # Fallback a búsqueda de texto simple
            return await self._keyword_search(query, limit)
    
    async def _cosine_similarity_search(
        self,
        query_embedding: List[float],
        limit: int,
        min_score: float
    ) -> List[Dict]:
        """
        Búsqueda por similitud de coseno con embeddings pre-calculados
        """
        try:
            # Obtener todos los documentos con embeddings
            cursor = self.db.knowledge.find(
                {"embedding": {"$exists": True}},
                {"_id": 0}
            ).limit(100)  # Limitar para eficiencia
            
            docs = await cursor.to_list(length=100)
            
            if not docs:
                logger.warning("No documents with embeddings found")
                return []
            
            # Calcular similitudes
            query_vec = np.array(query_embedding)
            results = []
            
            for doc in docs:
                doc_vec = np.array(doc.get("embedding", []))
                if len(doc_vec) == 0:
                    continue
                
                # Similitud de coseno
                similarity = np.dot(query_vec, doc_vec) / (
                    np.linalg.norm(query_vec) * np.linalg.norm(doc_vec)
                )
                
                if similarity >= min_score:
                    results.append({
                        "content": doc.get("content", ""),
                        "title": doc.get("title", "Sin título"),
                        "category": doc.get("category", "General"),
                        "score": float(similarity),
                        "metadata": doc.get("metadata", {})
                    })
            
            # Ordenar por score descendente
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:limit]
            
        except Exception as e:
            logger.error(f"Error in cosine similarity search: {str(e)}")
            return []
    
    async def _keyword_search(
        self,
        query: str,
        limit: int
    ) -> List[Dict]:
        """
        Búsqueda por palabras clave (fallback)
        """
        try:
            # Búsqueda de texto simple en MongoDB
            cursor = self.db.knowledge.find(
                {"$text": {"$search": query}},
                {"_id": 0, "score": {"$meta": "textScore"}}
            ).sort([("score", {"$meta": "textScore"})]).limit(limit)
            
            docs = await cursor.to_list(length=limit)
            
            return [{
                "content": doc.get("content", ""),
                "title": doc.get("title", "Sin título"),
                "category": doc.get("category", "General"),
                "score": doc.get("score", 0.0),
                "metadata": doc.get("metadata", {})
            } for doc in docs]
            
        except Exception as e:
            logger.error(f"Error in keyword search: {str(e)}")
            return []
    
    async def store_document(
        self,
        content: str,
        title: str,
        category: str = "General",
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Almacena un documento con su embedding en la base de conocimientos
        """
        try:
            # Generar embedding
            embedding = await self.embed_query(content)
            
            # Preparar documento
            doc = {
                "content": content,
                "title": title,
                "category": category,
                "embedding": embedding,
                "metadata": metadata or {}
            }
            
            # Insertar en MongoDB
            result = await self.db.knowledge.insert_one(doc)
            
            logger.info(f"Document stored: {title}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing document: {str(e)}")
            return False
    
    async def initialize_sample_data(self):
        """
        Inicializa datos de ejemplo si la base está vacía
        """
        try:
            count = await self.db.knowledge.count_documents({})
            if count > 0:
                logger.info(f"Knowledge base already has {count} documents")
                return
            
            # Datos de ejemplo de Prados de Paraíso
            sample_docs = [
                {
                    "title": "Posesión Legítima",
                    "category": "Propiedad",
                    "content": "La posesión legítima es el derecho que adquiere una persona sobre un inmueble al poseerlo de forma continua, pacífica y pública durante un período determinado. En Prados de Paraíso, los propietarios tienen posesión legítima desde 1998, avalada por documentación notarial y registral."
                },
                {
                    "title": "Proceso de Saneamiento",
                    "category": "Tramitación",
                    "content": "El saneamiento legal es el proceso mediante el cual se regulariza la situación jurídica de un inmueble. En Prados de Paraíso, este proceso incluye la obtención del título de propiedad individual tras completar el pago total del lote."
                },
                {
                    "title": "Diferencia entre Propiedad y Posesión",
                    "category": "Conceptos Legales",
                    "content": "La propiedad es el derecho pleno sobre un bien, mientras que la posesión es el ejercicio de hecho sobre el mismo. En Prados de Paraíso, los compradores inician con posesión legítima y obtienen la propiedad registral al completar sus pagos."
                },
                {
                    "title": "Entrega de Títulos",
                    "category": "Tramitación",
                    "content": "Los títulos de propiedad se entregan una vez completado el pago total del lote y cumplidos todos los requisitos legales. El proceso toma aproximadamente 3-6 meses después del último pago."
                },
                {
                    "title": "Documentación Legal",
                    "category": "Requisitos",
                    "content": "La documentación incluye: contrato de compraventa, constancia de posesión, comprobantes de pago, certificados de no adeudo, y documentos de identidad actualizados. Todo respaldado con actas notariales."
                }
            ]
            
            for doc in sample_docs:
                await self.store_document(
                    content=doc["content"],
                    title=doc["title"],
                    category=doc["category"]
                )
            
            logger.info(f"✅ Initialized {len(sample_docs)} sample documents")
            
        except Exception as e:
            logger.error(f"Error initializing sample data: {str(e)}")
