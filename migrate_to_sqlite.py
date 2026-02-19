"""
Script para migrar datos de MongoDB a SQLite
"""
import asyncio
import os
from motor.motor_asyncio import AsyncIOMotorClient
from services.sqlite_knowledge import SQLiteKnowledgeBase

async def migrate_mongodb_to_sqlite():
    """Migra todos los documentos de MongoDB a SQLite"""
    
    print("🔄 Iniciando migración de MongoDB a SQLite...")
    
    # Conectar a MongoDB
    mongo_url = os.getenv('MONGO_URL', 'mongodb://localhost:27017')
    db_name = os.getenv('DB_NAME', 'prados_legal_hub')
    
    try:
        client = AsyncIOMotorClient(mongo_url)
        db = client[db_name]
        
        # Obtener documentos de knowledge_base
        documents = await db.knowledge_base.find({}, {"_id": 0}).to_list(None)
        
        print(f"📦 Encontrados {len(documents)} documentos en MongoDB")
        
        # Inicializar SQLite
        sqlite_kb = SQLiteKnowledgeBase()
        
        # Limpiar base de datos existente
        print("🗑️  Limpiando base de datos SQLite existente...")
        sqlite_kb.clear_database()
        
        # Migrar documentos
        migrated = 0
        for doc in documents:
            text_chunk = doc.get('text_chunk', '')
            
            if text_chunk:
                # Crear título desde el inicio del texto
                titulo = text_chunk[:100] + '...' if len(text_chunk) > 100 else text_chunk
                
                # Agregar documento
                sqlite_kb.add_document(
                    titulo=titulo,
                    contenido=text_chunk,
                    metadata={'source': 'mongodb_migration'}
                )
                migrated += 1
                
                if migrated % 10 == 0:
                    print(f"  ✓ Migrados {migrated}/{len(documents)} documentos...")
        
        print(f"\n✅ Migración completada: {migrated} documentos migrados a SQLite")
        print(f"📁 Base de datos: /app/backend/prados.db")
        
        # Verificar
        count = sqlite_kb.count_documents()
        print(f"✅ Verificación: {count} documentos en SQLite")
        
        client.close()
        
    except Exception as e:
        print(f"❌ Error en migración: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(migrate_mongodb_to_sqlite())
