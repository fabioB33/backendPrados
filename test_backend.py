#!/usr/bin/env python3
"""
Test script for backend API
Run with: python test_backend.py
"""
import asyncio
import sys
import os
from pathlib import Path

# Add parent directory to path
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

# Set minimal env vars for testing (without MongoDB)
os.environ.setdefault('MONGO_URL', 'mongodb://localhost:27017')
os.environ.setdefault('DB_NAME', 'test_db')
os.environ.setdefault('OPENAI_API_KEY', 'test_key')
os.environ.setdefault('CORS_ORIGINS', '*')

async def test_imports():
    """Test that all imports work"""
    try:
        from server import app
        print("✅ Imports successful")
        print(f"✅ FastAPI app: {app.title}")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

async def test_routes():
    """Test that routes are registered"""
    try:
        from server import app
        
        routes = [route.path for route in app.routes]
        expected_routes = ['/api/', '/api/messages', '/api/text-chat', '/api/voice-agent']
        
        print(f"\n📋 Registered routes ({len(routes)} total):")
        for route in routes[:10]:  # Show first 10
            print(f"  - {route}")
        
        # Check key routes exist
        has_api_root = any('/api/' in route for route in routes)
        if has_api_root:
            print("✅ API routes registered")
            return True
        else:
            print("❌ API routes not found")
            return False
    except Exception as e:
        print(f"❌ Route test error: {e}")
        return False

def test_config():
    """Test configuration"""
    try:
        from server import OPENAI_API_KEY, ELEVENLABS_API_KEY
        
        print(f"\n⚙️  Configuration:")
        print(f"  - OPENAI_API_KEY: {'✅ Set' if OPENAI_API_KEY else '❌ Not set'}")
        print(f"  - ELEVENLABS_API_KEY: {'✅ Set' if ELEVENLABS_API_KEY else '❌ Not set'}")
        
        return True
    except Exception as e:
        print(f"❌ Config test error: {e}")
        return False

async def main():
    """Run all tests"""
    print("🧪 Testing Backend...\n")
    
    results = []
    
    # Test imports
    results.append(await test_imports())
    
    # Test routes
    results.append(await test_routes())
    
    # Test config
    results.append(test_config())
    
    # Summary
    print(f"\n{'='*50}")
    passed = sum(results)
    total = len(results)
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
