"""
Pytest configuration for Prados de Paraíso backend tests
"""
import pytest
import os

# Ensure environment variable is set
def pytest_configure(config):
    """Set up test environment"""
    backend_url = os.environ.get('REACT_APP_BACKEND_URL')
    if not backend_url:
        # Try to read from .env file
        env_path = '/app/frontend/.env'
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    if line.startswith('REACT_APP_BACKEND_URL='):
                        backend_url = line.split('=', 1)[1].strip()
                        os.environ['REACT_APP_BACKEND_URL'] = backend_url
                        break
    
    print(f"\n{'='*60}")
    print(f"Testing Backend URL: {os.environ.get('REACT_APP_BACKEND_URL', 'NOT SET')}")
    print(f"{'='*60}\n")


@pytest.fixture(scope="session", autouse=True)
def cleanup_test_data():
    """Cleanup TEST_ prefixed data after all tests complete"""
    yield
    # Cleanup logic would go here if needed
    print("\n✅ Test session completed")
