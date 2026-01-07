import pytest
from unittest.mock import MagicMock, AsyncMock
import sys
import os

# Ensure the root directory is in the path so we can import 'api'
sys.path.append(os.getcwd())
# Ensure src is in the path so we can import 'fairness_troops'
sys.path.append(os.path.join(os.getcwd(), 'src'))

# Mock modules that might require connection at import time or startup
# This is a bit aggressive but ensures we don't need a running DB for unit tests
sys.modules['api.database'] = MagicMock()
sys.modules['api.database'].engine = MagicMock()
sys.modules['api.database'].engine.begin = MagicMock()
sys.modules['api.database'].engine.begin.return_value.__aenter__ = AsyncMock()
sys.modules['api.database'].engine.begin.return_value.__aexit__ = AsyncMock()

sys.modules['api.cache'] = MagicMock()

# Now import app
from api.main import app
from fastapi.testclient import TestClient

@pytest.fixture
def client():
    # Helper to mock the lifespan if needed, but the sys.modules mock mostly covers it
    # We can also override dependencies here
    return TestClient(app)
