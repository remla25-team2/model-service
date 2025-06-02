import pytest
from unittest.mock import Mock, patch
import sys
import os

# Add app directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def test_simple_math():
    """A simple test to verify pytest is working."""
    assert 1 + 1 == 2


@patch('joblib.load')
@patch('lib_ml.preprocessing.TextPreprocessor.load')
def test_app_imports(mock_preprocessor_load, mock_joblib_load):
    """Test that the app module can be imported."""
    # Mock the dependencies
    mock_model = Mock()
    mock_preprocessor = Mock()
    mock_joblib_load.return_value = mock_model
    mock_preprocessor_load.return_value = mock_preprocessor
    
    # Import the app module
    from app import app
    
    # Basic assertion
    assert app.app is not None


@patch('joblib.load')
@patch('lib_ml.preprocessing.TextPreprocessor.load')
def test_health_endpoint(mock_preprocessor_load, mock_joblib_load):
    """Test the health check endpoint."""
    # Mock the dependencies
    mock_model = Mock()
    mock_preprocessor = Mock()
    mock_joblib_load.return_value = mock_model
    mock_preprocessor_load.return_value = mock_preprocessor
    
    # Import and create test client
    from app.app import app
    app.config['TESTING'] = True
    client = app.test_client()
    
    # Test the health endpoint
    response = client.get('/check_health')
    assert response.status_code == 200
    assert response.data.decode() == 'OK'