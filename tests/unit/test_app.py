"""
Unit tests for FastAPI App - Critical missing coverage (65 lines, 0% → 80%+)
"""
import pytest
import json
from unittest.mock import patch, Mock, AsyncMock, MagicMock
from fastapi.testclient import TestClient
from fastapi import FastAPI

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))


@patch('fastapi.staticfiles.StaticFiles', return_value=MagicMock())
class TestApp:
    """Test cases for FastAPI application"""
    
    @pytest.fixture
    def mock_rag_system(self):
        """Mock RAG system for testing"""
        mock_rag = Mock()
        mock_rag.query.return_value = ("Sample answer about AI", ["Source 1", "Source 2"])
        mock_rag.get_course_analytics.return_value = {
            "total_courses": 3,
            "total_chunks": 150,
            "course_titles": ["Course A", "Course B", "Course C"],
            "courses": [
                {"title": "Course A", "instructor": "Prof A", "lesson_count": 5},
                {"title": "Course B", "instructor": "Prof B", "lesson_count": 3}
            ]
        }
        return mock_rag
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration"""
        mock_cfg = Mock()
        mock_cfg.DOCS_PATH = "/fake/docs"
        return mock_cfg
    
    @patch('app.RAGSystem')
    @patch('app.config')
    def test_app_startup(self, mock_static_files, mock_config_module, mock_rag_system_class, mock_rag_system, mock_config):
        """Test application startup and initialization"""
        # Setup mocks
        mock_config_module.return_value = mock_config
        mock_rag_system_class.return_value = mock_rag_system
        
        # Import app after mocking (this triggers startup)
        from app import app
        
        # Verify RAG system was initialized with config
        mock_rag_system_class.assert_called_once_with(mock_config)
        
        # Verify documents were loaded
        mock_rag_system.add_course_folder.assert_called_once_with("/fake/docs", clear_existing=True)
        
        # Verify app is FastAPI instance
        assert isinstance(app, FastAPI)
    
    @patch('app.RAGSystem')  
    @patch('app.config')
    def test_query_endpoint_basic(self, mock_static_files, mock_config_module, mock_rag_system_class, mock_rag_system, mock_config):
        """Test basic query endpoint functionality"""
        # Setup mocks
        mock_config_module.return_value = mock_config
        mock_rag_system_class.return_value = mock_rag_system
        
        # Import and create test client
        from app import app
        client = TestClient(app)
        
        # Make request
        response = client.post(
            "/api/query",
            json={"query": "What is artificial intelligence?"}
        )
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert "sources" in data
        assert data["response"] == "Sample answer about AI"
        assert data["sources"] == ["Source 1", "Source 2"]
        
        # Verify RAG system was called correctly
        mock_rag_system.query.assert_called_once_with("What is artificial intelligence?", session_id=None)
    
    @patch('app.RAGSystem')
    @patch('app.config') 
    def test_query_endpoint_with_session(self, mock_static_files, mock_config_module, mock_rag_system_class, mock_rag_system, mock_config):
        """Test query endpoint with session ID"""
        mock_config_module.return_value = mock_config
        mock_rag_system_class.return_value = mock_rag_system
        
        from app import app
        client = TestClient(app)
        
        # Make request with session ID
        response = client.post(
            "/api/query",
            json={
                "query": "Follow up question",
                "session_id": "test-session-123"
            }
        )
        
        assert response.status_code == 200
        
        # Verify RAG system was called with session ID
        mock_rag_system.query.assert_called_once_with("Follow up question", session_id="test-session-123")
    
    @patch('app.RAGSystem')
    @patch('app.config')
    def test_query_endpoint_missing_query(self, mock_static_files, mock_config_module, mock_rag_system_class, mock_rag_system, mock_config):
        """Test query endpoint with missing query field"""
        mock_config_module.return_value = mock_config
        mock_rag_system_class.return_value = mock_rag_system
        
        from app import app
        client = TestClient(app)
        
        # Make request without query field
        response = client.post(
            "/api/query",
            json={"session_id": "test-session"}
        )
        
        # Should return 422 for validation error
        assert response.status_code == 422
    
    @patch('app.RAGSystem')
    @patch('app.config')
    def test_query_endpoint_rag_error(self, mock_static_files, mock_config_module, mock_rag_system_class, mock_rag_system, mock_config):
        """Test query endpoint when RAG system raises exception"""
        mock_config_module.return_value = mock_config
        mock_rag_system_class.return_value = mock_rag_system
        
        # Make RAG system raise an exception
        mock_rag_system.query.side_effect = Exception("RAG system error")
        
        from app import app
        client = TestClient(app)
        
        # Make request
        response = client.post(
            "/api/query",
            json={"query": "Test question"}
        )
        
        # Should return 500 for internal server error
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "RAG system error" in data["detail"]
    
    @patch('app.RAGSystem')
    @patch('app.config')
    def test_courses_endpoint(self, mock_static_files, mock_config_module, mock_rag_system_class, mock_rag_system, mock_config):
        """Test courses analytics endpoint"""
        mock_config_module.return_value = mock_config
        mock_rag_system_class.return_value = mock_rag_system
        
        from app import app
        client = TestClient(app)
        
        # Make request
        response = client.get("/api/courses")
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        
        assert "total_courses" in data
        assert "total_chunks" in data
        assert "course_titles" in data
        assert "courses" in data
        
        assert data["total_courses"] == 3
        assert data["total_chunks"] == 150
        assert len(data["course_titles"]) == 3
        assert len(data["courses"]) == 2
        
        # Verify RAG system was called
        mock_rag_system.get_course_analytics.assert_called_once()
    
    @patch('app.RAGSystem')
    @patch('app.config')
    def test_courses_endpoint_rag_error(self, mock_static_files, mock_config_module, mock_rag_system_class, mock_rag_system, mock_config):
        """Test courses endpoint when RAG system raises exception"""
        mock_config_module.return_value = mock_config
        mock_rag_system_class.return_value = mock_rag_system
        
        # Make RAG system raise an exception
        mock_rag_system.get_course_analytics.side_effect = Exception("Analytics error")
        
        from app import app
        client = TestClient(app)
        
        # Make request
        response = client.get("/api/courses")
        
        # Should return 500 for internal server error
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "Analytics error" in data["detail"]
    
    @patch('app.RAGSystem')
    @patch('app.config')
    def test_cors_middleware(self, mock_static_files, mock_config_module, mock_rag_system_class, mock_rag_system, mock_config):
        """Test CORS middleware is configured"""
        mock_config_module.return_value = mock_config
        mock_rag_system_class.return_value = mock_rag_system
        
        from app import app
        client = TestClient(app)
        
        # Make preflight request
        response = client.options(
            "/api/query",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type"
            }
        )
        
        # Should allow CORS
        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers
    
    @patch('app.RAGSystem')
    @patch('app.config')
    def test_query_endpoint_empty_query(self, mock_static_files, mock_config_module, mock_rag_system_class, mock_rag_system, mock_config):
        """Test query endpoint with empty query string"""
        mock_config_module.return_value = mock_config
        mock_rag_system_class.return_value = mock_rag_system
        
        from app import app
        client = TestClient(app)
        
        # Make request with empty query
        response = client.post(
            "/api/query",
            json={"query": ""}
        )
        
        # Should still process (RAG system handles empty queries)
        assert response.status_code == 200
        mock_rag_system.query.assert_called_once_with("", session_id=None)
    
    @patch('app.RAGSystem')
    @patch('app.config')
    def test_json_parsing_error(self, mock_static_files, mock_config_module, mock_rag_system_class, mock_rag_system, mock_config):
        """Test handling of malformed JSON"""
        mock_config_module.return_value = mock_config
        mock_rag_system_class.return_value = mock_rag_system
        
        from app import app
        client = TestClient(app)
        
        # Send malformed JSON
        response = client.post(
            "/api/query",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        # Should return 422 for JSON parsing error
        assert response.status_code == 422
    
    @patch('app.RAGSystem')
    @patch('app.config')  
    def test_unsupported_http_method(self, mock_static_files, mock_config_module, mock_rag_system_class, mock_rag_system, mock_config):
        """Test unsupported HTTP method on query endpoint"""
        mock_config_module.return_value = mock_config
        mock_rag_system_class.return_value = mock_rag_system
        
        from app import app
        client = TestClient(app)
        
        # Try GET on query endpoint (should be POST only)
        response = client.get("/api/query")
        
        # Should return 405 Method Not Allowed
        assert response.status_code == 405
    
    @patch('app.RAGSystem')
    @patch('app.config')
    def test_query_response_structure(self, mock_static_files, mock_config_module, mock_rag_system_class, mock_rag_system, mock_config):
        """Test query response has correct structure"""
        mock_config_module.return_value = mock_config
        mock_rag_system_class.return_value = mock_rag_system
        
        # Set specific return value
        mock_rag_system.query.return_value = ("Detailed answer", ["Source A", "Source B", "Source C"])
        
        from app import app
        client = TestClient(app)
        
        response = client.post("/api/query", json={"query": "test"})
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify exact response structure
        assert set(data.keys()) == {"response", "sources"}
        assert data["response"] == "Detailed answer"
        assert data["sources"] == ["Source A", "Source B", "Source C"]
        assert isinstance(data["sources"], list)
    
    @patch('app.RAGSystem')
    @patch('app.config')
    def test_courses_response_structure(self, mock_static_files, mock_config_module, mock_rag_system_class, mock_rag_system, mock_config):
        """Test courses response has correct structure"""
        mock_config_module.return_value = mock_config
        mock_rag_system_class.return_value = mock_rag_system
        
        from app import app
        client = TestClient(app)
        
        response = client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure matches expected analytics format
        required_keys = {"total_courses", "total_chunks", "course_titles", "courses"}
        assert set(data.keys()) == required_keys
        
        assert isinstance(data["total_courses"], int)
        assert isinstance(data["total_chunks"], int) 
        assert isinstance(data["course_titles"], list)
        assert isinstance(data["courses"], list)