"""
Test configuration and shared fixtures
"""
import pytest
import os
import tempfile
import shutil
from unittest.mock import Mock, MagicMock
from typing import Dict, Any

# Import the backend modules
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from models import Course, Lesson, CourseChunk
from vector_store import SearchResults
from session_manager import SessionManager
from config import Config


@pytest.fixture
def test_config():
    """Create test configuration"""
    config = Config()
    config.CHROMA_PATH = tempfile.mkdtemp()  # Use temp directory for tests
    config.ANTHROPIC_API_KEY = "test-api-key"
    config.ANTHROPIC_MODEL = "claude-3-sonnet-20240229"
    config.CHUNK_SIZE = 500
    config.CHUNK_OVERLAP = 50
    config.MAX_RESULTS = 5
    config.MAX_HISTORY = 5
    return config


@pytest.fixture
def sample_course():
    """Create a sample course for testing"""
    return Course(
        title="Test Course: Introduction to AI",
        course_link="https://example.com/test-course",
        instructor="Dr. Test",
        lessons=[
            Lesson(
                lesson_number=0,
                title="Introduction",
                lesson_link="https://example.com/lesson-0"
            ),
            Lesson(
                lesson_number=1,
                title="AI Fundamentals",
                lesson_link="https://example.com/lesson-1"
            )
        ]
    )


@pytest.fixture
def sample_course_chunks():
    """Create sample course chunks for testing"""
    return [
        CourseChunk(
            content="This is lesson 0 content about AI introduction.",
            course_title="Test Course: Introduction to AI",
            course_link="https://example.com/test-course",
            lesson_number=0,
            chunk_index=0
        ),
        CourseChunk(
            content="This is lesson 1 content about AI fundamentals and machine learning.",
            course_title="Test Course: Introduction to AI", 
            course_link="https://example.com/test-course",
            lesson_number=1,
            chunk_index=1
        ),
        CourseChunk(
            content="More content from lesson 1 covering deep learning concepts.",
            course_title="Test Course: Introduction to AI",
            course_link="https://example.com/test-course", 
            lesson_number=1,
            chunk_index=2
        )
    ]


@pytest.fixture
def sample_search_results():
    """Create sample search results for testing"""
    return SearchResults(
        documents=[
            "This is lesson 0 content about AI introduction.",
            "This is lesson 1 content about AI fundamentals."
        ],
        metadata=[
            {
                "course_title": "Test Course: Introduction to AI",
                "course_link": "https://example.com/test-course",
                "lesson_number": 0,
                "chunk_index": 0
            },
            {
                "course_title": "Test Course: Introduction to AI", 
                "course_link": "https://example.com/test-course",
                "lesson_number": 1,
                "chunk_index": 1
            }
        ],
        distances=[0.2, 0.4]
    )


@pytest.fixture
def mock_anthropic_client():
    """Create a mock Anthropic client for testing"""
    mock_client = Mock()
    
    # Mock successful response
    mock_response = Mock()
    mock_response.content = [Mock(text="This is a test AI response")]
    mock_response.stop_reason = "end_turn"
    
    mock_client.messages.create.return_value = mock_response
    
    return mock_client


@pytest.fixture
def mock_anthropic_tool_response():
    """Create a mock Anthropic response with tool use"""
    mock_response = Mock()
    
    # Mock tool use content block
    mock_tool_block = Mock()
    mock_tool_block.type = "tool_use"
    mock_tool_block.name = "search_course_content"
    mock_tool_block.id = "tool_use_123"
    mock_tool_block.input = {"query": "test query"}
    
    mock_response.content = [mock_tool_block]
    mock_response.stop_reason = "tool_use"
    
    return mock_response


@pytest.fixture
def sample_course_document():
    """Create sample course document content"""
    return """Course Title: Test Course: Introduction to AI
Course Link: https://example.com/test-course
Course Instructor: Dr. Test

Lesson 0: Introduction
This is the introduction to artificial intelligence. We will cover the basics and fundamentals.

Lesson 1: AI Fundamentals
This lesson covers AI fundamentals including machine learning and deep learning concepts.
We'll explore various algorithms and their applications.

Lesson 2: Advanced Topics
Advanced topics in AI including neural networks and natural language processing.
"""


@pytest.fixture
def test_session_manager():
    """Create a session manager for testing"""
    return SessionManager(max_history=3)


@pytest.fixture(autouse=True)
def cleanup_temp_dirs():
    """Clean up temporary directories after tests"""
    temp_dirs = []
    
    def add_temp_dir(path):
        temp_dirs.append(path)
    
    yield add_temp_dir
    
    # Cleanup
    for temp_dir in temp_dirs:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


@pytest.fixture
def mock_vector_store():
    """Create a mock vector store for testing"""
    mock_store = Mock()
    
    # Mock search results
    sample_results = SearchResults(
        documents=["Test document content"],
        metadata=[{
            "course_title": "Test Course",
            "course_link": "https://example.com/course",
            "lesson_number": 1
        }],
        distances=[0.3]
    )
    
    mock_store.search.return_value = sample_results
    mock_store.add_course_metadata.return_value = None
    mock_store.add_course_content.return_value = None
    
    return mock_store