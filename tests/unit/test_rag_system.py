"""
Unit tests for RAG System - Critical missing coverage (67 lines, 0% → 80%+)
"""
import pytest
import tempfile
import shutil
import os
from unittest.mock import patch, Mock, MagicMock, AsyncMock
from typing import List, Dict, Tuple
import asyncio

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))

from rag_system import RAGSystem
from models import Course, Lesson, CourseChunk


class TestRAGSystem:
    """Test cases for RAGSystem class"""
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing"""
        mock_cfg = Mock()
        mock_cfg.CHUNK_SIZE = 800
        mock_cfg.CHUNK_OVERLAP = 100
        mock_cfg.CHROMA_PATH = "/tmp/test_chroma"
        mock_cfg.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        mock_cfg.MAX_RESULTS = 5
        mock_cfg.ANTHROPIC_API_KEY = "test-key"
        mock_cfg.ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
        mock_cfg.MAX_HISTORY = 10
        return mock_cfg
    
    @pytest.fixture
    def sample_course(self):
        """Sample course for testing"""
        lessons = [
            Lesson(lesson_number=0, title="Introduction", content="Intro content", lesson_link="https://example.com/lesson0"),
            Lesson(lesson_number=1, title="Advanced Topics", content="Advanced content", lesson_link="https://example.com/lesson1")
        ]
        return Course(
            title="Sample AI Course",
            course_link="https://example.com/ai-course",
            instructor="Dr. AI Expert",
            lessons=lessons
        )
    
    @pytest.fixture
    def sample_chunks(self):
        """Sample course chunks for testing"""
        return [
            CourseChunk(
                content="Introduction to artificial intelligence concepts",
                course_title="Sample AI Course", 
                course_link="https://example.com/ai-course",
                lesson_number=0,
                chunk_index=0
            ),
            CourseChunk(
                content="Machine learning algorithms and techniques",
                course_title="Sample AI Course",
                course_link="https://example.com/ai-course", 
                lesson_number=1,
                chunk_index=1
            )
        ]
    
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    @patch('rag_system.ToolManager')
    @patch('rag_system.CourseSearchTool')
    def test_initialization(self, mock_search_tool, mock_tool_manager, mock_session_mgr, 
                           mock_ai_gen, mock_vector_store, mock_doc_proc, mock_config):
        """Test RAGSystem initialization"""
        
        rag = RAGSystem(mock_config)
        
        # Verify components were initialized with correct parameters
        mock_doc_proc.assert_called_once_with(800, 100)
        mock_vector_store.assert_called_once_with("/tmp/test_chroma", "all-MiniLM-L6-v2", 5)
        mock_ai_gen.assert_called_once_with("test-key", "claude-sonnet-4-20250514")
        mock_session_mgr.assert_called_once_with(10)
        
        # Verify tool setup
        mock_tool_manager.assert_called_once()
        mock_search_tool.assert_called_once_with(mock_vector_store.return_value)
        mock_tool_manager.return_value.register_tool.assert_called_once_with(mock_search_tool.return_value)
        
        # Verify attributes are set
        assert rag.config == mock_config
        assert rag.document_processor == mock_doc_proc.return_value
        assert rag.vector_store == mock_vector_store.return_value
        assert rag.ai_generator == mock_ai_gen.return_value
        assert rag.session_manager == mock_session_mgr.return_value
        assert rag.tool_manager == mock_tool_manager.return_value
        assert rag.search_tool == mock_search_tool.return_value
    
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    @patch('rag_system.ToolManager')
    @patch('rag_system.CourseSearchTool')
    def test_add_course_document(self, mock_search_tool, mock_tool_manager, mock_session_mgr, 
                                mock_ai_gen, mock_vector_store, mock_doc_proc, mock_config, 
                                sample_course, sample_chunks):
        """Test adding a single course document"""
        
        # Mock document processor to return sample data
        mock_doc_proc.return_value.process_course_document.return_value = (sample_course, sample_chunks)
        
        rag = RAGSystem(mock_config)
        result = rag.add_course_document("fake_course.txt")
        
        # Verify document processing was called
        mock_doc_proc.return_value.process_course_document.assert_called_once_with("fake_course.txt")
        
        # Verify vector store operations were called
        mock_vector_store.return_value.add_course_metadata.assert_called_once_with(sample_course)
        mock_vector_store.return_value.add_course_content.assert_called_once_with(sample_chunks)
        
        # Verify return value
        assert isinstance(result, tuple)
        assert len(result) == 2
        course, chunk_count = result
        assert course == sample_course
        assert chunk_count == len(sample_chunks)
    
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    @patch('rag_system.ToolManager')
    @patch('rag_system.CourseSearchTool')
    def test_add_course_document_error(self, mock_search_tool, mock_tool_manager, mock_session_mgr, 
                                      mock_ai_gen, mock_vector_store, mock_doc_proc, mock_config):
        """Test handling errors in add_course_document"""
        
        # Mock document processor to raise exception
        mock_doc_proc.return_value.process_course_document.side_effect = Exception("Processing failed")
        
        rag = RAGSystem(mock_config)
        
        # Should propagate the exception
        with pytest.raises(Exception) as exc_info:
            rag.add_course_document("bad_course.txt")
        
        assert "Processing failed" in str(exc_info.value)
    
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    @patch('rag_system.ToolManager')
    @patch('rag_system.CourseSearchTool')
    @patch('os.listdir')
    @patch('os.path.join')
    def test_add_course_folder(self, mock_join, mock_listdir, mock_search_tool, mock_tool_manager, 
                              mock_session_mgr, mock_ai_gen, mock_vector_store, mock_doc_proc, 
                              mock_config, sample_course, sample_chunks):
        """Test adding a folder of course documents"""
        
        # Mock file system
        mock_listdir.return_value = ['course1.txt', 'course2.txt', 'readme.md']
        mock_join.side_effect = lambda folder, file: f"{folder}/{file}"
        
        # Mock document processor to return sample data
        mock_doc_proc.return_value.process_course_document.return_value = (sample_course, sample_chunks)
        
        rag = RAGSystem(mock_config)
        result = rag.add_course_folder("fake_folder", clear_existing=False)
        
        # Verify folder was listed
        mock_listdir.assert_called_once_with("fake_folder")
        
        # Verify only .txt files were processed (2 calls, not 3)
        assert mock_doc_proc.return_value.process_course_document.call_count == 2
        
        # Verify vector store operations were called for each file
        assert mock_vector_store.return_value.add_course_metadata.call_count == 2
        assert mock_vector_store.return_value.add_course_content.call_count == 2
        
        # Verify return value
        assert isinstance(result, tuple)
        assert len(result) == 2
        courses_loaded, total_chunks = result
        assert courses_loaded == 2
        assert total_chunks == len(sample_chunks) * 2  # 2 files, same chunks each
    
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    @patch('rag_system.ToolManager')
    @patch('rag_system.CourseSearchTool')
    @patch('os.listdir')
    def test_add_course_folder_clear_existing(self, mock_listdir, mock_search_tool, mock_tool_manager, 
                                             mock_session_mgr, mock_ai_gen, mock_vector_store, 
                                             mock_doc_proc, mock_config, sample_course, sample_chunks):
        """Test adding course folder with clear_existing=True"""
        
        mock_listdir.return_value = ['course.txt']
        mock_doc_proc.return_value.process_course_document.return_value = (sample_course, sample_chunks)
        
        rag = RAGSystem(mock_config)
        rag.add_course_folder("fake_folder", clear_existing=True)
        
        # Verify clear was called before adding new content
        mock_vector_store.return_value.clear_all_data.assert_called_once()
    
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    @patch('rag_system.ToolManager')
    @patch('rag_system.CourseSearchTool')
    @patch('os.listdir')
    def test_add_course_folder_with_errors(self, mock_listdir, mock_search_tool, mock_tool_manager, 
                                          mock_session_mgr, mock_ai_gen, mock_vector_store, 
                                          mock_doc_proc, mock_config):
        """Test add_course_folder handles individual file errors gracefully"""
        
        mock_listdir.return_value = ['good.txt', 'bad.txt']
        
        # First call succeeds, second fails
        mock_doc_proc.return_value.process_course_document.side_effect = [
            (Mock(), []),  # Success
            Exception("Bad file")  # Error
        ]
        
        rag = RAGSystem(mock_config)
        # Should complete despite one file failing
        result = rag.add_course_folder("mixed_folder")
        
        # Should still process both files (error handling inside method)
        assert mock_doc_proc.return_value.process_course_document.call_count == 2
        
        # Result should reflect only successful file
        courses_loaded, total_chunks = result
        assert courses_loaded == 1  # Only one successful
    
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    @patch('rag_system.ToolManager')
    @patch('rag_system.CourseSearchTool')
    def test_query_basic(self, mock_search_tool, mock_tool_manager, mock_session_mgr, 
                        mock_ai_gen, mock_vector_store, mock_doc_proc, mock_config):
        """Test basic query processing"""
        
        # Mock AI generator response
        mock_ai_gen.return_value.generate_response.return_value = ("Generated answer", ["source1", "source2"])
        
        rag = RAGSystem(mock_config)
        result = rag.query("What is AI?")
        
        # Verify AI generation was called with correct parameters
        mock_ai_gen.return_value.generate_response.assert_called_once()
        call_args = mock_ai_gen.return_value.generate_response.call_args[0]
        assert call_args[0] == "What is AI?"
        assert call_args[1] == mock_tool_manager.return_value
        
        # Verify return value
        assert isinstance(result, tuple)
        assert len(result) == 2
        answer, sources = result
        assert answer == "Generated answer"
        assert sources == ["source1", "source2"]
    
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    @patch('rag_system.ToolManager')
    @patch('rag_system.CourseSearchTool')
    def test_query_with_session(self, mock_search_tool, mock_tool_manager, mock_session_mgr, 
                               mock_ai_gen, mock_vector_store, mock_doc_proc, mock_config):
        """Test query processing with session management"""
        
        mock_ai_gen.return_value.generate_response.return_value = ("Answer with context", [])
        mock_session_mgr.return_value.get_context.return_value = ["Previous question", "Previous answer"]
        
        rag = RAGSystem(mock_config)
        result = rag.query("Follow up question", session_id="test-session")
        
        # Verify session context was retrieved
        mock_session_mgr.return_value.get_context.assert_called_once_with("test-session")
        
        # Verify session was updated with new interaction
        mock_session_mgr.return_value.add_interaction.assert_called_once_with(
            "test-session", "Follow up question", "Answer with context"
        )
        
        # Verify AI generation included session context
        call_args = mock_ai_gen.return_value.generate_response.call_args[0]
        assert len(call_args) >= 3  # query, tools, context
        context = call_args[2]
        assert context == ["Previous question", "Previous answer"]
    
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    @patch('rag_system.ToolManager')
    @patch('rag_system.CourseSearchTool')
    def test_query_no_session(self, mock_search_tool, mock_tool_manager, mock_session_mgr, 
                             mock_ai_gen, mock_vector_store, mock_doc_proc, mock_config):
        """Test query processing without session ID"""
        
        mock_ai_gen.return_value.generate_response.return_value = ("Answer without context", [])
        
        rag = RAGSystem(mock_config)
        result = rag.query("Standalone question")
        
        # Verify session manager was not called
        mock_session_mgr.return_value.get_context.assert_not_called()
        mock_session_mgr.return_value.add_interaction.assert_not_called()
        
        # Verify AI generation was called with empty context
        call_args = mock_ai_gen.return_value.generate_response.call_args[0]
        if len(call_args) >= 3:
            context = call_args[2]
            assert context == []
    
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    @patch('rag_system.ToolManager')
    @patch('rag_system.CourseSearchTool')
    def test_query_ai_error(self, mock_search_tool, mock_tool_manager, mock_session_mgr, 
                           mock_ai_gen, mock_vector_store, mock_doc_proc, mock_config):
        """Test query handling when AI generator fails"""
        
        # Mock AI generator to raise exception
        mock_ai_gen.return_value.generate_response.side_effect = Exception("AI API Error")
        
        rag = RAGSystem(mock_config)
        
        # Should propagate the exception
        with pytest.raises(Exception) as exc_info:
            rag.query("Test question")
        
        assert "AI API Error" in str(exc_info.value)
    
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    @patch('rag_system.ToolManager')
    @patch('rag_system.CourseSearchTool')
    def test_get_course_analytics(self, mock_search_tool, mock_tool_manager, mock_session_mgr, 
                                 mock_ai_gen, mock_vector_store, mock_doc_proc, mock_config):
        """Test getting course analytics"""
        
        # Mock vector store analytics methods
        mock_vector_store.return_value.get_course_count.return_value = 3
        mock_vector_store.return_value.get_existing_course_titles.return_value = [
            "Course A", "Course B", "Course C"
        ]
        mock_vector_store.return_value.get_all_courses_metadata.return_value = [
            {"title": "Course A", "instructor": "Prof A", "lesson_count": 5},
            {"title": "Course B", "instructor": "Prof B", "lesson_count": 3},
            {"title": "Course C", "instructor": "Prof C", "lesson_count": 7}
        ]
        
        # Mock content collection count
        mock_content_collection = Mock()
        mock_content_collection.count.return_value = 45
        mock_vector_store.return_value.course_content = mock_content_collection
        
        rag = RAGSystem(mock_config)
        analytics = rag.get_course_analytics()
        
        # Verify analytics structure
        assert isinstance(analytics, dict)
        assert "total_courses" in analytics
        assert "total_chunks" in analytics
        assert "course_titles" in analytics
        assert "courses" in analytics
        
        # Verify values
        assert analytics["total_courses"] == 3
        assert analytics["total_chunks"] == 45
        assert len(analytics["course_titles"]) == 3
        assert len(analytics["courses"]) == 3
    
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    @patch('rag_system.ToolManager')
    @patch('rag_system.CourseSearchTool')
    def test_component_integration(self, mock_search_tool, mock_tool_manager, mock_session_mgr, 
                                  mock_ai_gen, mock_vector_store, mock_doc_proc, mock_config):
        """Test integration between components"""
        
        rag = RAGSystem(mock_config)
        
        # Verify all components are properly initialized and connected
        assert hasattr(rag, 'config')
        assert hasattr(rag, 'document_processor')
        assert hasattr(rag, 'vector_store')
        assert hasattr(rag, 'ai_generator')
        assert hasattr(rag, 'session_manager')
        assert hasattr(rag, 'tool_manager')
        assert hasattr(rag, 'search_tool')
        
        # Verify components are mocked instances
        assert rag.document_processor == mock_doc_proc.return_value
        assert rag.vector_store == mock_vector_store.return_value
        assert rag.ai_generator == mock_ai_gen.return_value
        assert rag.session_manager == mock_session_mgr.return_value
        assert rag.tool_manager == mock_tool_manager.return_value
        assert rag.search_tool == mock_search_tool.return_value
    
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    @patch('rag_system.ToolManager')
    @patch('rag_system.CourseSearchTool')
    def test_initialization_component_failure(self, mock_search_tool, mock_tool_manager, mock_session_mgr, 
                                             mock_ai_gen, mock_vector_store, mock_doc_proc, mock_config):
        """Test handling of component initialization failures"""
        
        # Mock vector store initialization to fail
        mock_vector_store.side_effect = Exception("VectorStore init failed")
        
        with pytest.raises(Exception) as exc_info:
            RAGSystem(mock_config)
        
        assert "VectorStore init failed" in str(exc_info.value)
    
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    @patch('rag_system.ToolManager')
    @patch('rag_system.CourseSearchTool')
    @patch('os.listdir')
    def test_add_course_folder_empty_directory(self, mock_listdir, mock_search_tool, mock_tool_manager, 
                                              mock_session_mgr, mock_ai_gen, mock_vector_store, 
                                              mock_doc_proc, mock_config):
        """Test adding course folder with no files"""
        
        mock_listdir.return_value = []
        
        rag = RAGSystem(mock_config)
        result = rag.add_course_folder("empty_folder")
        
        # Should return zero counts
        courses_loaded, total_chunks = result
        assert courses_loaded == 0
        assert total_chunks == 0
        
        # Should not call processing methods
        mock_doc_proc.return_value.process_course_document.assert_not_called()
    
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    @patch('rag_system.ToolManager')
    @patch('rag_system.CourseSearchTool')
    @patch('os.listdir')
    def test_add_course_folder_directory_error(self, mock_listdir, mock_search_tool, mock_tool_manager, 
                                              mock_session_mgr, mock_ai_gen, mock_vector_store, 
                                              mock_doc_proc, mock_config):
        """Test adding course folder when directory doesn't exist"""
        
        mock_listdir.side_effect = OSError("Directory not found")
        
        rag = RAGSystem(mock_config)
        
        # Should propagate the OS error
        with pytest.raises(OSError) as exc_info:
            rag.add_course_folder("nonexistent_folder")
        
        assert "Directory not found" in str(exc_info.value)