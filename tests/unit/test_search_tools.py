"""
Unit tests for Search Tools - Critical for hyperlinked sources functionality
"""
import pytest
from unittest.mock import Mock, MagicMock
from typing import Dict, Any

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))

from search_tools import CourseSearchTool, ToolManager, Tool
from vector_store import SearchResults
from models import CourseChunk


class TestCourseSearchTool:
    """Test cases for CourseSearchTool - Core functionality for hyperlinked sources"""
    
    @pytest.fixture
    def mock_vector_store(self):
        """Create mock vector store for testing"""
        return Mock()
    
    @pytest.fixture
    def search_tool(self, mock_vector_store):
        """Create CourseSearchTool instance"""
        return CourseSearchTool(mock_vector_store)
    
    @pytest.fixture
    def sample_search_results_with_links(self):
        """Create sample search results with course links for testing hyperlinks"""
        return SearchResults(
            documents=[
                "This is content from lesson 1 about AI fundamentals.",
                "This is content from lesson 2 about machine learning.",
                "More content from lesson 1 covering neural networks."
            ],
            metadata=[
                {
                    "course_title": "AI Course",
                    "course_link": "https://example.com/ai-course",
                    "lesson_number": 1,
                    "chunk_index": 0
                },
                {
                    "course_title": "AI Course", 
                    "course_link": "https://example.com/ai-course",
                    "lesson_number": 2,
                    "chunk_index": 1
                },
                {
                    "course_title": "AI Course",
                    "course_link": "https://example.com/ai-course", 
                    "lesson_number": 1,
                    "chunk_index": 2
                }
            ],
            distances=[0.1, 0.2, 0.3]
        )
    
    @pytest.fixture 
    def sample_search_results_no_links(self):
        """Create sample search results without course links"""
        return SearchResults(
            documents=["Content without links"],
            metadata=[{
                "course_title": "Course Without Link",
                "course_link": None,
                "lesson_number": 1,
                "chunk_index": 0
            }],
            distances=[0.1]
        )
    
    def test_get_tool_definition(self, search_tool):
        """Test that tool definition is correctly formatted for Anthropic"""
        definition = search_tool.get_tool_definition()
        
        # Verify structure
        assert "name" in definition
        assert "description" in definition
        assert "input_schema" in definition
        
        assert definition["name"] == "search_course_content"
        assert "search course materials" in definition["description"].lower()
        
        # Verify schema structure
        schema = definition["input_schema"]
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "required" in schema
        
        # Verify required and optional parameters
        properties = schema["properties"]
        assert "query" in properties
        assert "course_name" in properties
        assert "lesson_number" in properties
        assert schema["required"] == ["query"]
    
    def test_execute_simple_search(self, search_tool, mock_vector_store, sample_search_results_with_links):
        """Test basic search execution"""
        mock_vector_store.search.return_value = sample_search_results_with_links
        
        result = search_tool.execute("AI fundamentals")
        
        # Verify vector store was called correctly
        mock_vector_store.search.assert_called_once_with(
            query="AI fundamentals",
            course_name=None,
            lesson_number=None
        )
        
        # Verify result format
        assert isinstance(result, str)
        assert "AI Course" in result
        assert "Lesson 1" in result
        assert "Lesson 2" in result
        assert "AI fundamentals" in result
    
    def test_execute_search_with_filters(self, search_tool, mock_vector_store, sample_search_results_with_links):
        """Test search with course and lesson filters"""
        mock_vector_store.search.return_value = sample_search_results_with_links
        
        result = search_tool.execute("machine learning", course_name="AI Course", lesson_number=2)
        
        # Verify filters were passed
        mock_vector_store.search.assert_called_once_with(
            query="machine learning",
            course_name="AI Course", 
            lesson_number=2
        )
    
    def test_hyperlinked_sources_tracking(self, search_tool, mock_vector_store, sample_search_results_with_links):
        """CRITICAL TEST: Verify hyperlinked sources are correctly tracked"""
        mock_vector_store.search.return_value = sample_search_results_with_links
        
        # Execute search
        result = search_tool.execute("AI content")
        
        # Verify sources are tracked with links
        sources = search_tool.last_sources
        assert len(sources) == 2  # Should be deduplicated (lesson 1 appears twice)
        
        # Check first source
        source1 = sources[0]
        assert source1["title"] == "AI Course - Lesson 1"
        assert source1["link"] == "https://example.com/ai-course"
        
        # Check second source
        source2 = sources[1]
        assert source2["title"] == "AI Course - Lesson 2"
        assert source2["link"] == "https://example.com/ai-course"
    
    def test_source_deduplication(self, search_tool, mock_vector_store, sample_search_results_with_links):
        """CRITICAL TEST: Verify duplicate sources are removed"""
        mock_vector_store.search.return_value = sample_search_results_with_links
        
        # Execute search (sample data has lesson 1 appearing twice)
        result = search_tool.execute("AI content")
        
        # Verify deduplication worked
        sources = search_tool.last_sources
        source_titles = [s["title"] for s in sources]
        
        # Should only have unique sources
        assert len(sources) == 2
        assert "AI Course - Lesson 1" in source_titles
        assert "AI Course - Lesson 2" in source_titles
        assert source_titles.count("AI Course - Lesson 1") == 1  # No duplicates
    
    def test_sources_without_links(self, search_tool, mock_vector_store, sample_search_results_no_links):
        """Test sources without course links are handled correctly"""
        mock_vector_store.search.return_value = sample_search_results_no_links
        
        result = search_tool.execute("content")
        
        # Verify sources without links
        sources = search_tool.last_sources
        assert len(sources) == 1
        assert sources[0]["title"] == "Course Without Link - Lesson 1"
        assert sources[0]["link"] is None
    
    def test_empty_results(self, search_tool, mock_vector_store):
        """Test handling of empty search results"""
        empty_results = SearchResults(documents=[], metadata=[], distances=[])
        mock_vector_store.search.return_value = empty_results
        
        result = search_tool.execute("nonexistent content")
        
        assert "No relevant content found" in result
        assert search_tool.last_sources == []
    
    def test_empty_results_with_filters(self, search_tool, mock_vector_store):
        """Test empty results with filter information"""
        empty_results = SearchResults(documents=[], metadata=[], distances=[])
        mock_vector_store.search.return_value = empty_results
        
        result = search_tool.execute("content", course_name="Specific Course", lesson_number=5)
        
        assert "No relevant content found in course 'Specific Course' in lesson 5" in result
    
    def test_error_handling(self, search_tool, mock_vector_store):
        """Test error handling from vector store"""
        error_results = SearchResults(documents=[], metadata=[], distances=[], error="Database error")
        mock_vector_store.search.return_value = error_results
        
        result = search_tool.execute("query")
        
        assert result == "Database error"
    
    def test_result_formatting_with_headers(self, search_tool, mock_vector_store, sample_search_results_with_links):
        """Test that results are formatted with proper course/lesson headers"""
        mock_vector_store.search.return_value = sample_search_results_with_links
        
        result = search_tool.execute("AI content")
        
        # Verify headers are included
        assert "[AI Course - Lesson 1]" in result
        assert "[AI Course - Lesson 2]" in result
        
        # Verify content is included
        assert "AI fundamentals" in result
        assert "machine learning" in result
    
    def test_sources_reset_between_searches(self, search_tool, mock_vector_store, sample_search_results_with_links):
        """Test that sources are properly reset between searches"""
        mock_vector_store.search.return_value = sample_search_results_with_links
        
        # First search
        search_tool.execute("first query")
        first_sources = search_tool.last_sources.copy()
        
        # Second search with different results
        different_results = SearchResults(
            documents=["Different content"],
            metadata=[{
                "course_title": "Different Course",
                "course_link": "https://example.com/different",
                "lesson_number": 3,
                "chunk_index": 0
            }],
            distances=[0.1]
        )
        mock_vector_store.search.return_value = different_results
        
        search_tool.execute("second query")
        second_sources = search_tool.last_sources
        
        # Verify sources were updated, not appended
        assert len(second_sources) == 1
        assert second_sources[0]["title"] == "Different Course - Lesson 3"
        assert second_sources != first_sources


class TestToolManager:
    """Test cases for ToolManager"""
    
    @pytest.fixture
    def tool_manager(self):
        """Create ToolManager instance"""
        return ToolManager()
    
    @pytest.fixture
    def mock_tool(self):
        """Create mock tool for testing"""
        mock_tool = Mock(spec=Tool)
        mock_tool.get_tool_definition.return_value = {
            "name": "test_tool",
            "description": "Test tool",
            "input_schema": {"type": "object", "properties": {}}
        }
        mock_tool.execute.return_value = "Tool result"
        return mock_tool
    
    def test_register_tool(self, tool_manager, mock_tool):
        """Test tool registration"""
        tool_manager.register_tool(mock_tool)
        
        assert "test_tool" in tool_manager.tools
        assert tool_manager.tools["test_tool"] is mock_tool
    
    def test_register_tool_without_name(self, tool_manager):
        """Test registering tool without name raises error"""
        bad_tool = Mock(spec=Tool)
        bad_tool.get_tool_definition.return_value = {"description": "No name"}
        
        with pytest.raises(ValueError, match="Tool must have a 'name'"):
            tool_manager.register_tool(bad_tool)
    
    def test_get_tool_definitions(self, tool_manager, mock_tool):
        """Test getting all tool definitions"""
        tool_manager.register_tool(mock_tool)
        
        definitions = tool_manager.get_tool_definitions()
        
        assert len(definitions) == 1
        assert definitions[0]["name"] == "test_tool"
    
    def test_execute_tool(self, tool_manager, mock_tool):
        """Test tool execution"""
        tool_manager.register_tool(mock_tool)
        
        result = tool_manager.execute_tool("test_tool", param1="value1")
        
        assert result == "Tool result"
        mock_tool.execute.assert_called_once_with(param1="value1")
    
    def test_execute_nonexistent_tool(self, tool_manager):
        """Test executing nonexistent tool"""
        result = tool_manager.execute_tool("nonexistent_tool")
        
        assert "Tool 'nonexistent_tool' not found" in result
    
    def test_get_last_sources(self, tool_manager):
        """Test getting sources from tools"""
        # Create mock tool with sources
        mock_tool_with_sources = Mock()
        mock_tool_with_sources.get_tool_definition.return_value = {"name": "source_tool"}
        mock_tool_with_sources.last_sources = [{"title": "Test Source", "link": "https://test.com"}]
        
        # Create mock tool without sources
        mock_tool_no_sources = Mock()
        mock_tool_no_sources.get_tool_definition.return_value = {"name": "no_source_tool"}
        
        tool_manager.register_tool(mock_tool_with_sources)
        tool_manager.register_tool(mock_tool_no_sources)
        
        sources = tool_manager.get_last_sources()
        
        assert len(sources) == 1
        assert sources[0]["title"] == "Test Source"
    
    def test_reset_sources(self, tool_manager):
        """Test resetting sources from all tools"""
        # Create mock tool with sources
        mock_tool = Mock()
        mock_tool.get_tool_definition.return_value = {"name": "tool_with_sources"}
        mock_tool.last_sources = [{"title": "Source"}]
        
        tool_manager.register_tool(mock_tool)
        tool_manager.reset_sources()
        
        assert mock_tool.last_sources == []


class TestToolIntegration:
    """Integration tests for tools working together"""
    
    @pytest.fixture
    def sample_search_results_with_links(self):
        """Create sample search results with course links for testing hyperlinks"""
        return SearchResults(
            documents=[
                "This is content from lesson 1 about AI fundamentals.",
                "This is content from lesson 2 about machine learning."
            ],
            metadata=[
                {
                    "course_title": "AI Course",
                    "course_link": "https://example.com/ai-course",
                    "lesson_number": 1,
                    "chunk_index": 0
                },
                {
                    "course_title": "AI Course", 
                    "course_link": "https://example.com/ai-course",
                    "lesson_number": 2,
                    "chunk_index": 1
                }
            ],
            distances=[0.1, 0.2]
        )
    
    def test_search_tool_with_tool_manager(self, sample_search_results_with_links):
        """Test CourseSearchTool working through ToolManager"""
        # Create mock vector store locally
        from unittest.mock import Mock
        mock_vector_store = Mock()
        mock_vector_store.search.return_value = sample_search_results_with_links
        
        # Create and register tool
        search_tool = CourseSearchTool(mock_vector_store)
        tool_manager = ToolManager()
        tool_manager.register_tool(search_tool)
        
        # Execute through manager
        result = tool_manager.execute_tool("search_course_content", query="AI content")
        
        # Verify execution worked
        assert "AI Course" in result
        assert "Lesson 1" in result
        
        # Verify sources are available
        sources = tool_manager.get_last_sources()
        assert len(sources) == 2
        assert all("link" in source for source in sources)
        assert all(source["link"] == "https://example.com/ai-course" for source in sources)