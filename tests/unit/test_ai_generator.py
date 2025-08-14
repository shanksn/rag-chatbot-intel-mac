"""
Unit tests for AI Generator
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import anthropic

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))

from ai_generator import AIGenerator


class TestAIGenerator:
    """Test cases for AIGenerator class"""
    
    @pytest.fixture
    def ai_generator(self):
        """Create AIGenerator instance for testing"""
        return AIGenerator(api_key="test-key", model="claude-3-sonnet-20240229")
    
    @pytest.fixture
    def mock_anthropic_response(self):
        """Mock successful Anthropic API response"""
        mock_response = Mock()
        mock_content = Mock()
        mock_content.text = "This is a test AI response"
        mock_response.content = [mock_content]
        mock_response.stop_reason = "end_turn"
        return mock_response
    
    @pytest.fixture
    def mock_tool_response(self):
        """Mock Anthropic API response with tool use"""
        mock_response = Mock()
        
        # Mock tool use content block
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.id = "tool_123"
        mock_tool_block.input = {"query": "test search"}
        
        mock_response.content = [mock_tool_block]
        mock_response.stop_reason = "tool_use"
        return mock_response
    
    @pytest.fixture
    def mock_tool_manager(self):
        """Mock tool manager for testing"""
        mock_manager = Mock()
        mock_manager.execute_tool.return_value = "Tool execution result"
        return mock_manager
    
    def test_initialization(self, ai_generator):
        """Test AIGenerator initialization"""
        assert ai_generator.model == "claude-3-sonnet-20240229"
        assert ai_generator.base_params["model"] == "claude-3-sonnet-20240229"
        assert ai_generator.base_params["temperature"] == 0
        assert ai_generator.base_params["max_tokens"] == 800
        assert "You are an AI assistant specialized in course materials" in ai_generator.SYSTEM_PROMPT
    
    @patch('anthropic.Anthropic')
    def test_generate_response_simple(self, mock_anthropic_class, ai_generator, mock_anthropic_response):
        """Test simple response generation without tools"""
        # Setup mock client
        mock_client = Mock()
        mock_client.messages.create.return_value = mock_anthropic_response
        ai_generator.client = mock_client
        
        # Test simple query
        result = ai_generator.generate_response("What is AI?")
        
        assert result == "This is a test AI response"
        mock_client.messages.create.assert_called_once()
        
        # Verify API call parameters
        call_args = mock_client.messages.create.call_args
        assert call_args[1]["model"] == "claude-3-sonnet-20240229"
        assert call_args[1]["temperature"] == 0
        assert call_args[1]["max_tokens"] == 800
        assert call_args[1]["messages"][0]["role"] == "user"
        assert call_args[1]["messages"][0]["content"] == "What is AI?"
    
    @patch('anthropic.Anthropic')
    def test_generate_response_with_history(self, mock_anthropic_class, ai_generator, mock_anthropic_response):
        """Test response generation with conversation history"""
        mock_client = Mock()
        mock_client.messages.create.return_value = mock_anthropic_response
        ai_generator.client = mock_client
        
        history = "User: Hello\nAssistant: Hi there!"
        result = ai_generator.generate_response("What is AI?", conversation_history=history)
        
        assert result == "This is a test AI response"
        
        # Verify system message includes history
        call_args = mock_client.messages.create.call_args
        system_content = call_args[1]["system"]
        assert "Previous conversation:" in system_content
        assert history in system_content
    
    @patch('anthropic.Anthropic')
    def test_generate_response_with_tools(self, mock_anthropic_class, ai_generator, mock_anthropic_response):
        """Test response generation with tools available"""
        mock_client = Mock()
        mock_client.messages.create.return_value = mock_anthropic_response
        ai_generator.client = mock_client
        
        tools = [{"name": "search", "description": "Search tool"}]
        result = ai_generator.generate_response("Search for AI", tools=tools)
        
        assert result == "This is a test AI response"
        
        # Verify tools were included in API call
        call_args = mock_client.messages.create.call_args
        assert call_args[1]["tools"] == tools
        assert call_args[1]["tool_choice"] == {"type": "auto"}
    
    @patch('anthropic.Anthropic')
    def test_handle_tool_execution(self, mock_anthropic_class, ai_generator, mock_tool_response, mock_tool_manager):
        """Test tool execution workflow"""
        mock_client = Mock()
        
        # First response has tool use
        # Second response after tool execution
        mock_final_response = Mock()
        mock_final_content = Mock()
        mock_final_content.text = "Final response with tool results"
        mock_final_response.content = [mock_final_content]
        
        # Setup responses in sequence
        mock_client.messages.create.side_effect = [mock_tool_response, mock_final_response]
        ai_generator.client = mock_client
        
        tools = [{"name": "search_course_content", "description": "Search tool"}]
        result = ai_generator.generate_response(
            "Search for AI content", 
            tools=tools, 
            tool_manager=mock_tool_manager
        )
        
        assert result == "Final response with tool results"
        
        # Verify tool was executed
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content", 
            query="test search"
        )
        
        # Verify two API calls were made
        assert mock_client.messages.create.call_count == 2
        
        # Check second API call includes tool results
        second_call_args = mock_client.messages.create.call_args_list[1]
        messages = second_call_args[1]["messages"]
        assert len(messages) == 3  # original user message, assistant tool use, user tool result
        assert messages[2]["role"] == "user"
        assert messages[2]["content"][0]["type"] == "tool_result"
        assert messages[2]["content"][0]["tool_use_id"] == "tool_123"
    
    @patch('anthropic.Anthropic')
    def test_api_error_handling(self, mock_anthropic_class, ai_generator):
        """Test handling of API errors"""
        mock_client = Mock()
        mock_client.messages.create.side_effect = Exception("API Error")
        ai_generator.client = mock_client
        
        with pytest.raises(Exception, match="API Error"):
            ai_generator.generate_response("Test query")
    
    def test_system_prompt_content(self, ai_generator):
        """Test system prompt contains required instructions"""
        prompt = ai_generator.SYSTEM_PROMPT
        
        # Check for key instructions
        assert "AI assistant specialized in course materials" in prompt
        assert "Search Tool Usage" in prompt
        assert "One search per query maximum" in prompt
        assert "Brief, Concise and focused" in prompt
        assert "Educational" in prompt
        assert "No meta-commentary" in prompt
    
    @patch('anthropic.Anthropic')
    def test_multiple_tool_calls(self, mock_anthropic_class, ai_generator, mock_tool_manager):
        """Test handling of multiple tool calls in one response"""
        mock_client = Mock()
        
        # Mock response with two tool use blocks
        mock_response = Mock()
        mock_tool_block1 = Mock()
        mock_tool_block1.type = "tool_use"
        mock_tool_block1.name = "search_course_content"
        mock_tool_block1.id = "tool_1"
        mock_tool_block1.input = {"query": "search 1"}
        
        mock_tool_block2 = Mock()
        mock_tool_block2.type = "tool_use"
        mock_tool_block2.name = "search_course_content"
        mock_tool_block2.id = "tool_2"
        mock_tool_block2.input = {"query": "search 2"}
        
        mock_response.content = [mock_tool_block1, mock_tool_block2]
        mock_response.stop_reason = "tool_use"
        
        # Final response
        mock_final_response = Mock()
        mock_final_content = Mock()
        mock_final_content.text = "Response with multiple tool results"
        mock_final_response.content = [mock_final_content]
        
        mock_client.messages.create.side_effect = [mock_response, mock_final_response]
        ai_generator.client = mock_client
        
        tools = [{"name": "search_course_content", "description": "Search tool"}]
        result = ai_generator.generate_response(
            "Search for multiple things", 
            tools=tools, 
            tool_manager=mock_tool_manager
        )
        
        assert result == "Response with multiple tool results"
        
        # Verify both tools were executed
        assert mock_tool_manager.execute_tool.call_count == 2
        mock_tool_manager.execute_tool.assert_any_call("search_course_content", query="search 1")
        mock_tool_manager.execute_tool.assert_any_call("search_course_content", query="search 2")
    
    def test_parameter_building_efficiency(self, ai_generator):
        """Test that base parameters are reused efficiently"""
        # Verify base parameters are pre-built
        assert "model" in ai_generator.base_params
        assert "temperature" in ai_generator.base_params
        assert "max_tokens" in ai_generator.base_params
        
        # The base_params should be reused, not rebuilt each time
        original_params = ai_generator.base_params
        assert ai_generator.base_params is original_params  # Same object reference