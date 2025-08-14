"""
Unit tests for Session Manager
"""
import pytest

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))

from session_manager import SessionManager, Message


class TestMessage:
    """Test cases for Message dataclass"""
    
    def test_message_creation(self):
        """Test creating a message"""
        message = Message(role="user", content="Hello!")
        
        assert message.role == "user"
        assert message.content == "Hello!"
    
    def test_message_roles(self):
        """Test different message roles"""
        user_msg = Message(role="user", content="User message")
        assistant_msg = Message(role="assistant", content="Assistant response")
        
        assert user_msg.role == "user"
        assert assistant_msg.role == "assistant"
    
    def test_message_equality(self):
        """Test message equality comparison"""
        msg1 = Message(role="user", content="Same content")
        msg2 = Message(role="user", content="Same content")
        msg3 = Message(role="user", content="Different content")
        
        assert msg1 == msg2
        assert msg1 != msg3


class TestSessionManager:
    """Test cases for SessionManager"""
    
    @pytest.fixture
    def session_manager(self):
        """Create SessionManager with default settings"""
        return SessionManager()
    
    @pytest.fixture
    def limited_session_manager(self):
        """Create SessionManager with limited history"""
        return SessionManager(max_history=2)
    
    def test_initialization(self):
        """Test SessionManager initialization"""
        sm = SessionManager()
        
        assert sm.max_history == 5  # default value
        assert sm.sessions == {}
        assert sm.session_counter == 0
    
    def test_initialization_with_custom_max_history(self):
        """Test SessionManager with custom max_history"""
        sm = SessionManager(max_history=3)
        
        assert sm.max_history == 3
    
    def test_create_session(self, session_manager):
        """Test creating a new session"""
        session_id = session_manager.create_session()
        
        assert isinstance(session_id, str)
        assert session_id.startswith("session_")
        assert session_id in session_manager.sessions
        assert session_manager.sessions[session_id] == []
    
    def test_create_multiple_sessions(self, session_manager):
        """Test creating multiple sessions with unique IDs"""
        session1 = session_manager.create_session()
        session2 = session_manager.create_session()
        session3 = session_manager.create_session()
        
        # All should be different
        assert session1 != session2
        assert session2 != session3
        assert session1 != session3
        
        # All should exist in sessions dict
        assert session1 in session_manager.sessions
        assert session2 in session_manager.sessions
        assert session3 in session_manager.sessions
    
    def test_add_message_new_session(self, session_manager):
        """Test adding message to new session"""
        session_id = "new_session"
        
        session_manager.add_message(session_id, "user", "Hello!")
        
        # Session should be created automatically
        assert session_id in session_manager.sessions
        assert len(session_manager.sessions[session_id]) == 1
        
        message = session_manager.sessions[session_id][0]
        assert message.role == "user"
        assert message.content == "Hello!"
    
    def test_add_message_existing_session(self, session_manager):
        """Test adding messages to existing session"""
        session_id = session_manager.create_session()
        
        session_manager.add_message(session_id, "user", "First message")
        session_manager.add_message(session_id, "assistant", "First response")
        session_manager.add_message(session_id, "user", "Second message")
        
        messages = session_manager.sessions[session_id]
        assert len(messages) == 3
        
        assert messages[0].role == "user"
        assert messages[0].content == "First message"
        assert messages[1].role == "assistant"
        assert messages[1].content == "First response"
        assert messages[2].role == "user"
        assert messages[2].content == "Second message"
    
    def test_add_exchange(self, session_manager):
        """Test adding complete user-assistant exchange"""
        session_id = session_manager.create_session()
        
        session_manager.add_exchange(session_id, "What is AI?", "AI is artificial intelligence.")
        
        messages = session_manager.sessions[session_id]
        assert len(messages) == 2
        
        assert messages[0].role == "user"
        assert messages[0].content == "What is AI?"
        assert messages[1].role == "assistant"
        assert messages[1].content == "AI is artificial intelligence."
    
    def test_message_history_limit(self, limited_session_manager):
        """Test that message history respects max_history limit"""
        session_id = limited_session_manager.create_session()
        
        # Add more messages than the limit (max_history=2, so 4 total messages)
        for i in range(5):
            limited_session_manager.add_message(session_id, "user", f"Message {i}")
            limited_session_manager.add_message(session_id, "assistant", f"Response {i}")
        
        messages = limited_session_manager.sessions[session_id]
        
        # Should only keep the last 4 messages (2 exchanges)
        assert len(messages) == 4
        
        # Should keep the most recent messages
        assert messages[0].content == "Message 3"
        assert messages[1].content == "Response 3"
        assert messages[2].content == "Message 4"
        assert messages[3].content == "Response 4"
    
    def test_get_conversation_history_empty(self, session_manager):
        """Test getting history from empty session"""
        session_id = session_manager.create_session()
        
        history = session_manager.get_conversation_history(session_id)
        
        assert history is None
    
    def test_get_conversation_history_nonexistent(self, session_manager):
        """Test getting history from nonexistent session"""
        history = session_manager.get_conversation_history("nonexistent_session")
        
        assert history is None
    
    def test_get_conversation_history_none_session(self, session_manager):
        """Test getting history with None session_id"""
        history = session_manager.get_conversation_history(None)
        
        assert history is None
    
    def test_get_conversation_history_formatted(self, session_manager):
        """Test getting formatted conversation history"""
        session_id = session_manager.create_session()
        
        session_manager.add_message(session_id, "user", "Hello!")
        session_manager.add_message(session_id, "assistant", "Hi there!")
        session_manager.add_message(session_id, "user", "How are you?")
        
        history = session_manager.get_conversation_history(session_id)
        
        expected = "User: Hello!\nAssistant: Hi there!\nUser: How are you?"
        assert history == expected
    
    def test_get_conversation_history_capitalization(self, session_manager):
        """Test that role names are properly capitalized in history"""
        session_id = session_manager.create_session()
        
        session_manager.add_message(session_id, "user", "Test message")
        session_manager.add_message(session_id, "assistant", "Test response")
        
        history = session_manager.get_conversation_history(session_id)
        
        assert "User:" in history
        assert "Assistant:" in history
        assert "user:" not in history
        assert "assistant:" not in history
    
    def test_clear_session_existing(self, session_manager):
        """Test clearing existing session"""
        session_id = session_manager.create_session()
        session_manager.add_message(session_id, "user", "Message to be cleared")
        
        # Verify message exists
        assert len(session_manager.sessions[session_id]) == 1
        
        session_manager.clear_session(session_id)
        
        # Session should exist but be empty
        assert session_id in session_manager.sessions
        assert session_manager.sessions[session_id] == []
    
    def test_clear_session_nonexistent(self, session_manager):
        """Test clearing nonexistent session (should not error)"""
        session_manager.clear_session("nonexistent_session")
        
        # Should not raise error, should not create session
        assert "nonexistent_session" not in session_manager.sessions
    
    def test_multiple_sessions_isolation(self, session_manager):
        """Test that multiple sessions are properly isolated"""
        session1 = session_manager.create_session()
        session2 = session_manager.create_session()
        
        session_manager.add_message(session1, "user", "Message in session 1")
        session_manager.add_message(session2, "user", "Message in session 2")
        
        # Sessions should be isolated
        assert len(session_manager.sessions[session1]) == 1
        assert len(session_manager.sessions[session2]) == 1
        
        assert session_manager.sessions[session1][0].content == "Message in session 1"
        assert session_manager.sessions[session2][0].content == "Message in session 2"
    
    def test_session_counter_increment(self, session_manager):
        """Test that session counter increments properly"""
        initial_counter = session_manager.session_counter
        
        session1 = session_manager.create_session()
        assert session_manager.session_counter == initial_counter + 1
        
        session2 = session_manager.create_session()
        assert session_manager.session_counter == initial_counter + 2
        
        # Session IDs should reflect counter
        assert "1" in session1
        assert "2" in session2
    
    def test_conversation_flow_realistic(self, session_manager):
        """Test realistic conversation flow"""
        session_id = session_manager.create_session()
        
        # Simulate realistic conversation
        session_manager.add_exchange(
            session_id,
            "What is the MCP course about?",
            "The MCP course teaches about Model Context Protocol for AI applications."
        )
        
        session_manager.add_exchange(
            session_id,
            "How long is the course?",
            "The course has 8 lessons covering different aspects of MCP."
        )
        
        # Verify full conversation
        history = session_manager.get_conversation_history(session_id)
        
        assert "What is the MCP course about?" in history
        assert "Model Context Protocol" in history
        assert "How long is the course?" in history
        assert "8 lessons" in history
        
        # Verify proper formatting
        lines = history.split("\n")
        assert len(lines) == 4  # 2 exchanges = 4 lines
        assert lines[0].startswith("User:")
        assert lines[1].startswith("Assistant:")
        assert lines[2].startswith("User:")
        assert lines[3].startswith("Assistant:")


class TestSessionManagerIntegration:
    """Integration tests for SessionManager in context"""
    
    def test_new_chat_functionality(self, test_session_manager):
        """Test NEW CHAT button functionality simulation"""
        # Simulate ongoing conversation
        session_id = test_session_manager.create_session()
        test_session_manager.add_exchange(session_id, "Question 1", "Answer 1")
        test_session_manager.add_exchange(session_id, "Question 2", "Answer 2")
        
        # Verify conversation exists
        assert len(test_session_manager.sessions[session_id]) == 4
        
        # Simulate NEW CHAT - clear session
        test_session_manager.clear_session(session_id)
        
        # Verify conversation is cleared
        assert len(test_session_manager.sessions[session_id]) == 0
        assert test_session_manager.get_conversation_history(session_id) is None
    
    def test_context_management_for_ai(self, test_session_manager):
        """Test session manager provides proper context for AI"""
        session_id = test_session_manager.create_session()
        
        # Build conversation context
        test_session_manager.add_exchange(
            session_id,
            "Tell me about RAG systems",
            "RAG systems combine retrieval and generation for better AI responses."
        )
        
        test_session_manager.add_message(session_id, "user", "Can you give me an example?")
        
        # Get context for AI
        history = test_session_manager.get_conversation_history(session_id)
        
        # Should provide proper context for continuing conversation
        assert history is not None
        assert "RAG systems" in history
        assert "Can you give me an example?" in history
        
        # Format should be parseable by AI
        lines = history.split("\n")
        assert all(":" in line for line in lines)  # All lines should have role indicators