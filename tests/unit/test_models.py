"""
Unit tests for Pydantic Models
"""
import pytest
from pydantic import ValidationError

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))

from models import Course, Lesson, CourseChunk


class TestLesson:
    """Test cases for Lesson model"""
    
    def test_lesson_creation_minimal(self):
        """Test creating lesson with minimal required fields"""
        lesson = Lesson(lesson_number=1, title="Introduction")
        
        assert lesson.lesson_number == 1
        assert lesson.title == "Introduction"
        assert lesson.lesson_link is None
    
    def test_lesson_creation_with_link(self):
        """Test creating lesson with optional link"""
        lesson = Lesson(
            lesson_number=2,
            title="Advanced Topics",
            lesson_link="https://example.com/lesson-2"
        )
        
        assert lesson.lesson_number == 2
        assert lesson.title == "Advanced Topics"
        assert lesson.lesson_link == "https://example.com/lesson-2"
    
    def test_lesson_validation_missing_number(self):
        """Test validation fails without lesson_number"""
        with pytest.raises(ValidationError):
            Lesson(title="Test Lesson")
    
    def test_lesson_validation_missing_title(self):
        """Test validation fails without title"""
        with pytest.raises(ValidationError):
            Lesson(lesson_number=1)
    
    def test_lesson_serialization(self):
        """Test lesson can be serialized to dict"""
        lesson = Lesson(
            lesson_number=3,
            title="Final Lesson",
            lesson_link="https://example.com/lesson-3"
        )
        
        lesson_dict = lesson.model_dump()
        
        assert lesson_dict["lesson_number"] == 3
        assert lesson_dict["title"] == "Final Lesson"
        assert lesson_dict["lesson_link"] == "https://example.com/lesson-3"
    
    def test_lesson_from_dict(self):
        """Test creating lesson from dictionary"""
        lesson_data = {
            "lesson_number": 4,
            "title": "From Dict",
            "lesson_link": "https://example.com/lesson-4"
        }
        
        lesson = Lesson(**lesson_data)
        
        assert lesson.lesson_number == 4
        assert lesson.title == "From Dict"
        assert lesson.lesson_link == "https://example.com/lesson-4"


class TestCourse:
    """Test cases for Course model"""
    
    def test_course_creation_minimal(self):
        """Test creating course with minimal required fields"""
        course = Course(title="Test Course")
        
        assert course.title == "Test Course"
        assert course.course_link is None
        assert course.instructor is None
        assert course.lessons == []
    
    def test_course_creation_full(self):
        """Test creating course with all fields"""
        lessons = [
            Lesson(lesson_number=0, title="Introduction"),
            Lesson(lesson_number=1, title="Basics")
        ]
        
        course = Course(
            title="Complete Course",
            course_link="https://example.com/course",
            instructor="Dr. Test",
            lessons=lessons
        )
        
        assert course.title == "Complete Course"
        assert course.course_link == "https://example.com/course"
        assert course.instructor == "Dr. Test"
        assert len(course.lessons) == 2
        assert course.lessons[0].title == "Introduction"
    
    def test_course_validation_missing_title(self):
        """Test validation fails without title"""
        with pytest.raises(ValidationError):
            Course()
    
    def test_course_lessons_default_empty(self):
        """Test lessons defaults to empty list"""
        course = Course(title="Test Course")
        
        assert isinstance(course.lessons, list)
        assert len(course.lessons) == 0
    
    def test_course_serialization(self):
        """Test course serialization with lessons"""
        lesson = Lesson(lesson_number=1, title="Test Lesson")
        course = Course(
            title="Test Course",
            course_link="https://example.com/course",
            instructor="Test Instructor",
            lessons=[lesson]
        )
        
        course_dict = course.model_dump()
        
        assert course_dict["title"] == "Test Course"
        assert course_dict["course_link"] == "https://example.com/course"
        assert course_dict["instructor"] == "Test Instructor"
        assert len(course_dict["lessons"]) == 1
        assert course_dict["lessons"][0]["title"] == "Test Lesson"
    
    def test_course_add_lesson(self):
        """Test adding lessons to course"""
        course = Course(title="Test Course")
        lesson = Lesson(lesson_number=1, title="New Lesson")
        
        course.lessons.append(lesson)
        
        assert len(course.lessons) == 1
        assert course.lessons[0].title == "New Lesson"


class TestCourseChunk:
    """Test cases for CourseChunk model"""
    
    def test_chunk_creation_minimal(self):
        """Test creating chunk with minimal required fields"""
        chunk = CourseChunk(
            content="Test content",
            course_title="Test Course",
            chunk_index=0
        )
        
        assert chunk.content == "Test content"
        assert chunk.course_title == "Test Course"
        assert chunk.chunk_index == 0
        assert chunk.course_link is None
        assert chunk.lesson_number is None
    
    def test_chunk_creation_full(self):
        """Test creating chunk with all fields - CRITICAL for hyperlinked sources"""
        chunk = CourseChunk(
            content="Full test content",
            course_title="Full Test Course",
            course_link="https://example.com/full-course",
            lesson_number=2,
            chunk_index=5
        )
        
        assert chunk.content == "Full test content"
        assert chunk.course_title == "Full Test Course"
        assert chunk.course_link == "https://example.com/full-course"
        assert chunk.lesson_number == 2
        assert chunk.chunk_index == 5
    
    def test_chunk_validation_missing_content(self):
        """Test validation fails without content"""
        with pytest.raises(ValidationError):
            CourseChunk(course_title="Test Course", chunk_index=0)
    
    def test_chunk_validation_missing_course_title(self):
        """Test validation fails without course_title"""
        with pytest.raises(ValidationError):
            CourseChunk(content="Test content", chunk_index=0)
    
    def test_chunk_validation_missing_chunk_index(self):
        """Test validation fails without chunk_index"""
        with pytest.raises(ValidationError):
            CourseChunk(content="Test content", course_title="Test Course")
    
    def test_chunk_course_link_preservation(self):
        """CRITICAL TEST: Ensure course_link is preserved in chunks"""
        course_link = "https://example.com/test-course"
        
        chunk = CourseChunk(
            content="Content with link",
            course_title="Course with Link",
            course_link=course_link,
            lesson_number=1,
            chunk_index=0
        )
        
        # Verify link is preserved exactly
        assert chunk.course_link == course_link
        assert chunk.course_link is not None
    
    def test_chunk_serialization_with_links(self):
        """Test chunk serialization preserves course links"""
        chunk = CourseChunk(
            content="Serialization test content",
            course_title="Serialization Course",
            course_link="https://example.com/serialization",
            lesson_number=3,
            chunk_index=1
        )
        
        chunk_dict = chunk.model_dump()
        
        assert chunk_dict["content"] == "Serialization test content"
        assert chunk_dict["course_title"] == "Serialization Course"
        assert chunk_dict["course_link"] == "https://example.com/serialization"
        assert chunk_dict["lesson_number"] == 3
        assert chunk_dict["chunk_index"] == 1
    
    def test_chunk_from_dict_with_links(self):
        """Test creating chunk from dict preserves links"""
        chunk_data = {
            "content": "From dict content",
            "course_title": "From Dict Course",
            "course_link": "https://example.com/from-dict",
            "lesson_number": 4,
            "chunk_index": 2
        }
        
        chunk = CourseChunk(**chunk_data)
        
        assert chunk.content == "From dict content"
        assert chunk.course_title == "From Dict Course"
        assert chunk.course_link == "https://example.com/from-dict"
        assert chunk.lesson_number == 4
        assert chunk.chunk_index == 2
    
    def test_chunk_optional_fields_none(self):
        """Test chunk with None optional fields"""
        chunk = CourseChunk(
            content="Minimal chunk",
            course_title="Minimal Course",
            course_link=None,
            lesson_number=None,
            chunk_index=0
        )
        
        assert chunk.course_link is None
        assert chunk.lesson_number is None
    
    def test_chunk_json_serialization(self):
        """Test chunk can be serialized to JSON"""
        chunk = CourseChunk(
            content="JSON test content",
            course_title="JSON Course",
            course_link="https://example.com/json",
            lesson_number=1,
            chunk_index=0
        )
        
        json_str = chunk.model_dump_json()
        
        # Should be valid JSON string
        assert isinstance(json_str, str)
        assert '"course_link":"https://example.com/json"' in json_str
        assert '"lesson_number":1' in json_str


class TestModelIntegration:
    """Test models working together"""
    
    def test_course_with_lessons_and_chunks(self):
        """Test complete course structure"""
        # Create lessons
        lessons = [
            Lesson(lesson_number=0, title="Introduction", lesson_link="https://example.com/lesson-0"),
            Lesson(lesson_number=1, title="Advanced", lesson_link="https://example.com/lesson-1")
        ]
        
        # Create course
        course = Course(
            title="Integration Test Course",
            course_link="https://example.com/integration-course",
            instructor="Integration Instructor",
            lessons=lessons
        )
        
        # Create chunks that reference the course
        chunks = [
            CourseChunk(
                content="Introduction content",
                course_title=course.title,
                course_link=course.course_link,
                lesson_number=0,
                chunk_index=0
            ),
            CourseChunk(
                content="Advanced content",
                course_title=course.title,
                course_link=course.course_link,
                lesson_number=1,
                chunk_index=1
            )
        ]
        
        # Verify relationships
        assert len(course.lessons) == 2
        assert len(chunks) == 2
        
        # Verify course link propagation
        for chunk in chunks:
            assert chunk.course_link == course.course_link
            assert chunk.course_title == course.title
    
    def test_model_validation_cascade(self):
        """Test validation works properly across nested models"""
        # Invalid lesson should prevent course creation
        with pytest.raises(ValidationError):
            Course(
                title="Test Course",
                lessons=[
                    Lesson(lesson_number=1, title="Valid Lesson"),
                    {"invalid": "lesson"}  # This should fail validation
                ]
            )
    
    def test_model_defaults_consistency(self):
        """Test that model defaults are consistent"""
        course = Course(title="Default Test")
        chunk = CourseChunk(content="Default content", course_title="Default Course", chunk_index=0)
        lesson = Lesson(lesson_number=1, title="Default Lesson")
        
        # All optional link fields should default to None
        assert course.course_link is None
        assert chunk.course_link is None
        assert lesson.lesson_link is None
        
        # Lists should default to empty
        assert course.lessons == []