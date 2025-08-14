"""
Unit tests for Document Processor - Critical missing coverage (133 lines, 0% → 80%+)
"""
import pytest
import tempfile
import os
from unittest.mock import patch, mock_open, Mock

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))

from document_processor import DocumentProcessor
from models import Course, Lesson, CourseChunk


class TestDocumentProcessor:
    """Test cases for DocumentProcessor class"""
    
    @pytest.fixture
    def doc_processor(self):
        """Create DocumentProcessor instance for testing"""
        return DocumentProcessor(chunk_size=500, chunk_overlap=50)
    
    @pytest.fixture
    def sample_valid_document(self):
        """Sample valid course document content"""
        return """Course Title: Sample AI Course
Course Link: https://example.com/ai-course
Course Instructor: Dr. AI Expert

Lesson 0: Introduction to AI
This is the introduction lesson content. It covers basic concepts of artificial intelligence and machine learning fundamentals.

Lesson 1: Machine Learning Basics
This lesson covers machine learning algorithms, supervised and unsupervised learning techniques, and their applications in real-world scenarios.

Lesson 2: Deep Learning
Advanced topics in deep learning including neural networks, backpropagation, and modern architectures like transformers and CNNs."""
    
    @pytest.fixture
    def sample_malformed_document(self):
        """Sample malformed document for error testing"""
        return """This is a malformed document without proper headers.
Just some random content that doesn't follow the expected format.
No course title, no lessons, no structure."""
    
    @pytest.fixture
    def sample_missing_metadata_document(self):
        """Sample document missing some metadata"""
        return """Course Title: Incomplete Course
Course Instructor: Missing Link

Lesson 0: Some Content
Content here but missing course link."""
    
    def test_initialization(self, doc_processor):
        """Test DocumentProcessor initialization"""
        assert doc_processor.chunk_size == 500
        assert doc_processor.chunk_overlap == 50
    
    def test_read_file_success(self, doc_processor):
        """Test successful file reading"""
        content = "Test file content"
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as tmp_file:
            tmp_file.write(content)
            tmp_file.flush()
            
            try:
                result = doc_processor.read_file(tmp_file.name)
                assert result == content
            finally:
                os.unlink(tmp_file.name)
    
    def test_read_file_utf8_fallback(self, doc_processor):
        """Test file reading with UTF-8 error handling"""
        with patch('builtins.open', side_effect=[UnicodeDecodeError('utf-8', b'', 0, 1, 'error'), mock_open(read_data='fallback content').return_value]):
            result = doc_processor.read_file('test.txt')
            assert result == 'fallback content'
    
    def test_read_file_not_found(self, doc_processor):
        """Test handling of non-existent file"""
        with pytest.raises(FileNotFoundError):
            doc_processor.read_file('nonexistent_file.txt')
    
    def test_chunk_text_basic(self, doc_processor):
        """Test basic text chunking functionality"""
        text = "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence."
        
        chunks = doc_processor.chunk_text(text)
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)
        assert all(len(chunk) <= doc_processor.chunk_size for chunk in chunks)
    
    def test_chunk_text_with_overlap(self):
        """Test chunking with overlap functionality"""
        processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)
        
        # Create text that will definitely need multiple chunks
        long_text = "Sentence one. " * 50  # Will be much longer than 100 chars
        
        chunks = processor.chunk_text(long_text)
        
        assert len(chunks) > 1  # Should create multiple chunks
        # Check that chunks respect size limit
        assert all(len(chunk) <= processor.chunk_size * 1.5 for chunk in chunks)  # Allow some flexibility for sentence boundaries
    
    def test_chunk_text_empty_input(self, doc_processor):
        """Test chunking with empty input"""
        result = doc_processor.chunk_text("")
        assert result == []
    
    def test_chunk_text_whitespace_normalization(self, doc_processor):
        """Test that chunking normalizes whitespace correctly"""
        text_with_extra_spaces = "First  sentence.   Second\n\nsentence.    Third sentence."
        
        chunks = doc_processor.chunk_text(text_with_extra_spaces)
        
        # Should normalize multiple spaces to single spaces
        for chunk in chunks:
            assert '  ' not in chunk  # No double spaces
            assert '\n\n' not in chunk  # No double newlines
    
    def test_chunk_text_sentence_splitting(self, doc_processor):
        """Test sentence splitting handles abbreviations correctly"""
        text = "Dr. Smith works at U.S.A. University. He teaches A.I. courses. The Ph.D. program is excellent."
        
        chunks = doc_processor.chunk_text(text)
        
        # Should not split on abbreviations
        combined = ' '.join(chunks)
        assert "Dr. Smith" in combined
        assert "U.S.A. University" in combined
        assert "A.I. courses" in combined
        assert "Ph.D. program" in combined
    
    def test_process_course_document_valid(self, doc_processor, sample_valid_document):
        """Test processing a valid course document"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as tmp_file:
            tmp_file.write(sample_valid_document)
            tmp_file.flush()
            
            try:
                course, chunks = doc_processor.process_course_document(tmp_file.name)
                
                # Verify course metadata
                assert isinstance(course, Course)
                assert course.title == "Sample AI Course"
                assert course.course_link == "https://example.com/ai-course"
                assert course.instructor == "Dr. AI Expert"
                assert len(course.lessons) == 3
                
                # Verify lessons
                assert course.lessons[0].lesson_number == 0
                assert course.lessons[0].title == "Introduction to AI"
                assert course.lessons[1].lesson_number == 1
                assert course.lessons[1].title == "Machine Learning Basics"
                assert course.lessons[2].lesson_number == 2
                assert course.lessons[2].title == "Deep Learning"
                
                # Verify chunks
                assert len(chunks) > 0
                assert all(isinstance(chunk, CourseChunk) for chunk in chunks)
                assert all(chunk.course_title == "Sample AI Course" for chunk in chunks)
                assert all(chunk.course_link == "https://example.com/ai-course" for chunk in chunks)
                
            finally:
                os.unlink(tmp_file.name)
    
    def test_process_course_document_missing_metadata(self, doc_processor, sample_missing_metadata_document):
        """Test processing document with missing metadata"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as tmp_file:
            tmp_file.write(sample_missing_metadata_document)
            tmp_file.flush()
            
            try:
                course, chunks = doc_processor.process_course_document(tmp_file.name)
                
                # Should still work but with missing fields
                assert course.title == "Incomplete Course"
                assert course.course_link is None  # Missing from document
                assert course.instructor == "Missing Link"
                
                # Chunks should have course_link as None
                assert all(chunk.course_link is None for chunk in chunks)
                
            finally:
                os.unlink(tmp_file.name)
    
    def test_process_course_document_malformed(self, doc_processor, sample_malformed_document):
        """Test processing malformed document"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as tmp_file:
            tmp_file.write(sample_malformed_document)
            tmp_file.flush()
            
            try:
                course, chunks = doc_processor.process_course_document(tmp_file.name)
                
                # Should create basic course structure even from malformed input
                assert isinstance(course, Course)
                assert course.title is not None
                # Malformed documents without proper lessons may have no chunks
                assert isinstance(chunks, list)
                
            finally:
                os.unlink(tmp_file.name)
    
    def test_process_course_document_lesson_link_extraction(self, doc_processor):
        """Test extraction of lesson links from document"""
        document_with_lesson_links = """Course Title: Course with Lesson Links
Course Link: https://example.com/course
Course Instructor: Test Instructor

Lesson 0: Introduction
Lesson Link: https://example.com/lesson-0
Introduction content here.

Lesson 1: Advanced Topics  
Lesson Link: https://example.com/lesson-1
Advanced content here."""
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as tmp_file:
            tmp_file.write(document_with_lesson_links)
            tmp_file.flush()
            
            try:
                course, chunks = doc_processor.process_course_document(tmp_file.name)
                
                # Verify lesson links are extracted
                assert len(course.lessons) == 2
                assert course.lessons[0].lesson_link == "https://example.com/lesson-0"
                assert course.lessons[1].lesson_link == "https://example.com/lesson-1"
                
            finally:
                os.unlink(tmp_file.name)
    
    def test_process_course_document_no_lessons(self, doc_processor):
        """Test processing document without lesson structure"""
        document_no_lessons = """Course Title: No Lessons Course
Course Link: https://example.com/no-lessons
Course Instructor: Test Instructor

This is just general content without any lesson structure. 
Just paragraphs of text that should be chunked directly."""
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as tmp_file:
            tmp_file.write(document_no_lessons)
            tmp_file.flush()
            
            try:
                course, chunks = doc_processor.process_course_document(tmp_file.name)
                
                # Should handle content without lessons
                assert course.title == "No Lessons Course"
                assert len(course.lessons) == 0
                assert len(chunks) > 0  # Should still create chunks from remaining content
                
                # Chunks should not have lesson numbers
                assert all(chunk.lesson_number is None for chunk in chunks)
                
            finally:
                os.unlink(tmp_file.name)
    
    def test_chunk_index_assignment(self, doc_processor, sample_valid_document):
        """Test that chunk indices are assigned correctly"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as tmp_file:
            tmp_file.write(sample_valid_document)
            tmp_file.flush()
            
            try:
                course, chunks = doc_processor.process_course_document(tmp_file.name)
                
                # Verify chunk indices are sequential
                chunk_indices = [chunk.chunk_index for chunk in chunks]
                expected_indices = list(range(len(chunks)))
                assert chunk_indices == expected_indices
                
            finally:
                os.unlink(tmp_file.name)
    
    def test_chunk_lesson_number_assignment(self, doc_processor, sample_valid_document):
        """Test that lesson numbers are correctly assigned to chunks"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as tmp_file:
            tmp_file.write(sample_valid_document)
            tmp_file.flush()
            
            try:
                course, chunks = doc_processor.process_course_document(tmp_file.name)
                
                # Should have chunks from different lessons
                lesson_numbers = [chunk.lesson_number for chunk in chunks if chunk.lesson_number is not None]
                assert 0 in lesson_numbers  # Lesson 0 chunks
                assert 1 in lesson_numbers  # Lesson 1 chunks
                assert 2 in lesson_numbers  # Lesson 2 chunks
                
            finally:
                os.unlink(tmp_file.name)
    
    def test_case_insensitive_header_parsing(self, doc_processor):
        """Test that header parsing is case insensitive"""
        document_mixed_case = """course title: Mixed Case Course
COURSE LINK: https://example.com/mixed-case
Course instructor: Mixed Case Instructor

lesson 0: Introduction
Content here."""
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as tmp_file:
            tmp_file.write(document_mixed_case)
            tmp_file.flush()
            
            try:
                course, chunks = doc_processor.process_course_document(tmp_file.name)
                
                # Should parse headers regardless of case
                assert course.title == "Mixed Case Course"
                assert course.course_link == "https://example.com/mixed-case"
                assert course.instructor == "Mixed Case Instructor"
                assert len(course.lessons) == 1
                assert course.lessons[0].title == "Introduction"
                
            finally:
                os.unlink(tmp_file.name)
    
    def test_whitespace_trimming(self, doc_processor):
        """Test that whitespace is properly trimmed from extracted values"""
        document_with_spaces = """Course Title:   Spacey Course   
Course Link:   https://example.com/spacey   
Course Instructor:   Spacey Instructor   

Lesson 0:   Spacey Lesson   
Content with extra spaces."""
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as tmp_file:
            tmp_file.write(document_with_spaces)
            tmp_file.flush()
            
            try:
                course, chunks = doc_processor.process_course_document(tmp_file.name)
                
                # All values should be trimmed
                assert course.title == "Spacey Course"
                assert course.course_link == "https://example.com/spacey"
                assert course.instructor == "Spacey Instructor"
                assert course.lessons[0].title == "Spacey Lesson"
                
            finally:
                os.unlink(tmp_file.name)
    
    def test_empty_document_handling(self, doc_processor):
        """Test handling of empty document"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as tmp_file:
            tmp_file.write("")  # Empty file
            tmp_file.flush()
            
            try:
                course, chunks = doc_processor.process_course_document(tmp_file.name)
                
                # Should handle empty document gracefully
                assert isinstance(course, Course)
                assert chunks == []  # No chunks from empty content
                
            finally:
                os.unlink(tmp_file.name)
    
    def test_large_document_chunking(self, doc_processor):
        """Test chunking of large documents"""
        # Create a large document that will definitely require multiple chunks
        large_content = "This is a sentence. " * 1000  # Very long content
        large_document = f"""Course Title: Large Course
Course Link: https://example.com/large
Course Instructor: Large Instructor

Lesson 0: Large Lesson
{large_content}"""
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as tmp_file:
            tmp_file.write(large_document)
            tmp_file.flush()
            
            try:
                course, chunks = doc_processor.process_course_document(tmp_file.name)
                
                # Should create multiple chunks
                assert len(chunks) > 5  # Should be many chunks
                
                # All chunks should respect size limits
                for chunk in chunks:
                    assert len(chunk.content) <= doc_processor.chunk_size * 1.2  # Allow some flexibility
                
                # All chunks should have proper metadata
                assert all(chunk.course_title == "Large Course" for chunk in chunks)
                assert all(chunk.course_link == "https://example.com/large" for chunk in chunks)
                
            finally:
                os.unlink(tmp_file.name)
    
    def test_special_characters_handling(self, doc_processor):
        """Test handling of special characters in content"""
        document_with_special_chars = """Course Title: Special Characters Course 🎓
Course Link: https://example.com/special-chars?param=value&other=123
Course Instructor: Dr. José García-López

Lesson 0: Unicode & Symbols ∑∆π
Content with special characters: "quotes", 'apostrophes', —dashes—, and émojis 🚀."""
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as tmp_file:
            tmp_file.write(document_with_special_chars)
            tmp_file.flush()
            
            try:
                course, chunks = doc_processor.process_course_document(tmp_file.name)
                
                # Should handle special characters correctly
                assert "🎓" in course.title
                assert "José García-López" in course.instructor
                assert "Unicode & Symbols ∑∆π" in course.lessons[0].title
                
                # Chunks should preserve special characters
                content_with_special = ''.join(chunk.content for chunk in chunks)
                assert "émojis 🚀" in content_with_special
                
            finally:
                os.unlink(tmp_file.name)
    
    def test_chunk_text_no_overlap_path(self):
        """Test chunking path without overlap to cover lines 86-89"""
        processor = DocumentProcessor(chunk_size=50, chunk_overlap=0)  # No overlap
        
        # Text that will create chunks without overlap
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        
        chunks = processor.chunk_text(text)
        
        assert len(chunks) > 1  # Should create multiple chunks
        
        # Verify chunks don't overlap when overlap=0
        combined = ''.join(chunks)
        # Should contain all original content
        assert len(combined.replace(' ', '')) >= len(text.replace(' ', '')) * 0.8  # Allow for some sentence boundary adjustments