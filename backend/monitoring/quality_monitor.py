"""
ML and Data Quality Monitoring for RAG System
Prevents ML debt accumulation through proactive monitoring
"""
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
from pathlib import Path
import requests
from urllib.parse import urlparse

from models import Course, CourseChunk
from vector_store import SearchResults


@dataclass 
class QualityMetrics:
    """Container for quality metrics"""
    timestamp: datetime
    document_count: int
    chunk_count: int
    avg_chunk_size: float
    broken_links: List[str]
    missing_metadata: List[str]
    duplicate_chunks: int
    retrieval_latency: float
    embedding_consistency_score: float


@dataclass
class DocumentQualityReport:
    """Quality report for course documents"""
    file_path: str
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    metadata_completeness: float
    content_quality_score: float


class DocumentQualityMonitor:
    """Monitors document quality and prevents data debt"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.link_cache: Dict[str, Tuple[bool, datetime]] = {}
        self.cache_ttl = timedelta(hours=24)  # Cache link checks for 24 hours
    
    def validate_document_structure(self, doc_path: str) -> DocumentQualityReport:
        """
        Validate that document follows expected RAG format
        Prevents ML debt from malformed training data
        """
        errors = []
        warnings = []
        
        try:
            with open(doc_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            return DocumentQualityReport(
                file_path=doc_path,
                is_valid=False,
                errors=[f"Cannot read file: {e}"],
                warnings=[],
                metadata_completeness=0.0,
                content_quality_score=0.0
            )
        
        lines = content.strip().split('\n')
        
        # Check required headers
        required_headers = ['Course Title:', 'Course Link:', 'Course Instructor:']
        metadata_found = 0
        
        for header in required_headers:
            if any(line.startswith(header) for line in lines[:10]):  # Check first 10 lines
                metadata_found += 1
            else:
                errors.append(f"Missing required header: {header}")
        
        # Check for lesson structure
        lesson_pattern = r'^Lesson \d+:'
        import re
        lesson_matches = [line for line in lines if re.match(lesson_pattern, line)]
        
        if not lesson_matches:
            errors.append("No lessons found (expected 'Lesson N:' format)")
        else:
            # Check lesson numbering consistency
            lesson_numbers = []
            for match in lesson_matches:
                try:
                    num = int(re.search(r'Lesson (\d+):', match).group(1))
                    lesson_numbers.append(num)
                except:
                    warnings.append(f"Invalid lesson number format: {match}")
            
            # Check for gaps in lesson numbering
            if lesson_numbers:
                expected = list(range(min(lesson_numbers), max(lesson_numbers) + 1))
                if sorted(lesson_numbers) != expected:
                    warnings.append(f"Non-sequential lesson numbers: {lesson_numbers}")
        
        # Calculate quality scores
        metadata_completeness = metadata_found / len(required_headers)
        
        # Content quality heuristics
        content_quality_score = self._calculate_content_quality(content)
        
        # Check course link accessibility
        course_link = self._extract_course_link(content)
        if course_link and not self._is_link_accessible(course_link):
            warnings.append(f"Course link may be inaccessible: {course_link}")
        
        is_valid = len(errors) == 0 and metadata_completeness >= 0.8
        
        return DocumentQualityReport(
            file_path=doc_path,
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            metadata_completeness=metadata_completeness,
            content_quality_score=content_quality_score
        )
    
    def _calculate_content_quality(self, content: str) -> float:
        """Calculate content quality score (0-1)"""
        score = 1.0
        
        # Penalize very short content
        if len(content) < 500:
            score -= 0.3
        
        # Penalize excessive repetition
        words = content.lower().split()
        if len(set(words)) / len(words) < 0.3:  # Low word diversity
            score -= 0.2
        
        # Check for reasonable sentence structure
        sentences = content.split('.')
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
        if avg_sentence_length < 3 or avg_sentence_length > 50:
            score -= 0.2
        
        return max(0.0, score)
    
    def _extract_course_link(self, content: str) -> Optional[str]:
        """Extract course link from document content"""
        for line in content.split('\n'):
            if line.startswith('Course Link:'):
                return line.split('Course Link:', 1)[1].strip()
        return None
    
    def _is_link_accessible(self, url: str) -> bool:
        """Check if URL is accessible (with caching)"""
        now = datetime.now()
        
        # Check cache first
        if url in self.link_cache:
            is_accessible, cached_time = self.link_cache[url]
            if now - cached_time < self.cache_ttl:
                return is_accessible
        
        # Check accessibility
        try:
            response = requests.head(url, timeout=10, allow_redirects=True)
            is_accessible = response.status_code < 400
        except:
            is_accessible = False
        
        # Cache result
        self.link_cache[url] = (is_accessible, now)
        return is_accessible


class ChunkQualityMonitor:
    """Monitors chunk quality and prevents embedding debt"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def analyze_chunk_quality(self, chunks: List[CourseChunk]) -> Dict[str, Any]:
        """Analyze quality of course chunks to prevent ML debt"""
        if not chunks:
            return {"error": "No chunks to analyze"}
        
        # Size analysis
        chunk_sizes = [len(chunk.content) for chunk in chunks]
        avg_size = sum(chunk_sizes) / len(chunk_sizes)
        size_std = (sum((s - avg_size) ** 2 for s in chunk_sizes) / len(chunk_sizes)) ** 0.5
        
        # Content diversity analysis
        unique_content = set(chunk.content for chunk in chunks)
        duplication_rate = 1 - (len(unique_content) / len(chunks))
        
        # Metadata completeness
        chunks_with_links = sum(1 for chunk in chunks if chunk.course_link)
        link_completeness = chunks_with_links / len(chunks)
        
        # Course distribution
        course_distribution = {}
        for chunk in chunks:
            course_distribution[chunk.course_title] = course_distribution.get(chunk.course_title, 0) + 1
        
        # Quality flags
        flags = []
        if avg_size < 100:
            flags.append("SMALL_CHUNKS: Average chunk size very small")
        if avg_size > 2000:
            flags.append("LARGE_CHUNKS: Average chunk size very large")
        if size_std > avg_size * 0.8:
            flags.append("INCONSISTENT_SIZES: High variation in chunk sizes")
        if duplication_rate > 0.1:
            flags.append("HIGH_DUPLICATION: Significant duplicate content detected")
        if link_completeness < 0.8:
            flags.append("MISSING_LINKS: Many chunks missing course links")
        
        return {
            "total_chunks": len(chunks),
            "unique_chunks": len(unique_content),
            "avg_chunk_size": avg_size,
            "chunk_size_std": size_std,
            "duplication_rate": duplication_rate,
            "link_completeness": link_completeness,
            "course_distribution": course_distribution,
            "quality_flags": flags,
            "overall_score": self._calculate_overall_score(avg_size, duplication_rate, link_completeness)
        }
    
    def _calculate_overall_score(self, avg_size: float, duplication_rate: float, link_completeness: float) -> float:
        """Calculate overall chunk quality score (0-1)"""
        score = 1.0
        
        # Size penalty
        if avg_size < 200 or avg_size > 1500:
            score -= 0.2
        
        # Duplication penalty
        score -= min(duplication_rate * 2, 0.4)  # Max 40% penalty
        
        # Link completeness bonus
        score = score * (0.6 + 0.4 * link_completeness)  # Scale by link completeness
        
        return max(0.0, min(1.0, score))


class RetrievalQualityMonitor:
    """Monitors retrieval quality to prevent ML performance debt"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.query_log: List[Dict] = []
    
    def log_query_result(self, query: str, results: SearchResults, response_time: float):
        """Log query and results for quality analysis"""
        entry = {
            "timestamp": datetime.now(),
            "query": query,
            "result_count": len(results.documents),
            "avg_distance": sum(results.distances) / len(results.distances) if results.distances else 1.0,
            "response_time": response_time,
            "has_results": not results.is_empty()
        }
        self.query_log.append(entry)
        
        # Keep only last 1000 entries
        if len(self.query_log) > 1000:
            self.query_log = self.query_log[-1000:]
    
    def get_retrieval_metrics(self, hours: int = 24) -> Dict[str, Any]:
        """Get retrieval quality metrics for the last N hours"""
        cutoff = datetime.now() - timedelta(hours=hours)
        recent_queries = [q for q in self.query_log if q["timestamp"] > cutoff]
        
        if not recent_queries:
            return {"error": "No recent queries found"}
        
        # Calculate metrics
        total_queries = len(recent_queries)
        successful_queries = sum(1 for q in recent_queries if q["has_results"])
        success_rate = successful_queries / total_queries
        
        avg_response_time = sum(q["response_time"] for q in recent_queries) / total_queries
        avg_distance = sum(q["avg_distance"] for q in recent_queries if q["has_results"]) / successful_queries if successful_queries > 0 else 1.0
        
        # Quality flags
        flags = []
        if success_rate < 0.8:
            flags.append("LOW_SUCCESS_RATE: Many queries returning no results")
        if avg_response_time > 2.0:
            flags.append("SLOW_RETRIEVAL: Average response time too high")
        if avg_distance > 0.7:
            flags.append("POOR_RELEVANCE: Search results not very relevant")
        
        return {
            "period_hours": hours,
            "total_queries": total_queries,
            "success_rate": success_rate,
            "avg_response_time": avg_response_time,
            "avg_relevance_score": 1 - avg_distance,  # Convert distance to relevance
            "quality_flags": flags,
            "overall_health": "GOOD" if not flags else "NEEDS_ATTENTION"
        }


class SystemHealthMonitor:
    """Overall system health monitoring to prevent technical debt"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.doc_monitor = DocumentQualityMonitor()
        self.chunk_monitor = ChunkQualityMonitor()
        self.retrieval_monitor = RetrievalQualityMonitor()
    
    def generate_health_report(self, docs_path: str, chunks: List[CourseChunk]) -> Dict[str, Any]:
        """Generate comprehensive system health report"""
        report = {
            "timestamp": datetime.now(),
            "document_quality": {},
            "chunk_quality": {},
            "retrieval_quality": {},
            "overall_status": "UNKNOWN"
        }
        
        # Document quality assessment
        try:
            doc_files = list(Path(docs_path).glob("*.txt"))
            doc_reports = []
            for doc_file in doc_files:
                doc_report = self.doc_monitor.validate_document_structure(str(doc_file))
                doc_reports.append(doc_report)
            
            valid_docs = sum(1 for r in doc_reports if r.is_valid)
            report["document_quality"] = {
                "total_documents": len(doc_reports),
                "valid_documents": valid_docs,
                "validation_rate": valid_docs / len(doc_reports) if doc_reports else 0,
                "avg_metadata_completeness": sum(r.metadata_completeness for r in doc_reports) / len(doc_reports) if doc_reports else 0,
                "reports": [asdict(r) for r in doc_reports]
            }
        except Exception as e:
            self.logger.error(f"Document quality assessment failed: {e}")
            report["document_quality"]["error"] = str(e)
        
        # Chunk quality assessment
        try:
            report["chunk_quality"] = self.chunk_monitor.analyze_chunk_quality(chunks)
        except Exception as e:
            self.logger.error(f"Chunk quality assessment failed: {e}")
            report["chunk_quality"]["error"] = str(e)
        
        # Retrieval quality assessment
        try:
            report["retrieval_quality"] = self.retrieval_monitor.get_retrieval_metrics()
        except Exception as e:
            self.logger.error(f"Retrieval quality assessment failed: {e}")
            report["retrieval_quality"]["error"] = str(e)
        
        # Overall status determination
        issues = []
        if report["document_quality"].get("validation_rate", 0) < 0.8:
            issues.append("DOCUMENT_QUALITY")
        if report["chunk_quality"].get("overall_score", 0) < 0.7:
            issues.append("CHUNK_QUALITY")
        if report["retrieval_quality"].get("success_rate", 0) < 0.8:
            issues.append("RETRIEVAL_QUALITY")
        
        if not issues:
            report["overall_status"] = "HEALTHY"
        elif len(issues) == 1:
            report["overall_status"] = "WARNING"
        else:
            report["overall_status"] = "CRITICAL"
        
        report["issues"] = issues
        
        return report