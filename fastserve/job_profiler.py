"""
Job Profiler for FastServe Skip-Join MLFQ Scheduler

Analyzes incoming inference jobs to predict their execution characteristics
and determine optimal initial queue placement to minimize job completion times.
"""

import logging
import time
import hashlib
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
import numpy as np
from enum import Enum

from .models import InferenceJob, Priority
from .config import get_config

logger = logging.getLogger(__name__)


class JobComplexity(Enum):
    """Job complexity classification"""
    SIMPLE = 0      # Short prompts, likely quick completion
    MODERATE = 1    # Medium prompts, moderate execution time
    COMPLEX = 2     # Long prompts, likely longer execution
    HEAVY = 3       # Very complex prompts, definite long execution


@dataclass
class JobProfile:
    """Profile characteristics of a job"""
    complexity: JobComplexity
    predicted_tokens: int
    predicted_duration: float
    confidence: float
    features: Dict[str, float] = field(default_factory=dict)


@dataclass
class HistoricalData:
    """Historical execution data for profiling"""
    prompt_hash: str
    input_length: int
    output_length: int
    execution_time: float
    completion_time: float
    preemptions: int
    queue_assignments: List[int] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


class JobProfiler:
    """
    Intelligent Job Profiler for FastServe
    
    Key responsibilities:
    - Analyze incoming jobs to predict execution characteristics
    - Maintain historical performance data
    - Provide optimal initial queue assignments
    - Adapt predictions based on observed performance
    """
    
    def __init__(self, config=None):
        self.config = config or get_config()
        
        # Historical data storage
        self.historical_data: Dict[str, HistoricalData] = {}
        self.recent_completions: deque = deque(maxlen=1000)
        self.pattern_cache: Dict[str, JobProfile] = {}
        
        # Feature extractors
        self.complexity_keywords = {
            'simple': ['yes', 'no', 'what', 'when', 'where', 'who'],
            'moderate': ['explain', 'describe', 'summarize', 'compare'],
            'complex': ['analyze', 'evaluate', 'create', 'generate', 'write'],
            'heavy': ['comprehensive', 'detailed', 'thorough', 'complete']
        }
        
        # Statistics and adaptation
        self.prediction_accuracy = defaultdict(list)
        self.queue_performance = defaultdict(lambda: defaultdict(list))
        
        # Learning parameters
        self.learning_rate = 0.1
        self.min_samples_for_prediction = 5
        self.cache_ttl = 3600  # 1 hour cache expiry
        
        logger.info("Job Profiler initialized")
    
    def profile_job(self, job: InferenceJob) -> JobProfile:
        """
        Analyze a job and create its profile for queue assignment
        
        Args:
            job: The inference job to profile
            
        Returns:
            JobProfile with predicted characteristics
        """
        # Extract features from the job
        features = self._extract_features(job)
        
        # Check for similar historical patterns
        prompt_pattern = self._get_prompt_pattern(job.prompt)
        
        if prompt_pattern in self.pattern_cache:
            cached_profile = self.pattern_cache[prompt_pattern]
            # Verify cache freshness
            if time.time() - cached_profile.features.get('cached_at', 0) < self.cache_ttl:
                logger.debug(f"Using cached profile for job {job.job_id}")
                return cached_profile
        
        # Predict job characteristics
        complexity = self._predict_complexity(features)
        predicted_tokens = self._predict_output_length(features)
        predicted_duration = self._predict_execution_time(features)
        confidence = self._calculate_confidence(features, prompt_pattern)
        
        # Create profile
        profile = JobProfile(
            complexity=complexity,
            predicted_tokens=predicted_tokens,
            predicted_duration=predicted_duration,
            confidence=confidence,
            features=features
        )
        
        # Cache the profile
        features['cached_at'] = time.time()
        self.pattern_cache[prompt_pattern] = profile
        
        logger.debug(f"Profiled job {job.job_id}: {complexity.name}, "
                    f"predicted_tokens={predicted_tokens}, "
                    f"predicted_duration={predicted_duration:.2f}s, "
                    f"confidence={confidence:.2f}")
        
        return profile
    
    def get_optimal_initial_queue(self, job: InferenceJob, profile: JobProfile) -> int:
        """
        Determine optimal initial queue for job based on its profile
        
        Args:
            job: The inference job
            profile: Job profile from profiling
            
        Returns:
            Optimal initial queue ID (0 = highest priority)
        """
        # Base assignment rules
        if profile.complexity == JobComplexity.SIMPLE:
            base_queue = 0  # High priority for quick jobs
        elif profile.complexity == JobComplexity.MODERATE:
            base_queue = 0  # Start high, will demote if needed
        elif profile.complexity == JobComplexity.COMPLEX:
            base_queue = 1  # Skip highest priority to avoid blocking
        else:  # HEAVY
            base_queue = 2  # Start in lower priority
        
        # Adjust based on predicted characteristics
        if profile.predicted_duration > 5.0:  # Long-running jobs
            base_queue = min(base_queue + 1, 3)
        elif profile.predicted_tokens > 200:  # High output jobs
            base_queue = min(base_queue + 1, 3)
        
        # Historical performance adjustment
        pattern = self._get_prompt_pattern(job.prompt)
        if pattern in self.historical_data:
            historical = self.historical_data[pattern]
            # If this pattern typically gets preempted a lot, start lower
            if historical.preemptions > 2:
                base_queue = min(base_queue + 1, 3)
        
        # Confidence-based adjustment
        if profile.confidence < 0.5:
            # Low confidence predictions - be conservative
            base_queue = min(base_queue + 1, 3)
        
        # System load adjustment (if available)
        base_queue = self._adjust_for_system_load(base_queue)
        
        logger.debug(f"Assigned job {job.job_id} to initial queue {base_queue}")
        return base_queue
    
    def update_with_completion_data(self, job: InferenceJob, 
                                   actual_execution_time: float,
                                   actual_tokens: int) -> None:
        """
        Update profiler with actual job completion data for learning
        
        Args:
            job: The completed job
            actual_execution_time: Actual time taken
            actual_tokens: Actual tokens generated
        """
        pattern = self._get_prompt_pattern(job.prompt)
        
        # Store historical data
        historical = HistoricalData(
            prompt_hash=pattern,
            input_length=job.input_length,
            output_length=actual_tokens,
            execution_time=actual_execution_time,
            completion_time=job.get_completion_time() or 0.0,
            preemptions=job.preemption_count
        )
        
        self.historical_data[pattern] = historical
        self.recent_completions.append(historical)
        
        # Update prediction accuracy
        if pattern in self.pattern_cache:
            cached_profile = self.pattern_cache[pattern]
            
            # Calculate prediction errors
            token_error = abs(cached_profile.predicted_tokens - actual_tokens)
            time_error = abs(cached_profile.predicted_duration - actual_execution_time)
            
            self.prediction_accuracy['tokens'].append(token_error)
            self.prediction_accuracy['time'].append(time_error)
            
            # Adaptive learning - update weights based on accuracy
            self._update_prediction_weights(cached_profile, actual_execution_time, actual_tokens)
        
        # Clean old cache entries
        self._cleanup_cache()
        
        logger.debug(f"Updated profiler with completion data for job {job.job_id}")
    
    def _extract_features(self, job: InferenceJob) -> Dict[str, float]:
        """Extract numerical features from job for prediction"""
        prompt = job.prompt.lower()
        
        features = {
            'input_length': float(job.input_length),
            'max_tokens': float(job.max_new_tokens),
            'temperature': job.temperature,
            'top_p': job.top_p,
            'top_k': float(job.top_k),
            'word_count': float(len(prompt.split())),
            'char_count': float(len(prompt)),
            'sentence_count': float(prompt.count('.') + prompt.count('!') + prompt.count('?')),
            'question_marks': float(prompt.count('?')),
            'complexity_score': self._calculate_complexity_score(prompt)
        }
        
        return features
    
    def _calculate_complexity_score(self, prompt: str) -> float:
        """Calculate a complexity score for the prompt"""
        score = 0.0
        prompt_lower = prompt.lower()
        
        # Keyword-based scoring
        for complexity_level, keywords in self.complexity_keywords.items():
            for keyword in keywords:
                if keyword in prompt_lower:
                    if complexity_level == 'simple':
                        score += 0.25
                    elif complexity_level == 'moderate':
                        score += 0.5
                    elif complexity_level == 'complex':
                        score += 0.75
                    elif complexity_level == 'heavy':
                        score += 1.0
        
        # Length-based adjustment
        word_count = len(prompt.split())
        if word_count > 100:
            score += 0.5
        elif word_count > 50:
            score += 0.25
        
        return min(score, 3.0)  # Cap at 3.0
    
    def _predict_complexity(self, features: Dict[str, float]) -> JobComplexity:
        """Predict job complexity based on features"""
        complexity_score = features.get('complexity_score', 0.0)
        input_length = features.get('input_length', 0)
        max_tokens = features.get('max_tokens', 0)
        
        # Weighted scoring
        total_score = (
            complexity_score * 0.4 +
            (input_length / 50.0) * 0.3 +
            (max_tokens / 100.0) * 0.3
        )
        
        if total_score < 0.5:
            return JobComplexity.SIMPLE
        elif total_score < 1.0:
            return JobComplexity.MODERATE
        elif total_score < 2.0:
            return JobComplexity.COMPLEX
        else:
            return JobComplexity.HEAVY
    
    def _predict_output_length(self, features: Dict[str, float]) -> int:
        """Predict expected output token count"""
        # Simple heuristic - can be improved with ML models
        max_tokens = features.get('max_tokens', 100)
        complexity_score = features.get('complexity_score', 1.0)
        input_length = features.get('input_length', 10)
        
        # Estimate based on complexity and input length
        base_estimate = min(max_tokens, int(input_length * complexity_score * 0.8))
        
        return max(10, base_estimate)  # Minimum 10 tokens
    
    def _predict_execution_time(self, features: Dict[str, float]) -> float:
        """Predict expected execution time in seconds"""
        predicted_tokens = self._predict_output_length(features)
        complexity_score = features.get('complexity_score', 1.0)
        
        # Base time estimation (tokens per second varies by complexity)
        base_rate = 20.0  # tokens per second for simple jobs
        adjusted_rate = base_rate / (1.0 + complexity_score * 0.5)
        
        estimated_time = predicted_tokens / adjusted_rate
        
        return max(0.5, estimated_time)  # Minimum 0.5 seconds
    
    def _calculate_confidence(self, features: Dict[str, float], pattern: str) -> float:
        """Calculate confidence in predictions"""
        base_confidence = 0.5
        
        # Increase confidence if we have historical data
        if pattern in self.historical_data:
            base_confidence += 0.3
        
        # Increase confidence for simpler prompts
        complexity_score = features.get('complexity_score', 1.0)
        if complexity_score < 1.0:
            base_confidence += 0.2
        
        return min(1.0, base_confidence)
    
    def _get_prompt_pattern(self, prompt: str) -> str:
        """Generate a pattern hash for similar prompts"""
        # Simple pattern matching - can be enhanced with semantic similarity
        normalized = prompt.lower().strip()
        
        # Create a hash of the first 100 characters for pattern matching
        pattern_text = normalized[:100] if len(normalized) > 100 else normalized
        
        return hashlib.md5(pattern_text.encode()).hexdigest()[:16]
    
    def _adjust_for_system_load(self, base_queue: int) -> int:
        """Adjust queue assignment based on current system load"""
        # Simple load balancing - can be enhanced with actual load metrics
        recent_count = len(self.recent_completions)
        
        if recent_count > 50:  # High load
            return min(base_queue + 1, 3)
        elif recent_count < 10:  # Low load
            return max(base_queue - 1, 0)
        
        return base_queue
    
    def _update_prediction_weights(self, profile: JobProfile, 
                                  actual_time: float, actual_tokens: int) -> None:
        """Update internal prediction weights based on accuracy"""
        # Simple adaptive learning - can be enhanced with proper ML
        time_error = abs(profile.predicted_duration - actual_time) / max(actual_time, 1.0)
        token_error = abs(profile.predicted_tokens - actual_tokens) / max(actual_tokens, 1.0)
        
        # Log for potential model updates
        logger.debug(f"Prediction errors - Time: {time_error:.2f}, Tokens: {token_error:.2f}")
    
    def _cleanup_cache(self) -> None:
        """Clean up old cache entries"""
        current_time = time.time()
        expired_keys = []
        
        for key, profile in self.pattern_cache.items():
            cached_at = profile.features.get('cached_at', 0)
            if current_time - cached_at > self.cache_ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.pattern_cache[key]
        
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def get_statistics(self) -> Dict[str, float]:
        """Get profiler performance statistics"""
        stats = {
            'total_patterns': len(self.historical_data),
            'cache_size': len(self.pattern_cache),
            'recent_completions': len(self.recent_completions)
        }
        
        if self.prediction_accuracy['tokens']:
            stats['avg_token_error'] = np.mean(self.prediction_accuracy['tokens'])
            stats['avg_time_error'] = np.mean(self.prediction_accuracy['time'])
        
        return stats 