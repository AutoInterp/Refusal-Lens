"""
Refusal Detector Module

Combines prompt generation and response classification to detect and analyze
model refusal behavior in a consistent manner.
"""

import logging
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from .prompt_template import PromptCategory, PromptTemplate, PromptTemplateLibrary
from .refusal_classifier import ClassificationResult, RefusalCategory, RefusalClassifier

logger = logging.getLogger(__name__)


class DetectionStatus(Enum):
    """Status of the refusal detection."""

    SUCCESS = "success"
    FAILED = "failed"
    ERROR = "error"


@dataclass
class DetectionResult:
    """
    Result of a refusal detection operation.

    Contains the prompt, model response, classification, and metadata.
    """

    prompt: str
    response: str
    classification: ClassificationResult | None
    topic: str
    category: PromptCategory
    detection_status: DetectionStatus = DetectionStatus.SUCCESS
    error_message: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_refusal(self) -> bool:
        """Check if the response is a refusal (strong or soft)."""
        if self.classification is None:
            return False
        return self.classification.category in (
            RefusalCategory.STRONG_REFUSAL,
            RefusalCategory.SOFT_REFUSAL,
        )

    @property
    def is_strong_refusal(self) -> bool:
        """Check if the response is a strong refusal."""
        if self.classification is None:
            return False
        return self.classification.category == RefusalCategory.STRONG_REFUSAL

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "prompt": self.prompt,
            "response": self.response,
            "classification": self.classification.to_dict()
            if self.classification
            else None,
            "topic": self.topic,
            "category": self.category.value,
            "detection_status": self.detection_status.value,
            "error_message": self.error_message,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "is_refusal": self.is_refusal,
            "is_strong_refusal": self.is_strong_refusal,
        }


class RefusalDetector:
    """
    Detects and analyzes model refusal behavior.

    Combines prompt generation with response classification to provide
    consistent detection of refusal behavior across different prompt framings.
    """

    def __init__(
        self,
        classifier: RefusalClassifier | None = None,
        template_library: PromptTemplateLibrary | None = None,
        model: Any | None = None,
        tokenizer: Any | None = None,
    ):
        """
        Initialize the RefusalDetector.

        Args:
            classifier: RefusalClassifier instance. Creates default if None.
            template_library: PromptTemplateLibrary instance. Creates default if None.
            model: Model to generate responses (optional).
            tokenizer: Tokenizer for the model (optional).
        """
        self.classifier = classifier or RefusalClassifier()
        self.template_library = template_library or PromptTemplateLibrary()
        self.model = model
        self.tokenizer = tokenizer
        self.logger = logging.getLogger(self.__class__.__name__)

    def set_model(self, model: Any, tokenizer: Any) -> None:
        """
        Set the model and tokenizer for response generation.

        Args:
            model: The language model.
            tokenizer: The tokenizer for the model.
        """
        self.model = model
        self.tokenizer = tokenizer

    def detect(
        self,
        prompt: str,
        topic: str = "",
        category: PromptCategory = PromptCategory.DIRECT_HARM,
        generate_response: Callable[[str], str] | None = None,
    ) -> DetectionResult:
        """
        Detect refusal behavior for a given prompt.

        Args:
            prompt: The prompt to test.
            topic: The topic being tested.
            category: The prompt category.
            generate_response: Optional function to generate response.
                           Uses model's generate_response if not provided.

        Returns:
            DetectionResult with classification and metadata.
        """
        try:
            # Generate response if generator provided
            if generate_response:
                response = generate_response(prompt)
            elif self.model and self.tokenizer:
                response = self._generate_response(prompt)
            else:
                msg = "No response generation method available"
                raise ValueError(msg)

            # Classify the response
            classification = self.classifier.classify(response)

            return DetectionResult(
                prompt=prompt,
                response=response,
                classification=classification,
                topic=topic,
                category=category,
                detection_status=DetectionStatus.SUCCESS,
            )

        except Exception as e:
            msg = f"Error during detection: {e}"
            self.logger.error(msg)
            return DetectionResult(
                prompt=prompt,
                response="",
                classification=ClassificationResult(
                    category=RefusalCategory.UNKNOWN,
                    confidence=0.0,
                    pattern_counts={
                        "strong_refusal": 0,
                        "soft_refusal": 0,
                        "acceptance": 0,
                    },
                    is_valid=False,
                    validation_message=str(e),
                ),
                topic=topic,
                category=category,
                detection_status=DetectionStatus.ERROR,
                error_message=str(e),
            )

    def detect_from_template(
        self,
        template: PromptTemplate,
        generate_response: Callable[[str], str] | None = None,
    ) -> DetectionResult:
        """
        Detect refusal behavior using a prompt template.

        Args:
            template: The PromptTemplate to use.
            generate_response: Optional function to generate response.

        Returns:
            DetectionResult with classification and metadata.
        """
        return self.detect(
            prompt=template.template,
            topic=template.topic,
            category=template.category,
            generate_response=generate_response,
        )

    def detect_batch(
        self,
        prompts: list[str],
        topics: list[str] | None = None,
        categories: list[PromptCategory] | None = None,
        generate_response: Callable[[str], str] | None = None,
    ) -> list[DetectionResult]:
        """
        Detect refusal behavior for multiple prompts.

        Args:
            prompts: List of prompts to test.
            topics: List of topics (one per prompt).
            categories: List of categories (one per prompt).
            generate_response: Optional function to generate response.

        Returns:
            List of DetectionResult objects.
        """
        results = []
        topics = topics or ["" for _ in prompts]
        categories = categories or [PromptCategory.DIRECT_HARM for _ in prompts]

        for prompt, topic, category in zip(prompts, topics, categories, strict=False):
            result = self.detect(
                prompt=prompt,
                topic=topic,
                category=category,
                generate_response=generate_response,
            )
            results.append(result)

        return results

    def detect_category_comparison(
        self,
        topic: str,
        topic_request: str,
        harmful_keywords: list[str],
        topic_info: str | None = None,
        generate_response: Callable[[str], str] | None = None,
    ) -> list[DetectionResult]:
        """
        Test refusal behavior across all categories for a single topic.

        This is useful for comparing how different prompt framings affect
        the model's refusal behavior.

        Args:
            topic: The harmful topic being tested.
            topic_request: The request to make about the topic.
            harmful_keywords: Keywords indicating harmful content.
            topic_info: Information about the topic for framing.
            generate_response: Optional function to generate response.

        Returns:
            List of DetectionResult objects, one for each category.
        """
        templates = self.template_library.generate_prompts_for_topic(
            topic=topic,
            topic_request=topic_request,
            harmful_keywords=harmful_keywords,
            topic_info=topic_info,
        )

        results = []
        for template in templates:
            result = self.detect_from_template(
                template=template,
                generate_response=generate_response,
            )
            results.append(result)

        return results

    def get_category_statistics(self, results: list[DetectionResult]) -> dict[str, Any]:
        """
        Get statistics for detection results grouped by category.

        Args:
            results: List of DetectionResult objects.

        Returns:
            Dictionary with statistics per category.
        """

        stats: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "total": 0,
                "strong_refusals": 0,
                "soft_refusals": 0,
                "accepts": 0,
                "errors": 0,
                "avg_confidence": 0.0,
            }
        )

        for result in results:
            category = result.category.value
            stats[category]["total"] += 1

            if result.detection_status == DetectionStatus.ERROR:
                stats[category]["errors"] += 1
            elif result.is_strong_refusal and result.classification:
                stats[category]["strong_refusals"] += 1
                stats[category]["avg_confidence"] += result.classification.confidence
            elif result.is_refusal and result.classification:
                stats[category]["soft_refusals"] += 1
                stats[category]["avg_confidence"] += result.classification.confidence
            elif result.classification:
                stats[category]["accepts"] += 1
                stats[category]["avg_confidence"] += result.classification.confidence

        # Calculate averages
        for _category, data in stats.items():
            if data["total"] > 0:
                data["avg_confidence"] /= data["total"]
            data["refusal_rate"] = (
                (data["strong_refusals"] + data["soft_refusals"]) / data["total"]
                if data["total"] > 0
                else 0.0
            )
            data["strong_refusal_rate"] = (
                data["strong_refusals"] / data["total"] if data["total"] > 0 else 0.0
            )

        return dict(stats)

    def _generate_response(self, prompt: str, max_tokens: int = 150) -> str:
        """
        Generate a response using the configured model.

        Args:
            prompt: The prompt to send to the model.
            max_tokens: Maximum tokens to generate.

        Returns:
            The generated response.
        """
        if not self.model or not self.tokenizer:
            msg = "Model and tokenizer not set"
            raise ValueError(msg)

        # This is a simplified version - in practice you'd want to handle
        # special tokens, chat templates, etc.
        inputs = self.tokenizer(prompt, return_tensors="pt")

        # Move to same device as model
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with self.model.generation_context():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )

        response: str = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove the prompt from the response
        if response.startswith(prompt):
            response = response[len(prompt) :].strip()

        return response

    def validate_consistency(self, results: list[DetectionResult]) -> dict[str, Any]:
        """
        Validate that classification is consistent across similar prompts.

        Args:
            results: List of DetectionResult objects.

        Returns:
            Dictionary with consistency metrics.
        """

        # Group by topic
        topic_results: dict[str, list[DetectionResult]] = defaultdict(list)
        for result in results:
            topic_results[result.topic].append(result)

        inconsistencies = []
        for topic, topic_results_list in topic_results.items():
            # Check if same topic has different classifications
            # Filter out results with no classification
            valid_results = [r for r in topic_results_list if r.classification]
            categories = []
            for r in valid_results:
                if r.classification:
                    categories.append(r.classification.category)
            if len(set(categories)) > 1:
                inconsistencies.append(
                    {
                        "topic": topic,
                        "classifications": [c.value for c in categories],
                        "count": len(categories),
                    }
                )

        return {
            "total_topics": len(topic_results),
            "inconsistent_topics": len(inconsistencies),
            "inconsistencies": inconsistencies,
            "consistency_rate": (
                1 - len(inconsistencies) / len(topic_results) if topic_results else 1.0
            ),
        }
