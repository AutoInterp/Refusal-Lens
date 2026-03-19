"""
Experiment Runner Module

Provides functionality for running refusal detection experiments with
progress tracking, error handling, and result persistence.
"""

import json
import logging
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from .prompt_template import PromptCategory, PromptTemplate
from .refusal_detector import DetectionResult, DetectionStatus, RefusalDetector

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for an experiment run."""

    name: str = "refusal_detection_experiment"
    save_interval: int = 5
    max_retries: int = 3
    retry_delay: float = 1.0
    output_dir: Path | None = None
    verbose: bool = True


@dataclass
class ExperimentResult:
    """Result of an entire experiment."""

    config: ExperimentConfig
    results: list[DetectionResult]
    start_time: str
    end_time: str
    duration_seconds: float
    statistics: dict[str, Any]
    errors: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "config": {
                "name": self.config.name,
                "save_interval": self.config.save_interval,
                "max_retries": self.config.max_retries,
            },
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": self.duration_seconds,
            "total_prompts": len(self.results),
            "statistics": self.statistics,
            "errors": self.errors,
            "results": [r.to_dict() for r in self.results],
        }

    def save(self, filepath: Path) -> None:
        """Save experiment results to a JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

        logger.info("Experiment results saved to %s", filepath)


class ExperimentRunner:
    """
    Runs refusal detection experiments with progress tracking and error handling.
    """

    def __init__(
        self,
        detector: RefusalDetector,
        config: ExperimentConfig | None = None,
    ):
        """
        Initialize the ExperimentRunner.

        Args:
            detector: RefusalDetector instance to use for detection.
            config: Experiment configuration. Uses defaults if None.
        """
        self.detector = detector
        self.config = config or ExperimentConfig()
        self.logger = logging.getLogger(self.__class__.__name__)

    def run(
        self,
        prompts: list[PromptTemplate] | list[str],
        generate_response: Callable[[str], str] | None = None,
        topics: list[str] | None = None,
        categories: list[PromptCategory] | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> ExperimentResult:
        """
        Run the experiment on a list of prompts.

        Args:
            prompts: List of prompts or PromptTemplates to test.
            generate_response: Function to generate model responses.
            topics: Optional list of topics (one per prompt).
            categories: Optional list of categories (one per prompt).
            progress_callback: Optional callback for progress updates.

        Returns:
            ExperimentResult with all detection results and statistics.
        """
        start_time = time.time()
        start_timestamp = datetime.now().isoformat()

        results: list[DetectionResult] = []
        errors: list[dict[str, Any]] = []

        total = len(prompts)

        if self.config.verbose:
            self.logger.info("Starting experiment with %s prompts...", total)

        for idx, prompt in enumerate(prompts):
            try:
                # Determine prompt type and extract needed info
                if isinstance(prompt, PromptTemplate):
                    result = self.detector.detect_from_template(
                        template=prompt,
                        generate_response=generate_response,
                    )
                else:
                    topic = topics[idx] if topics and idx < len(topics) else ""
                    category = (
                        categories[idx]
                        if categories and idx < len(categories)
                        else PromptCategory.DIRECT_HARM
                    )
                    result = self.detector.detect(
                        prompt=cast(str, prompt),
                        topic=topic,
                        category=category,
                        generate_response=generate_response,
                    )

                results.append(result)

                # Progress callback
                if progress_callback:
                    progress_callback(idx + 1, total)

                # Auto-save at intervals
                if (
                    self.config.output_dir
                    and (idx + 1) % self.config.save_interval == 0
                ):
                    self._save_progress(results, errors, idx + 1)

                if self.config.verbose and (idx + 1) % 10 == 0:
                    self.logger.info("Processed %s/%s prompts", idx + 1, total)

            except Exception as e:
                self.logger.error("Error processing prompt %s: %s", idx, e)
                errors.append(
                    {
                        "prompt_index": idx,
                        "error": str(e),
                        "timestamp": datetime.now().isoformat(),
                    }
                )

                # Create error result
                error_result = DetectionResult(
                    prompt=str(prompt),
                    response="",
                    classification=None,  # Will be handled in to_dict
                    topic=topics[idx] if topics else "",
                    category=categories[idx]
                    if categories
                    else PromptCategory.DIRECT_HARM,
                    detection_status=DetectionStatus.ERROR,
                    error_message=str(e),
                )
                results.append(error_result)

        end_time = time.time()
        end_timestamp = datetime.now().isoformat()
        duration = end_time - start_time

        # Calculate statistics
        statistics = self._calculate_statistics(results)

        if self.config.verbose:
            self.logger.info("Experiment complete in %.2fs", duration)
            self.logger.info(
                "Results: %s successful, %s failed, %s errors",
                statistics["successful"],
                statistics["failed"],
                len(errors),
            )

        # Create experiment result
        experiment_result = ExperimentResult(
            config=self.config,
            results=results,
            start_time=start_timestamp,
            end_time=end_timestamp,
            duration_seconds=duration,
            statistics=statistics,
            errors=errors,
        )

        # Save final results if output directory specified
        if self.config.output_dir:
            output_path = self.config.output_dir / f"{self.config.name}_results.json"
            experiment_result.save(output_path)

        return experiment_result

    def run_category_comparison(
        self,
        topic: str,
        topic_request: str,
        harmful_keywords: list[str],
        topic_info: str | None = None,
        generate_response: Callable[[str], str] | None = None,
    ) -> list[DetectionResult]:
        """
        Run a category comparison experiment for a single topic.

        Tests all prompt categories (Direct Harm, Contextual/Educational,
        Roleplay, Hypothetical/Fictional) to compare refusal behavior.

        Args:
            topic: The harmful topic being tested.
            topic_request: The request to make about the topic.
            harmful_keywords: Keywords indicating harmful content.
            topic_info: Information about the topic for framing.
            generate_response: Function to generate model responses.

        Returns:
            List of DetectionResult objects, one for each category.
        """
        return self.detector.detect_category_comparison(
            topic=topic,
            topic_request=topic_request,
            harmful_keywords=harmful_keywords,
            topic_info=topic_info,
            generate_response=generate_response,
        )

    def run_batch_topics(
        self,
        topics: list[dict[str, Any]],
        generate_response: Callable[[str], str] | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> ExperimentResult:
        """
        Run experiments on multiple topics across all categories.

        Args:
            topics: List of topic dictionaries with keys:
                   - topic: The harmful topic
                   - topic_request: The request to make
                   - harmful_keywords: Keywords indicating harmful content
                   - topic_info: Optional info for framing
            generate_response: Function to generate model responses.
            progress_callback: Optional callback for progress updates.

        Returns:
            ExperimentResult with all detection results.
        """
        all_prompts: list[PromptTemplate] = []
        all_topics: list[str] = []
        all_categories: list[PromptCategory] = []

        # Generate prompts for all topics and categories
        for topic_data in topics:
            prompts = self.detector.template_library.generate_prompts_for_topic(
                topic=topic_data["topic"],
                topic_request=topic_data["topic_request"],
                harmful_keywords=topic_data.get("harmful_keywords", []),
                topic_info=topic_data.get("topic_info"),
            )

            for prompt in prompts:
                all_prompts.append(prompt)
                all_topics.append(topic_data["topic"])
                all_categories.append(prompt.category)

        return self.run(
            prompts=all_prompts,
            generate_response=generate_response,
            topics=all_topics,
            categories=all_categories,
            progress_callback=progress_callback,
        )

    def _calculate_statistics(self, results: list[DetectionResult]) -> dict[str, Any]:
        """Calculate statistics from detection results."""
        total = len(results)
        if total == 0:
            return {"total": 0}

        successful = sum(
            1 for r in results if r.detection_status == DetectionStatus.SUCCESS
        )
        failed = sum(1 for r in results if r.detection_status == DetectionStatus.FAILED)
        errors = sum(1 for r in results if r.detection_status == DetectionStatus.ERROR)

        # Count classifications
        category_counts = Counter(
            r.classification.category.value for r in results if r.classification
        )

        # Get category statistics from detector
        category_stats = self.detector.get_category_statistics(results)

        # Calculate confidence statistics
        confidences = [r.classification.confidence for r in results if r.classification]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        return {
            "total": total,
            "successful": successful,
            "failed": failed,
            "errors": errors,
            "category_counts": dict(category_counts),
            "category_statistics": category_stats,
            "average_confidence": avg_confidence,
            "refusal_rate": (
                sum(1 for r in results if r.is_refusal) / total if total > 0 else 0.0
            ),
            "strong_refusal_rate": (
                sum(1 for r in results if r.is_strong_refusal) / total
                if total > 0
                else 0.0
            ),
        }

    def _save_progress(
        self,
        results: list[DetectionResult],
        errors: list[dict[str, Any]],
        current_count: int,
    ) -> None:
        """Save progress to a temporary file."""
        if not self.config.output_dir:
            return

        progress_path = self.config.output_dir / f"{self.config.name}_progress.json"

        progress_data = {
            "current_count": current_count,
            "errors": errors,
            "results": [r.to_dict() for r in results[-self.config.save_interval :]],
        }

        with open(progress_path, "w") as f:
            json.dump(progress_data, f, indent=2)

        self.logger.debug("Progress saved to %s", progress_path)
