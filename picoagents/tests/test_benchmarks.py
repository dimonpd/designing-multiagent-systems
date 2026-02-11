"""
Tests for the benchmark system.

Tests cover:
- AgentConfig creation and serialization
- BenchmarkTask and BenchmarkDataset
- BenchmarkResults and comparison utilities
- BenchmarkMiddleware metrics collection
- BenchmarkRunner execution
- Built-in dataset loading
- Analysis formatting
"""

import json
import tempfile
from pathlib import Path
from typing import List

import pytest

from picoagents.eval.benchmarks import (
    AgentConfig,
    BenchmarkDataset,
    BenchmarkMiddleware,
    BenchmarkResults,
    BenchmarkRunner,
    BenchmarkTask,
    BenchmarkTarget,
    CallableTarget,
    PicoAgentTarget,
    TargetResult,
    TargetSummary,
    TaskResult,
    format_file_read_analysis,
    format_summary_table,
    format_task_breakdown,
    list_builtin_datasets,
    load_builtin_dataset,
)
from picoagents.eval._base import BaseEvalJudge
from picoagents.messages import AssistantMessage, UserMessage
from picoagents.types import EvalScore, EvalTask, EvalTrajectory, Usage


# =============================================================================
# Test Fixtures and Helpers
# =============================================================================


class MockJudge(BaseEvalJudge):
    """Mock judge that returns configurable scores."""

    def __init__(self, score: float = 8.0):
        super().__init__(name="mock_judge")
        self.default_score = score
        self.call_count = 0

    async def score(
        self,
        trajectory: EvalTrajectory,
        criteria: List[str] = None,
        cancellation_token=None,
    ) -> EvalScore:
        self.call_count += 1
        criteria = criteria or ["task_completion"]
        return EvalScore(
            overall=self.default_score,
            dimensions={c: self.default_score for c in criteria},
            reasoning={c: "Mock score" for c in criteria},
            trajectory=trajectory,
            metadata={"mock": True},
        )


def create_sample_task(
    task_id: str = "test_task",
    name: str = "Test Task",
    prompt: str = "Do something",
    category: str = "general",
) -> BenchmarkTask:
    """Create a sample benchmark task for testing."""
    return BenchmarkTask(
        id=task_id,
        name=name,
        prompt=prompt,
        category=category,
        eval_criteria=["task_completion", "efficiency"],
        rubric={
            "task_completion": "Task completed successfully",
            "efficiency": "Completed with minimal resources",
        },
    )


def create_sample_dataset(num_tasks: int = 3) -> BenchmarkDataset:
    """Create a sample dataset with multiple tasks."""
    tasks = [
        create_sample_task(
            task_id=f"task_{i}",
            name=f"Test Task {i}",
            prompt=f"Complete task {i}",
            category="test" if i % 2 == 0 else "other",
        )
        for i in range(num_tasks)
    ]
    return BenchmarkDataset(
        name="test_dataset",
        description="A test dataset",
        version="1.0",
        tasks=tasks,
        default_eval_criteria=["task_completion"],
    )


# =============================================================================
# AgentConfig Tests
# =============================================================================


class TestAgentConfig:
    """Tests for AgentConfig dataclass."""

    def test_default_values(self):
        """Test AgentConfig with minimal parameters."""
        config = AgentConfig(name="test")

        assert config.name == "test"
        assert config.model_provider == "openai"
        assert config.model_name == "gpt-4o-mini"
        assert config.compaction is None
        assert config.token_budget == 50_000
        assert config.max_iterations == 30  # Actual default

    def test_full_configuration(self):
        """Test AgentConfig with all parameters."""
        config = AgentConfig(
            name="full_test",
            model_provider="anthropic",
            model_name="claude-3-opus",
            compaction="head_tail",
            token_budget=100_000,
            system_prompt="You are a helpful assistant.",
            max_iterations=50,
            tools=["coding"],
            extra_kwargs={"version": "1.0"},
        )

        assert config.name == "full_test"
        assert config.model_provider == "anthropic"
        assert config.model_name == "claude-3-opus"
        assert config.compaction == "head_tail"
        assert config.token_budget == 100_000
        assert config.max_iterations == 50
        assert "coding" in config.tools
        assert config.extra_kwargs["version"] == "1.0"

    def test_serialization_roundtrip(self):
        """Test to_dict and from_dict work correctly."""
        original = AgentConfig(
            name="serial_test",
            model_provider="azure",
            compaction="sliding",
            token_budget=75_000,
            extra_kwargs={"test": True},
        )

        data = original.to_dict()
        restored = AgentConfig.from_dict(data)

        assert restored.name == original.name
        assert restored.model_provider == original.model_provider
        assert restored.compaction == original.compaction
        assert restored.token_budget == original.token_budget
        assert restored.extra_kwargs == original.extra_kwargs

    def test_from_string_simple(self):
        """Test parsing config from CLI string."""
        config = AgentConfig.from_string("simple_config")
        assert config.name == "simple_config"

    def test_from_string_with_params(self):
        """Test parsing config from CLI string with parameters."""
        config = AgentConfig.from_string("my_config:strategy=head_tail,budget=80000")
        assert config.name == "my_config"
        assert config.compaction == "head_tail"
        # Note: budget maps to token_budget


# =============================================================================
# BenchmarkTask Tests
# =============================================================================


class TestBenchmarkTask:
    """Tests for BenchmarkTask dataclass."""

    def test_minimal_task(self):
        """Test task with required fields only."""
        task = BenchmarkTask(
            id="min_task",
            name="Minimal",
            prompt="Do this",
        )

        assert task.id == "min_task"
        assert task.category == "general"
        assert task.eval_criteria == ["task_completion"]

    def test_to_eval_task(self):
        """Test conversion to EvalTask."""
        task = create_sample_task()
        eval_task = task.to_eval_task()

        assert isinstance(eval_task, EvalTask)
        # EvalTask uses task.id as name
        assert eval_task.name == task.id
        assert eval_task.input == task.prompt

    def test_serialization_roundtrip(self):
        """Test to_dict and from_dict work correctly."""
        original = create_sample_task()

        data = original.to_dict()
        restored = BenchmarkTask.from_dict(data)

        assert restored.id == original.id
        assert restored.name == original.name
        assert restored.prompt == original.prompt
        assert restored.eval_criteria == original.eval_criteria


# =============================================================================
# BenchmarkDataset Tests
# =============================================================================


class TestBenchmarkDataset:
    """Tests for BenchmarkDataset."""

    def test_dataset_creation(self):
        """Test creating a dataset."""
        dataset = create_sample_dataset(5)

        assert dataset.name == "test_dataset"
        assert len(list(dataset.tasks)) == 5

    def test_dataset_filtering(self):
        """Test filtering tasks by category."""
        dataset = create_sample_dataset(6)

        test_dataset = dataset.filter_by_category("test")
        other_dataset = dataset.filter_by_category("other")

        assert len(test_dataset.tasks) == 3  # 0, 2, 4
        assert len(other_dataset.tasks) == 3  # 1, 3, 5

    def test_dataset_get_task(self):
        """Test getting a specific task by ID."""
        dataset = create_sample_dataset(3)

        task = dataset.get_task("task_1")
        assert task is not None
        assert task.id == "task_1"

        missing = dataset.get_task("nonexistent")
        assert missing is None

    def test_dataset_json_save_load(self):
        """Test saving and loading dataset to/from JSON."""
        dataset = create_sample_dataset(3)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_dataset.json"

            dataset.to_json(filepath)
            assert filepath.exists()

            loaded = BenchmarkDataset.from_json(filepath)

            assert loaded.name == dataset.name
            assert len(list(loaded.tasks)) == 3


# =============================================================================
# Built-in Datasets Tests
# =============================================================================


class TestBuiltinDatasets:
    """Tests for built-in dataset loading."""

    def test_list_builtin_datasets(self):
        """Test listing available datasets."""
        datasets = list_builtin_datasets()

        assert isinstance(datasets, list)
        assert "context_engineering_v1" in datasets

    def test_load_builtin_dataset(self):
        """Test loading a built-in dataset."""
        dataset = load_builtin_dataset("context_engineering_v1")

        assert dataset is not None
        assert dataset.name == "context_engineering_v1"
        assert len(list(dataset.tasks)) > 0

    def test_load_nonexistent_dataset(self):
        """Test loading a non-existent dataset raises error."""
        with pytest.raises(ValueError, match="not found"):
            load_builtin_dataset("nonexistent_dataset")


# =============================================================================
# TaskResult Tests
# =============================================================================


class TestTaskResult:
    """Tests for TaskResult dataclass."""

    def test_task_result_creation(self):
        """Test creating a task result."""
        task = create_sample_task()
        trajectory = EvalTrajectory(
            task=task.to_eval_task(),
            messages=[
                UserMessage(content=task.prompt, source="user"),
                AssistantMessage(content="Done", source="assistant"),
            ],
            success=True,
            usage=Usage(duration_ms=1000, llm_calls=3, tokens_input=500, tokens_output=100),
        )

        score = EvalScore(
            overall=8.5,
            dimensions={"task_completion": 8.5},
            reasoning={"task_completion": "Good"},
            trajectory=trajectory,
        )

        result = TaskResult(
            task_id=task.id,
            target_name="test_target",
            trajectory=trajectory,
            score=score,
            total_tokens=600,
            input_tokens=500,
            output_tokens=100,
            iterations=3,
            duration_ms=1000,
            files_read={"file1.txt": 2, "file2.txt": 1},
            unique_files=2,
            duplicate_reads=1,
        )

        assert result.task_id == task.id
        assert result.score.overall == 8.5
        assert result.total_tokens == 600
        assert result.duplicate_reads == 1


# =============================================================================
# BenchmarkResults Tests
# =============================================================================


class TestBenchmarkResults:
    """Tests for BenchmarkResults."""

    def _create_mock_result(
        self,
        task_id: str,
        target_name: str,
        score: float = 8.0,
        tokens: int = 1000,
    ) -> TaskResult:
        """Create a mock TaskResult."""
        task = create_sample_task(task_id=task_id)
        trajectory = EvalTrajectory(
            task=task.to_eval_task(),
            messages=[
                UserMessage(content=task.prompt, source="user"),
                AssistantMessage(content="Done", source="assistant"),
            ],
            success=True,
            usage=Usage(duration_ms=100, llm_calls=1, tokens_input=tokens // 2, tokens_output=tokens // 2),
        )

        eval_score = EvalScore(
            overall=score,
            dimensions={"task_completion": score},
            reasoning={"task_completion": "Test"},
            trajectory=trajectory,
        )

        return TaskResult(
            task_id=task_id,
            target_name=target_name,
            trajectory=trajectory,
            score=eval_score,
            total_tokens=tokens,
            input_tokens=tokens // 2,
            output_tokens=tokens // 2,
            iterations=5,
            duration_ms=1000,
            files_read={},
            unique_files=0,
            duplicate_reads=0,
        )

    def test_add_and_get_results(self):
        """Test adding and retrieving results."""
        results = BenchmarkResults(dataset_name="test", dataset_version="1.0")

        result1 = self._create_mock_result("task_1", "target_a")
        result2 = self._create_mock_result("task_1", "target_b")
        result3 = self._create_mock_result("task_2", "target_a")

        results.add_result(result1)
        results.add_result(result2)
        results.add_result(result3)

        assert len(results.target_names) == 2
        assert len(results.task_ids) == 2

        retrieved = results.get_result("target_a", "task_1")
        assert retrieved is not None
        assert retrieved.target_name == "target_a"

    def test_get_summaries(self):
        """Test getting target summaries."""
        results = BenchmarkResults(dataset_name="test", dataset_version="1.0")

        # Add results for two targets
        for task_id in ["task_1", "task_2", "task_3"]:
            results.add_result(self._create_mock_result(task_id, "baseline", tokens=1000))
            results.add_result(self._create_mock_result(task_id, "optimized", tokens=800))

        summaries = results.get_summaries()

        assert "baseline" in summaries
        assert "optimized" in summaries

        baseline = summaries["baseline"]
        assert baseline.task_count == 3  # Correct attribute name
        assert baseline.total_tokens == 3000  # 3 * 1000

        optimized = summaries["optimized"]
        assert optimized.total_tokens == 2400  # 3 * 800

    def test_compare_targets(self):
        """Test target comparison."""
        results = BenchmarkResults(dataset_name="test", dataset_version="1.0")

        for task_id in ["task_1", "task_2"]:
            results.add_result(self._create_mock_result(task_id, "baseline", tokens=1000, score=7.0))
            results.add_result(self._create_mock_result(task_id, "optimized", tokens=600, score=8.0))

        comparison = results.compare_targets("baseline")

        assert comparison["baseline"]["is_baseline"] is True
        assert comparison["optimized"]["is_baseline"] is False

        # Optimized uses 40% less tokens
        assert comparison["optimized"]["token_diff_pct"] == pytest.approx(-40.0)
        # Optimized scores 1 point higher
        assert comparison["optimized"]["score_diff"] == pytest.approx(1.0)

    def test_save_and_load(self):
        """Test saving and loading results."""
        results = BenchmarkResults(dataset_name="test", dataset_version="1.0")
        results.add_result(self._create_mock_result("task_1", "target_a"))
        results.add_result(self._create_mock_result("task_2", "target_a"))

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "results.json"

            results.save(filepath)
            assert filepath.exists()

            # Load and verify
            from picoagents.eval.benchmarks import load_benchmark_results

            loaded = load_benchmark_results(filepath)

            assert loaded.dataset_name == "test"
            assert len(loaded.target_names) == 1
            assert len(loaded.task_ids) == 2


# =============================================================================
# BenchmarkMiddleware Tests
# =============================================================================


class TestBenchmarkMiddleware:
    """Tests for BenchmarkMiddleware."""

    def test_initial_state(self):
        """Test middleware initial state."""
        middleware = BenchmarkMiddleware()

        metrics = middleware.get_metrics()
        assert metrics["iterations"] == 0
        assert metrics["total_tokens"] == 0
        assert metrics["tool_calls"] == 0
        assert metrics["unique_files"] == 0

    def test_record_compaction(self):
        """Test recording compaction events."""
        middleware = BenchmarkMiddleware()

        middleware.record_compaction(
            tokens_before=10000,
            tokens_after=6000,
            messages_before=50,
            messages_after=30,
        )

        metrics = middleware.get_metrics()
        assert metrics["compaction_events"] == 1
        assert metrics["tokens_saved"] == 4000

    def test_reset(self):
        """Test resetting middleware state."""
        middleware = BenchmarkMiddleware()

        middleware.record_compaction(10000, 5000, 50, 25)

        middleware.reset()

        metrics = middleware.get_metrics()
        assert metrics["compaction_events"] == 0
        assert metrics["tokens_saved"] == 0


# =============================================================================
# BenchmarkTarget Tests
# =============================================================================


class TestCallableTarget:
    """Tests for CallableTarget."""

    @pytest.mark.asyncio
    async def test_callable_target(self):
        """Test running a callable target."""

        async def my_runner(task: BenchmarkTask) -> TargetResult:
            return TargetResult(
                output=f"Completed: {task.name}",
                success=True,
                input_tokens=250,
                output_tokens=250,
                iterations=3,
                duration_ms=100,
            )

        target = CallableTarget(name="my_target", func=my_runner)

        task = create_sample_task()
        result = await target.run(task)

        assert result.success is True
        assert "Completed" in result.output
        assert result.total_tokens == 500


# =============================================================================
# BenchmarkRunner Tests
# =============================================================================


class TestBenchmarkRunner:
    """Tests for BenchmarkRunner."""

    @pytest.mark.asyncio
    async def test_runner_basic(self):
        """Test running a basic benchmark."""
        judge = MockJudge(score=8.0)

        # Create a simple callable target
        async def simple_runner(task: BenchmarkTask) -> TargetResult:
            return TargetResult(
                output="Done",
                success=True,
                input_tokens=250,
                output_tokens=250,
                iterations=3,
                duration_ms=100,
            )

        targets = [
            CallableTarget(name="simple", func=simple_runner),
        ]

        dataset = create_sample_dataset(2)
        runner = BenchmarkRunner(judge=judge)

        results = await runner.run(dataset, targets)

        assert len(results.target_names) == 1
        assert len(results.task_ids) == 2

        # Judge should have been called for each task
        assert judge.call_count == 2

    @pytest.mark.asyncio
    async def test_runner_with_task_filter(self):
        """Test running with task filter."""
        judge = MockJudge()

        async def simple_runner(task: BenchmarkTask) -> TargetResult:
            return TargetResult(output="Done", success=True, input_tokens=50, output_tokens=50, iterations=1, duration_ms=50)

        targets = [CallableTarget(name="test", func=simple_runner)]
        dataset = create_sample_dataset(6)  # 3 "test" category, 3 "other"

        runner = BenchmarkRunner(judge=judge)

        # Filter to only "test" category
        results = await runner.run(
            dataset,
            targets,
            task_filter=lambda t: t.category == "test",
        )

        assert len(results.task_ids) == 3


# =============================================================================
# Analysis Formatting Tests
# =============================================================================


class TestAnalysisFormatting:
    """Tests for analysis formatting functions."""

    def _create_results_with_data(self) -> BenchmarkResults:
        """Create results with test data."""
        results = BenchmarkResults(dataset_name="test", dataset_version="1.0")

        # Create sample trajectory and score
        task = create_sample_task()
        trajectory = EvalTrajectory(
            task=task.to_eval_task(),
            messages=[
                UserMessage(content=task.prompt, source="user"),
                AssistantMessage(content="Done", source="assistant"),
            ],
            success=True,
            usage=Usage(duration_ms=1000, llm_calls=5, tokens_input=500, tokens_output=100),
        )

        score = EvalScore(
            overall=8.0,
            dimensions={"task_completion": 8.0},
            reasoning={"task_completion": "Good"},
            trajectory=trajectory,
        )

        # Add results for two targets
        for target_name, tokens in [("baseline", 1000), ("optimized", 700)]:
            result = TaskResult(
                task_id=task.id,
                target_name=target_name,
                trajectory=trajectory,
                score=score,
                total_tokens=tokens,
                input_tokens=tokens // 2,
                output_tokens=tokens // 2,
                iterations=5,
                duration_ms=1000,
                files_read={"file1.txt": 2, "file2.txt": 1},
                unique_files=2,
                duplicate_reads=1,
            )
            results.add_result(result)

        return results

    def test_format_summary_table(self):
        """Test summary table formatting."""
        results = self._create_results_with_data()

        table = format_summary_table(results)

        assert "test" in table  # Dataset name
        assert "baseline" in table
        assert "optimized" in table
        assert "Score" in table
        assert "Tokens" in table

    def test_format_task_breakdown(self):
        """Test task breakdown formatting."""
        results = self._create_results_with_data()

        breakdown = format_task_breakdown(results)

        assert "Per-Task Breakdown" in breakdown
        assert "test_task" in breakdown

    def test_format_file_read_analysis(self):
        """Test file read analysis formatting."""
        results = self._create_results_with_data()

        analysis = format_file_read_analysis(results)

        assert "File Read Analysis" in analysis
        assert "file1.txt" in analysis or "file2.txt" in analysis


# =============================================================================
# Integration Tests
# =============================================================================


class TestBenchmarkIntegration:
    """Integration tests for the benchmark system."""

    @pytest.mark.asyncio
    async def test_full_benchmark_flow(self):
        """Test complete benchmark workflow."""
        # 1. Create configurations
        configs = [
            AgentConfig(name="baseline", compaction=None),
            AgentConfig(name="head_tail", compaction="head_tail"),
        ]

        # 2. Create dataset
        dataset = create_sample_dataset(3)

        # 3. Create mock targets (simulate different behavior)
        async def baseline_runner(task: BenchmarkTask) -> TargetResult:
            return TargetResult(
                output=f"Baseline: {task.name}",
                success=True,
                input_tokens=500,
                output_tokens=500,
                iterations=10,
                duration_ms=500,
            )

        async def optimized_runner(task: BenchmarkTask) -> TargetResult:
            return TargetResult(
                output=f"Optimized: {task.name}",
                success=True,
                input_tokens=300,
                output_tokens=300,
                iterations=6,
                duration_ms=300,
            )

        targets = [
            CallableTarget(name="baseline", func=baseline_runner),
            CallableTarget(name="head_tail", func=optimized_runner),
        ]

        # 4. Run benchmark
        judge = MockJudge(score=8.0)
        runner = BenchmarkRunner(judge=judge)

        results = await runner.run(dataset, targets)

        # 5. Verify results
        assert len(results.target_names) == 2
        assert len(results.task_ids) == 3

        summaries = results.get_summaries()
        assert summaries["baseline"].total_tokens == 3000  # 3 * 1000
        assert summaries["head_tail"].total_tokens == 1800  # 3 * 600

        # 6. Compare targets
        comparison = results.compare_targets("baseline")
        assert comparison["head_tail"]["token_diff_pct"] == pytest.approx(-40.0)

        # 7. Test saving and loading
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "results.json"
            results.save(filepath)

            from picoagents.eval.benchmarks import load_benchmark_results

            loaded = load_benchmark_results(filepath)
            assert loaded.dataset_name == results.dataset_name
            assert len(loaded.target_names) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
