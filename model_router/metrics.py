"""Metrics for routing decisions."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RoutingMetrics:
    """Metrics for routing decisions."""

    total_requests: int = 0
    requests_by_task_type: dict[str, int] = field(default_factory=dict)
    requests_by_model: dict[str, int] = field(default_factory=dict)
    fallback_count: int = 0
    errors_by_model: dict[str, int] = field(default_factory=dict)
    avg_classification_time_ms: float = 0.0
    classification_times: list[float] = field(default_factory=list)

    def record_request(
        self,
        task_type: str,
        model_id: str,
        classification_time_ms: float,
        used_fallback: bool = False,
    ) -> None:
        self.total_requests += 1
        self.requests_by_task_type[task_type] = self.requests_by_task_type.get(task_type, 0) + 1
        self.requests_by_model[model_id] = self.requests_by_model.get(model_id, 0) + 1
        if used_fallback:
            self.fallback_count += 1
        self.classification_times.append(classification_time_ms)
        self.avg_classification_time_ms = (
            sum(self.classification_times) / len(self.classification_times)
            if self.classification_times
            else 0.0
        )

    def record_error(self, model_id: str) -> None:
        self.errors_by_model[model_id] = self.errors_by_model.get(model_id, 0) + 1

    def to_dict(self) -> dict:
        return {
            "total_requests": self.total_requests,
            "requests_by_task_type": self.requests_by_task_type,
            "requests_by_model": self.requests_by_model,
            "fallback_count": self.fallback_count,
            "fallback_rate": self.fallback_count / max(self.total_requests, 1),
            "errors_by_model": self.errors_by_model,
            "avg_classification_time_ms": round(self.avg_classification_time_ms, 2),
        }


metrics = RoutingMetrics()
