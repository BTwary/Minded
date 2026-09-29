"""Production Autonomous AI Data Analyst Orchestrator powered by InvestigationController."""
from typing import Any, Dict, List, Optional
import pandas as pd

from apps.api.src.ai.providers.base import BaseAIProvider
from apps.api.src.ai.tools.registry import ToolRegistry
from apps.api.src.ai.runtime import InvestigationRuntime
from packages.analytics_core.src.runtime.controller import InvestigationController
from packages.schemas.src.analysis import AnalysisResponse
from packages.shared.src.constants import MAX_ORCHESTRATION_STEPS


class AutonomousOrchestrator:
    """Autonomous AI Data Analyst Orchestrator routing to InvestigationController and runtime adapters."""

    def __init__(
        self,
        ai_provider: BaseAIProvider,
        tool_registry: Optional[ToolRegistry] = None,
        controller: Optional[InvestigationController] = None,
    ):
        self.provider = ai_provider
        self.tool_registry = tool_registry
        self.controller = controller or InvestigationController(ai_provider=self.provider)
        self.runtime = InvestigationRuntime(ai_provider=self.provider, controller=self.controller)

    def run_investigation(
        self,
        question: str,
        project_id: str,
        datasets: Dict[str, pd.DataFrame],
        business_metrics: Optional[List[Dict[str, Any]]] = None,
        max_steps: int = MAX_ORCHESTRATION_STEPS,
    ) -> AnalysisResponse:
        """Delegate investigation execution through the canonical runtime pipeline."""
        return self.runtime.execute_investigation(
            question=question,
            project_id=project_id,
            datasets=datasets,
            business_metrics=business_metrics,
            max_steps=max_steps,
        )
