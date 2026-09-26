"""DbtSemanticLayerCompiler: Ingests dbt manifest.json and maps metric definitions to SemanticWorldModel."""
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DbtMetricDefinition:
    """Formal business metric definition extracted from dbt manifest.json."""
    name: str
    label: str
    base_model: str
    expression: str
    calculation_method: str
    filters: List[Dict[str, Any]] = field(default_factory=list)
    canonical_dimensions: List[str] = field(default_factory=list)


class DbtSemanticLayerCompiler:
    """Parses dbt manifest.json and translates metric logic into semantic constraints."""

    @staticmethod
    def parse_manifest(manifest_path: str) -> Dict[str, DbtMetricDefinition]:
        """Ingests dbt manifest.json and extracts semantic metric definitions."""
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        metrics_map: Dict[str, DbtMetricDefinition] = {}

        # dbt manifest places metrics under the 'metrics' root key
        for node_id, metric_data in manifest.get("metrics", {}).items():
            name = metric_data.get("name")
            if not name:
                continue

            # Extract base model reference (e.g. ref('fct_subscriptions'))
            model_ref = metric_data.get("model", "")
            base_model = model_ref.replace("ref('", "").replace("')", "") if "ref(" in model_ref else model_ref

            # Parse dbt filters
            filters = []
            for f_item in metric_data.get("filters", []):
                filters.append({
                    "field": f_item.get("field"),
                    "operator": f_item.get("operator", "="),
                    "value": f_item.get("value"),
                })

            metrics_map[name] = DbtMetricDefinition(
                name=name,
                label=metric_data.get("label", name),
                base_model=base_model,
                expression=metric_data.get("expression", "*"),
                calculation_method=metric_data.get("calculation_method", "count"),
                filters=filters,
                canonical_dimensions=metric_data.get("dimensions", []),
            )

        return metrics_map

    @staticmethod
    def inject_dbt_constraints(semantic_resolution: Any, dbt_metrics: Dict[str, DbtMetricDefinition]):
        """Overrides heuristic semantic guesses with authoritative dbt definitions."""
        target = getattr(semantic_resolution, "target_metric", None)
        if target and target in dbt_metrics:
            dbt_def = dbt_metrics[target]
            semantic_resolution.mandatory_filters = dbt_def.filters
            semantic_resolution.authoritative_source = f"dbt:{dbt_def.base_model}"
