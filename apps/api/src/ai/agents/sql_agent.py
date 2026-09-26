"""SQL Generation Agent."""
import json
import re
from typing import Any, Dict, Optional
from apps.api.src.ai.providers.base import BaseAIProvider
from packages.prompts.sql.v1 import SQL_AGENT_SYSTEM_PROMPT_V1


class SQLAgent:
    """Agent that translates analytical intent into safe, deterministic DuckDB SQL queries."""

    def __init__(self, provider: BaseAIProvider):
        self.provider = provider

    def generate_sql(
        self,
        task_description: str,
        table_schemas: Dict[str, Any],
        business_metrics: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate safe read-only SQL query."""
        if not getattr(self.provider, "is_ai_enabled", False):
            return self._create_fallback_sql(task_description, table_schemas)

        prompt = (
            f"Analytical Task: {task_description}\n\n"
            f"Available Tables and Schemas:\n{json.dumps(table_schemas, indent=2)}\n\n"
            f"Generate a single, read-only DuckDB SQL query to compute the required metrics."
        )

        try:
            resp = self.provider.generate(prompt=prompt, system_prompt=SQL_AGENT_SYSTEM_PROMPT_V1)
            # Clean SQL from markdown
            sql = resp.strip()
            match = re.search(r"```(?:sql)?\s*([\s\S]*?)\s*```", sql, re.IGNORECASE)
            if match:
                sql = match.group(1).strip()
            return sql.rstrip(";")
        except Exception:
            return self._create_fallback_sql(task_description, table_schemas)

    def _create_fallback_sql(self, task: str, schemas: Dict[str, Any]) -> str:
        """Generate deterministic fallback query based on schema."""
        table_name = list(schemas.keys())[0] if schemas else "sales"
        cols = schemas.get(table_name, {}).get("columns", [])
        col_names = [c["name"] for c in cols] if cols else []

        lower_t = task.lower()

        if "month" in lower_t or "date" in lower_t:
            date_col = next((c for c in col_names if "date" in c.lower()), "order_date")
            metric_col = next((c for c in col_names if any(m in c.lower() for m in ["revenue", "sales", "amount"])), "revenue")
            return f"SELECT strftime('%Y-%m', CAST({date_col} AS DATE)) AS month, SUM({metric_col}) AS total_revenue, COUNT(*) AS transaction_count FROM {table_name} GROUP BY 1 ORDER BY 1;"
        
        if "region" in lower_t:
            reg_col = next((c for c in col_names if "region" in c.lower()), "region")
            metric_col = next((c for c in col_names if any(m in c.lower() for m in ["revenue", "sales", "amount"])), "revenue")
            return f"SELECT {reg_col}, SUM({metric_col}) AS total_revenue, COUNT(*) AS transaction_count FROM {table_name} GROUP BY 1 ORDER BY 2 DESC;"

        return f"SELECT * FROM {table_name} LIMIT 100;"
