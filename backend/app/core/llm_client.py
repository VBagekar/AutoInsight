"""LLM integration layer — Nemotron chart planning, preprocessing, and validation.

The LLM is intentionally constrained to choose chart types and fields from the
real dataset schema.  It NEVER produces data values.  Every number shown in the
UI is computed locally by DashboardBuilder from the user's cleaned dataframe.
"""
import json
import logging
import re
from difflib import get_close_matches
from typing import Generator, Dict, Any, List, Optional

from openai import OpenAI
from app.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Allowed chart types — the finite, supported catalog.
# ---------------------------------------------------------------------------
ALLOWED_CHART_TYPES = {
    "Area Graph", "Line Graph", "Bar Chart",
    "Donut Chart", "Scatterplot", "Histogram",
}

# Loose aliases the LLM might return — mapped to canonical names.
_CHART_TYPE_ALIASES: Dict[str, str] = {
    "area": "Area Graph", "area chart": "Area Graph", "area graph": "Area Graph",
    "line": "Line Graph", "line chart": "Line Graph", "line graph": "Line Graph",
    "bar": "Bar Chart", "bar chart": "Bar Chart", "bar graph": "Bar Chart",
    "column chart": "Bar Chart", "column": "Bar Chart",
    "donut": "Donut Chart", "donut chart": "Donut Chart",
    "pie": "Donut Chart", "pie chart": "Donut Chart",
    "scatter": "Scatterplot", "scatterplot": "Scatterplot", "scatter plot": "Scatterplot",
    "histogram": "Histogram", "hist": "Histogram",
}


class NemotronLLMClient:
    def __init__(self) -> None:
        self.api_key = settings.NVIDIA_API_KEY
        self.base_url = settings.NVIDIA_BASE_URL
        self.model = settings.NVIDIA_MODEL

        if self.api_key:
            self.client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key
            )
        else:
            self.client = None

    # ------------------------------------------------------------------
    #  MASTER PROMPT BUILDER
    # ------------------------------------------------------------------
    def _build_master_prompt(
        self,
        summary: Dict[str, Any],
        query: str,
        conversation_context: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Build a comprehensive, dataset-aware prompt for accurate chart planning.

        Sections:
        A. Rich dataset context (columns, types, roles, stats, sample values)
        B. Chart-selection decision rules
        C. Query-type handling matrix
        D. Strict output constraints
        E. Edge-case handling
        F. Few-shot examples
        G. Conversation context (if follow-up)
        """
        numeric = summary.get("numeric_columns", [])
        dates = summary.get("date_columns", [])
        categories = summary.get("categorical_columns", [])
        primary_kpi = summary.get("primary_kpi")
        col_meta = summary.get("col_meta", {})
        row_count = summary.get("row_count", "unknown")
        quality = summary.get("quality_score", "unknown")

        # --- A. Rich dataset context ---
        col_details: List[str] = []
        for col_info in col_meta.values() if isinstance(col_meta, dict) else []:
            name = col_info.get("name", "?")
            dtype = col_info.get("dtype", "?")
            role = col_info.get("semantic_role", "unknown")
            missing_pct = col_info.get("missing_pct", 0)
            detail = f"  - {name} (type={dtype}, role={role}, missing={missing_pct}%"
            if dtype in ("int64", "float64") or "int" in str(dtype) or "float" in str(dtype):
                detail += f", min={col_info.get('min', '?')}, max={col_info.get('max', '?')}, mean={col_info.get('mean', '?')}"
            if col_info.get("sample_values"):
                samples = col_info["sample_values"][:5]
                detail += f", samples={samples}"
            if col_info.get("unique_count") is not None:
                detail += f", unique={col_info['unique_count']}"
            detail += ")"
            col_details.append(detail)

        if not col_details:
            # Fallback if col_meta is not populated — use column lists
            for c in numeric:
                col_details.append(f"  - {c} (numeric)")
            for c in dates:
                col_details.append(f"  - {c} (date)")
            for c in categories:
                col_details.append(f"  - {c} (categorical)")

        schema_block = "\n".join(col_details) if col_details else "  (no column metadata available)"

        # --- G. Conversation context ---
        conv_block = ""
        if conversation_context:
            recent = conversation_context[-5:]  # last 5 turns
            turns = "\n".join([f"  User: {t['query']}" for t in recent])
            conv_block = (
                "\n\nCONVERSATION HISTORY (most recent queries — the current query may be a follow-up):\n"
                f"{turns}\n"
                "If the current query references 'that', 'it', 'those', or is a short refinement "
                "(e.g. 'by region', 'in Q3', 'what about profit'), treat it as a follow-up to the "
                "last query and reuse/refine the previous chart selections.\n"
            )

        # --- Build complete prompt ---
        prompt = f"""You are an expert AI Data Scientist and Visualization Architect for the AutoInsights platform.
Your ONLY job is to select the best chart types and map them to real columns from the user's dataset.
You must NEVER invent data values, percentages, statistics, or numbers.

=== DATASET CONTEXT ===
Rows: {row_count}
Data quality score: {quality}%
Primary KPI: {primary_kpi or 'not detected'}

Numeric columns: {json.dumps(numeric)}
Date columns: {json.dumps(dates)}
Categorical columns: {json.dumps(categories)}

Detailed column metadata:
{schema_block}
{conv_block}
=== CHART SELECTION RULES (follow strictly) ===
1. Time trend / "over time" / "monthly" / "yearly" + date column exists → Line Graph or Area Graph (x=date column, y=metric)
2. Category breakdown / "by region" / "by product" → Bar Chart (x=categorical column, y=metric)
3. Composition / "share" / "proportion" / "percentage" → Donut Chart (x=categorical column, y=metric)
4. Correlation / "relationship" / "scatter" + 2 numeric columns → Scatterplot (x=numeric1, y=numeric2)
5. Distribution / "spread" / "histogram" + numeric column → Histogram (y=metric)
6. Comparison of 2+ metrics over time → multiple Line Graphs
7. "Top N" / "best" / "worst" / "highest" / "lowest" → Bar Chart sorted by value
8. Geographic question + geo column exists → Bar Chart by geo dimension
9. If user explicitly names a chart type → use that exact type
10. If query is ambiguous → default to Bar Chart + one supporting chart

=== QUERY TYPE DETECTION ===
- "trend" / "over time" / "monthly" / "yearly" / "growth" / "decline" → needs date on x-axis
- "by" / "per" / "across" / "breakdown" / "split" → needs categorical on x-axis
- "vs" / "compare" / "versus" / "difference" → comparison layout (multiple charts or overlays)
- "share" / "proportion" / "percent" / "composition" → Donut Chart
- "distribution" / "spread" / "histogram" / "range" → Histogram
- "correlation" / "relationship" / "scatter" / "between X and Y" → Scatterplot
- "top" / "bottom" / "best" / "worst" / "highest" / "lowest" → sorted Bar Chart
- "forecast" / "predict" / "future" / "next" → Line Graph with trend extension
- "everything" / "overview" / "analyze" / "dashboard" → provide 4-5 diverse charts covering trend + breakdown + composition + correlation

=== STRICT OUTPUT CONSTRAINTS ===
1. ONLY use column names that appear in the schema above — NEVER invent column names.
2. x_axis MUST be a real column name from the schema, or an empty string if not applicable.
3. y_axis MUST be a real NUMERIC column name from the schema.
4. type MUST be exactly one of: Area Graph, Line Graph, Bar Chart, Donut Chart, Scatterplot, Histogram
5. NEVER include computed data values, percentages, statistics, or numbers in insight_tooltip.
6. insight_tooltip must describe WHAT the chart will show (e.g. "Shows how Sales vary across Region"), NOT computed results.
7. title should be descriptive and specific to the dataset columns used.
8. Return 1 to 5 charts. The FIRST chart should directly answer the user's question. Remaining charts provide supporting context.

=== EDGE CASES ===
- No date columns exist → do NOT suggest Line Graph or Area Graph
- No categorical columns exist → use Histogram or Scatterplot only
- Only 1 numeric column → suggest Histogram of it, or if categories exist, Bar Chart
- User asks about a column not in the schema → pick the closest matching column name
- Vague query ("analyze everything", "show me insights") → return overview: trend + breakdown + composition + scatter (4 charts)
- Greeting or non-analytical query ("hi", "hello", "help") → return empty charts array

=== EXAMPLES ===

Example 1:
Schema: numeric=[Sales, Profit, Orders], dates=[Date], categories=[Region, Category]
Query: "show sales trend over time"
Answer: {{"charts":[{{"title":"Sales Trend Over Time","type":"Area Graph","x_axis":"Date","y_axis":"Sales","insight_tooltip":"Displays how Sales values change over time periods"}}]}}

Example 2:
Schema: numeric=[Revenue, Cost], dates=[], categories=[Department, City]
Query: "compare revenue by department"
Answer: {{"charts":[{{"title":"Revenue by Department","type":"Bar Chart","x_axis":"Department","y_axis":"Revenue","insight_tooltip":"Compares Revenue across different Department categories"}},{{"title":"Department Revenue Share","type":"Donut Chart","x_axis":"Department","y_axis":"Revenue","insight_tooltip":"Shows each Department proportional contribution to total Revenue"}}]}}

Example 3:
Schema: numeric=[Price, Quantity, Rating], dates=[Order_Date], categories=[Brand]
Query: "is there a relationship between price and rating?"
Answer: {{"charts":[{{"title":"Price vs Rating Correlation","type":"Scatterplot","x_axis":"Price","y_axis":"Rating","insight_tooltip":"Explores the relationship between Price and Rating across data points"}}]}}

Example 4:
Schema: numeric=[Salary], dates=[], categories=[Gender, Department]
Query: "analyze everything"
Answer: {{"charts":[{{"title":"Salary by Department","type":"Bar Chart","x_axis":"Department","y_axis":"Salary","insight_tooltip":"Compares Salary across departments"}},{{"title":"Department Salary Share","type":"Donut Chart","x_axis":"Department","y_axis":"Salary","insight_tooltip":"Proportional Salary contribution by Department"}},{{"title":"Salary by Gender","type":"Bar Chart","x_axis":"Gender","y_axis":"Salary","insight_tooltip":"Compares Salary distribution between Gender groups"}},{{"title":"Salary Distribution","type":"Histogram","x_axis":"","y_axis":"Salary","insight_tooltip":"Shows the frequency distribution of Salary values"}}]}}

=== YOUR TASK ===
User's question: "{query}"

Return ONLY a valid JSON object in this exact format (no markdown, no explanation, no extra text):
{{"charts":[{{"title":"...","type":"...","x_axis":"...","y_axis":"...","insight_tooltip":"..."}}]}}
"""
        return prompt

    # ------------------------------------------------------------------
    #  CHART SPEC VALIDATION & REPAIR
    # ------------------------------------------------------------------
    def _normalize_chart_type(self, raw_type: str) -> Optional[str]:
        """Map a raw chart type string to a canonical allowed type."""
        if raw_type in ALLOWED_CHART_TYPES:
            return raw_type
        lowered = raw_type.strip().lower()
        if lowered in _CHART_TYPE_ALIASES:
            return _CHART_TYPE_ALIASES[lowered]
        # Fuzzy substring match
        for alias, canonical in _CHART_TYPE_ALIASES.items():
            if alias in lowered or lowered in alias:
                return canonical
        return None

    def _fuzzy_match_column(self, requested: str, available: List[str]) -> Optional[str]:
        """Try to match a possibly misspelled column name to a real one."""
        if not requested or not available:
            return None
        # Exact match (case-insensitive)
        lower_map = {c.lower(): c for c in available}
        if requested.lower() in lower_map:
            return lower_map[requested.lower()]
        # Close match
        matches = get_close_matches(requested.lower(), [c.lower() for c in available], n=1, cutoff=0.6)
        if matches:
            return lower_map[matches[0]]
        return None

    def _validate_and_repair_spec(self, spec: Dict[str, Any], summary: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Validate a single chart spec. Attempt repair; return None if unfixable."""
        if not isinstance(spec, dict):
            return None

        all_columns = (
            summary.get("numeric_columns", [])
            + summary.get("date_columns", [])
            + summary.get("categorical_columns", [])
        )
        numeric_columns = summary.get("numeric_columns", [])

        # --- Validate and repair chart type ---
        raw_type = spec.get("type", "")
        canonical_type = self._normalize_chart_type(str(raw_type))
        if not canonical_type:
            logger.warning("LLM returned unknown chart type '%s' — dropping spec", raw_type)
            return None
        spec["type"] = canonical_type

        # --- Validate and repair x_axis ---
        x_axis = spec.get("x_axis", "")
        if x_axis and isinstance(x_axis, str) and x_axis.strip():
            matched = self._fuzzy_match_column(x_axis, all_columns)
            if matched:
                spec["x_axis"] = matched
            else:
                # Can't find this column — clear it, DashboardBuilder will pick default
                logger.warning("LLM x_axis '%s' not in dataset — clearing", x_axis)
                spec["x_axis"] = ""

        # --- Validate and repair y_axis ---
        y_axis = spec.get("y_axis", "")
        if y_axis and isinstance(y_axis, str) and y_axis.strip():
            matched = self._fuzzy_match_column(y_axis, numeric_columns)
            if matched:
                spec["y_axis"] = matched
            else:
                # Try all columns as fallback
                matched_any = self._fuzzy_match_column(y_axis, all_columns)
                if matched_any:
                    spec["y_axis"] = matched_any
                else:
                    # Use primary KPI as fallback
                    primary = summary.get("primary_kpi")
                    if primary:
                        spec["y_axis"] = primary
                    elif numeric_columns:
                        spec["y_axis"] = numeric_columns[0]
                    else:
                        logger.warning("LLM y_axis '%s' not in dataset and no fallback — dropping", y_axis)
                        return None

        # --- Ensure title and tooltip are strings ---
        if not spec.get("title") or not isinstance(spec.get("title"), str):
            x = spec.get("x_axis", "")
            y = spec.get("y_axis", "")
            spec["title"] = f"{y} by {x}" if x else f"{y} overview"

        if not spec.get("insight_tooltip") or not isinstance(spec.get("insight_tooltip"), str):
            spec["insight_tooltip"] = f"Visualizes {spec.get('y_axis', 'data')} patterns."

        return spec

    # ------------------------------------------------------------------
    #  CHART PLAN GENERATION (with master prompt + validation)
    # ------------------------------------------------------------------
    def generate_chart_plan(
        self,
        dataset_summary: Dict[str, Any],
        user_query: str,
        conversation_context: Optional[List[Dict[str, str]]] = None,
    ) -> List[Dict[str, Any]]:
        """Ask Nemotron for semantic chart choices using the master prompt.

        The returned fields are validated, repaired where possible, and then
        passed to DashboardBuilder which performs every aggregation locally.
        This means a model response cannot fabricate a number shown to a user.
        """
        if not self.client or not self.api_key:
            return []

        prompt = self._build_master_prompt(dataset_summary, user_query, conversation_context)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                top_p=0.95,
                max_tokens=2048,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": True},
                    "reasoning_budget": 2048,
                },
            )
            content = response.choices[0].message.content or ""
            content = content.strip()

            # Strip markdown code fences if present
            if "```" in content:
                content = content.replace("```json", "").replace("```", "").strip()

            # Try to extract JSON object from possibly noisy output
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                content = json_match.group(0)

            parsed = json.loads(content)
            charts = parsed.get("charts", []) if isinstance(parsed, dict) else []
            if not isinstance(charts, list):
                return []

            # --- Validate and repair every spec ---
            validated: List[Dict[str, Any]] = []
            for spec in charts:
                repaired = self._validate_and_repair_spec(spec, dataset_summary)
                if repaired:
                    validated.append(repaired)

            logger.info(
                "LLM chart plan: %d specs received, %d passed validation",
                len(charts), len(validated),
            )
            return validated

        except json.JSONDecodeError as e:
            logger.warning("LLM returned invalid JSON: %s", e)
            return []
        except Exception as e:
            logger.warning("LLM chart plan request failed: %s", e)
            return []

    # ------------------------------------------------------------------
    #  PREPROCESSING ACTION (Person 1's copilot — unchanged)
    # ------------------------------------------------------------------
    def generate_preprocessing_action(self, user_command: str, columns: List[str], column_types: Dict[str, str]) -> Dict[str, Any]:
        """Ask Nemotron to translate a natural-language command into a single preprocessing action.

        The model only selects from the allowed action types and must reference only real column names.
        """
        if not self.client or not self.api_key:
            raise ValueError("LLM client not configured; cannot generate preprocessing action.")

        # Build a concise schema description for the prompt
        cols_desc = ", ".join([f"{c} ({column_types.get(c, 'unknown')})" for c in columns])
        action_schema = (
            "Allowed actions (choose exactly one):\n"
            "1. drop_column: params {\"column\": \"<existing column>\"}\n"
            "2. rename_column: params {\"column\": \"<existing column>\", \"new_name\": \"<new name>\"}\n"
            "3. fill_missing: params {\"column\": \"<existing column>\", \"strategy\": \"mean|median|mode|constant|zero|forward_fill\", \"value\": <optional, required only for constant>}\n"
            "4. change_type: params {\"column\": \"<existing column>\", \"target_type\": \"int|float|string|datetime\"}\n"
            "5. add_column: params {\"new_column\": \"<new column name>\", \"expression\": \"<safe arithmetic expression using existing numeric columns, e.g., revenue - cost or 2 * revenue>\"}\n"
            "6. filter_rows: params {\"column\": \"<existing column>\", \"operator\": \"==|!=|>|<|>=|<=\", \"value\": <value>}\n"
        )

        prompt = (
            "You are a data preprocessing planner. Convert the user's command into ONE allowed action.\n"
            "Use ONLY column names from the provided schema. Return JSON only with fields: type, params, explanation.\n\n"
            f"Columns: {cols_desc}\n\n"
            f"{action_schema}\n"
            f"User command: {user_command}"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                top_p=0.9,
                max_tokens=800,
                extra_body={"chat_template_kwargs": {"enable_thinking": True}, "reasoning_budget": 1024},
            )
            content = response.choices[0].message.content or ""
            content = content.strip()
            if "```" in content:
                content = content.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(content)

            # Basic validation
            if not isinstance(parsed, dict):
                raise ValueError("LLM did not return a JSON object.")
            action_type = parsed.get("type")
            allowed_types = {"drop_column", "rename_column", "fill_missing", "change_type", "add_column", "filter_rows"}
            if action_type not in allowed_types:
                raise ValueError(f"LLM returned invalid action type: {action_type}")
            if "params" not in parsed or not isinstance(parsed["params"], dict):
                raise ValueError("LLM response missing params object.")
            if "explanation" not in parsed or not isinstance(parsed["explanation"], str):
                raise ValueError("LLM response missing explanation string.")
            return parsed
        except Exception as e:
            # Re-raise as clear error for caller
            raise ValueError(f"Failed to parse preprocessing action from LLM: {e}")


nemotron_client = NemotronLLMClient()
