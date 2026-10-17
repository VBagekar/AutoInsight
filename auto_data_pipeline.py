import os
import re
import sys
import json
import logging
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# =============================================================================
# UNIVERSAL ENTERPRISE MISSING-VALUE DETECTOR & NORMALIZER
# =============================================================================
class UniversalNullNormalizer:
    """
    Enterprise-grade missing-value detector, mojibake cleaner, and normaliser.
    """

    CANONICAL_MISSING_TOKENS: Set[str] = {
        "",
        "null", "none", "nil", "nan", "nat", "void",
        "na", "n/a", "n/a.", "n.a", "n.a.", "n-a", "n_a", "n a",
        "unknown", "unk", "unknwon", "unkn",
        "missing", "miss", "<missing>", "(missing)", "[missing]",
        "undefined", "undef",
        "unavailable", "not available", "not_available",
        "unspecified", "not specified", "not_specified",
        "unrecorded", "not recorded", "not_recorded",
        "unreported", "not reported", "not_reported",
        "unassigned", "not assigned", "not_assigned",
        "not applicable", "not_applicable", "n.ap", "n.ap.",
        "not known", "not_known",
        "not provided", "not_provided",
        "not entered", "not_entered",
        "not collected", "not_collected",
        "not defined", "not_defined",
        "not stated", "not_stated",
        "not given", "not_given",
        "not found", "not_found",
        "not listed", "not_listed",
        "not determined", "not_determined",
        "not set", "not_set",
        "no data", "no_data", "nodata",
        "no value", "no_value",
        "no info", "no_info",
        "no information", "no_information",
        "none provided", "none_provided",
        "none available", "none_available",
        "tbd", "tba", "pending", "blank", "empty", "error",
        "inf", "-inf", "+inf", "infinity", "-infinity",
        "<na>", "<null>", "<none>", "<nil>", "<unk>", "<unknown>",
        "[na]", "[null]", "[none]", "[nil]",
        "(na)", "(null)", "(none)", "(nil)",
    }

    MISSING_REGEX_PATTERNS: Tuple[str, ...] = (
        r"^\s*$",
        r"^[-_?*/\\~.|+=:;,#!@$%^&`'\"]{1,10}$",
        r"^n\s*[/.\-_\s]?\s*a\.?$",
        r"^not[\s_\-]+(available|applicable|provided|specified|recorded|reported|assigned|known|entered|collected|defined|stated|given|found|listed|determined|set|valid|present)$",
        r"^no[\s_\-]+(data|value|info|information|record|entry|response|answer|result|input)$",
        r"^(null|none|nan|nil|missing|unknown|unk|undefined|unavailable|unspecified|unrecorded|unreported|unassigned|error)[\s_.\-0-9]*$",
        r"^#(n/a|value!|ref!|div/0!|name\?|null!|num!|error!)$",
        r"^[+-]?(inf|infinity|nan)$",
    )

    _COMPILED_REGEX = re.compile(
        "|".join(f"(?:{p})" for p in MISSING_REGEX_PATTERNS), re.IGNORECASE
    )
    _WRAPPER_STRIP_REGEX = re.compile(r"^[\s<>\[\](){}'_~\-]+|[\s<>\[\](){}'_~\-]+$")
    _WHITESPACE_NORMALIZE_REGEX = re.compile(r"[\s_]+")
    _MOJIBAKE_REGEX = re.compile(r"[\u00c2\u00a0\xa0\u200b\ufeff]+")

    @classmethod
    def clean_text_scalar(cls, val: Any) -> str:
        s = cls._MOJIBAKE_REGEX.sub(" ", str(val))
        return re.sub(r"\s+", " ", s).strip()

    @classmethod
    def is_missing_scalar(cls, val: Any) -> bool:
        if val is None:
            return True
        try:
            if pd.isna(val):
                return True
        except (TypeError, ValueError):
            pass

        if isinstance(val, (int, float, np.number)) and not isinstance(val, bool):
            return bool(np.isnan(val) or np.isinf(val))

        s = cls.clean_text_scalar(val)
        if not s:
            return True

        s_lower = s.lower()
        if s_lower in cls.CANONICAL_MISSING_TOKENS:
            return True

        try:
            float_val = float(s.replace(",", ""))
            if np.isnan(float_val) or np.isinf(float_val):
                return True
            return False
        except ValueError:
            pass

        unwrapped = cls._WRAPPER_STRIP_REGEX.sub("", s)
        normalised = cls._WHITESPACE_NORMALIZE_REGEX.sub(" ", unwrapped).strip().lower()
        if not normalised or normalised in cls.CANONICAL_MISSING_TOKENS:
            return True

        return bool(cls._COMPILED_REGEX.fullmatch(normalised))

    @classmethod
    def normalize_series(cls, series: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            return series.replace([np.inf, -np.inf], np.nan)

        orig_null_mask = series.isna()
        str_series = (
            series.astype(str)
            .str.replace(cls._MOJIBAKE_REGEX, " ", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

        numeric_check = pd.to_numeric(str_series.str.replace(",", "", regex=False), errors="coerce")
        is_valid_number = numeric_check.notna() & (~np.isinf(numeric_check))

        unwrapped = str_series.str.replace(cls._WRAPPER_STRIP_REGEX, "", regex=True)
        normalised = (
            unwrapped
            .str.replace(cls._WHITESPACE_NORMALIZE_REGEX, " ", regex=True)
            .str.strip()
            .str.lower()
        )

        is_missing_mask = (
            orig_null_mask
            | (str_series == "")
            | str_series.str.lower().isin(cls.CANONICAL_MISSING_TOKENS)
            | (
                ~is_valid_number
                & (
                    normalised.isin(cls.CANONICAL_MISSING_TOKENS)
                    | normalised.str.fullmatch(cls._COMPILED_REGEX, na=True)
                )
            )
        )

        return str_series.where(~is_missing_mask, np.nan)

    @classmethod
    def safe_fill_value(
        cls,
        series: pd.Series,
        fallback: str = "Standard",
    ) -> str:
        clean_s = cls.normalize_series(series).dropna()
        if clean_s.empty:
            return fallback

        mode_candidates = clean_s.value_counts().index.tolist()
        for candidate in mode_candidates:
            if not cls.is_missing_scalar(candidate):
                return str(candidate)

        return fallback


class DataPipelineConfig:
    """Configuration parameters for the AutoInsight Data Cleaning & EDA Pipeline."""
    def __init__(
        self,
        missing_col_threshold: float = 0.60,
        high_cardinality_limit: int = 15,
        top_n_categories: int = 10,
        iqr_multiplier: float = 1.5,
        zscore_threshold: float = 3.0,
        handle_outliers: bool = False,
        protect_formula_cols_from_clipping: bool = True,
        reconcile_math_relations: bool = True,
        fill_categorical_na: bool = True,
        boolean_na_fill: str = "False",
        categorical_fallback_label: str = "Standard",
        max_numeric_plots: int = 12,
        max_categorical_plots: int = 9,
        max_scatter_pairs: int = 4,
        max_timeseries_cols: int = 3,
        generate_interactive_html: bool = True,
        generate_senior_ds_report: bool = True,
        purge_legacy_pngs: bool = True,
        dpi: int = 150,
        theme: str = "dark",
    ):
        self.missing_col_threshold = missing_col_threshold
        self.high_cardinality_limit = high_cardinality_limit
        self.top_n_categories = top_n_categories
        self.iqr_multiplier = iqr_multiplier
        self.zscore_threshold = zscore_threshold
        self.handle_outliers = handle_outliers
        self.protect_formula_cols_from_clipping = protect_formula_cols_from_clipping
        self.reconcile_math_relations = reconcile_math_relations
        self.fill_categorical_na = fill_categorical_na
        self.boolean_na_fill = boolean_na_fill
        self.categorical_fallback_label = categorical_fallback_label
        self.max_numeric_plots = max_numeric_plots
        self.max_categorical_plots = max_categorical_plots
        self.max_scatter_pairs = max_scatter_pairs
        self.max_timeseries_cols = max_timeseries_cols
        self.generate_interactive_html = generate_interactive_html
        self.generate_senior_ds_report = generate_senior_ds_report
        self.purge_legacy_pngs = purge_legacy_pngs
        self.dpi = dpi
        self.theme = theme


class EnterpriseDataPipeline:
    """
    Automated Data Cleaning, Algebraic Reconciliation, Statistical EDA, and
    Publication-Grade Matplotlib/Seaborn Visual Component Engine.
    """

    def __init__(self, config: Optional[DataPipelineConfig] = None, output_dir: str = "pipeline_output"):
        self.config = config or DataPipelineConfig()
        self.output_dir = Path(output_dir)
        self.eda_dir = self.output_dir / "eda_reports"

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.eda_dir.mkdir(parents=True, exist_ok=True)

        self.logger = self._setup_logger()
        self.df: Optional[pd.DataFrame] = None
        self.raw_missing_summary: Dict[str, Dict[str, Any]] = {}
        self.eda_summary: Dict[str, Any] = {}
        self.integrity_audit: Dict[str, Any] = {}
        self.interactive_components: Dict[str, Any] = {}

        self.numeric_cols: List[str] = []
        self.categorical_cols: List[str] = []
        self.datetime_cols: List[str] = []
        self.id_or_constant_cols: List[str] = []
        self.formula_triplets: List[Tuple[str, str, str]] = []
        self.additive_triplets: List[Tuple[str, str, str]] = []
        self.formula_protected_cols: Set[str] = set()

        self._fitted_iqr_bounds: Dict[str, Tuple[float, float]] = {}

        if self.config.purge_legacy_pngs:
            self._remove_any_png_files()

    def _setup_logger(self) -> logging.Logger:
        logger_name = f"AutoInsightPipeline_{id(self)}"
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        if logger.handlers:
            logger.handlers.clear()

        formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        fh = logging.FileHandler(self.output_dir / "pipeline_execution.log", mode="w", encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        return logger

    def _remove_any_png_files(self) -> int:
        removed = 0
        for png_path in list(self.output_dir.rglob("*.png")) + list(self.output_dir.rglob("*.PNG")):
            try:
                png_path.unlink()
                removed += 1
            except OSError:
                pass
        return removed

    @classmethod
    def _is_identifier_column_name(cls, col_name: str) -> bool:
        tokens = set(re.split(r"[_\s\-]+", col_name.lower().strip("_")))
        if tokens & {"id", "uuid", "guid", "phone", "mobile", "tel", "zip", "postal", "ssn", "pin", "no"}:
            return True
        return any(
            col_name.lower().endswith(suffix)
            for suffix in ("_id", "_no", "_phone", "_code", "_zip", "_pin", "_ssn")
        )

    def load_data(self, file_path: str, source_name: Optional[str] = None) -> pd.DataFrame:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")

        display_name = source_name or path.name
        self.logger.info(f"Loading dataset from: {display_name}")
        ext = path.suffix.lower()

        if ext == ".csv":
            try:
                self.df = pd.read_csv(path, low_memory=False, encoding="utf-8-sig")
            except UnicodeDecodeError:
                self.df = pd.read_csv(path, encoding="latin1", low_memory=False)
        elif ext in [".xlsx", ".xls"]:
            self.df = pd.read_excel(path)
        elif ext == ".json":
            self.df = pd.read_json(path)
        elif ext == ".parquet":
            self.df = pd.read_parquet(path)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

        self.logger.info(f"Dataset loaded. Initial Shape: {self.df.shape[0]} rows, {self.df.shape[1]} columns.")
        return self.df

    def standardize_and_infer_types(self) -> None:
        self.logger.info("Standardizing column names, normalizing missing tokens, and inferring data types...")

        raw_cols = (
            self.df.columns.astype(str)
            .str.strip()
            .str.lower()
            .str.replace(r"[^\w\s]", "", regex=True)
            .str.replace(r"\s+", "_", regex=True)
            .str.strip("_")
        )
        deduped_cols: List[str] = []
        seen_counts: Dict[str, int] = {}
        for col in raw_cols:
            base = col if col else "column"
            if base not in seen_counts:
                seen_counts[base] = 1
                deduped_cols.append(base)
            else:
                seen_counts[base] += 1
                deduped_cols.append(f"{base}_{seen_counts[base]}")
        self.df.columns = deduped_cols

        for col in self.df.select_dtypes(include=["bool", "boolean"]).columns:
            self.df[col] = self.df[col].astype("object")

        for col in self.df.select_dtypes(include=["object", "string"]).columns:
            self.df[col] = UniversalNullNormalizer.normalize_series(self.df[col])

        n_rows = max(1, len(self.df))
        self.raw_missing_summary = {}
        for col in self.df.columns:
            m_cnt = int(self.df[col].isna().sum())
            if m_cnt > 0:
                self.raw_missing_summary[col] = {
                    "count": m_cnt,
                    "percentage": round(float(m_cnt / n_rows * 100), 2),
                }

        for col in self.df.select_dtypes(include=["object", "string"]).columns:
            non_null = self.df[col].dropna()
            if non_null.empty:
                continue

            unique_lower = set(non_null.astype(str).str.lower().unique())
            if unique_lower.issubset({"true", "false", "yes", "no"}):
                bool_map = {"true": "True", "false": "False", "yes": "True", "no": "False"}
                mask_valid = self.df[col].notna()
                self.df.loc[mask_valid, col] = (
                    self.df.loc[mask_valid, col].astype(str).str.lower().map(bool_map)
                )
                continue

            cleaned_num = (
                non_null.astype(str)
                .str.replace(r"^\((.*)\)$", r"-\1", regex=True)
                .str.replace(r"[$,₹€£%\s]", "", regex=True)
            )
            converted_num = pd.to_numeric(cleaned_num, errors="coerce")
            if converted_num.notnull().mean() > 0.85:
                full_cleaned = (
                    self.df[col].astype(str)
                    .str.replace(r"^\((.*)\)$", r"-\1", regex=True)
                    .str.replace(r"[$,₹€£%\s]", "", regex=True)
                )
                self.df[col] = pd.to_numeric(full_cleaned, errors="coerce")
                self.logger.info(f"Inferred column '{col}' as Numeric.")
                continue

            if non_null.astype(str).str.len().mean() >= 6:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    converted_dt = pd.to_datetime(non_null, errors="coerce")
                if converted_dt.notnull().mean() > 0.85:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        self.df[col] = pd.to_datetime(self.df[col], errors="coerce")
                    self.logger.info(f"Inferred column '{col}' as Datetime.")

        self._categorize_columns()

    def _categorize_columns(self) -> None:
        self.numeric_cols = []
        self.categorical_cols = []
        self.datetime_cols = []
        self.id_or_constant_cols = []

        n_rows = len(self.df)

        for col in self.df.columns:
            n_unique = self.df[col].nunique(dropna=True)

            if n_unique <= 1:
                self.id_or_constant_cols.append(col)
                continue

            if pd.api.types.is_datetime64_any_dtype(self.df[col]):
                self.datetime_cols.append(col)
            elif pd.api.types.is_numeric_dtype(self.df[col]) and not pd.api.types.is_bool_dtype(self.df[col]):
                is_id_name = self._is_identifier_column_name(col)
                if n_rows > 50 and n_unique == n_rows and is_id_name:
                    self.id_or_constant_cols.append(col)
                elif is_id_name and (n_unique / max(1, n_rows) > 0.50 or self.df[col].dropna().abs().median() > 1_000_000):
                    self.df[col] = self.df[col].astype(str)
                    self.categorical_cols.append(col)
                else:
                    self.numeric_cols.append(col)
            else:
                if n_rows > 50 and n_unique == n_rows and ("id" in col.lower() or "uuid" in col.lower()):
                    self.id_or_constant_cols.append(col)
                else:
                    self.categorical_cols.append(col)

        self.logger.info(
            f"Column Classification -> Numeric: {len(self.numeric_cols)} | "
            f"Categorical: {len(self.categorical_cols)} | "
            f"Datetime: {len(self.datetime_cols)} | "
            f"ID/Constant: {len(self.id_or_constant_cols)}"
        )

    def _detect_formulas(self, df: pd.DataFrame) -> None:
        self.formula_triplets = []
        self.additive_triplets = []
        self.formula_protected_cols = set()

        if not self.config.reconcile_math_relations or len(self.numeric_cols) < 3:
            return

        num_cols = [c for c in self.numeric_cols if c in df.columns]
        complete_df = df[num_cols].dropna()
        if len(complete_df) < 10:
            return

        used_targets: Set[str] = set()
        for i in range(len(num_cols)):
            for j in range(i + 1, len(num_cols)):
                for k in range(len(num_cols)):
                    if k == i or k == j:
                        continue
                    col_a, col_b, col_c = num_cols[i], num_cols[j], num_cols[k]
                    if col_c in used_targets:
                        continue

                    product = complete_df[col_a] * complete_df[col_b]
                    mult_matches = np.isclose(product, complete_df[col_c], rtol=1e-3, atol=1e-2)
                    if mult_matches.mean() > 0.98:
                        self.formula_triplets.append((col_a, col_b, col_c))
                        self.formula_protected_cols.update([col_a, col_b, col_c])
                        used_targets.add(col_c)
                        self.logger.info(
                            f"Detected exact deterministic formula: '{col_a}' * '{col_b}' = '{col_c}'."
                        )
                        continue

                    summation = complete_df[col_a] + complete_df[col_b]
                    add_matches = np.isclose(summation, complete_df[col_c], rtol=1e-3, atol=1e-2)
                    if add_matches.mean() > 0.98:
                        self.additive_triplets.append((col_a, col_b, col_c))
                        self.formula_protected_cols.update([col_a, col_b, col_c])
                        used_targets.add(col_c)
                        self.logger.info(
                            f"Detected exact additive formula: '{col_a}' + '{col_b}' = '{col_c}'."
                        )

    def clean_structure(self) -> None:
        self.logger.info("Executing structural cleaning...")

        initial_rows = len(self.df)
        self.df.drop_duplicates(inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        dropped_dups = initial_rows - len(self.df)
        if dropped_dups > 0:
            self.logger.info(f"Removed {dropped_dups} exact duplicate rows.")

        if self.id_or_constant_cols:
            self.df.drop(columns=self.id_or_constant_cols, inplace=True, errors="ignore")
            self.logger.info(f"Dropped uninformative ID/constant columns: {self.id_or_constant_cols}")

        null_ratios = self.df.isnull().mean()
        cols_to_drop = list(null_ratios[null_ratios > self.config.missing_col_threshold].index)
        if cols_to_drop:
            self.df.drop(columns=cols_to_drop, inplace=True)
            self.logger.info(f"Dropped high-null columns: {cols_to_drop}")

        self._categorize_columns()
        self._detect_formulas(self.df)

    def _recover_numeric_via_lookup(self, df: pd.DataFrame) -> pd.DataFrame:
        for num_col in self.numeric_cols:
            if num_col not in df.columns or not df[num_col].isna().any():
                continue
            for cat_col in self.categorical_cols:
                if cat_col not in df.columns or self._is_identifier_column_name(cat_col):
                    continue
                n_uniq = df[cat_col].nunique(dropna=True)
                if n_uniq < 5 or n_uniq > 500:
                    continue
                valid_sub = df[[cat_col, num_col]].dropna()
                if valid_sub.empty:
                    continue
                grouped_nunique = valid_sub.groupby(cat_col)[num_col].nunique()
                if (grouped_nunique == 1).mean() >= 0.95:
                    lookup_map = valid_sub.groupby(cat_col)[num_col].first().to_dict()
                    mask_miss = df[num_col].isna() & df[cat_col].notna()
                    if mask_miss.any():
                        mapped = df.loc[mask_miss, cat_col].map(lookup_map)
                        n_rec = int(mapped.notna().sum())
                        df.loc[mask_miss, num_col] = mapped
                        if n_rec > 0:
                            self.logger.info(
                                f"Recovered {n_rec} exact '{num_col}' values via '{cat_col}' lookup."
                            )
        return df

    def _recover_categorical_via_lookup(self, df: pd.DataFrame) -> pd.DataFrame:
        for target_col in self.categorical_cols:
            if target_col not in df.columns or not df[target_col].isna().any():
                continue
            if self._is_identifier_column_name(target_col) or "name" in target_col.lower():
                continue

            other_cats = [
                c for c in self.categorical_cols
                if c != target_col and c in df.columns and 2 <= df[c].nunique(dropna=True) <= 50
            ]
            for cat_key in other_cats:
                for num_key in self.numeric_cols:
                    if num_key not in df.columns:
                        continue
                    valid_sub = df[[cat_key, num_key, target_col]].dropna()
                    if valid_sub.empty:
                        continue

                    grouped = valid_sub.groupby([cat_key, num_key])[target_col].nunique()
                    if (grouped == 1).mean() >= 0.90:
                        lookup_map = (
                            valid_sub.groupby([cat_key, num_key])[target_col]
                            .agg(lambda x: x.mode().iloc[0])
                            .to_dict()
                        )
                        mask_miss = df[target_col].isna() & df[cat_key].notna() & df[num_key].notna()
                        if mask_miss.any():
                            keys = list(zip(df.loc[mask_miss, cat_key], df.loc[mask_miss, num_key]))
                            mapped = pd.Series(keys, index=df.index[mask_miss]).map(lookup_map)
                            n_rec = int(mapped.notna().sum())
                            df.loc[mask_miss, target_col] = mapped
                            if n_rec > 0:
                                self.logger.info(
                                    f"Recovered {n_rec} exact '{target_col}' values via ({cat_key}, {num_key}) lookup."
                                )
        return df

    def prepare_for_visualization(self) -> pd.DataFrame:
        df = self.df.copy()

        for col in df.select_dtypes(include=["object", "string"]).columns:
            df[col] = UniversalNullNormalizer.normalize_series(df[col])

        for col_a, col_b, col_c in self.formula_triplets:
            if all(c in df.columns for c in [col_a, col_b, col_c]):
                mask_c = df[col_c].isna() & df[col_a].notna() & df[col_b].notna()
                n_c = int(mask_c.sum())
                if n_c > 0:
                    df.loc[mask_c, col_c] = (df.loc[mask_c, col_a] * df.loc[mask_c, col_b]).round(4)

                mask_a = df[col_a].isna() & df[col_c].notna() & df[col_b].notna() & (df[col_b] != 0)
                n_a = int(mask_a.sum())
                if n_a > 0:
                    df.loc[mask_a, col_a] = (df.loc[mask_a, col_c] / df.loc[mask_a, col_b]).round(4)

                mask_b = df[col_b].isna() & df[col_c].notna() & df[col_a].notna() & (df[col_a] != 0)
                n_b = int(mask_b.sum())
                if n_b > 0:
                    df.loc[mask_b, col_b] = (df.loc[mask_b, col_c] / df.loc[mask_b, col_a]).round(4)

                self.logger.info(
                    f"Algebraically reconstructed exact values -> {col_a}: {n_a}, {col_b}: {n_b}, {col_c}: {n_c}"
                )
                self.integrity_audit["formula_reconciliation"] = {
                    "formula": f"{col_a} * {col_b} = {col_c}",
                    "reconstructed_counts": {col_a: n_a, col_b: n_b, col_c: n_c},
                }

        for col_a, col_b, col_c in self.additive_triplets:
            if all(c in df.columns for c in [col_a, col_b, col_c]):
                mask_c = df[col_c].isna() & df[col_a].notna() & df[col_b].notna()
                if mask_c.any():
                    df.loc[mask_c, col_c] = (df.loc[mask_c, col_a] + df.loc[mask_c, col_b]).round(4)
                mask_a = df[col_a].isna() & df[col_c].notna() & df[col_b].notna()
                if mask_a.any():
                    df.loc[mask_a, col_a] = (df.loc[mask_a, col_c] - df.loc[mask_a, col_b]).round(4)
                mask_b = df[col_b].isna() & df[col_c].notna() & df[col_a].notna()
                if mask_b.any():
                    df.loc[mask_b, col_b] = (df.loc[mask_b, col_c] - df.loc[mask_b, col_a]).round(4)

        df = self._recover_numeric_via_lookup(df)

        low_card_group_cats = [
            c for c in self.categorical_cols
            if c in df.columns and 2 <= df[c].nunique(dropna=True) <= 30
        ]
        for col_a, col_b, col_c in self.formula_triplets:
            if all(c in df.columns for c in [col_a, col_b, col_c]):
                for factor_col in [col_a, col_b]:
                    if df[factor_col].isna().any():
                        valid_vals = df[factor_col].dropna()
                        is_int_col = not valid_vals.empty and (valid_vals % 1 == 0).all()
                        if low_card_group_cats:
                            grp_med = df.groupby(low_card_group_cats[0])[factor_col].transform("median")
                            df[factor_col] = df[factor_col].fillna(grp_med)
                        df[factor_col] = df[factor_col].fillna(valid_vals.median() if not valid_vals.empty else 1.0)
                        if is_int_col:
                            df[factor_col] = df[factor_col].round(0)

                mask_c_null = df[col_c].isna()
                if mask_c_null.any():
                    df.loc[mask_c_null, col_c] = (
                        df.loc[mask_c_null, col_a] * df.loc[mask_c_null, col_b]
                    ).round(4)

        for col in self.numeric_cols:
            if col in df.columns and df[col].isna().any():
                med = df[col].median()
                df[col] = df[col].fillna(0.0 if pd.isna(med) else med)

        df = self._recover_categorical_via_lookup(df)

        for col in self.categorical_cols:
            if col not in df.columns or not df[col].isna().any():
                continue

            df[col] = UniversalNullNormalizer.normalize_series(df[col])
            if not df[col].isna().any():
                continue

            valid_lower = set(df[col].dropna().astype(str).str.lower().unique())
            if valid_lower and valid_lower.issubset({"true", "false"}):
                df[col] = df[col].fillna(self.config.boolean_na_fill)
                continue

            grp_candidates = [c for c in low_card_group_cats if c != col]
            if grp_candidates:
                grp_col = grp_candidates[0]
                grp_modes = (
                    df.dropna(subset=[grp_col, col])
                    .groupby(grp_col)[col]
                    .agg(lambda s: s.value_counts().index[0] if not s.empty else np.nan)
                    .to_dict()
                )
                mask_na = df[col].isna()
                df.loc[mask_na, col] = df.loc[mask_na, grp_col].map(grp_modes)

            if df[col].isna().any():
                safe_val = UniversalNullNormalizer.safe_fill_value(
                    df[col], fallback=self.config.categorical_fallback_label
                )
                df[col] = df[col].fillna(safe_val)

        for col in self.datetime_cols:
            if col in df.columns and df[col].isna().any():
                df[col] = df[col].ffill().bfill()

        if self.config.handle_outliers:
            for col in self.numeric_cols:
                if col not in df.columns or df[col].nunique() <= 2:
                    continue
                if self.config.protect_formula_cols_from_clipping and col in self.formula_protected_cols:
                    continue
                q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
                iqr = q3 - q1
                if iqr > 0:
                    low = q1 - self.config.iqr_multiplier * iqr
                    high = q3 + self.config.iqr_multiplier * iqr
                    self._fitted_iqr_bounds[col] = (low, high)
                    df[col] = df[col].clip(lower=low, upper=high)

        for col in df.columns:
            if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
                df[col] = UniversalNullNormalizer.normalize_series(df[col])
                if df[col].isna().any():
                    valid_lower = set(df[col].dropna().astype(str).str.lower().unique())
                    if valid_lower and valid_lower.issubset({"true", "false"}):
                        df[col] = df[col].fillna(self.config.boolean_na_fill)
                    else:
                        safe_val = UniversalNullNormalizer.safe_fill_value(
                            df[col], fallback=self.config.categorical_fallback_label
                        )
                        df[col] = df[col].fillna(safe_val)
            elif pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
                if df[col].isna().any():
                    med = df[col].median()
                    df[col] = df[col].fillna(0.0 if pd.isna(med) else med)
                valid_num = df[col].dropna()
                if not valid_num.empty and (valid_num % 1 == 0).all():
                    df[col] = df[col].astype(int)

        self.df = df
        return df

    @classmethod
    def pick_primary_measure(cls, num_cols: List[str], formula_triplets: Optional[List[Tuple[str, str, str]]] = None) -> Optional[str]:
        if formula_triplets:
            return formula_triplets[0][2]
        if not num_cols:
            return None
        priority_keywords = ["total", "spent", "revenue", "sales", "price", "amount", "profit", "income", "score"]
        for kw in priority_keywords:
            matches = [c for c in num_cols if kw in c.lower() and not cls._is_identifier_column_name(c)]
            if matches:
                return matches[0]
        non_id = [c for c in num_cols if not cls._is_identifier_column_name(c)]
        return non_id[0] if non_id else num_cols[0]

    def perform_eda(self) -> Dict[str, Any]:
        self.logger.info("Executing Automated Exploratory Data Analysis (EDA) & Anomaly Detection...")

        missing_counts = self.df.isnull().sum()
        missing_pct = (missing_counts / max(1, len(self.df)) * 100).round(2)

        self.eda_summary["dataset_overview"] = {
            "total_rows": int(self.df.shape[0]),
            "total_columns": int(self.df.shape[1]),
            "total_missing_cells": int(missing_counts.sum()),
            "duplicate_rows": int(self.df.duplicated().sum()),
            "memory_usage_mb": round(float(self.df.memory_usage(deep=True).sum() / (1024 ** 2)), 3),
            "column_types": {
                "numeric": self.numeric_cols,
                "categorical": self.categorical_cols,
                "datetime": self.datetime_cols,
                "id_or_constant": self.id_or_constant_cols,
            },
        }

        self.eda_summary["raw_missing_values"] = self.raw_missing_summary
        self.eda_summary["missing_values"] = {
            col: {"count": int(missing_counts[col]), "percentage": float(missing_pct[col])}
            for col in self.df.columns if missing_counts[col] > 0
        }

        num_stats: Dict[str, Any] = {}
        anomaly_report: Dict[str, Any] = {}
        for col in self.numeric_cols:
            if col not in self.df.columns:
                continue
            series = self.df[col].dropna()
            if series.empty:
                continue
            q1, q3 = float(series.quantile(0.25)), float(series.quantile(0.75))
            iqr = q3 - q1
            low_fence = q1 - 1.5 * iqr
            high_fence = q3 + 1.5 * iqr
            iqr_outliers = int(((series < low_fence) | (series > high_fence)).sum())

            mean_val = float(series.mean())
            std_val = float(series.std()) if len(series) > 1 else 0.0
            z_outliers = int((np.abs((series - mean_val) / std_val) > self.config.zscore_threshold).sum()) if std_val > 0 else 0

            num_stats[col] = {
                "mean": round(mean_val, 4),
                "std": round(std_val, 4),
                "min": round(float(series.min()), 4),
                "q1": round(q1, 4),
                "median": round(float(series.median()), 4),
                "q3": round(q3, 4),
                "max": round(float(series.max()), 4),
                "skewness": round(float(series.skew()), 4),
                "kurtosis": round(float(series.kurtosis()), 4),
                "iqr_outliers_count": iqr_outliers,
                "zscore_outliers_count": z_outliers,
                "iqr_lower_fence": round(low_fence, 4),
                "iqr_upper_fence": round(high_fence, 4),
            }
            anomaly_report[col] = {
                "iqr_anomalies": iqr_outliers,
                "iqr_anomaly_pct": round(iqr_outliers / len(series) * 100, 2),
                "zscore_anomalies": z_outliers,
                "lower_bound": round(low_fence, 4),
                "upper_bound": round(high_fence, 4),
            }

        self.eda_summary["numeric_summary"] = num_stats
        self.eda_summary["anomaly_report"] = anomaly_report

        cat_stats: Dict[str, Any] = {}
        for col in self.categorical_cols:
            if col not in self.df.columns:
                continue
            series = self.df[col].dropna()
            top_vals = series.value_counts().head(5).to_dict()
            cat_stats[col] = {
                "unique_categories": int(series.nunique()),
                "top_5_frequent": {str(k): int(v) for k, v in top_vals.items()},
            }
        self.eda_summary["categorical_summary"] = cat_stats
        self.eda_summary["automated_insights"] = self._generate_automated_insights()

        with open(self.eda_dir / "eda_summary.json", "w", encoding="utf-8") as f:
            json.dump(self.eda_summary, f, indent=4)

        return self.eda_summary

    def _generate_automated_insights(self) -> List[Dict[str, str]]:
        insights: List[Dict[str, str]] = []
        if self.df is None or self.df.empty:
            return insights

        primary_measure = self.pick_primary_measure(self.numeric_cols, self.formula_triplets)

        if primary_measure and primary_measure in self.df.columns:
            total_val = float(self.df[primary_measure].sum())
            mean_val = float(self.df[primary_measure].mean())
            median_val = float(self.df[primary_measure].median())
            q1_v = self.eda_summary["numeric_summary"][primary_measure]["q1"]
            q3_v = self.eda_summary["numeric_summary"][primary_measure]["q3"]
            insights.append({
                "category": "KPI Headline",
                "title": f"Cumulative {primary_measure.replace('_', ' ').title()} Reaches ${total_val:,.2f}",
                "detail": (
                    f"Across {len(self.df):,} verified records, {primary_measure} averages ${mean_val:,.2f} "
                    f"(median ${median_val:,.2f}, interquartile range ${q1_v:,.2f}–${q3_v:,.2f})."
                ),
            })

            low_card = [
                c for c in self.categorical_cols
                if c in self.df.columns
                and not self._is_identifier_column_name(c)
                and 2 <= self.df[c].nunique() <= 35
            ]
            for cat in low_card[:4]:
                grp = self.df.groupby(cat)[primary_measure].agg(["sum", "mean", "count"]).sort_values("sum", ascending=False)
                if not grp.empty:
                    top_name = str(grp.index[0])
                    top_sum = float(grp.iloc[0]["sum"])
                    share_pct = (top_sum / total_val * 100) if total_val > 0 else 0.0
                    bot_name = str(grp.index[-1])
                    bot_sum = float(grp.iloc[-1]["sum"])
                    insights.append({
                        "category": f"Segment Leader ({cat})",
                        "title": f"'{top_name}' Leads {cat.replace('_', ' ').title()} with {share_pct:.1f}% Share",
                        "detail": (
                            f"'{top_name}' generated ${top_sum:,.2f} across {int(grp.iloc[0]['count']):,} records "
                            f"(mean ${float(grp.iloc[0]['mean']):,.2f}), outperforming '{bot_name}' (${bot_sum:,.2f})."
                        ),
                    })

        return insights

    def generate_visualizations(self) -> Dict[str, Any]:
        self.logger.info("Generating visual component specifications & Senior Data Scientist Report...")

        removed_pngs = self._remove_any_png_files()
        if removed_pngs > 0:
            self.logger.info(f"Purged {removed_pngs} legacy static PNG file(s) from output directory.")

        components: List[Dict[str, Any]] = []
        num_cols = [c for c in self.numeric_cols if c in self.df.columns][: self.config.max_numeric_plots]
        primary_measure = self.pick_primary_measure(num_cols, self.formula_triplets)

        for col in num_cols:
            s = self.df[col].dropna()
            if s.empty:
                continue
            counts, bin_edges = np.histogram(s, bins=min(25, max(10, int(s.nunique()))))
            bin_labels = [f"{bin_edges[i]:.1f}–{bin_edges[i+1]:.1f}" for i in range(len(counts))]
            stats = self.eda_summary.get("numeric_summary", {}).get(col, {})
            components.append({
                "component_id": f"comp_dist_{col}",
                "title": f"Distribution & Parametric Quantile Profile: {col}",
                "type": "distribution_and_box",
                "column": col,
                "bins": [round(float((bin_edges[i] + bin_edges[i+1]) / 2), 3) for i in range(len(counts))],
                "bin_ranges": bin_labels,
                "counts": [int(c) for c in counts],
                "stats": stats,
            })

        if len(num_cols) >= 2:
            corr_df = self.df[num_cols].corr().round(4)
            components.append({
                "component_id": "comp_correlation_matrix",
                "title": "Multi-Feature Pearson Correlation Matrix",
                "type": "heatmap",
                "columns": num_cols,
                "z_matrix": corr_df.values.tolist(),
            })

        cat_cols = [
            c for c in self.categorical_cols
            if c in self.df.columns and not self._is_identifier_column_name(c)
        ][: self.config.max_categorical_plots]

        for col in cat_cols:
            vc = self.df[col].value_counts().head(self.config.top_n_categories)
            item_data: Dict[str, Any] = {
                "component_id": f"comp_cat_{col}",
                "title": f"Segment Breakdown: {col}",
                "type": "categorical_bar",
                "column": col,
                "labels": [str(k) for k in vc.index],
                "counts": [int(v) for v in vc.values],
            }
            if primary_measure:
                rev_series = self.df.groupby(col)[primary_measure].agg(["sum", "mean", "median"]).reindex(vc.index)
                item_data["measure"] = primary_measure
                item_data["measure_sum"] = [round(float(v), 2) for v in rev_series["sum"].fillna(0)]
                item_data["measure_mean"] = [round(float(v), 2) for v in rev_series["mean"].fillna(0)]
                item_data["measure_median"] = [round(float(v), 2) for v in rev_series["median"].fillna(0)]
            components.append(item_data)

        self.interactive_components = {
            "render_mode": "matplotlib_seaborn_vector_components",
            "static_png_count": 0,
            "generated_components_count": len(components),
            "components": components,
        }

        comp_json_path = self.eda_dir / "interactive_visual_components.json"
        with open(comp_json_path, "w", encoding="utf-8") as f:
            json.dump(self.interactive_components, f, indent=2)
        self.logger.info(f"Saved {len(components)} visual component specs to: {comp_json_path.name}")

        return self.interactive_components

    def verify_integrity(self, raw_df: pd.DataFrame, processed_df: pd.DataFrame) -> Dict[str, Any]:
        formula_check_passed = True
        max_formula_discrepancy = 0.0

        for col_a, col_b, col_c in self.formula_triplets:
            if all(c in processed_df.columns for c in [col_a, col_b, col_c]):
                complete = processed_df[[col_a, col_b, col_c]].dropna()
                if not complete.empty:
                    diff = np.abs(complete[col_a] * complete[col_b] - complete[col_c])
                    max_formula_discrepancy = max(max_formula_discrepancy, float(diff.max()))
                    if max_formula_discrepancy >= 0.05:
                        formula_check_passed = False

        remaining_missing_tokens = 0
        stray_token_detail: Dict[str, int] = {}
        for col in processed_df.select_dtypes(include=["object", "string"]).columns:
            re_checked = UniversalNullNormalizer.normalize_series(processed_df[col])
            n_stray = int(re_checked.isna().sum())
            remaining_missing_tokens += n_stray
            if n_stray > 0:
                stray_token_detail[col] = n_stray

        png_files_remaining = len(list(self.output_dir.rglob("*.png"))) + len(list(self.output_dir.rglob("*.PNG")))
        remaining_nulls = int(processed_df.isna().sum().sum())
        row_preserved = bool(len(raw_df.drop_duplicates()) == len(processed_df))

        health_score = 100.0
        if remaining_nulls > 0:
            health_score -= min(40.0, remaining_nulls * 0.5)
        if remaining_missing_tokens > 0:
            health_score -= min(30.0, remaining_missing_tokens * 1.0)
        if not formula_check_passed:
            health_score -= 20.0
        if not row_preserved:
            health_score -= 10.0

        audit = {
            "data_health_score": round(max(0.0, health_score), 1),
            "row_count_preserved": row_preserved,
            "original_rows": int(len(raw_df)),
            "processed_rows": int(len(processed_df)),
            "remaining_null_cells": remaining_nulls,
            "remaining_missing_token_strings": remaining_missing_tokens,
            "stray_token_columns": stray_token_detail,
            "formula_consistency_verified": formula_check_passed,
            "max_formula_discrepancy": round(max_formula_discrepancy, 6),
            "static_png_files_in_output": png_files_remaining,
            "interactive_components_generated": int(self.interactive_components.get("generated_components_count", 0)),
            "processed_columns_count": int(processed_df.shape[1]),
            "processed_columns": list(processed_df.columns),
        }
        self.integrity_audit.update(audit)

        with open(self.eda_dir / "data_integrity_audit.json", "w", encoding="utf-8") as f:
            json.dump(self.integrity_audit, f, indent=4)

        self.logger.info(
            f"Self-Verification Complete -> Health Score: {audit['data_health_score']}/100 | "
            f"Formula Consistent: {formula_check_passed} (Max discrepancy: {max_formula_discrepancy:.6f}) | "
            f"Static PNGs in Output: {png_files_remaining}"
        )
        return self.integrity_audit

    def run_pipeline(
        self,
        input_file: str,
        output_csv_name: str = "cleaned_visualization_data.csv",
        source_name: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, Path]:
        self.logger.info("=" * 72)
        self.logger.info("INITIATING AUTOINSIGHT DATA CLEANING & EDA PIPELINE")
        self.logger.info("=" * 72)

        self.load_data(input_file, source_name=source_name)
        raw_snapshot = self.df.copy()

        self.standardize_and_infer_types()
        self.clean_structure()
        self.prepare_for_visualization()
        self.perform_eda()
        self.generate_visualizations()

        output_path = self.output_dir / output_csv_name
        self.df.to_csv(output_path, index=False, encoding="utf-8")

        self.verify_integrity(raw_snapshot, self.df)

        if self.config.generate_senior_ds_report:
            try:
                from auto_report_pipeline import SeniorDataScientistReportPipeline
                report_pipe = SeniorDataScientistReportPipeline(output_dir=str(self.output_dir))
                report_pipe.generate_report(
                    df=self.df,
                    eda_summary=self.eda_summary,
                    integrity_audit=self.integrity_audit,
                    dataset_name=source_name or Path(input_file).name,
                )
            except Exception as exc:
                self.logger.warning(f"Report pipeline skipped: {exc}")

        self.logger.info(
            f"Saved cleaned dataset to '{output_path}' "
            f"(Shape: {self.df.shape[0]} rows, {self.df.shape[1]} columns)."
        )
        self.logger.info("=" * 72)
        self.close()
        return self.df, output_path

    def close(self):
        """Release and close all logging handlers to avoid file locking on Windows."""
        if hasattr(self, "logger") and self.logger:
            for handler in list(self.logger.handlers):
                try:
                    handler.close()
                    self.logger.removeHandler(handler)
                except Exception:
                    pass

    def __del__(self):
        self.close()


EnterpriseDataPipelineV5 = EnterpriseDataPipeline
EnterpriseDataPipelineV3 = EnterpriseDataPipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoInsight Data Cleaning + EDA Pipeline")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input dataset")
    parser.add_argument("--output-dir", "-o", type=str, default="pipeline_output", help="Directory for outputs and reports")
    parser.add_argument("--output-csv", "-c", type=str, default="cleaned_visualization_data.csv", help="Filename for cleaned CSV")
    parser.add_argument("--clip-outliers", action="store_true", help="Clip IQR outliers (off by default to keep real values)")
    parser.add_argument("--top-n", type=int, default=10, help="Top-N categories to include")
    parser.add_argument("--theme", type=str, default="dark", help="Visual theme (dark or light)")

    args = parser.parse_args()

    config = DataPipelineConfig(
        handle_outliers=args.clip_outliers,
        top_n_categories=args.top_n,
        theme=args.theme,
    )

    pipeline = EnterpriseDataPipeline(config=config, output_dir=args.output_dir)
    pipeline.run_pipeline(input_file=args.input, output_csv_name=args.output_csv)
