# ⚡ AutoInsight — Enterprise AI BI & Senior Data Scientist Report Suite

**AutoInsight** is a public deploy-ready automated data cleaning, algebraic reconciliation, exploratory data analysis (EDA), Senior Data Scientist reporting, and resizable Business Intelligence platform powered by **Matplotlib & Seaborn**.

## 🚀 Core Capabilities

1. **Dual Enterprise Pipelines**
   - **Data Cleaning & EDA Pipeline (`auto_data_pipeline.py`):** Normalizes missing tokens (`UniversalNullNormalizer`), cleans encoding mojibake, reconciles deterministic formulas (`A * B = C` and `A + B = C`), filters out non-analytical identifier columns (such as `phone`, `dealer_no`, `car_id`), and computes parametric/non-parametric statistics.
   - **Senior Data Scientist Report Pipeline (`auto_report_pipeline.py`):** Generates a publication-grade **Senior Data Scientist Report** (`senior_data_scientist_report.html`) featuring:
     1. **Executive Data Summary & Statistical Profile Table**
     2. **Publication-Grade Matplotlib & Seaborn Visualizations Paired with Detailed Analytical Insights** (Empirical Statistical Finding, Senior Data Scientist Interpretation, and Strategic/Modeling Recommendation)
     3. **Short Executive Conclusion & Strategic Action Plan**

2. **Publication-Grade Matplotlib & Seaborn Visuals (Light & Dark Mode)**
   - Uses **Matplotlib** and **Seaborn** with smart label wrapping, automatic horizontal bar orientation for long category names, direct data callouts, and `tight_layout` to prevent any overlapping labels.
   - Supports full **Light Mode** (`☀️`) and **Dark Mode** (`🌙`) switching with crisp contrast across both the UI and charts.

3. **PowerBI-Style Drag-to-Resize Visual Cards & Full-Space Utilization**
   - Every visual component card on the dashboard can be resized smoothly by dragging its corner/edges (`↘`), just like PowerBI.
   - The AI Co-Pilot panel is paired with live companion Matplotlib/Seaborn visual components directly below the chatbot so zero vacant space is wasted, and can also be collapsed via a 1-click toggle.
   - AI credentials load automatically from `.env`.

## 🛠️ Quick Start

```bash
pip install -r requirements.txt

# 1. Run Data Cleaning Pipeline (also triggers Senior Data Scientist Report Pipeline)
python auto_data_pipeline.py --input path/to/dataset.csv --output-dir pipeline_output

# 2. Or Run Senior Data Scientist Report Pipeline Standalone
python auto_report_pipeline.py --input pipeline_output/cleaned_visualization_data.csv --output-dir pipeline_output

# 3. Launch Interactive BI & Report Suite
streamlit run auto_insight_dashboard.py
```
