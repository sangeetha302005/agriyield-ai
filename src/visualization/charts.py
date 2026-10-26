"""
src/visualization/charts.py
---------------------------
Modular visualization library for AgriYield AI.
Provides interactive Plotly visualizations and static charts designed with
a modern agricultural intelligence aesthetic (forest green, muted sage, off-white).
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Brand color system
COLORS = {
    "primary": "#1E3A2B",       # Deep Forest Green
    "secondary": "#2D5A3F",     # Muted Pine Green
    "accent": "#D4A373",        # Warm Harvest Amber / Gold
    "sage": "#8FBC8F",          # Soft Sage
    "light_bg": "#F8FAF8",      # Crisp Off-White
    "card_bg": "#FFFFFF",       # Pure White
    "border": "#E2E8F0",        # Subtle Slate Border
    "text_dark": "#1A2520",     # Charcoal Green
    "text_muted": "#556B60",    # Muted Olive
    "highlight_red": "#B84A39", # Warning Coral
    "highlight_blue": "#3E7B99" # Waterway Blue
}

FONT_FAMILY = "Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif"


def apply_agri_theme(fig: go.Figure, title: str = "", height: int = 420) -> go.Figure:
    """Apply consistent styling across all Plotly figures."""
    fig.update_layout(
        title={
            "text": f"<b>{title}</b>" if title else "",
            "font": {"size": 16, "color": COLORS["text_dark"], "family": FONT_FAMILY},
            "x": 0.02,
            "xanchor": "left"
        },
        paper_bgcolor=COLORS["card_bg"],
        plot_bgcolor=COLORS["light_bg"],
        font={"family": FONT_FAMILY, "color": COLORS["text_dark"], "size": 12},
        height=height,
        margin={"l": 45, "r": 25, "t": 50, "b": 40},
        xaxis={
            "gridcolor": "#E9ECEF",
            "linecolor": "#CBD5E1",
            "tickfont": {"size": 11, "color": COLORS["text_muted"]},
            "title_font": {"size": 12, "color": COLORS["text_dark"]}
        },
        yaxis={
            "gridcolor": "#E9ECEF",
            "linecolor": "#CBD5E1",
            "tickfont": {"size": 11, "color": COLORS["text_muted"]},
            "title_font": {"size": 12, "color": COLORS["text_dark"]}
        },
        legend={
            "bgcolor": "rgba(255,255,255,0.85)",
            "bordercolor": COLORS["border"],
            "borderwidth": 1,
            "font": {"size": 11, "color": COLORS["text_dark"]}
        },
        hoverlabel={
            "bgcolor": COLORS["primary"],
            "font_size": 12,
            "font_family": FONT_FAMILY,
            "font_color": "#FFFFFF"
        }
    )
    return fig


def plot_historical_yield_trends(
    df: pd.DataFrame,
    crop: str = "Rice",
    state: Optional[str] = None
) -> go.Figure:
    """Interactive line chart tracking yield (t/ha) and cultivated area over years."""
    sub = df[df["Crop"] == crop].copy()
    if state and state != "All India":
        sub = sub[sub["State_Name"] == state]

    if sub.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data matching current filters", showarrow=False)
        return apply_agri_theme(fig, f"Historical Trend: {crop}")

    yearly = sub.groupby("Crop_Year").agg(
        Mean_Yield=("Yield", "mean"),
        Total_Area=("Area", "sum"),
        Total_Production=("Production", "sum")
    ).reset_index()

    fig = go.Figure()

    # Yield line
    fig.add_trace(go.Scatter(
        x=yearly["Crop_Year"],
        y=yearly["Mean_Yield"],
        name="Average Yield (Tonnes/Ha)",
        mode="lines+markers",
        line={"color": COLORS["secondary"], "width": 3},
        marker={"size": 7, "color": COLORS["primary"]}
    ))

    # Area bar on secondary y-axis
    fig.add_trace(go.Bar(
        x=yearly["Crop_Year"],
        y=yearly["Total_Area"],
        name="Cultivated Area (Ha)",
        yaxis="y2",
        marker={"color": "rgba(212, 163, 115, 0.35)"}
    ))

    title_text = f"Annual Yield & Cultivated Area: {crop}" + (f" ({state})" if state and state != "All India" else " (All India)")

    fig.update_layout(
        yaxis2={
            "title": "Cultivated Area (Hectares)",
            "overlaying": "y",
            "side": "right",
            "showgrid": False,
            "title_font": {"size": 11, "color": COLORS["accent"]}
        }
    )
    fig.update_yaxes(title_text="Yield (Tonnes / Hectare)")
    fig.update_xaxes(title_text="Crop Year", tickmode="linear", dtick=2)

    return apply_agri_theme(fig, title_text)


def plot_crop_distribution(
    df: pd.DataFrame,
    metric: str = "Production",
    top_n: int = 12
) -> go.Figure:
    """Bar chart comparing major crops by Area or Production."""
    top_crops = (
        df.groupby("Crop")[metric]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
    )

    unit = "Tonnes" if metric == "Production" else "Hectares"
    fig = px.bar(
        top_crops,
        x=metric,
        y="Crop",
        orientation="h",
        color=metric,
        color_continuous_scale=[[0, "#E8F0EA"], [0.5, COLORS["sage"]], [1.0, COLORS["primary"]]],
        labels={metric: f"Total {metric} ({unit})", "Crop": "Agricultural Crop"}
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
    return apply_agri_theme(fig, f"Top {top_n} Crops by Historical {metric} ({unit})")


def plot_season_yield_distribution(
    df: pd.DataFrame,
    crop: Optional[str] = None
) -> go.Figure:
    """Box plot of crop yields categorized across the 6 official agricultural seasons."""
    data = df if not crop or crop == "All Crops" else df[df["Crop"] == crop]
    # Filter extreme visual outliers for plot readability
    q99 = data["Yield"].quantile(0.98)
    sub = data[data["Yield"] <= q99]

    fig = px.box(
        sub,
        x="Season",
        y="Yield",
        color="Season",
        color_discrete_sequence=[COLORS["primary"], COLORS["secondary"], COLORS["accent"],
                                 COLORS["sage"], COLORS["highlight_blue"], "#7E8D85"],
        labels={"Yield": "Yield (Tonnes / Ha)", "Season": "Agricultural Season"}
    )
    fig.update_layout(showlegend=False)
    title = f"Yield Distribution Across Seasons ({crop if crop and crop != 'All Crops' else 'All Crops'})"
    return apply_agri_theme(fig, title)


def plot_top_producing_states(
    df: pd.DataFrame,
    crop: Optional[str] = None,
    top_n: int = 10
) -> go.Figure:
    """Horizontal bar chart showing top agricultural states for a given crop."""
    data = df if not crop or crop == "All Crops" else df[df["Crop"] == crop]
    state_totals = (
        data.groupby("State_Name")
        .agg(
            Total_Production=("Production", "sum"),
            Mean_Yield=("Yield", "mean")
        )
        .sort_values("Total_Production", ascending=False)
        .head(top_n)
        .reset_index()
    )

    fig = go.Figure(go.Bar(
        x=state_totals["Total_Production"],
        y=state_totals["State_Name"],
        orientation="h",
        marker={
            "color": state_totals["Mean_Yield"],
            "colorscale": [[0, "#C7D8C6"], [1, COLORS["primary"]]],
            "colorbar": {"title": "Avg Yield (t/ha)", "len": 0.8}
        },
        text=state_totals["Mean_Yield"].apply(lambda v: f"Avg: {v:.2f} t/ha"),
        textposition="auto"
    ))
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    fig.update_xaxes(title_text="Total Production (Tonnes)")
    title = f"Top {top_n} Producing States" + (f" for {crop}" if crop and crop != "All Crops" else "")
    return apply_agri_theme(fig, title)


def plot_actual_vs_predicted(
    y_true: pd.Series,
    y_pred: np.ndarray,
    sample_size: int = 1200
) -> go.Figure:
    """Scatter plot comparing actual vs predicted crop yields with 45-degree parity line."""
    if len(y_true) > sample_size:
        idx = np.random.RandomState(42).choice(len(y_true), size=sample_size, replace=False)
        actual = y_true.iloc[idx].values
        pred = y_pred[idx]
    else:
        actual = y_true.values
        pred = y_pred

    abs_error = np.abs(actual - pred)
    max_val = max(float(np.percentile(actual, 99)), float(np.percentile(pred, 99))) * 1.05

    fig = go.Figure()

    # 45-degree reference line
    fig.add_trace(go.Scatter(
        x=[0, max_val],
        y=[0, max_val],
        mode="lines",
        name="Perfect Prediction (y = x)",
        line={"color": COLORS["accent"], "dash": "dash", "width": 2}
    ))

    # Scatter points
    fig.add_trace(go.Scatter(
        x=actual,
        y=pred,
        mode="markers",
        name="Out-of-Time Test Predictions",
        marker={
            "size": 5,
            "color": abs_error,
            "colorscale": [[0, COLORS["secondary"]], [0.5, COLORS["accent"]], [1.0, COLORS["highlight_red"]]],
            "colorbar": {"title": "Abs Error (t/ha)", "len": 0.8},
            "opacity": 0.65
        },
        hovertemplate="Actual: %{x:.2f} t/ha<br>Predicted: %{y:.2f} t/ha<br>Abs Error: %{marker.color:.2f} t/ha"
    ))

    fig.update_xaxes(title_text="Actual Harvest Yield (Tonnes / Ha)", range=[0, max_val])
    fig.update_yaxes(title_text="Model Predicted Yield (Tonnes / Ha)", range=[0, max_val])

    return apply_agri_theme(fig, "Actual vs. Predicted Yield (Out-of-Time Test Set 2012–2015)")


def plot_residuals(
    y_true: pd.Series,
    y_pred: np.ndarray
) -> go.Figure:
    """Histogram and distribution of prediction residuals (Actual - Predicted)."""
    residuals = y_true.values - y_pred
    # Filter extreme visual outliers for histogram clarity
    p01, p99 = np.percentile(residuals, [1, 99])
    sub_res = residuals[(residuals >= p01) & (residuals <= p99)]

    fig = go.Figure(go.Histogram(
        x=sub_res,
        nbinsx=60,
        marker={"color": COLORS["secondary"], "line": {"color": "#FFFFFF", "width": 0.5}},
        opacity=0.85,
        name="Residuals"
    ))

    # Zero error vertical reference
    fig.add_vline(x=0, line_width=2, line_dash="dash", line_color=COLORS["accent"])

    fig.update_xaxes(title_text="Residual Error: (Actual - Predicted) [Tonnes / Ha]")
    fig.update_yaxes(title_text="Frequency")

    return apply_agri_theme(fig, "Residual Error Distribution (Check for Systemic Bias)")


def plot_model_comparison(benchmark_results: Dict[str, Dict[str, float]]) -> go.Figure:
    """Grouped bar chart comparing MAE, RMSE, and R2 across model candidates."""
    models = list(benchmark_results.keys())
    maes = [benchmark_results[m]["MAE"] for m in models]
    rmses = [benchmark_results[m]["RMSE"] for m in models]
    r2s = [benchmark_results[m]["R2"] for m in models]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=models,
        y=maes,
        name="MAE (Tonnes/Ha)",
        marker={"color": COLORS["secondary"]}
    ))
    fig.add_trace(go.Bar(
        x=models,
        y=rmses,
        name="RMSE (Tonnes/Ha)",
        marker={"color": COLORS["accent"]}
    ))
    fig.add_trace(go.Bar(
        x=models,
        y=r2s,
        name="R² Score",
        marker={"color": COLORS["primary"]}
    ))

    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Metric Score")
    return apply_agri_theme(fig, "Model Candidate Comparison (Lower MAE/RMSE, Higher R² is Better)")


def plot_data_quality_breakdown(summary: Dict[str, Any]) -> go.Figure:
    """Donut chart illustrating data pipeline audit & retention."""
    labels = ["Valid Supervised Records", "Flagged Yield Outliers", "Missing Production Records"]
    values = [
        summary.get("valid_yield_records", 242171),
        summary.get("outlier_yield_records_flagged", 190),
        summary.get("missing_production_count", 3730)
    ]
    colors = [COLORS["primary"], COLORS["accent"], COLORS["highlight_red"]]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        marker={"colors": colors},
        textinfo="label+percent",
        insidetextorientation="radial"
    )])
    return apply_agri_theme(fig, "Raw Dataset Audit: Record Integrity & Quality Partitioning")


# -----------------------------------------------------------------------------
# Dynamic Visualizations for Custom Uploaded Datasets
# -----------------------------------------------------------------------------
def plot_uploaded_category_bar(
    df: pd.DataFrame, 
    cat_col: str, 
    num_col: str, 
    agg: str = "sum", 
    top_n: int = 12
) -> go.Figure:
    """Horizontal/Vertical bar chart for top categories in uploaded data."""
    if df.empty or cat_col not in df.columns or num_col not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="Insufficient data for category chart", showarrow=False)
        return apply_agri_theme(fig, "Category Distribution")

    # Safe aggregation
    clean_sub = df.dropna(subset=[cat_col, num_col]).copy()
    clean_sub[num_col] = pd.to_numeric(clean_sub[num_col], errors="coerce")
    clean_sub = clean_sub.dropna(subset=[num_col])

    if agg == "mean":
        grouped = clean_sub.groupby(cat_col)[num_col].mean().reset_index()
        title_metric = f"Average {num_col}"
    elif agg == "count":
        grouped = clean_sub.groupby(cat_col)[num_col].count().reset_index()
        title_metric = f"Record Count"
    else:
        grouped = clean_sub.groupby(cat_col)[num_col].sum().reset_index()
        title_metric = f"Total {num_col}"

    grouped = grouped.sort_values(num_col, ascending=False).head(top_n)

    fig = go.Figure(go.Bar(
        x=grouped[cat_col],
        y=grouped[num_col],
        marker={
            "color": grouped[num_col],
            "colorscale": [[0, "#86EFAC"], [0.5, "#22C55E"], [1, "#14532D"]],
            "line": {"color": "#14532D", "width": 1}
        },
        text=grouped[num_col].apply(lambda v: f"{v:,.1f}" if abs(v) < 1000 else f"{v:,.0f}"),
        textposition="outside",
        hovertemplate=f"<b>%{{x}}</b><br>{title_metric}: %{{y:,.2f}}<extra></extra>"
    ))
    fig.update_layout(xaxis_tickangle=-30)
    fig.update_xaxes(title_text=cat_col)
    fig.update_yaxes(title_text=title_metric)
    return apply_agri_theme(fig, f"Top {len(grouped)} {cat_col} by {title_metric}", height=430)


def plot_uploaded_scatter(
    df: pd.DataFrame, 
    x_col: str, 
    y_col: str, 
    color_col: Optional[str] = None
) -> go.Figure:
    """Scatter plot illustrating correlation / distribution between two numeric features."""
    clean_sub = df.dropna(subset=[x_col, y_col]).copy()
    clean_sub[x_col] = pd.to_numeric(clean_sub[x_col], errors="coerce")
    clean_sub[y_col] = pd.to_numeric(clean_sub[y_col], errors="coerce")
    clean_sub = clean_sub.dropna(subset=[x_col, y_col])

    # Sample if very large
    if len(clean_sub) > 4000:
        clean_sub = clean_sub.sample(4000, random_state=42)

    fig = go.Figure()
    if color_col and color_col in clean_sub.columns:
        # Group by top categories
        top_cats = clean_sub[color_col].value_counts().head(6).index.tolist()
        for cat in top_cats:
            cat_data = clean_sub[clean_sub[color_col] == cat]
            fig.add_trace(go.Scatter(
                x=cat_data[x_col],
                y=cat_data[y_col],
                mode="markers",
                name=str(cat),
                marker={"size": 6, "opacity": 0.75}
            ))
    else:
        fig.add_trace(go.Scatter(
            x=clean_sub[x_col],
            y=clean_sub[y_col],
            mode="markers",
            marker={"size": 6, "color": "#16A34A", "opacity": 0.7, "line": {"width": 0.5, "color": "#14532D"}},
            name="Records"
        ))

    fig.update_xaxes(title_text=x_col)
    fig.update_yaxes(title_text=y_col)
    return apply_agri_theme(fig, f"Relationship: {x_col} vs {y_col}", height=430)


def plot_uploaded_distribution(
    df: pd.DataFrame, 
    num_col: str, 
    group_col: Optional[str] = None
) -> go.Figure:
    """Box plot or histogram of numeric distribution."""
    clean_sub = df.dropna(subset=[num_col]).copy()
    clean_sub[num_col] = pd.to_numeric(clean_sub[num_col], errors="coerce")
    clean_sub = clean_sub.dropna(subset=[num_col])

    fig = go.Figure()
    if group_col and group_col in clean_sub.columns:
        top_cats = clean_sub[group_col].value_counts().head(8).index.tolist()
        palette = ["#16A34A", "#2563EB", "#D97706", "#7C3AED", "#0891B2", "#E11D48", "#475569", "#059669"]
        for idx, cat in enumerate(top_cats):
            cat_data = clean_sub[clean_sub[group_col] == cat]
            fig.add_trace(go.Box(
                y=cat_data[num_col],
                name=str(cat),
                marker_color=palette[idx % len(palette)],
                boxpoints="outliers"
            ))
        title = f"{num_col} Distribution across Top {group_col}"
    else:
        fig.add_trace(go.Box(
            y=clean_sub[num_col],
            name=num_col,
            marker_color="#16A34A",
            boxpoints="outliers"
        ))
        title = f"Overall Distribution of {num_col}"

    fig.update_yaxes(title_text=num_col)
    return apply_agri_theme(fig, title, height=430)


def plot_uploaded_time_series(
    df: pd.DataFrame, 
    time_col: str, 
    num_col: str, 
    agg: str = "mean"
) -> go.Figure:
    """Chronological line plot for uploaded data."""
    clean_sub = df.dropna(subset=[time_col, num_col]).copy()
    clean_sub[num_col] = pd.to_numeric(clean_sub[num_col], errors="coerce")
    clean_sub = clean_sub.dropna(subset=[num_col])

    if agg == "sum":
        grouped = clean_sub.groupby(time_col)[num_col].sum().reset_index()
        title_metric = f"Total {num_col}"
    else:
        grouped = clean_sub.groupby(time_col)[num_col].mean().reset_index()
        title_metric = f"Average {num_col}"

    grouped = grouped.sort_values(time_col)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=grouped[time_col],
        y=grouped[num_col],
        mode="lines+markers",
        line={"color": "#16A34A", "width": 3},
        marker={"size": 8, "color": "#14532D"},
        name=title_metric
    ))
    fig.update_xaxes(title_text=time_col)
    fig.update_yaxes(title_text=title_metric)
    return apply_agri_theme(fig, f"{title_metric} Progression over {time_col}", height=430)

