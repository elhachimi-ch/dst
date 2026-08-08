import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns


class Chart:
    """
    A unified data-visualisation toolkit combining a stateful Seaborn/Plotly API
    with standalone scientific plotting helpers.

    Stateful workflow
    -----------------
    >>> c = Chart(df, column4x="x", chart_type="line")
    >>> c.add_data_to_show(data_column="y")
    >>> c.config(title="My Chart", y_label="Units")
    >>> c.show()
    >>> c.save("output.png")

    Standalone helpers
    ------------------
    >>> Chart().viz_timeseries(...)
    >>> Chart().xai_heat_maps(...)
    >>> Chart().scatter(...)
    """

    chart_type_list = [
        "line", "bar", "box", "swarm", "strip_swarm", "count",
        "scatter", "dist", "point", "pair", "correlation_map",
        "reg", "heat_map", "pie",
    ]

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        dataframe: Optional[pd.DataFrame] = None,
        column4x=None,
        chart_type: str = "pair",
        group_by: Optional[str] = None,
        columns_names_list: Optional[List[str]] = None,
        plotly: bool = False,
        results_monitoring_folder: Optional[Union[str, Path]] = None,
    ):
        self.dataframe = dataframe
        self.column4x = dataframe.index if (column4x is None and dataframe is not None) else column4x
        self.chart_type = chart_type
        self.group_by = group_by
        self.columns_names_list = columns_names_list
        self.plotly = plotly
        self.results_monitoring_folder = (
            Path(results_monitoring_folder) if results_monitoring_folder else Path(".")
        )
        self.ax = None
        self.fig = None
        sns.set_theme(color_codes=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _save_figure(fig: plt.Figure, path: Union[str, Path], dpi: int = 300) -> Path:
        """Save *fig* to *path*, creating parent directories as needed."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {out}")
        return out

    @staticmethod
    def _annotate_heatmap(
        ax: plt.Axes,
        data: np.ndarray,
        fmt: str = "{:.0f}",
        fontsize: int = 8,
        color: str = "black",
        fontfamily: str = "DejaVu Sans",
    ):
        """Write *fmt*-formatted text in every cell of an imshow heatmap."""
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(
                    j, i, fmt.format(data[i, j]),
                    ha="center", va="center",
                    fontsize=fontsize, color=color, fontfamily=fontfamily,
                )

    # ------------------------------------------------------------------
    # Stateful API
    # ------------------------------------------------------------------

    def add_data_to_show(
        self,
        data_column=None,
        column4hover=None,
        column4size=None,
        y_column=None,
        color=None,
        labels=None,
    ):
        ct = self.chart_type
        df = self.dataframe

        if self.plotly:
            if ct == "line":
                self.fig = px.line(df, x=self.column4x, y=data_column,
                                   color=self.group_by, hover_name=column4hover)
            elif ct in ("scatter", "reg"):
                self.fig = px.scatter(df, x=self.column4x, y=data_column,
                                      color=self.group_by, size=column4size,
                                      hover_name=column4hover)
            elif ct == "pair":
                self.fig = px.scatter_matrix(
                    df, dimensions=self.columns_names_list, color=self.group_by
                )
            elif ct == "bar":
                self.fig = px.bar(df, x=self.column4x, y=data_column, color=self.group_by)
            elif ct == "pie":
                self.fig = px.pie(df, values=data_column, names=labels)
            else:
                raise ValueError(f"chart_type '{ct}' is not supported in Plotly mode.")
        else:
            if ct == "line":
                self.ax = sns.lineplot(
                    data=df, x=self.column4x, y=data_column,
                    markers=True, hue=self.group_by,
                )
            elif ct == "bar":
                sns.set_theme(style="whitegrid")
                self.ax = sns.barplot(
                    data=df, x=self.column4x, y=data_column, hue=self.group_by
                )
                self.ax.xaxis.set_major_locator(plticker.MultipleLocator(base=1.0))
            elif ct == "box":
                self.ax = sns.boxplot(
                    data=df, x=self.column4x, y=data_column, hue=self.group_by
                )
            elif ct == "swarm":
                self.ax = sns.swarmplot(
                    data=df, x=self.column4x, y=data_column, hue=self.group_by
                )
            elif ct == "strip_swarm":
                self.ax = sns.stripplot(
                    data=df, x=self.column4x, y=data_column, hue=self.group_by
                )
            elif ct == "count":
                self.ax = sns.countplot(data=df, x=self.column4x, hue=self.group_by)
            elif ct == "scatter":
                self.ax = sns.scatterplot(
                    data=df, x=self.column4x, y=data_column, hue=self.group_by
                )
            elif ct == "reg":
                # sns.regplot does not support hue; use lmplot for grouped regression
                self.ax = sns.regplot(data=df, x=self.column4x, y=data_column)
            elif ct == "dist":
                self.ax = sns.histplot(
                    df[data_column], kde=True,
                    hue=self.group_by if self.group_by else None,
                )
            elif ct == "point":
                self.ax = sns.pointplot(
                    data=df, x=self.column4x, y=data_column, hue=self.group_by
                )
            elif ct == "pair":
                self.ax = sns.pairplot(df, hue=self.group_by, vars=self.columns_names_list)
            elif ct == "correlation_map":
                corr = df.select_dtypes(include="number").corr()
                mask = np.triu(np.ones_like(corr, dtype=bool))
                cmap = sns.diverging_palette(230, 20, as_cmap=True)
                self.ax = sns.heatmap(
                    corr, mask=mask, cmap=cmap, center=0,
                    square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, annot=True,
                )
            elif ct == "heat_map":
                corr = df.select_dtypes(include="number").corr()
                self.ax = sns.heatmap(corr, annot=True, center=0.0)
            elif ct == "pie":
                colors = sns.color_palette("pastel")
                self.ax = plt.pie(data_column, labels=labels, colors=colors, autopct="%.0f%%")
            else:
                raise ValueError(
                    f"Unsupported chart_type: '{ct}'. Choose from {self.chart_type_list}"
                )

        return self.fig if self.plotly else self.ax

    def plot_on_map(
        self,
        iso_locations_column: Optional[str] = None,
        circle_size_column: Optional[str] = None,
        animation_frame_column: Optional[str] = None,
        hover_name_column: Optional[str] = None,
        projection: str = "natural earth",
        scope: str = "world",
    ):
        self.fig = px.scatter_geo(
            self.dataframe,
            locations=iso_locations_column,
            size=circle_size_column,
            animation_frame=animation_frame_column,
            hover_name=hover_name_column,
            color=self.group_by,
            projection=projection,
            scope=scope,
        )
        return self.fig

    def plot_colored_map(
        self,
        iso_locations_column: Optional[str] = None,
        color_column: Optional[str] = None,
        animation_frame_column: str = "Year",
        scope: str = "world",
        hover_name_column: Optional[str] = None,
    ):
        self.fig = px.choropleth(
            self.dataframe,
            locations=iso_locations_column,
            scope=scope,
            color=color_column,
            hover_name=hover_name_column,
            color_continuous_scale=px.colors.sequential.Plasma,
            animation_frame=animation_frame_column,
            projection="natural earth",
        )
        return self.fig

    def show(self):
        if self.plotly:
            self.fig.show()
        else:
            plt.show()

    def config(
        self,
        title: str = "",
        x_label: str = "X",
        y_label: str = "Y",
        x_limit_i=None,
        x_limit_f=None,
        y_limit_i=None,
        y_limit_f=None,
        interval: Optional[float] = None,
        x_rotation_angle: int = 90,
        y_rotation_angle: int = 0,
        title_font_size: int = 29,
        x_label_font_size: int = 13,
        y_label_font_size: int = 13,
        x_font_size: int = 11,
        y_font_size: int = 11,
    ):
        if self.plotly:
            self.fig.update_layout(title_text=title)
        else:
            ax = self.ax
            if ax is None:
                raise RuntimeError("No chart to configure — call add_data_to_show() first.")
            if isinstance(ax, plt.Axes):
                ax.set_title(title, fontsize=title_font_size)
                ax.set_xlabel(x_label, fontsize=x_label_font_size)
                ax.set_ylabel(y_label, fontsize=y_label_font_size)
                plt.xlim(x_limit_i, x_limit_f)
                plt.ylim(y_limit_i, y_limit_f)
                plt.xticks(rotation=x_rotation_angle, fontsize=x_font_size)
                plt.yticks(rotation=y_rotation_angle, fontsize=y_font_size)
                if interval is not None:
                    ax.xaxis.set_major_locator(plticker.MultipleLocator(base=interval))
            else:
                # Grid types (PairGrid, ClusterGrid, etc.) — set suptitle at minimum
                try:
                    ax.figure.suptitle(title, fontsize=title_font_size)
                except AttributeError:
                    pass

    def save(self, chart_path: str = "output.png", transparent: bool = False, dpi: int = 300):
        """Save the current figure to *chart_path*."""
        if self.plotly:
            if hasattr(self.fig, "write_image"):
                self.fig.write_image(chart_path)
            else:
                raise RuntimeError(
                    "Plotly figure does not support write_image — install kaleido: pip install kaleido"
                )
        else:
            if self.ax is None:
                raise RuntimeError("No figure to save — call add_data_to_show() first.")
            figure = (
                self.ax.get_figure()
                if isinstance(self.ax, plt.Axes)
                else self.ax.figure
            )
            figure.savefig(chart_path, transparent=transparent, bbox_inches="tight", dpi=dpi)

    # ------------------------------------------------------------------
    # Standalone scientific plotting helpers
    # ------------------------------------------------------------------

    def scatter(
        self,
        x: Union[pd.Series, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        *,
        color: str = "#1f77b4",
        alpha: float = 0.6,
        marker_size: float = 20,
        add_regression: bool = True,
        add_1to1: bool = False,
        add_metrics: bool = True,
        xlabel: str = "Observed",
        ylabel: str = "Predicted",
        title: str = "Scatter Plot",
        figsize: Tuple[float, float] = (6, 6),
        dpi: int = 150,
        output_path: Optional[str] = None,
        show: bool = False,
    ) -> plt.Figure:
        """
        Scatter plot with optional 1:1 line, OLS regression line and performance metrics.

        Parameters
        ----------
        x, y : array-like
            Observed and predicted (or any two series).
        add_regression : bool
            Overlay OLS regression line.
        add_1to1 : bool
            Overlay the perfect 1:1 identity line.
        add_metrics : bool
            Annotate R², RMSE, and MAE inside the plot.
        output_path : str, optional
            If given, save the figure to this path.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.scatter(x, y, s=marker_size, color=color, alpha=alpha)

        xy_min = min(x.min(), y.min())
        xy_max = max(x.max(), y.max())

        if add_1to1:
            ax.plot([xy_min, xy_max], [xy_min, xy_max], "k--", lw=1.2, label="1:1")
        if add_regression:
            m, b = np.polyfit(x, y, 1)
            xs = np.linspace(xy_min, xy_max, 200)
            ax.plot(xs, m * xs + b, color="crimson", lw=1.5, label=f"OLS (slope={m:.2f})")

        if add_metrics:
            r2 = float(np.corrcoef(x, y)[0, 1] ** 2)
            rmse = float(np.sqrt(np.mean((x - y) ** 2)))
            mae = float(np.mean(np.abs(x - y)))
            ax.text(
                0.05, 0.95,
                f"R²={r2:.3f}\nRMSE={rmse:.3f}\nMAE={mae:.3f}",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
            )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if add_1to1 or add_regression:
            ax.legend(fontsize=8)
        fig.tight_layout()

        if output_path:
            self._save_figure(fig, output_path, dpi=dpi)
        if show:
            plt.show()
        plt.close(fig)
        return fig

    def residuals_plot(
        self,
        observed: Union[pd.Series, np.ndarray],
        predicted: Union[pd.Series, np.ndarray],
        *,
        color: str = "#2ca02c",
        alpha: float = 0.5,
        marker_size: float = 15,
        xlabel: str = "Predicted",
        ylabel: str = "Residual",
        title: str = "Residuals vs. Predicted",
        figsize: Tuple[float, float] = (7, 4),
        dpi: int = 150,
        output_path: Optional[str] = None,
        show: bool = False,
    ) -> plt.Figure:
        """
        Residuals (observed − predicted) vs. predicted scatter plot.

        A horizontal zero line and, if statsmodels is available, a LOWESS
        trend line are added automatically.
        """
        obs = np.asarray(observed, dtype=float)
        pred = np.asarray(predicted, dtype=float)
        mask = np.isfinite(obs) & np.isfinite(pred)
        pred, obs = pred[mask], obs[mask]
        resid = obs - pred

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.scatter(pred, resid, s=marker_size, color=color, alpha=alpha)
        ax.axhline(0, color="black", linewidth=1.2, linestyle="--")

        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess
            order = np.argsort(pred)
            smoothed = lowess(resid[order], pred[order], frac=0.4)
            ax.plot(smoothed[:, 0], smoothed[:, 1], color="crimson", lw=1.5, label="Trend")
            ax.legend(fontsize=8)
        except ImportError:
            pass

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        fig.tight_layout()

        if output_path:
            self._save_figure(fig, output_path, dpi=dpi)
        if show:
            plt.show()
        plt.close(fig)
        return fig

    def histogram(
        self,
        data: Union[pd.Series, np.ndarray, pd.DataFrame],
        *,
        columns: Optional[List[str]] = None,
        bins: Union[int, str] = "auto",
        kde: bool = True,
        color: Optional[str] = None,
        xlabel: str = "Value",
        ylabel: str = "Count",
        title: str = "Distribution",
        figsize: Tuple[float, float] = (7, 4),
        dpi: int = 150,
        output_path: Optional[str] = None,
        show: bool = False,
    ) -> plt.Figure:
        """
        Histogram with optional KDE overlay.

        Accepts a Series/1-D array (single histogram) or a DataFrame
        (one overlapping histogram per numeric column).
        """
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        if isinstance(data, pd.DataFrame):
            cols = columns or list(data.select_dtypes(include="number").columns)
            for col in cols:
                sns.histplot(data[col].dropna(), kde=kde, bins=bins, label=col, ax=ax, alpha=0.6)
            ax.legend()
        else:
            s = pd.Series(np.asarray(data, dtype=float)).dropna()
            sns.histplot(s, kde=kde, bins=bins, color=color, ax=ax)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        fig.tight_layout()

        if output_path:
            self._save_figure(fig, output_path, dpi=dpi)
        if show:
            plt.show()
        plt.close(fig)
        return fig

    def violin(
        self,
        data: pd.DataFrame,
        x: str,
        y: str,
        *,
        hue: Optional[str] = None,
        split: bool = False,
        palette: str = "Set2",
        inner: str = "box",
        title: str = "Violin Plot",
        xlabel: str = "",
        ylabel: str = "",
        figsize: Tuple[float, float] = (9, 5),
        dpi: int = 150,
        output_path: Optional[str] = None,
        show: bool = False,
    ) -> plt.Figure:
        """Violin plot wrapping seaborn.violinplot with a consistent save/show API."""
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        sns.violinplot(data=data, x=x, y=y, hue=hue, split=split,
                       palette=palette, inner=inner, ax=ax)
        ax.set_title(title)
        ax.set_xlabel(xlabel or x)
        ax.set_ylabel(ylabel or y)
        fig.tight_layout()

        if output_path:
            self._save_figure(fig, output_path, dpi=dpi)
        if show:
            plt.show()
        plt.close(fig)
        return fig

    def feature_importance(
        self,
        importances: Union[pd.Series, Dict[str, float]],
        *,
        top_n: Optional[int] = None,
        color: str = "#1f77b4",
        title: str = "Feature Importance",
        xlabel: str = "Importance",
        figsize: Tuple[float, float] = (8, 5),
        dpi: int = 150,
        output_path: Optional[str] = None,
        show: bool = False,
    ) -> plt.Figure:
        """
        Horizontal bar chart of feature importances, sorted descending.

        Parameters
        ----------
        importances : pd.Series or dict
            ``{feature_name: importance_value}`` mapping.
        top_n : int, optional
            Show only the top-N most important features.
        """
        if isinstance(importances, dict):
            importances = pd.Series(importances)
        importances = importances.sort_values(ascending=True)
        if top_n is not None:
            importances = importances.iloc[-top_n:]

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.barh(importances.index, importances.values, color=color)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.margins(y=0.02)
        fig.tight_layout()

        if output_path:
            self._save_figure(fig, output_path, dpi=dpi)
        if show:
            plt.show()
        plt.close(fig)
        return fig

    def confusion_matrix(
        self,
        cm: np.ndarray,
        class_labels: Optional[List[str]] = None,
        *,
        normalize: bool = False,
        cmap: str = "Blues",
        title: str = "Confusion Matrix",
        figsize: Tuple[float, float] = (6, 5),
        dpi: int = 150,
        output_path: Optional[str] = None,
        show: bool = False,
    ) -> plt.Figure:
        """
        Plot a confusion matrix heatmap.

        Parameters
        ----------
        cm : np.ndarray, shape (n_classes, n_classes)
            Confusion matrix (rows = true, cols = predicted).
        class_labels : list of str, optional
            Labels for each class.
        normalize : bool
            Row-normalise the matrix before plotting (values become proportions).
        """
        cm = np.asarray(cm, dtype=float)
        if normalize:
            row_sums = cm.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums == 0, 1, row_sums)
            cm = cm / row_sums
            fmt, cbar_label = ".2f", "Proportion"
        else:
            fmt, cbar_label = ".0f", "Count"

        n = cm.shape[0]
        labels = class_labels or [str(i) for i in range(n)]

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        im = ax.imshow(cm, cmap=cmap, aspect="auto")
        fig.colorbar(im, ax=ax, shrink=0.85, label=cbar_label)

        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title)

        thresh = cm.max() / 2.0
        for i in range(n):
            for j in range(n):
                ax.text(
                    j, i, f"{cm[i, j]:{fmt}}",
                    ha="center", va="center", fontsize=9,
                    color="white" if cm[i, j] > thresh else "black",
                )

        fig.tight_layout()

        if output_path:
            self._save_figure(fig, output_path, dpi=dpi)
        if show:
            plt.show()
        plt.close(fig)
        return fig

    def learning_curves(
        self,
        train_scores: Union[List[float], np.ndarray],
        val_scores: Union[List[float], np.ndarray],
        *,
        metric_name: str = "Loss",
        train_label: str = "Train",
        val_label: str = "Validation",
        log_scale: bool = False,
        colors: Tuple[str, str] = ("#1f77b4", "#d62728"),
        figsize: Tuple[float, float] = (8, 4),
        dpi: int = 150,
        output_path: Optional[str] = None,
        show: bool = False,
    ) -> plt.Figure:
        """
        Plot training and validation learning curves (e.g., loss/accuracy per epoch).

        Parameters
        ----------
        train_scores, val_scores : array-like
            Per-epoch metric values.
        log_scale : bool
            Use a log-scale y-axis (useful for loss curves).
        """
        epochs = np.arange(1, len(train_scores) + 1)
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.plot(epochs, train_scores, color=colors[0], lw=2, label=train_label)
        ax.plot(epochs, val_scores, color=colors[1], lw=2, linestyle="--", label=val_label)
        if log_scale:
            ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric_name)
        ax.set_title(f"Learning Curves — {metric_name}")
        ax.legend()
        ax.margins(x=0)
        fig.tight_layout()

        if output_path:
            self._save_figure(fig, output_path, dpi=dpi)
        if show:
            plt.show()
        plt.close(fig)
        return fig

    def taylor_diagram(
        self,
        references: Union[float, List[float]],
        stds: List[float],
        correlations: List[float],
        labels: List[str],
        *,
        ref_label: str = "Reference",
        title: str = "Taylor Diagram",
        normalize: bool = True,
        figsize: Tuple[float, float] = (7, 6),
        dpi: int = 150,
        markers: Optional[List[str]] = None,
        colors: Optional[List[str]] = None,
        output_path: Optional[str] = None,
        show: bool = False,
    ) -> plt.Figure:
        """
        Taylor Diagram for comparing multiple model outputs against a reference.

        Each model is a point in polar coordinates:
          - Radial distance = (normalised) standard deviation
          - Angle          = arccos(correlation)

        Parameters
        ----------
        references : float or list of float
            Reference standard deviation(s). A scalar is broadcast to all models.
        stds : list of float
            Standard deviations of each model.
        correlations : list of float
            Pearson correlations with the reference.
        labels : list of str
            Name for each model point.
        normalize : bool
            Normalise all stds by the mean reference std (reference plots at radius 1).
        """
        if isinstance(references, (int, float)):
            references = [float(references)] * len(stds)
        ref_std = float(np.mean(references))
        norm_stds = [s / ref_std for s in stds] if normalize else list(stds)
        ref_radius = 1.0 if normalize else ref_std
        angles = [np.arccos(max(-1.0, min(1.0, c))) for c in correlations]

        _colors = colors or plt.rcParams["axes.prop_cycle"].by_key()["color"]
        _markers = markers or ["o", "s", "D", "^", "v", "<", ">", "p", "*", "h"]

        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111, polar=True)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_thetamin(0)
        ax.set_thetamax(90)

        # Reference arc
        theta_arc = np.linspace(0, np.pi / 2, 200)
        ax.plot(theta_arc, [ref_radius] * 200, "k--", lw=1, label=ref_label)
        ax.plot([0], [ref_radius], "k*", ms=10, label=f"{ref_label} (std={ref_std:.2f})")

        # Model points
        for i, (a, r, lbl) in enumerate(zip(angles, norm_stds, labels)):
            ax.scatter(
                a, r, s=70,
                color=_colors[i % len(_colors)],
                marker=_markers[i % len(_markers)],
                zorder=5, label=lbl,
            )

        corr_ticks = [0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
        ax.set_xticks([np.arccos(c) for c in corr_ticks])
        ax.set_xticklabels([str(c) for c in corr_ticks], fontsize=8)
        ax.set_ylabel("Normalised Std" if normalize else "Std", labelpad=30)
        ax.set_title(title, pad=20)
        ax.legend(loc="lower left", bbox_to_anchor=(1.05, 0), fontsize=8, frameon=False)
        fig.tight_layout()

        if output_path:
            self._save_figure(fig, output_path, dpi=dpi)
        if show:
            plt.show()
        plt.close(fig)
        return fig

    # ------------------------------------------------------------------
    # Standalone domain-specific methods
    # ------------------------------------------------------------------

    def xai_heat_maps(
        self,
        input_root: str,
        output_dir: Optional[str] = None,
        ipcc_regions: Optional[Sequence[str]] = None,
        feature_name_map: Optional[Dict[str, str]] = None,
        shap_prefix: str = "mean_abs_shap_",
        fi_prefix: str = "model_feature_importance_",
        xgb_folder_name: str = "xgboost",
        cat_folder_name: str = "catboost",
        dpi: int = 300,
        figsize_main: Tuple[int, int] = (16, 11),
        cmap_rank: str = "Blues_r",
        cmap_agreement: str = "RdBu_r",
        rank_font_family: str = "DejaVu Sans",
        rank_font_size: int = 8,
        rank_font_color: str = "black",
        x_y_labels_font_size: int = 10,
        plot_title: bool = True,
        show: bool = False,
    ) -> Dict:
        """
        Generate weight-coloured rank heatmaps for XGBoost and CatBoost XAI results.

        Produces 4 importance heatmaps (2 models × SHAP / Feature-Importance) and
        2 rank-agreement heatmaps, each saved as a separate PNG.

        Expected directory layout
        -------------------------
        input_root/
          xgboost/
            mean_abs_shap_<region>.csv          columns: feature, mean_abs_shap
            model_feature_importance_<region>.csv  columns: feature, model_importance
          catboost/
            ...same...

        Returns
        -------
        dict
            All rank and weight DataFrames, keyed by descriptive name.
        """
        _DEFAULT_REGIONS = ["caf", "neaf", "waf", "seaf", "med", "sah", "esaf", "wsaf", "mdg"]
        _DEFAULT_FEATURE_MAP = {
            "ws_mean_aligned_ws_mean": "Ws",
            "rs_sum_mj_aligned_rs_sum_mj": "Rs",
            "ta_max_celsius_aligned_ta_max_celsius": "Ta_max",
            "ta_min_celsius_aligned_ta_min_celsius": "Ta_min",
            "rh_min_aligned_rh_min": "RH_min",
            "rh_max_aligned_rh_max": "RH_max",
        }
        # Must match the values in _DEFAULT_FEATURE_MAP
        _PREFERRED_ORDER = ["Rs", "Ws", "Ta_max", "Ta_min", "RH_min", "RH_max"]

        if ipcc_regions is None:
            ipcc_regions = _DEFAULT_REGIONS
        if feature_name_map is None:
            feature_name_map = _DEFAULT_FEATURE_MAP

        input_root = Path(input_root)
        xgb_dir = input_root / xgb_folder_name
        cat_dir = input_root / cat_folder_name
        _out_dir = (
            Path(output_dir) if output_dir else self.results_monitoring_folder / "xai_heatmaps"
        )
        _out_dir.mkdir(parents=True, exist_ok=True)
        figsize_single = (figsize_main[0] / 2, figsize_main[1] / 2)

        # ---------- CSV loaders ---------- #

        def _rename(feat: str) -> str:
            return feature_name_map.get(feat, feat)

        def _load_shap(path: Path) -> pd.Series:
            if not path.exists():
                raise FileNotFoundError(f"Missing: {path}")
            df = pd.read_csv(path)
            missing = {"feature", "mean_abs_shap"} - set(df.columns)
            if missing:
                raise ValueError(f"{path} is missing columns: {missing}")
            df["feature"] = df["feature"].astype(str).map(_rename)
            return pd.Series(df["mean_abs_shap"].values, index=df["feature"].values, dtype=float)

        def _load_fi(path: Path) -> pd.Series:
            if not path.exists():
                raise FileNotFoundError(f"Missing: {path}")
            df = pd.read_csv(path)
            missing = {"feature", "model_importance"} - set(df.columns)
            if missing:
                raise ValueError(f"{path} is missing columns: {missing}")
            df["feature"] = df["feature"].astype(str).map(_rename)
            return pd.Series(df["model_importance"].values, index=df["feature"].values, dtype=float)

        # ---------- Feature discovery ---------- #

        def _discover_features(model_dir: Path) -> set:
            feats: set = set()
            for region in ipcc_regions:
                for loader, prefix in [(_load_shap, shap_prefix), (_load_fi, fi_prefix)]:
                    p = model_dir / f"{prefix}{region}.csv"
                    if p.exists():
                        feats.update(loader(p).index.tolist())
            return feats

        all_feats = sorted(_discover_features(xgb_dir) | _discover_features(cat_dir))
        if not all_feats:
            raise RuntimeError("No features found in XGBoost/CatBoost folders.")
        all_feats = (
            [f for f in _PREFERRED_ORDER if f in all_feats]
            + [f for f in all_feats if f not in _PREFERRED_ORDER]
        )

        # ---------- Table builders (single pass per model) ---------- #

        regions_upper = [r.upper() for r in ipcc_regions]

        def _rank(s: pd.Series) -> pd.Series:
            return (
                s.reindex(all_feats)
                .fillna(-np.inf)
                .rank(method="dense", ascending=False)
                .astype(int)
            )

        def _build_tables(model_dir: Path):
            shap_rank = pd.DataFrame(index=regions_upper, columns=all_feats, dtype=float)
            fi_rank   = pd.DataFrame(index=regions_upper, columns=all_feats, dtype=float)
            shap_wt   = pd.DataFrame(index=regions_upper, columns=all_feats, dtype=float)
            fi_wt     = pd.DataFrame(index=regions_upper, columns=all_feats, dtype=float)

            for region in ipcc_regions:
                ru = region.upper()
                shap_s = _load_shap(model_dir / f"{shap_prefix}{region}.csv")
                fi_s   = _load_fi(model_dir / f"{fi_prefix}{region}.csv")

                shap_rank.loc[ru] = _rank(shap_s).values
                fi_rank.loc[ru]   = _rank(fi_s).values
                shap_wt.loc[ru]   = shap_s.reindex(all_feats).fillna(0.0).values
                fi_wt.loc[ru]     = fi_s.reindex(all_feats).fillna(0.0).values

            return (
                shap_rank.astype(int), fi_rank.astype(int),
                shap_wt.astype(float), fi_wt.astype(float),
            )

        xgb_shap_rank, xgb_fi_rank, xgb_shap_wt, xgb_fi_wt = _build_tables(xgb_dir)
        cat_shap_rank, cat_fi_rank, cat_shap_wt, cat_fi_wt = _build_tables(cat_dir)
        xgb_agreement = xgb_fi_rank - xgb_shap_rank
        cat_agreement = cat_fi_rank - cat_shap_rank

        # ---------- Save CSVs ---------- #

        for name, df in [
            ("xgboost_shap_rank",               xgb_shap_rank),
            ("xgboost_model_importance_rank",    xgb_fi_rank),
            ("catboost_shap_rank",               cat_shap_rank),
            ("catboost_model_importance_rank",   cat_fi_rank),
            ("xgboost_rank_agreement",           xgb_agreement),
            ("catboost_rank_agreement",          cat_agreement),
            ("xgboost_shap_weights",             xgb_shap_wt),
            ("xgboost_model_importance_weights", xgb_fi_wt),
            ("catboost_shap_weights",            cat_shap_wt),
            ("catboost_model_importance_weights", cat_fi_wt),
        ]:
            df.to_csv(_out_dir / f"{name}.csv")

        # ---------- Plotting ---------- #

        def _plot_weight_heatmap(
            wt_df: pd.DataFrame, rank_df: pd.DataFrame,
            title: str, filename: str, cbar_label: str,
        ):
            fig, ax = plt.subplots(figsize=figsize_single, dpi=dpi)
            vals = wt_df.values.astype(float)
            vmin, vmax = float(vals.min()), float(vals.max())
            if vmin == vmax:
                vmax = vmin + 1.0
            im = ax.imshow(vals, aspect="auto", cmap=cmap_rank, vmin=vmin, vmax=vmax)
            if plot_title:
                ax.set_title(title, fontsize=12, pad=8)
            ax.set_xticks(np.arange(wt_df.shape[1]))
            ax.set_xticklabels(wt_df.columns, rotation=45, ha="right",
                                fontsize=x_y_labels_font_size, fontfamily=rank_font_family)
            ax.set_yticks(np.arange(wt_df.shape[0]))
            ax.set_yticklabels(wt_df.index, fontsize=x_y_labels_font_size,
                                fontfamily=rank_font_family)
            self._annotate_heatmap(
                ax, rank_df.values.astype(float), fmt="{:.0f}",
                fontsize=rank_font_size, color=rank_font_color, fontfamily=rank_font_family,
            )
            ax.set_xticks(np.arange(-0.5, wt_df.shape[1], 1), minor=True)
            ax.set_yticks(np.arange(-0.5, wt_df.shape[0], 1), minor=True)
            ax.grid(which="minor", color="white", linestyle="-", linewidth=1)
            ax.tick_params(which="minor", bottom=False, left=False)
            fig.colorbar(im, ax=ax, shrink=0.86, pad=0.02, label=cbar_label)
            fig.tight_layout()
            self._save_figure(fig, _out_dir / filename, dpi=dpi)
            if show:
                plt.show()
            plt.close(fig)

        def _plot_agreement_heatmap(df: pd.DataFrame, title: str, filename: str):
            fig, ax = plt.subplots(figsize=figsize_single, dpi=dpi)
            vmax = max(int(np.abs(df.values).max()), 1)
            im = ax.imshow(df.values, aspect="auto", cmap=cmap_agreement, vmin=-vmax, vmax=vmax)
            if plot_title:
                ax.set_title(title, fontsize=12, pad=8)
            ax.set_xticks(np.arange(df.shape[1]))
            ax.set_xticklabels(df.columns, rotation=45, ha="right",
                                fontsize=x_y_labels_font_size, fontfamily=rank_font_family)
            ax.set_yticks(np.arange(df.shape[0]))
            ax.set_yticklabels(df.index, fontsize=x_y_labels_font_size,
                                fontfamily=rank_font_family)
            self._annotate_heatmap(
                ax, df.values.astype(float), fmt="{:.0f}",
                fontsize=rank_font_size, color=rank_font_color, fontfamily=rank_font_family,
            )
            ax.set_xticks(np.arange(-0.5, df.shape[1], 1), minor=True)
            ax.set_yticks(np.arange(-0.5, df.shape[0], 1), minor=True)
            ax.grid(which="minor", color="white", linestyle="-", linewidth=1)
            ax.tick_params(which="minor", bottom=False, left=False)
            fig.colorbar(im, ax=ax, shrink=0.9, pad=0.02, label="Rank difference")
            fig.tight_layout()
            self._save_figure(fig, _out_dir / filename, dpi=dpi)
            if show:
                plt.show()
            plt.close(fig)

        _plot_weight_heatmap(xgb_shap_wt, xgb_shap_rank, "XGBoost SHAP importance",
                              "xgboost_shap_heatmap.png", "SHAP importance")
        _plot_weight_heatmap(xgb_fi_wt,   xgb_fi_rank,   "XGBoost feature importance",
                              "xgboost_fi_heatmap.png",   "Feature importance")
        _plot_weight_heatmap(cat_shap_wt, cat_shap_rank, "CatBoost SHAP importance",
                              "catboost_shap_heatmap.png", "SHAP importance")
        _plot_weight_heatmap(cat_fi_wt,   cat_fi_rank,   "CatBoost feature importance",
                              "catboost_fi_heatmap.png",   "Feature importance")
        _plot_agreement_heatmap(
            xgb_agreement,
            "XGBoost rank agreement\n(feature importance − SHAP)",
            "xgboost_rank_agreement_heatmap.png",
        )
        _plot_agreement_heatmap(
            cat_agreement,
            "CatBoost rank agreement\n(feature importance − SHAP)",
            "catboost_rank_agreement_heatmap.png",
        )

        return {
            "xgb_shap_rank":      xgb_shap_rank,
            "xgb_fi_rank":        xgb_fi_rank,
            "cat_shap_rank":      cat_shap_rank,
            "cat_fi_rank":        cat_fi_rank,
            "xgb_shap_weights":   xgb_shap_wt,
            "xgb_fi_weights":     xgb_fi_wt,
            "cat_shap_weights":   cat_shap_wt,
            "cat_fi_weights":     cat_fi_wt,
            "xgb_rank_agreement": xgb_agreement,
            "cat_rank_agreement": cat_agreement,
        }

    def viz_timeseries(
        self,
        input_folder: str,
        reference_timeseries_path: str,
        *,
        output_dir: Optional[str] = None,
        output_filename: str = "timeseries.png",
        save: bool = True,
        dpi_save: int = 600,
        show: bool = False,
        date_col: str = "date",
        value_col: str = "value",
        ref_date_col: Optional[str] = None,
        ref_value_col: Optional[str] = None,
        ref_label: str = "Reference",
        value_min: float = 0.0,
        value_max: float = 20.0,
        use_smoothing: bool = True,
        smooth_method: str = "sg",
        smooth_window: int = 7,
        show_raw_data: bool = True,
        raw_opacity: float = 0.18,
        raw_point_size: float = 6.0,
        raw_label: str = "Reference (raw)",
        figsize: Tuple[float, float] = (12, 5),
        dpi: int = 200,
        title: str = "Daily Time Series",
        y_label: str = "Value",
        x_label: str = "Date",
        date_format: str = "%b",
        month_interval: int = 1,
        show_grid: bool = False,
        tight_layout_rect: Tuple[float, float, float, float] = (0, 0.10, 1, 1),
        ref_color: str = "black",
        ref_linewidth: float = 2.6,
        series_linewidth: float = 1.8,
        colors: Optional[List[str]] = None,
        linestyles: Optional[List[str]] = None,
        legend_title: str = "Series",
        legend_ncol: int = 3,
        legend_loc: str = "upper center",
        legend_bbox_to_anchor: Tuple[float, float] = (0.5, -0.14),
        legend_frameon: bool = False,
    ) -> Optional[str]:
        """
        Plot multiple time-series CSV files against a reference time series.

        Parameters
        ----------
        input_folder : str
            Folder containing CSV files; each file becomes one comparison series
            (filename stem used as the label).
        reference_timeseries_path : str
            Path to the reference/benchmark CSV.

        Returns
        -------
        str or None
            Absolute path of the saved figure, or ``None`` if ``save=False``.
        """
        _default_colors = [
            "#1f77b4", "#2ca02c", "#d62728", "#ff7f0e",
            "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
            "#bcbd22", "#17becf",
        ]
        ref_date_col = ref_date_col or date_col
        ref_value_col = ref_value_col or value_col
        color_cycle = colors or _default_colors
        ls_cycle = linestyles or ["--"]

        def _smooth(series: pd.Series) -> pd.Series:
            s = series.clip(lower=value_min, upper=value_max).copy()
            if not use_smoothing or smooth_method.lower() == "no" or smooth_window < 3:
                return s
            method = smooth_method.lower()
            if method == "median":
                return s.rolling(smooth_window, center=True,
                                 min_periods=max(3, smooth_window // 2)).median()
            if method == "sg":
                from scipy.signal import savgol_filter
                win = smooth_window if smooth_window % 2 == 1 else smooth_window + 1
                return pd.Series(
                    savgol_filter(s.interpolate("linear"), win, 2), index=s.index
                )
            return s.rolling(smooth_window, center=True,
                             min_periods=max(3, smooth_window // 2)).mean()

        def _load(path: Path, dc: str, vc: str) -> pd.Series:
            df = pd.read_csv(path)
            for col in (dc, vc):
                if col not in df.columns:
                    raise KeyError(
                        f"Column '{col}' not found in {path}. Available: {list(df.columns)}"
                    )
            df[dc] = pd.to_datetime(df[dc], errors="coerce")
            df = df.dropna(subset=[dc]).sort_values(dc)
            df[vc] = pd.to_numeric(df[vc], errors="coerce")
            return df.set_index(dc)[vc]

        ref_raw    = _load(Path(reference_timeseries_path), ref_date_col, ref_value_col)
        ref_raw    = ref_raw.clip(lower=value_min, upper=value_max)
        ref_smooth = _smooth(ref_raw)

        csv_files = sorted(Path(input_folder).glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in: {input_folder}")

        comparison: List[Tuple[str, pd.Series]] = [
            (fp.stem, _smooth(_load(fp, date_col, value_col))) for fp in csv_files
        ]

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        if show_raw_data:
            ax.scatter(
                ref_raw.index, ref_raw.values,
                s=raw_point_size, color=ref_color, alpha=raw_opacity,
                linewidths=0, label=raw_label, zorder=1,
            )
        ax.plot(
            ref_smooth.index, ref_smooth.values,
            linewidth=ref_linewidth, color=ref_color, label=ref_label, zorder=2,
        )
        for idx, (label, series) in enumerate(comparison):
            ax.plot(
                series.index, series.values,
                linewidth=series_linewidth,
                color=color_cycle[idx % len(color_cycle)],
                linestyle=ls_cycle[idx % len(ls_cycle)],
                label=label, zorder=3,
            )

        ax.set_title(title, pad=10)
        ax.set_ylabel(y_label)
        ax.set_xlabel(x_label)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=month_interval))
        ax.xaxis.set_major_formatter(mdates.DateFormatter(date_format))
        ax.margins(x=0)
        ax.grid(show_grid)

        y_all = np.concatenate(
            [ref_smooth.to_numpy()] + [s.to_numpy() for _, s in comparison]
        )
        y_all = y_all[np.isfinite(y_all)]
        if y_all.size:
            y0, y1 = float(np.min(y_all)), float(np.max(y_all))
            pad = 0.20 * (y1 - y0 + 1e-6)
            ax.set_ylim(max(0.0, y0 - pad), y1 + pad)

        ax.legend(
            title=legend_title, loc=legend_loc,
            bbox_to_anchor=legend_bbox_to_anchor,
            ncol=legend_ncol, frameon=legend_frameon,
        )
        plt.tight_layout(rect=list(tight_layout_rect))

        out_path = None
        if save:
            _out_dir = Path(output_dir) if output_dir else Path(input_folder)
            _out_dir.mkdir(parents=True, exist_ok=True)
            out_path = str(_out_dir / output_filename)
            fig.savefig(out_path, dpi=dpi_save, bbox_inches="tight")
            print(f"Saved: {out_path}")

        if show:
            plt.show()
        plt.close(fig)
        return out_path

