import logging
import types

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from scipy.ndimage import gaussian_filter1d

from ..miniscope import get_loader_result

logger = logging.getLogger("datajoint")

try:
    from caiman.source_extraction.cnmf.estimates import Estimates  # noqa: E402
except ModuleNotFoundError:
    logger.error("Please install CaImAn")


class QualityMetricFigs(object):
    def __init__(
        self,
        mini: types.ModuleType,
        key: dict = None,
        scale: float = 1,
        fig_width: int = 800,
        dark_mode: bool = False,
    ):
        """Initialize QC metric class

        Args:
            mini (module): datajoint module with a QualityMetric table
            key (dict, optional): key from mini.QualityMetric table. Defaults to None.
            scale (float, optional): Scale at which to render figure. Defaults to 1.4.
            fig_width (int, optional): Figure width in pixels. Defaults to 800.
            dark_mode (bool, optional): Set background to black, foreground white.
                Default False, black on white.
        """
        self._mini = mini
        self._key = (self._mini.Curation & key).fetch1("KEY")
        self._scale = scale
        self._plots = {}  # Empty default to defer set to dict property below
        self._fig_width = fig_width
        self._dark_mode = dark_mode
        self._estimates = None
        self._component_list = []
        self._components = pd.DataFrame()  # Empty default
        self._x_fmt = dict(showgrid=False, zeroline=False, linewidth=2, ticks="outside")
        self._y_fmt = dict(showgrid=False, linewidth=0, zeroline=True, visible=False)
        self._no_data_text = "No data available"  # What to show when no data in table
        self._null_series = pd.Series(np.nan)  # What to substitute when no data

    @property
    def key(self) -> dict:
        """Key in mini.Curation table"""
        return self._key

    @key.setter  # Allows `cls.property = new_item` notation
    def key(self, key: dict):
        """Use class_instance.key = your_key to reset key"""
        if key not in self._mini.Curation.fetch("KEY"):
            # If not already full key, check if uniquely identifies entry
            key = (self._mini.Curation & key).fetch1("KEY")
            self._estimates = None  # Refresh estimates
        self._key = key

    @key.deleter  # Allows `del cls.property` to clear key
    def key(self):
        """Use del class_instance.key to clear key"""
        logger.info("Cleared key")
        self._key = None

    @property
    def estimates(self) -> Estimates:
        if not self._estimates:
            method, loaded_result = get_loader_result(self._key, self._mini.Curation)
            assert (
                method == "caiman"
            ), f"Quality figures for {method} not yet implemented. Try CaImAn."

            self._estimates = loaded_result.cnmf.estimates
        return self._estimates

    @property
    def component_list(self) -> list:
        if not self._component_list:
            self._component_list = ["r_values", "SNR_comp", "cnn_preds"]
        return self._component_list

    @component_list.setter
    def component_list(self, component_list: list):
        self._component_list = component_list
        self._components = pd.DataFrame()  # Reset components

    @property
    def components(self) -> pd.DataFrame:
        """Pandas dataframe of QC metrics"""
        if not self._key:
            return self._null_series

        if self._components.empty:
            empty_array = np.zeros((self.estimates.C.shape[0],))

            self._components = pd.DataFrame(
                data=[
                    getattr(self.estimates, attrib, empty_array).astype(np.float64)
                    for attrib in self.component_list
                ],
                index=self.component_list,
            ).T

        return self._components

    def _format_fig(
        self, fig: go.Figure = None, scale: float = None, ratio: float = 1.0
    ) -> go.Figure:
        """Return formatted figure or apply formatting to existing figure

        Args:
            fig (go.Figure, optional): Apply formatting to this plotly graph object
                Figure to apply formatting. Defaults to empty.
            scale (float, optional): Scale to render figure. Defaults to scale from
                class init, 1.
            ratio (float, optional): Figure aspect ratio width/height . Defaults to 1.

        Returns:
            go.Figure: Formatted figure
        """
        if not fig:
            fig = go.Figure()
        if not scale:
            scale = self._scale

        width = self._fig_width * scale

        return fig.update_layout(
            template="plotly_dark" if self._dark_mode else "simple_white",
            width=width,
            height=width / ratio,
            margin=dict(l=20 * scale, r=20 * scale, t=40 * scale, b=40 * scale),
            showlegend=False,
        )

    def _empty_fig(
        self, text="Select a key to visualize QC metrics", scale=None
    ) -> go.Figure:
        """Return figure object for when no key is provided"""
        if not scale:
            scale = self._scale

        return (
            self._format_fig(scale=scale)
            .add_annotation(text=text, showarrow=False)
            .update_layout(xaxis=self._y_fmt, yaxis=self._y_fmt)
        )

    def _plot_metric(
        self,
        data: pd.DataFrame,
        bins: np.ndarray,
        scale: float = None,
        fig: go.Figure = None,
        **trace_kwargs,
    ) -> go.Figure:
        """Plot histogram using bins provided

        Args:
            data (pd.DataFrame): Data to be plotted, from QC metric
            bins (np.ndarray): Array of bins to use for histogram
            scale (float, optional): Scale to render figure. Defaults to scale from
                class initialization.
            fig (go.Figure, optional): Add trace to this figure. Defaults to empty
                formatted figure.

        Returns:
            go.Figure: Histogram plot
        """
        if not scale:
            scale = self._scale
        if not fig:
            fig = self._format_fig(scale=scale)

        if not data.isnull().all():
            histogram, histogram_bins = np.histogram(data, bins=bins, density=True)
        else:
            # To quiet divide by zero error when no data
            histogram, histogram_bins = np.ndarray(0), np.ndarray(0)

        return fig.add_trace(
            go.Scatter(
                x=histogram_bins[:-1],
                y=gaussian_filter1d(histogram, 1),  # TODO: remove smoothing
                mode="lines",
                line=dict(color="rgb(0, 160, 223)", width=2 * scale),  # DataJoint Blue
                hovertemplate="%{x:.2f}<br>%{y:.2f}<extra></extra>",
            ),
            **trace_kwargs,
        )

    def get_single_fig(self, fig_name: str, scale: float = None) -> go.Figure:
        """Return a single figure of the plots listed in the plot_list property

        Args:
            fig_name (str): Name of figure to be rendered
            scale (float, optional): Scale to render fig. Defaults to scale at class
                init, 1.

        Returns:
            go.Figure: Histogram plot
        """
        if not self._key:
            return self._empty_fig()
        if not scale:
            scale = self._scale

        fig_dict = self.plots.get(fig_name, dict()) if self._key else dict()
        data = fig_dict.get("data", self._null_series)
        bins = fig_dict.get("bins", np.linspace(0, 0, 0))
        vline = fig_dict.get("vline", None)

        if data.isnull().all():
            return self._empty_fig(text=self._no_data_text)

        fig = (
            self._plot_metric(data=data, bins=bins, scale=scale)
            .update_layout(xaxis=self._x_fmt, yaxis=self._y_fmt)
            .update_layout(  # Add title
                title=dict(text=fig_dict.get("xaxis", " "), xanchor="center", x=0.5),
                font=dict(size=12 * scale),
            )
        )

        if vline:
            fig.add_vline(x=vline, line_width=2 * scale, line_dash="dash")

        return fig

    def get_grid(self, n_columns: int = 3, scale: float = 1.0) -> go.Figure:
        """Plot grid of histograms as subplots in go.Figure using n_columns

        Args:
            n_columns (int, optional): Number of columns in grid. Defaults to 4.
            scale (float, optional): Scale to render fig. Defaults to scale at class
                init, 1.

        Returns:
            go.Figure: grid of available plots
        """
        from plotly.subplots import make_subplots

        if not self._key:
            return self._empty_fig()
        if not scale:
            scale = self._scale

        n_rows = int(np.ceil(len(self.plots) / n_columns))

        fig = self._format_fig(
            fig=make_subplots(
                rows=n_rows,
                cols=n_columns,
                shared_xaxes=False,
                shared_yaxes=False,
                vertical_spacing=(0.5 / n_rows),
            ),
            scale=scale,
            ratio=(n_columns / n_rows),
        ).update_layout(  # Global title
            title=dict(text="Histograms of Quality Metrics", xanchor="center", x=0.5),
            font=dict(size=12 * scale),
        )

        for idx, plot in enumerate(self._plots.values()):  # Each subplot
            this_row = int(np.floor(idx / n_columns) + 1)
            this_col = idx % n_columns + 1
            data = plot.get("data", self._null_series)
            vline = plot.get("vline", None)
            if data.isnull().all():
                vline = None  # If no data, don't want vline either
                fig["layout"].update(
                    annotations=[
                        dict(
                            xref=f"x{idx+1}",
                            yref=f"y{idx+1}",
                            text=self._no_data_text,
                            showarrow=False,
                        ),
                    ]
                )
            fig = self._plot_metric(  # still need to plot empty to cal y_vals min/max
                data=data,
                bins=plot["bins"],
                fig=fig,
                row=this_row,
                col=this_col,
                scale=scale,
            )
            fig.update_xaxes(
                title=dict(text=plot["xaxis"], font_size=11 * scale),
                row=this_row,
                col=this_col,
            )
            if vline:
                y_vals = fig.to_dict()["data"][idx]["y"]
                fig.add_shape(  # Add overlay WRT whole fig
                    go.layout.Shape(
                        type="line",
                        yref="paper",
                        xref="x",  # relative to subplot x
                        x0=vline,
                        y0=min(y_vals),
                        x1=vline,
                        y1=max(y_vals),
                        line=dict(width=2 * scale),
                    ),
                    row=this_row,
                    col=this_col,
                )

        return fig.update_xaxes(**self._x_fmt).update_yaxes(**self._y_fmt)

    @property
    def plot_list(self):
        """List of plots that can be rendered individually by name or as grid"""
        if not self._plots:
            _ = self.plots
        return [plot for plot in self._plots]

    def _default_bins(self, component: pd.Series, nbins: int = 10) -> np.ndarray:
        """Default bins for rendered histograms

        Args:
            component (pd.Series): Pandas series of which we'll use min and max
            nbins (int, optional): Number of bins to use. Defaults to 10.

        Returns:
            numpy.ndarray: numpy linspace(min(component), max(component), nbins)
        """
        values = self.components.get(component, self._null_series).replace(
            [np.inf, -np.inf], np.nan
        )
        return np.linspace(min(values), max(values), nbins)

    @property
    def plots(self) -> dict:
        """Set of plots available to be rendered"""
        if not self._plots:
            self._plots = {
                "r_values": {
                    "xaxis": "R Values",
                    "data": self.components.get("r_values", self._null_series),
                    "bins": self._default_bins("r_values"),
                    "vline": 0.85,
                },
                "SNR": {
                    "xaxis": "SNR",
                    "data": self.components.get("SNR_comp", self._null_series),
                    "bins": self._default_bins("SNR_comp"),
                    "vline": 2,
                },
                "cnn_preds": {
                    "xaxis": "CNN Preds",
                    "data": self.components.get("cnn_preds", self._null_series),
                    "bins": self._default_bins("cnn_preds"),
                    "vline": 0.1,
                },
            }
        return self._plots

    @plots.setter
    def plots(self, new_plot_dict: dict):
        """Adds or updates plot item in the set to be rendered.

        plot items are structured as followed: dict with name key, embedded dict with
            xaxis: string x-axis label
            data: pandas dataframe to be plotted
            bins: numpy ndarray of bin cutoffs for histogram
        """
        _ = self.plots
        [self._plots.update({k: v}) for k, v in new_plot_dict.items()]

    def remove_plot(self, plot_name):
        """Removes an item from the set of plots"""
        _ = self._plots.pop(plot_name)
