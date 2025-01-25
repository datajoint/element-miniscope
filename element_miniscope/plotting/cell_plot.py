import datajoint as dj
import plotly.graph_objects as go
import numpy as np


def plot_cell_overlayed_image(
    miniscope_module, segmentation_key, fig_height=600, fig_width=600
) -> go.Figure:

    average_image, max_projection_image, mask_ids, mask_xpix, mask_ypix = figure_data(
        miniscope_module, segmentation_key
    )

    fig = go.Figure()

    # Add heatmaps for the average and max projection images
    fig.add_trace(
        go.Heatmap(
            z=average_image,
            colorscale=[[0, "black"], [1, "white"]],
            showscale=False,
            visible=True,  # Initially visible
        )
    )

    fig.add_trace(
        go.Heatmap(
            z=max_projection_image,
            colorscale=[[0, "black"], [1, "white"]],
            showscale=False,
            visible=False,  # Initially hidden
        )
    )

    roi_colors = [
        f"rgb({np.random.randint(0, 256)}, {np.random.randint(0, 256)}, {np.random.randint(0, 256)})"
        for _ in range(mask_ids.size)
    ]

    # Add scatter traces for ROI contours
    for xpix, ypix, color, roi_id in zip(mask_xpix, mask_ypix, roi_colors, mask_ids):
        fig.add_trace(
            go.Scatter(
                x=xpix,
                y=ypix,
                mode="lines",
                line=dict(color=color, width=2),
                hoverinfo="text",
                text=[f"ROI {roi_id}"] * len(xpix),  # Display ROI ID on hover
                showlegend=False,  # Hide legend for ROI contours
                opacity=0.7,  # Adjust line transparency
            )
        )

    fig.update_layout(
        title=dict(
            text="Summary Image",
            x=0.5,
            xanchor="center",
            font=dict(size=18),
        ),
        updatemenus=[
            {
                "buttons": [
                    {
                        "label": "Average Projection",
                        "method": "update",
                        "args": [{"visible": [True, False] + [True] * mask_ids.size}],
                    },
                    {
                        "label": "Max Projection",
                        "method": "update",
                        "args": [{"visible": [False, True] + [True] * mask_ids.size}],
                    },
                ],
                "direction": "down",
                "showactive": True,
                "x": 0.5,
                "xanchor": "center",
                "y": 1.1,
            }
        ],
        height=fig_height,
        width=fig_width,
        margin=dict(t=fig_height / 6, b=40),
    )

    return fig


def figure_data(miniscope_module, segmentation_key):
    miniscope = dj.create_virtual_module("miniscope", miniscope_module)

    average_image = np.squeeze(
        (miniscope.MotionCorrection.Summary & segmentation_key).fetch1("average_image")
    )
    max_projection_image = np.squeeze(
        (miniscope.MotionCorrection.Summary & segmentation_key).fetch1("max_proj_image")
    )
    mask_ids, mask_xpix, mask_ypix = (
        miniscope.Segmentation.Mask & segmentation_key
    ).fetch("mask", "mask_xpix", "mask_ypix")

    return average_image, max_projection_image, mask_ids, mask_xpix, mask_ypix
