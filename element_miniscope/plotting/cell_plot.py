import datajoint as dj
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import colorsys


def plot_cell_overlayed_image(
    miniscope_module, segmentation_key, fig_height=600, fig_width=600, mask_saturation=0.7, mask_color_value=1
) -> go.Figure:

    average_image, max_projection_image, mask_ids, mask_xpix, mask_ypix = figure_data(
        miniscope_module, segmentation_key
    )

    roi_hsv_colors = [np.random.rand() for _ in range(mask_ids.size)]

    # Convert HSV colors to RGB for Plotly
    roi_rgb_colors = []
    for hue in roi_hsv_colors:
        rgb = colorsys.hsv_to_rgb(hue, mask_saturation, mask_color_value)
        rgb_scaled = tuple(int(c * 255) for c in rgb) 
        roi_rgb_colors.append(f"rgb{rgb_scaled}")  # Convert to Plotly-compatible format

    fig = px.imshow(
    average_image,  # Initial image (can toggle later)
    color_continuous_scale="gray",
    labels={"color": "Intensity"},
    )
    fig.update_coloraxes(showscale=False)
    # Add scatter traces for ROI contours (keeping HSV logic internally)
    for xpix, ypix, color, roi_id in zip(mask_xpix, mask_ypix, roi_rgb_colors, mask_ids):
        fig.add_trace(
            go.Scatter(
                x=xpix,
                y=ypix,
                mode="lines",
                line=dict(color=color, width=2),  # Use HSV-based color (converted to RGB)
                name=f"ROI {roi_id}",
                hoverinfo="text",
                text=[f"ROI {roi_id}"] * len(xpix),
                showlegend=False,
                opacity=0.5,
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
                        "label": "Average Image",
                        "method": "update",
                        "args": [{"z": [average_image]}],
                    },
                    {
                        "label": "Max Projection Image",
                        "method": "update",
                        "args": [{"z": [max_projection_image]}],
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

    average_image = np.squeeze(
        (miniscope_module.MotionCorrection.Summary & segmentation_key).fetch1("average_image")
    )
    max_projection_image = np.squeeze(
        (miniscope_module.MotionCorrection.Summary & segmentation_key).fetch1("max_proj_image")
    )
    mask_ids, mask_xpix, mask_ypix = (
        miniscope_module.Segmentation.Mask & segmentation_key
    ).fetch("mask", "mask_xpix", "mask_ypix")

    return average_image, max_projection_image, mask_ids, mask_xpix, mask_ypix
