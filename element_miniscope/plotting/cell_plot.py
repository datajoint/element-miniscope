import colorsys
import numpy as np
import plotly.graph_objects as go


def plot_cell_overlayed_image(
    miniscope_module,
    segmentation_key,
    fig_height=600,
    fig_width=600,
    mask_saturation=0.7,
    mask_color_value=1,
    **kwargs,
) -> go.Figure:
    """Generate a Plotly figure with an overlayed summary image and ROI masks."""

    # Fetch data
    average_image, max_projection_image, mask_ids, mask_xpix, mask_ypix = figure_data(
        miniscope_module, segmentation_key
    )

    average_image = normalize_image(average_image, **kwargs)
    max_projection_image = normalize_image(max_projection_image, **kwargs)

    # Generate random HSV colors and convert to RGB
    roi_hsv_colors = [np.random.rand() for _ in range(mask_ids.size)]
    roi_rgb_colors = [
        f"rgb{tuple(int(c * 255) for c in colorsys.hsv_to_rgb(hue, mask_saturation, mask_color_value))}"
        for hue in roi_hsv_colors
    ]

    fig = go.Figure()

    # Use `go.Heatmap()` instead of `go.Image()` for better JSON handling
    fig.add_trace(go.Heatmap(z=average_image, colorscale="gray", showscale=False))

    for xpix, ypix, color, roi_id in zip(
        mask_xpix, mask_ypix, roi_rgb_colors, mask_ids
    ):
        fig.add_trace(
            go.Scatter(
                x=xpix,
                y=ypix,
                mode="lines",
                line=dict(color=color, width=2),
                name=f"ROI {roi_id}",
                hoverinfo="text",
                text=[f"ROI {roi_id}"] * len(xpix),
                showlegend=False,
                opacity=0.5,
            )
        )

    # Update layout
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
    """Fetch data for generating a figure."""
    average_image = np.squeeze(
        (miniscope_module.MotionCorrection.Summary & segmentation_key).fetch1(
            "average_image"
        )
    )
    max_projection_image = np.squeeze(
        (miniscope_module.MotionCorrection.Summary & segmentation_key).fetch1(
            "max_proj_image"
        )
    )
    mask_ids, mask_xpix, mask_ypix = (
        miniscope_module.Segmentation.Mask & segmentation_key
    ).fetch("mask", "mask_xpix", "mask_ypix")

    return average_image, max_projection_image, mask_ids, mask_xpix, mask_ypix


def normalize_image(image, low_q=0, high_q=1):
    """Normalize image to [0,1] based on quantile clipping."""
    q_min, q_max = np.quantile(image, [low_q, high_q])
    image = np.clip(image, q_min, q_max)
    
    return ((image - q_min) / (q_max - q_min) * 255).astype(np.uint8)
