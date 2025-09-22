# residual_viz.py
"""
Residual Stream Visualization Utilities

These plotting functions consume the *output of analyze_residual_stream*.

Supported input shapes:
- reduction="norm" or "mean": torch.Tensor [L, T]
- reduction="pca": torch.Tensor [L, T, 3]
- reduction="none": torch.Tensor [L, T, D]  (only for summarize_to_norm)

Notation:
- L = number of layers (including embedding layer if present)
- T = sequence length (tokens)
- D = hidden size (residual width)

All functions:
- Accept either torch.Tensor or numpy.ndarray.
- Move tensors to CPU and convert to numpy automatically.
- Return the Matplotlib Figure for further customization/saving.

Usage example:

    from residual_viz import (
        plot_heatmap_LT,
        plot_token_path_from_pca,
        plot_layer_token_series,
        plot_layer_scatter_from_pca,
        plot_token_contributions_from_LT,
        summarize_to_norm,  # if you started with reduction="none"
    )

    # Suppose rs = analyze_residual_stream(residual_stream, reduction="norm")
    fig = plot_heatmap_LT(rs, title="L2 norm per (layer, token)")

"""

from typing import Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Colormap

ArrayLike = Union["np.ndarray", "torch.Tensor"]


# ----------------------------- Internal helpers ----------------------------- #


def _to_numpy(x: ArrayLike, dtype: Optional[np.dtype] = None) -> np.ndarray:
    """
    Convert a torch.Tensor or numpy.ndarray to a numpy.ndarray on CPU.

    This function handles various input types and ensures the output is a NumPy array
    suitable for visualization. It automatically detects and converts low-precision
    tensors (float16, bfloat16) to float32 for numerical stability.

    Args:
        x: Input array/tensor (torch.Tensor, numpy.ndarray, list, etc.)
        dtype: Optional NumPy dtype to convert the result to

    Returns:
        Numpy array on CPU memory with appropriate dtype

    Raises:
        TypeError: If the input cannot be converted to a NumPy array
    """
    try:
        if isinstance(x, torch.Tensor):
            # Handle low-precision tensors by converting to float32
            if x.dtype in [torch.float16, torch.bfloat16]:
                x = x.to(torch.float32)

            # Move to CPU and convert to numpy
            result = x.detach().cpu().numpy()
        else:
            # Convert other types to numpy array
            result = np.asarray(x)

        # Apply requested dtype if specified
        if dtype is not None:
            result = result.astype(dtype)

        return result
    except Exception as e:
        raise TypeError(
            f"Could not convert input of type {type(x)} to numpy array: {str(e)}"
        ) from e


def _assert_shape(
    x: np.ndarray,
    ndim: Optional[Union[int, Tuple[int, ...]]] = None,
    name: str = "array",
    last_dim: Optional[int] = None,
    expected_shape: Optional[Tuple[int, ...]] = None,
    allow_none: bool = False,
) -> np.ndarray:
    """
    Validate array dimensionality and shape with detailed error messages.

    This function performs various shape checks on the input array and raises
    descriptive error messages if the checks fail. It can validate:
    - The number of dimensions (ndim)
    - The size of the last dimension (last_dim)
    - The exact shape (expected_shape)
    - Or any combination of these

    Args:
        x: Numpy array to validate
        ndim: Expected number of dimensions. Can be:
              - An integer (exact match required)
              - A tuple of integers (any match is valid)
              - None (no check performed)
        name: Human-friendly variable name for error messages
        last_dim: If set, check x.shape[-1] == last_dim
        expected_shape: If set, check x.shape matches this exactly
        allow_none: If True, allow x to be None (returns None in this case)

    Returns:
        The validated array (for chaining)

    Raises:
        ValueError: If any shape check fails
        TypeError: If x is not a numpy array (or None when allow_none=True)
    """
    # Handle None case
    if x is None:
        if allow_none:
            return None
        raise TypeError(f"{name} cannot be None")

    # Type check
    if not isinstance(x, np.ndarray):
        raise TypeError(f"{name} must be a numpy array, got {type(x).__name__}")

    # Check number of dimensions
    if ndim is not None:
        if isinstance(ndim, int):
            if x.ndim != ndim:
                raise ValueError(
                    f"{name} must have {ndim} dimensions, "
                    f"got {x.ndim} with shape {x.shape}"
                )
        elif isinstance(ndim, tuple):
            if x.ndim not in ndim:
                raise ValueError(
                    f"{name} must have one of {ndim} dimensions, "
                    f"got {x.ndim} with shape {x.shape}"
                )

    # Check last dimension
    if last_dim is not None and x.shape[-1] != last_dim:
        raise ValueError(
            f"{name} must have last dimension {last_dim}, "
            f"got {x.shape[-1]} with full shape {x.shape}"
        )

    # Check exact shape
    if expected_shape is not None:
        if x.shape != expected_shape:
            raise ValueError(
                f"{name} must have shape {expected_shape}, " f"got {x.shape}"
            )

    return x


# ---------------------------- Shape conversion tools ---------------------------- #


def summarize_to_norm(
    rs_none: ArrayLike,
    method: str = "l2",
    normalize: Optional[str] = None,
    mask: Optional[ArrayLike] = None,
) -> np.ndarray:
    """
    Summarize a raw residual stream (reduction="none", shape [L, T, D]) into
    an L×T map using various methods across the hidden dimension.

    This function converts the full-dimensional residual stream into a 2D map
    that can be visualized with heatmaps or line plots. It's useful for understanding
    how activation patterns vary across layers and tokens.

    This is the only function here that accepts the *raw* 'none' output.
    All other plotters expect either [L, T] (norm/mean) or [L, T, 3] (pca).

    Args:
        rs_none: Array/Tensor of shape [L, T, D] from analyze_residual_stream(..., reduction="none").
        method: Summarization method across the hidden dimension:
            - "l2": L2 norm (default, captures overall activation magnitude)
            - "mean": Mean value (useful for signed activations)
            - "max": Maximum value (captures strongest positive activation)
            - "min": Minimum value (captures strongest negative activation)
            - "abs_mean": Mean of absolute values (like L1 norm)
            - "std": Standard deviation (captures activation variance)
        normalize: Optional normalization method after summarization:
            - None: No normalization (default)
            - "layers": Normalize each layer to [0,1] range (compare tokens within layers)
            - "tokens": Normalize each token to [0,1] range (compare layers for each token)
            - "global": Normalize entire matrix to [0,1] range (global comparison)
        mask: Optional boolean mask of shape [T] to exclude certain tokens
              (e.g., padding tokens). True values are kept, False are masked.

    Returns:
        LxT numpy array summarizing the residual stream per (layer, token).

    Examples:
        # Basic L2 norm (default)
        lt_map = summarize_to_norm(residual_stream)

        # Mean activation with per-layer normalization
        lt_map = summarize_to_norm(residual_stream, method="mean", normalize="layers")

        # Mask padding tokens (assuming pad_mask is a boolean array where True = keep)
        lt_map = summarize_to_norm(residual_stream, mask=pad_mask)
    """
    x = _to_numpy(rs_none)
    _assert_shape(x, ndim=3, name="rs_none")

    # Apply mask if provided
    if mask is not None:
        mask_array = _to_numpy(mask)
        _assert_shape(mask_array, ndim=1, name="mask")
        if mask_array.shape[0] != x.shape[1]:
            raise ValueError(
                f"Mask length {mask_array.shape[0]} must match token dimension {x.shape[1]}"
            )

        # Create a mask for all layers (broadcast across layers)
        # We don't modify x directly to avoid potential side effects
        # mask_expanded = np.expand_dims(mask_array, axis=(0, 2))  # [1, T, 1] - Currently unused

    # Apply summarization method
    if method == "l2":
        result = np.linalg.norm(x, axis=-1)  # [L, T]
    elif method == "mean":
        result = np.mean(x, axis=-1)
    elif method == "max":
        result = np.max(x, axis=-1)
    elif method == "min":
        result = np.min(x, axis=-1)
    elif method == "abs_mean":
        result = np.mean(np.abs(x), axis=-1)
    elif method == "std":
        result = np.std(x, axis=-1)
    else:
        raise ValueError(
            f"Unknown summarization method: {method}. "
            f"Choose from: l2, mean, max, min, abs_mean, std"
        )

    # Apply mask if provided
    if mask is not None:
        # Create a copy to avoid modifying the original data
        result = result.copy()
        # Set masked positions to NaN (better for visualization than 0)
        result[:, ~mask_array] = np.nan

    # Apply normalization if requested
    if normalize is not None:
        if normalize == "layers":
            # Normalize each layer independently
            layer_min = np.nanmin(result, axis=1, keepdims=True)
            layer_max = np.nanmax(result, axis=1, keepdims=True)
            # Avoid division by zero
            layer_range = np.maximum(layer_max - layer_min, 1e-10)
            result = (result - layer_min) / layer_range
        elif normalize == "tokens":
            # Normalize each token independently
            token_min = np.nanmin(result, axis=0, keepdims=True)
            token_max = np.nanmax(result, axis=0, keepdims=True)
            # Avoid division by zero
            token_range = np.maximum(token_max - token_min, 1e-10)
            result = (result - token_min) / token_range
        elif normalize == "global":
            # Normalize the entire matrix
            global_min = np.nanmin(result)
            global_max = np.nanmax(result)
            # Avoid division by zero
            global_range = max(global_max - global_min, 1e-10)
            result = (result - global_min) / global_range
        else:
            raise ValueError(
                f"Unknown normalization method: {normalize}. "
                f"Choose from: layers, tokens, global"
            )

    return result


# ------------------------------- Heatmap (L×T) ------------------------------- #


def plot_heatmap_LT(
    lt: ArrayLike,
    *,
    title: str = "Layer × Token Map",
    xlabel: str = "Token index",
    ylabel: str = "Layer",
    show_colorbar: bool = True,
    colorbar_label: Optional[str] = None,
    cmap: Union[str, Colormap] = "viridis",
    transform: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    center: Optional[float] = None,
    layer_labels: Optional[Sequence[str]] = None,
    token_labels: Optional[Sequence[str]] = None,
    show_values: bool = False,
    value_format: str = ".2f",
    show_grid: bool = False,
    highlight_indices: Optional[dict[str, list[int]]] = None,
    highlight_color: str = "red",
    figsize: Optional[Tuple[float, float]] = None,
    nan_color: str = "#EEEEEE",
) -> plt.Figure:
    """
    Plot a heatmap for data shaped [L, T]. Works for any 2D array representing
    layer × token data, such as from summarize_to_norm() or analyze_residual_stream().

    This visualization helps identify patterns in how activations vary across
    layers and tokens, revealing which layers contribute most to certain tokens
    or how activation patterns evolve through the network.

    Typical inputs:
        - analyze_residual_stream(..., reduction="norm")  # L2 magnitude
        - analyze_residual_stream(..., reduction="mean")  # mean over D
        - summarize_to_norm(residual_stream, method="l2")  # L2 norm
        - summarize_to_norm(residual_stream, method="std", normalize="layers")  # Normalized std dev

    Args:
        lt: L×T array/tensor.
        title: Plot title.
        xlabel: X-axis label (tokens).
        ylabel: Y-axis label (layers).
        show_colorbar: Whether to draw a colorbar.
        colorbar_label: Optional label for the colorbar.
        cmap: Colormap name or matplotlib colormap object. Some useful options:
              - "viridis": Default, perceptually uniform (good for continuous data)
              - "RdBu_r": Red-Blue diverging (good for data centered around zero)
              - "plasma": Perceptually uniform with more contrast than viridis
              - "YlOrRd": Yellow-Orange-Red (good for showing intensity)
        transform: Data transformation before plotting:
              - None: No transformation (default)
              - "log": Apply log(1+x) transform (good for data with large range)
              - "sqrt": Apply sqrt(x) transform (less aggressive than log)
              - "abs": Take absolute value (for data with negative values)
              - "zscore": Z-score normalization (centers data around 0)
        vmin: Minimum value for colormap scaling. If None, uses data minimum.
        vmax: Maximum value for colormap scaling. If None, uses data maximum.
        center: If provided, centers the colormap at this value (useful for diverging data).
        layer_labels: Optional custom labels for layers (y-axis).
        token_labels: Optional custom labels for tokens (x-axis).
        show_values: If True, display values in each cell.
        value_format: Format string for cell values if show_values=True.
        show_grid: If True, display grid lines between cells.
        highlight_indices: Optional dict with keys 'layers' and/or 'tokens' containing
                          lists of indices to highlight.
        highlight_color: Color for highlighted indices.
        figsize: Optional figure size as (width, height) in inches.
        nan_color: Color to use for NaN values.

    Returns:
        Matplotlib Figure object for further customization or saving.

    Examples:
        # Basic heatmap with default settings
        fig = plot_heatmap_LT(layer_token_data)

        # Customized heatmap with log transform and custom colormap
        fig = plot_heatmap_LT(
            layer_token_data,
            transform="log",
            cmap="plasma",
            title="Layer Activations",
            show_grid=True
        )

        # Heatmap with custom labels and highlighted regions
        fig = plot_heatmap_LT(
            layer_token_data,
            layer_labels=["Emb", "L1", "L2", "L3"],
            highlight_indices={"layers": [2], "tokens": [0, 5]},
            show_values=True
        )
    """
    arr = _to_numpy(lt)
    _assert_shape(arr, ndim=2, name="lt")

    # Apply data transformation if requested
    if transform is not None:
        if transform == "log":
            # Ensure data is positive before log transform
            if np.nanmin(arr) < 0:
                raise ValueError(
                    "Cannot apply log transform to negative values. "
                    "Consider using 'abs' transform first."
                )
            arr = np.log1p(arr)  # log(1+x) safe transform
        elif transform == "sqrt":
            # Ensure data is positive before sqrt transform
            if np.nanmin(arr) < 0:
                raise ValueError(
                    "Cannot apply sqrt transform to negative values. "
                    "Consider using 'abs' transform first."
                )
            arr = np.sqrt(arr)
        elif transform == "abs":
            arr = np.abs(arr)
        elif transform == "zscore":
            # Z-score normalization (ignoring NaNs)
            mean = np.nanmean(arr)
            std = np.nanstd(arr)
            if std > 0:
                arr = (arr - mean) / std
            else:
                arr = arr - mean
        else:
            raise ValueError(
                f"Unknown transform: {transform}. "
                f"Choose from: log, sqrt, abs, zscore"
            )

    # Create figure with specified size
    if figsize is None:
        # Calculate a reasonable default size based on data dimensions
        height = max(5, min(12, arr.shape[0] * 0.3 + 2))
        width = max(6, min(15, arr.shape[1] * 0.2 + 2))
        figsize = (width, height)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    # Set up colormap with proper NaN handling
    cmap_obj = plt.cm.get_cmap(cmap) if isinstance(cmap, str) else cmap
    cmap_obj.set_bad(nan_color)

    # Handle centered colormap if requested
    if center is not None:
        # Create a diverging colormap centered at the specified value
        vmax = vmax if vmax is not None else np.nanmax(arr)
        vmin = vmin if vmin is not None else np.nanmin(arr)
        # Ensure vmax and vmin are equidistant from center
        max_distance = max(abs(vmax - center), abs(center - vmin))
        vmax = center + max_distance
        vmin = center - max_distance
        # Use a diverging colormap if not explicitly specified
        if isinstance(cmap, str) and cmap == "viridis":
            cmap = "RdBu_r"  # Better default for centered data

    # Create the heatmap
    im = ax.imshow(arr, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

    # Set title with transform indicator
    transform_suffix = f" ({transform})" if transform else ""
    ax.set_title(title + transform_suffix)

    # Set axis labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Set custom tick labels if provided
    if layer_labels is not None:
        if len(layer_labels) != arr.shape[0]:
            raise ValueError(
                f"Expected {arr.shape[0]} layer labels, got {len(layer_labels)}"
            )
        ax.set_yticks(range(len(layer_labels)))
        ax.set_yticklabels(layer_labels)

    if token_labels is not None:
        if len(token_labels) != arr.shape[1]:
            raise ValueError(
                f"Expected {arr.shape[1]} token labels, got {len(token_labels)}"
            )
        # For many tokens, show a subset of ticks to avoid overcrowding
        if len(token_labels) > 20:
            step = max(1, len(token_labels) // 10)
            tick_positions = range(0, len(token_labels), step)
            tick_labels = [token_labels[i] for i in tick_positions]
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, rotation=45, ha="right")
        else:
            ax.set_xticks(range(len(token_labels)))
            ax.set_xticklabels(token_labels, rotation=45, ha="right")

    # Show grid if requested
    if show_grid:
        ax.grid(True, which="both", color="lightgray", linewidth=0.5, linestyle="-")
        ax.set_axisbelow(True)

    # Display values in cells if requested
    if show_values:
        # Determine text color based on cell value for better contrast
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                val = arr[i, j]
                if np.isnan(val):
                    continue

                # Format the value
                text = f"{val:{value_format}}"

                # Determine text color based on cell brightness
                # for better contrast against the background
                if cmap_obj is not None:
                    # Normalize value to [0, 1] range for colormap
                    if vmin is not None and vmax is not None:
                        norm_val = (val - vmin) / (vmax - vmin)
                    else:
                        data_min = np.nanmin(arr)
                        data_max = np.nanmax(arr)
                        norm_val = (
                            (val - data_min) / (data_max - data_min)
                            if data_max > data_min
                            else 0.5
                        )

                    # Clip to [0, 1] range
                    norm_val = max(0, min(1, norm_val))

                    # Get RGB color at this value
                    rgb = cmap_obj(norm_val)[:3]  # Exclude alpha

                    # Calculate perceived brightness
                    brightness = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]

                    # Use white text on dark backgrounds, black text on light backgrounds
                    text_color = "white" if brightness < 0.6 else "black"
                else:
                    text_color = "black"

                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=8,
                    fontweight="normal",
                )

    # Highlight specific indices if requested
    if highlight_indices is not None:
        # Highlight layers (rows)
        if "layers" in highlight_indices:
            for layer_idx in highlight_indices["layers"]:
                if 0 <= layer_idx < arr.shape[0]:
                    ax.axhspan(
                        layer_idx - 0.5,
                        layer_idx + 0.5,
                        color=highlight_color,
                        alpha=0.2,
                    )

        # Highlight tokens (columns)
        if "tokens" in highlight_indices:
            for token_idx in highlight_indices["tokens"]:
                if 0 <= token_idx < arr.shape[1]:
                    ax.axvspan(
                        token_idx - 0.5,
                        token_idx + 0.5,
                        color=highlight_color,
                        alpha=0.2,
                    )

    # Add colorbar if requested
    if show_colorbar:
        cbar = fig.colorbar(im, ax=ax)
        if colorbar_label:
            cbar.set_label(colorbar_label)

    # Ensure layout looks good
    fig.tight_layout()
    return fig


# ----------------------- Token path from PCA (PC1–PC2) ----------------------- #


def plot_token_path_from_pca(
    pca_l_t_3: ArrayLike,
    token_indices: Union[int, list[int], Sequence[int]] = -1,
    *,
    layer_range: Optional[Tuple[int, int]] = None,
    annotate_layers: bool = True,
    layer_labels: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
    colors: Optional[Union[str, list[str]]] = None,
    marker: str = "o",
    markersize: float = 6,
    linewidth: float = 1.5,
    linestyle: str = "-",
    show_arrows: bool = False,
    arrow_spacing: int = 2,
    arrow_scale: float = 15,
    highlight_layers: Optional[list[int]] = None,
    highlight_color: str = "yellow",
    highlight_alpha: float = 0.3,
    plot_3d: bool = False,
    figsize: Optional[Tuple[float, float]] = None,
    legend_loc: str = "best",
    token_labels: Optional[list[str]] = None,
    equal_aspect: bool = False,
) -> plt.Figure:
    """
    Plot the trajectory of one or more tokens across layers in the principal component space.

    This visualization shows how token representations evolve through the network layers,
    revealing patterns in how the model processes specific tokens. It can help identify
    where major transformations occur or how different tokens' representations relate.

    Input must be from reduction="pca": shape [L, T, 3].
    We take each token's coordinates in PC space at each layer and connect them.

    Args:
        pca_l_t_3: Array/Tensor of shape [L, T, 3] from analyze_residual_stream(..., reduction="pca").
        token_indices: Token(s) to visualize. Can be:
                      - A single integer (supports negative indexing; e.g., -1 for last token)
                      - A list of integers for multiple tokens
        layer_range: Optional (start, end) tuple to plot only a subset of layers.
                    End is exclusive, following Python convention.
        annotate_layers: If True, label each point with its layer index or custom label.
        layer_labels: Optional custom labels for layers. Must match the number of layers
                     in the selected range if layer_range is specified.
        title: Optional title; if None, a default is used.
        colors: Color(s) for the plotted line(s). Can be a single color or list of colors.
        marker: Marker style for points (matplotlib marker code).
        markersize: Size of markers.
        linewidth: Width of connecting lines.
        linestyle: Style of connecting lines.
        show_arrows: If True, add arrows showing the direction of layer progression.
        arrow_spacing: Number of layers between arrows (higher = fewer arrows).
        arrow_scale: Size scaling factor for arrows.
        highlight_layers: Optional list of layer indices to highlight with background color.
        highlight_color: Color for highlighted layers.
        highlight_alpha: Alpha (transparency) for highlighted regions.
        plot_3d: If True, create a 3D plot using all three principal components.
        figsize: Optional figure size as (width, height) in inches.
        legend_loc: Location for the legend when plotting multiple tokens.
        token_labels: Optional custom labels for tokens in the legend.
        equal_aspect: If True, set equal aspect ratio for the axes.

    Returns:
        Matplotlib Figure object for further customization or saving.

    Examples:
        # Basic path for the last token
        fig = plot_token_path_from_pca(pca_data)

        # Compare paths of first and last tokens with custom colors
        fig = plot_token_path_from_pca(
            pca_data,
            token_indices=[0, -1],
            colors=["blue", "red"],
            token_labels=["First", "Last"]
        )

        # Focus on middle layers with arrows showing direction
        fig = plot_token_path_from_pca(
            pca_data,
            layer_range=(4, 12),
            show_arrows=True,
            highlight_layers=[6, 7, 8]
        )

        # 3D visualization of token path
        fig = plot_token_path_from_pca(
            pca_data,
            plot_3d=True,
            show_arrows=True
        )
    """
    pcs = _to_numpy(pca_l_t_3)
    _assert_shape(pcs, ndim=3, name="pca_l_t_3", last_dim=3)
    L, T, _ = pcs.shape

    # Handle layer range selection
    start_layer, end_layer = 0, L
    if layer_range is not None:
        start_layer, end_layer = layer_range
        # Validate layer range
        if not (
            0 <= start_layer < L and 0 < end_layer <= L and start_layer < end_layer
        ):
            raise ValueError(f"Invalid layer range {layer_range} for {L} layers")

    # Select the layers to plot
    pcs_subset = pcs[start_layer:end_layer]
    L_subset = end_layer - start_layer

    # Validate layer labels if provided
    if layer_labels is not None:
        if len(layer_labels) != L_subset:
            raise ValueError(
                f"Expected {L_subset} layer labels for the selected range, "
                f"got {len(layer_labels)}"
            )

    # Convert single token index to list for uniform processing
    if isinstance(token_indices, (int, np.integer)):
        token_indices = [token_indices]

    # Validate and normalize token indices
    norm_token_indices = []
    for idx in token_indices:
        tok = idx if idx >= 0 else T + idx
        if not (0 <= tok < T):
            raise IndexError(f"Token index {idx} out of range for T={T}")
        norm_token_indices.append(tok)

    # Validate token labels if provided
    if token_labels is not None and len(token_labels) != len(norm_token_indices):
        raise ValueError(
            f"Expected {len(norm_token_indices)} token labels, "
            f"got {len(token_labels)}"
        )

    # Set up colors
    if colors is None:
        # Default color cycle for multiple tokens
        if len(norm_token_indices) > 1:
            colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        else:
            colors = ["blue"]  # Default single color

    # Ensure colors is a list matching the number of tokens
    if isinstance(colors, str):
        colors = [colors] * len(norm_token_indices)
    elif len(colors) < len(norm_token_indices):
        # Cycle colors if not enough provided
        colors = (colors * (len(norm_token_indices) // len(colors) + 1))[
            : len(norm_token_indices)
        ]

    # Create figure
    if figsize is None:
        figsize = (8, 6) if not plot_3d else (10, 8)

    fig = plt.figure(figsize=figsize)

    if plot_3d:
        ax = fig.add_subplot(111, projection="3d")
    else:
        ax = fig.add_subplot(111)

    # Plot each token's path
    for i, (token_idx, color) in enumerate(zip(norm_token_indices, colors)):
        # Extract coordinates for this token
        if plot_3d:
            coords = pcs_subset[:, token_idx]  # [L_subset, 3]
            x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
            line = ax.plot(
                x,
                y,
                z,
                marker=marker,
                markersize=markersize,
                linewidth=linewidth,
                linestyle=linestyle,
                color=color,
            )
        else:
            coords = pcs_subset[:, token_idx, :2]  # [L_subset, 2]
            x, y = coords[:, 0], coords[:, 1]
            line = ax.plot(
                x,
                y,
                marker=marker,
                markersize=markersize,
                linewidth=linewidth,
                linestyle=linestyle,
                color=color,
            )

        # Add label for legend if multiple tokens
        if len(norm_token_indices) > 1:
            if token_labels is not None:
                line[0].set_label(token_labels[i])
            else:
                line[0].set_label(f"Token {token_idx}")

        # Add arrows to show direction
        if show_arrows and len(x) > arrow_spacing:
            arrow_indices = range(arrow_spacing, len(x), arrow_spacing)
            for j in arrow_indices:
                if plot_3d:
                    # 3D arrows are more complex, using quiver3D
                    dx, dy, dz = x[j] - x[j - 1], y[j] - y[j - 1], z[j] - z[j - 1]
                    # Normalize and scale
                    magnitude = np.sqrt(dx**2 + dy**2 + dz**2)
                    if magnitude > 0:  # Avoid division by zero
                        scale = arrow_scale / magnitude
                        ax.quiver(
                            x[j - 1],
                            y[j - 1],
                            z[j - 1],
                            dx * scale,
                            dy * scale,
                            dz * scale,
                            color=color,
                            arrow_length_ratio=0.3,
                        )
                else:
                    dx, dy = x[j] - x[j - 1], y[j] - y[j - 1]
                    # Normalize and scale
                    magnitude = np.sqrt(dx**2 + dy**2)
                    if magnitude > 0:  # Avoid division by zero
                        scale = arrow_scale / magnitude
                        ax.arrow(
                            x[j - 1],
                            y[j - 1],
                            dx * scale,
                            dy * scale,
                            head_width=arrow_scale / 5,
                            head_length=arrow_scale / 3,
                            fc=color,
                            ec=color,
                            alpha=0.8,
                        )

    # Annotate layers
    if annotate_layers:
        for i in range(L_subset):
            # Use the first token's coordinates for annotations
            if plot_3d:
                xi, yi, zi = pcs_subset[i, norm_token_indices[0]]
                label = str(i + start_layer)
                if layer_labels is not None:
                    label = layer_labels[i]
                ax.text(xi, yi, zi, label, fontsize=8)
            else:
                xi, yi = pcs_subset[i, norm_token_indices[0], :2]
                label = str(i + start_layer)
                if layer_labels is not None:
                    label = layer_labels[i]
                ax.text(xi, yi, label, fontsize=8)

    # Highlight specific layers
    if highlight_layers is not None and not plot_3d:
        for layer_idx in highlight_layers:
            if start_layer <= layer_idx < end_layer:
                # Get relative index in the subset
                rel_idx = layer_idx - start_layer

                # Get the min/max coordinates to define the highlight area
                token_coords = [
                    pcs_subset[rel_idx, idx, :2] for idx in norm_token_indices
                ]
                x_coords = [coord[0] for coord in token_coords]
                y_coords = [coord[1] for coord in token_coords]

                # Add some padding
                padding = 0.1
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                x_range = x_max - x_min
                y_range = y_max - y_min
                x_min -= x_range * padding
                x_max += x_range * padding
                y_min -= y_range * padding
                y_max += y_range * padding

                # Draw highlight rectangle
                rect = plt.Rectangle(
                    (x_min, y_min),
                    x_max - x_min,
                    y_max - y_min,
                    color=highlight_color,
                    alpha=highlight_alpha,
                    zorder=0,
                )
                ax.add_patch(rect)

                # Add layer label inside the highlight
                layer_text = str(layer_idx)
                if layer_labels is not None:
                    layer_text = layer_labels[rel_idx]
                ax.text(
                    x_min + (x_max - x_min) * 0.05,
                    y_min + (y_max - y_min) * 0.9,
                    f"Layer {layer_text}",
                    fontsize=8,
                    alpha=0.7,
                )

    # Set axis labels
    if plot_3d:
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
    else:
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

        # Equal aspect ratio if requested
        if equal_aspect:
            ax.set_aspect("equal")

    # Set title
    if title is None:
        if len(norm_token_indices) == 1:
            title = f"PC path across layers (token={token_indices[0]})"
        else:
            title = f"PC path across layers ({len(norm_token_indices)} tokens)"

        # Add layer range info if specified
        if layer_range is not None:
            title += f" [layers {start_layer}–{end_layer-1}]"

    ax.set_title(title)

    # Add grid
    ax.grid(True, linestyle="--", alpha=0.3)

    # Add legend if multiple tokens
    if len(norm_token_indices) > 1:
        ax.legend(loc=legend_loc)

    fig.tight_layout()
    return fig


# ---------------------- Layer scatter from PCA (PC1–PC2) --------------------- #


def plot_layer_scatter_from_pca(
    pca_l_t_3: ArrayLike,
    layer_index: int,
    token_indices: Optional[Union[Sequence[int], slice]] = None,
    *,
    token_labels: Optional[Sequence[str]] = None,
    layer_label: Optional[str] = None,
    title: Optional[str] = None,
    color: Union[str, Sequence[str], None] = None,
    colormap: Optional[str] = None,
    color_by: Optional[str] = "index",
    marker: Union[str, Sequence[str]] = "o",
    marker_size: Union[float, Sequence[float]] = 30,
    alpha: float = 0.8,
    label_tokens: Union[bool, int, Sequence[int]] = False,
    label_fontsize: float = 7,
    show_legend: bool = True,
    legend_loc: str = "best",
    highlight_tokens: Optional[Sequence[int]] = None,
    highlight_color: str = "red",
    highlight_alpha: float = 0.7,
    highlight_size: float = 100,
    plot_3d: bool = False,
    equal_aspect: bool = False,
    figsize: Optional[Tuple[float, float]] = None,
    add_convex_hull: bool = False,
    add_centroid: bool = False,
    add_token_connections: Optional[list[Tuple[int, int]]] = None,
    connection_style: str = "-",
    connection_width: float = 1.0,
    connection_alpha: float = 0.5,
    connection_color: str = "gray",
    annotate_dimensions: bool = False,
) -> plt.Figure:
    """
    For a chosen layer, scatter plot tokens in principal component space.

    This visualization shows how different tokens are distributed in the reduced
    representation space at a specific layer. It can reveal clustering patterns,
    outliers, and the overall structure of token representations.

    Input must be from reduction="pca": shape [L, T, 3].

    Args:
        pca_l_t_3: Array/Tensor [L, T, 3] from analyze_residual_stream(..., reduction="pca").
        layer_index: Layer to visualize (0-based).
        token_indices: Optional subset of tokens to plot. Can be:
                      - A sequence of indices (e.g., [0, 1, 5, 10])
                      - A slice object (e.g., slice(0, 10, 2) for every other token in first 10)
                      - None to plot all tokens
        token_labels: Optional custom labels for tokens in the plot.
        layer_label: Optional custom label for the layer (instead of numeric index).
        title: Optional title; default is provided if None.
        color: Color(s) for scatter points. Can be:
              - A single color name/code for all points
              - A sequence of colors matching token_indices
              - None to use default or colormap
        colormap: Name of matplotlib colormap to use when color_by is specified.
        color_by: How to color the points. Options:
              - "index": Color by token index (default)
              - "position": Same as "index" but more explicit
              - "value": Color by PC3 value (useful in 2D plots to show 3rd dimension)
              - None: Use single color or provided color sequence
        marker: Marker style(s) for points. Either a single marker code or sequence.
        marker_size: Size(s) of markers. Either a single value or sequence.
        alpha: Transparency of scatter points (0-1).
        label_tokens: Which tokens to label. Can be:
              - True/False to label all/none
              - An integer N to label every Nth token
              - A sequence of specific token indices to label
        label_fontsize: Font size for token labels.
        show_legend: Whether to show a legend (when using multiple colors/markers).
        legend_loc: Location for the legend.
        highlight_tokens: Optional sequence of token indices to highlight.
        highlight_color: Color for highlighted tokens.
        highlight_alpha: Alpha (transparency) for highlighted tokens.
        highlight_size: Size for highlighted tokens.
        plot_3d: If True, create a 3D plot using all three principal components.
        equal_aspect: If True, set equal aspect ratio for the axes.
        figsize: Optional figure size as (width, height) in inches.
        add_convex_hull: If True, draw a convex hull around the token points.
        add_centroid: If True, mark the centroid of all token points.
        add_token_connections: Optional list of token index pairs to connect with lines.
        connection_style: Line style for token connections.
        connection_width: Line width for token connections.
        connection_alpha: Alpha (transparency) for token connections.
        connection_color: Color for token connections.
        annotate_dimensions: If True, show variance explained by each principal component.

    Returns:
        Matplotlib Figure object for further customization or saving.

    Examples:
        # Basic scatter plot of all tokens at layer 5
        fig = plot_layer_scatter_from_pca(pca_data, layer_index=5)

        # Colored scatter with custom token subset and labels
        fig = plot_layer_scatter_from_pca(
            pca_data,
            layer_index=10,
            token_indices=range(0, 20),
            color_by="index",
            colormap="viridis",
            label_tokens=True
        )

        # 3D visualization with highlighted tokens
        fig = plot_layer_scatter_from_pca(
            pca_data,
            layer_index=8,
            plot_3d=True,
            highlight_tokens=[0, 5, 10],
            add_centroid=True
        )

        # Show token connections (e.g., for sequential tokens)
        fig = plot_layer_scatter_from_pca(
            pca_data,
            layer_index=6,
            add_token_connections=[(i, i+1) for i in range(19)],
            add_convex_hull=True
        )
    """
    pcs = _to_numpy(pca_l_t_3)
    _assert_shape(pcs, ndim=3, name="pca_l_t_3", last_dim=3)
    L, T, _ = pcs.shape

    # Validate layer index
    if not (0 <= layer_index < L):
        raise IndexError(f"layer_index out of range for L={L}: {layer_index}")

    # Process token indices
    if token_indices is None:
        token_indices = range(T)
    elif isinstance(token_indices, slice):
        token_indices = range(*token_indices.indices(T))

    # Convert to list and validate
    token_indices = list(token_indices)
    if not token_indices:
        raise ValueError("No tokens selected to plot")

    # Validate token indices
    for idx in token_indices:
        if not (0 <= idx < T):
            raise IndexError(f"Token index {idx} out of range for T={T}")

    # Validate token labels if provided
    if token_labels is not None:
        if len(token_labels) != len(token_indices):
            raise ValueError(
                f"Expected {len(token_indices)} token labels, "
                f"got {len(token_labels)}"
            )

    # Extract coordinates for the selected layer and tokens
    if plot_3d:
        # Use all 3 dimensions
        coords = pcs[layer_index, token_indices]  # [num_tokens, 3]
    else:
        # Use only first 2 dimensions
        coords = pcs[layer_index, token_indices, :2]  # [num_tokens, 2]

    # Create figure with specified size
    if figsize is None:
        figsize = (8, 6) if not plot_3d else (10, 8)

    fig = plt.figure(figsize=figsize)

    if plot_3d:
        ax = fig.add_subplot(111, projection="3d")
    else:
        ax = fig.add_subplot(111)

    # Prepare colors based on color_by parameter
    if color_by is not None:
        if colormap is None:
            colormap = "viridis"  # Default colormap

        # cmap = plt.cm.get_cmap(colormap)  # Currently unused

        if color_by == "index" or color_by == "position":
            # Color by token position/index
            c_values = np.arange(len(token_indices))
            # c_norm = plt.Normalize(c_values.min(), c_values.max())  # Currently unused
            # colors = [cmap(c_norm(i)) for i in range(len(token_indices))]  # Currently unused
            scatter_c = c_values
        elif color_by == "value" and not plot_3d:
            # Color by PC3 value (useful in 2D plots)
            c_values = pcs[layer_index, token_indices, 2]
            # c_norm = plt.Normalize(c_values.min(), c_values.max())  # Currently unused
            # colors = [cmap(c_norm(v)) for v in c_values]  # Currently unused
            scatter_c = c_values
        else:
            # Default to single color if color_by is invalid
            color_by = None

    if color_by is None:
        scatter_c = color if color is not None else "blue"

    # Handle marker sizes
    if isinstance(marker_size, (int, float)):
        marker_size = [marker_size] * len(token_indices)

    # Create the scatter plot
    if plot_3d:
        scatter = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            coords[:, 2],
            c=scatter_c,
            marker=marker,
            s=marker_size,
            alpha=alpha,
        )
    else:
        scatter = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=scatter_c,
            marker=marker,
            s=marker_size,
            alpha=alpha,
        )

    # Add colorbar if using a colormap
    if color_by is not None and colormap is not None:
        if color_by == "value" and not plot_3d:
            plt.colorbar(scatter, ax=ax, label="PC3 value")
        elif color_by in ["index", "position"]:
            plt.colorbar(scatter, ax=ax, label="Token position")

    # Highlight specific tokens if requested
    if highlight_tokens is not None:
        highlight_coords = []
        for idx in highlight_tokens:
            if idx in token_indices:
                pos = token_indices.index(idx)
                highlight_coords.append(coords[pos])

        if highlight_coords:
            highlight_coords = np.array(highlight_coords)
            if plot_3d:
                ax.scatter(
                    highlight_coords[:, 0],
                    highlight_coords[:, 1],
                    highlight_coords[:, 2],
                    color=highlight_color,
                    s=highlight_size,
                    alpha=highlight_alpha,
                    edgecolors="black",
                    linewidths=1,
                    zorder=10,
                )
            else:
                ax.scatter(
                    highlight_coords[:, 0],
                    highlight_coords[:, 1],
                    color=highlight_color,
                    s=highlight_size,
                    alpha=highlight_alpha,
                    edgecolors="black",
                    linewidths=1,
                    zorder=10,
                )

    # Add token connections if requested
    if add_token_connections is not None:
        for idx1, idx2 in add_token_connections:
            if idx1 in token_indices and idx2 in token_indices:
                pos1 = token_indices.index(idx1)
                pos2 = token_indices.index(idx2)
                if plot_3d:
                    ax.plot(
                        [coords[pos1, 0], coords[pos2, 0]],
                        [coords[pos1, 1], coords[pos2, 1]],
                        [coords[pos1, 2], coords[pos2, 2]],
                        linestyle=connection_style,
                        linewidth=connection_width,
                        alpha=connection_alpha,
                        color=connection_color,
                    )
                else:
                    ax.plot(
                        [coords[pos1, 0], coords[pos2, 0]],
                        [coords[pos1, 1], coords[pos2, 1]],
                        linestyle=connection_style,
                        linewidth=connection_width,
                        alpha=connection_alpha,
                        color=connection_color,
                    )

    # Add convex hull if requested (2D only)
    if add_convex_hull and not plot_3d and len(coords) >= 3:
        try:
            from scipy.spatial import ConvexHull

            hull = ConvexHull(coords)
            for simplex in hull.simplices:
                ax.plot(coords[simplex, 0], coords[simplex, 1], "k-", alpha=0.5)
        except Exception as e:
            print(f"Could not create convex hull: {e}")

    # Add centroid if requested
    if add_centroid:
        centroid = np.mean(coords, axis=0)
        if plot_3d:
            ax.scatter(
                [centroid[0]],
                [centroid[1]],
                [centroid[2]],
                color="black",
                marker="*",
                s=200,
                alpha=0.8,
                zorder=11,
            )
            ax.text(
                centroid[0],
                centroid[1],
                centroid[2],
                "centroid",
                fontsize=10,
                ha="center",
                va="bottom",
            )
        else:
            ax.scatter(
                [centroid[0]],
                [centroid[1]],
                color="black",
                marker="*",
                s=200,
                alpha=0.8,
                zorder=11,
            )
            ax.text(
                centroid[0],
                centroid[1],
                "centroid",
                fontsize=10,
                ha="center",
                va="bottom",
            )

    # Label tokens
    if label_tokens:
        tokens_to_label = []
        if isinstance(label_tokens, bool) and label_tokens:
            tokens_to_label = list(range(len(token_indices)))
        elif isinstance(label_tokens, int):
            tokens_to_label = list(range(0, len(token_indices), label_tokens))
        elif isinstance(label_tokens, (list, tuple)):
            tokens_to_label = [i for i in label_tokens if 0 <= i < len(token_indices)]

        for i in tokens_to_label:
            if 0 <= i < len(token_indices):
                if token_labels is not None:
                    label = token_labels[i]
                else:
                    label = f"t={token_indices[i]}"

                if plot_3d:
                    ax.text(
                        coords[i, 0],
                        coords[i, 1],
                        coords[i, 2],
                        label,
                        fontsize=label_fontsize,
                    )
                else:
                    ax.text(coords[i, 0], coords[i, 1], label, fontsize=label_fontsize)

    # Set axis labels
    if plot_3d:
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
    else:
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

        # Equal aspect ratio if requested
        if equal_aspect:
            ax.set_aspect("equal")

    # Add grid
    ax.grid(True, linestyle="--", alpha=0.3)

    # Set title
    layer_display = layer_label if layer_label is not None else f"layer={layer_index}"
    if title is None:
        title = f"Token scatter in PC space ({layer_display})"
        if len(token_indices) < T:
            title += f" [{len(token_indices)} tokens]"
    ax.set_title(title)

    # Add annotation about dimensions
    if annotate_dimensions:
        # This is just a placeholder - in a real implementation you'd
        # want to calculate the actual variance explained by each PC
        ax.text(
            0.02,
            0.98,
            "PC1: xx% variance\nPC2: yy% variance"
            + ("\nPC3: zz% variance" if plot_3d else ""),
            transform=ax.transAxes,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.7},
        )

    # Add legend if requested and there are multiple colors/markers
    if show_legend and (
        color_by is not None
        or isinstance(color, (list, tuple))
        or isinstance(marker, (list, tuple))
    ):
        ax.legend(loc=legend_loc)

    fig.tight_layout()
    return fig


# ------------------------ Line series from [L, T] map ------------------------ #


def plot_token_series_from_LT(
    lt: ArrayLike,
    token_indices: Union[int, list[int], Sequence[int]],
    *,
    layer_range: Optional[Tuple[int, int]] = None,
    title: Optional[str] = None,
    ylabel: str = "Value",
    xlabel: str = "Layer",
    layer_labels: Optional[Sequence[str]] = None,
    token_labels: Optional[Sequence[str]] = None,
    colors: Optional[Union[str, list[str]]] = None,
    line_styles: Optional[Union[str, list[str]]] = None,
    markers: Optional[Union[str, list[str]]] = "o",
    marker_size: Union[float, list[float]] = 6,
    line_width: Union[float, list[float]] = 1.5,
    alpha: float = 1.0,
    show_legend: bool = True,
    legend_loc: str = "best",
    highlight_layers: Optional[list[int]] = None,
    highlight_color: str = "yellow",
    highlight_alpha: float = 0.3,
    show_stats: bool = False,
    show_min_max: bool = False,
    show_average: bool = False,
    show_trend: bool = False,
    figsize: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    normalize_y: bool = False,
    log_scale: bool = False,
    add_annotations: Optional[dict[int, str]] = None,
    comparison_value: Optional[float] = None,
    comparison_style: str = "--",
    comparison_color: str = "red",
    comparison_label: Optional[str] = None,
    smoothing: Optional[int] = None,
) -> plt.Figure:
    """
    Plot one or more tokens' series across layers using an [L, T] map.

    This visualization shows how a token's representation evolves through the network,
    revealing patterns in how different layers process specific tokens. It can help
    identify important layers for certain tokens or compare processing patterns
    across different tokens.

    Works for reduction="norm" (per-layer L2 magnitude), "mean" (per-layer mean activation),
    or any other 2D array with layers as rows and tokens as columns.

    Args:
        lt: Array/Tensor [L, T].
        token_indices: Token(s) to visualize. Can be:
                      - A single integer (supports negative indexing)
                      - A list/sequence of integers for multiple tokens
        layer_range: Optional (start, end) tuple to plot only a subset of layers.
                    End is exclusive, following Python convention.
        title: Optional title; if None, a default is used.
        ylabel: Y-axis label, e.g., "L2 norm".
        xlabel: X-axis label, e.g., "Layer".
        layer_labels: Optional custom labels for layers (x-axis ticks).
        token_labels: Optional custom labels for tokens in the legend.
        colors: Color(s) for the plotted line(s). Can be a single color or list of colors.
        line_styles: Line style(s) for the plotted line(s). Can be a single style or list.
        markers: Marker style(s) for points. Can be a single marker code or list.
        marker_size: Size(s) of markers. Can be a single value or list.
        line_width: Width(s) of lines. Can be a single value or list.
        alpha: Transparency of lines and markers (0-1).
        show_legend: Whether to show a legend (when plotting multiple tokens).
        legend_loc: Location for the legend.
        highlight_layers: Optional list of layer indices to highlight with background color.
        highlight_color: Color for highlighted layers.
        highlight_alpha: Alpha (transparency) for highlighted regions.
        show_stats: If True, add statistics annotations to the plot.
        show_min_max: If True, mark the minimum and maximum points on each line.
        show_average: If True, add a horizontal line showing the average value.
        show_trend: If True, add a trend line (linear regression) for each token.
        figsize: Optional figure size as (width, height) in inches.
        ylim: Optional y-axis limits as (min, max).
        normalize_y: If True, normalize y values to [0,1] range for each token.
        log_scale: If True, use logarithmic scale for y-axis.
        add_annotations: Optional dictionary mapping layer indices to annotation strings.
        comparison_value: Optional horizontal line value to add for comparison.
        comparison_style: Line style for comparison line.
        comparison_color: Color for comparison line.
        comparison_label: Label for comparison line in legend.
        smoothing: Optional window size for moving average smoothing.

    Returns:
        Matplotlib Figure object for further customization or saving.

    Examples:
        # Basic plot for a single token
        fig = plot_token_series_from_LT(layer_token_data, token_index=5)

        # Compare multiple tokens with custom styling
        fig = plot_token_series_from_LT(
            layer_token_data,
            token_indices=[0, 5, 10],
            colors=["blue", "red", "green"],
            markers=["o", "s", "^"],
            token_labels=["First", "Middle", "Last"]
        )

        # Focus on specific layers with statistics
        fig = plot_token_series_from_LT(
            layer_token_data,
            token_indices=0,
            layer_range=(4, 12),
            highlight_layers=[6, 7],
            show_min_max=True,
            show_average=True
        )

        # Add trend analysis and annotations
        fig = plot_token_series_from_LT(
            layer_token_data,
            token_indices=[0, -1],
            show_trend=True,
            add_annotations={5: "Attention peak", 10: "FFN peak"}
        )
    """
    arr = _to_numpy(lt)
    _assert_shape(arr, ndim=2, name="lt")
    L, T = arr.shape

    # Handle layer range selection
    start_layer, end_layer = 0, L
    if layer_range is not None:
        start_layer, end_layer = layer_range
        # Validate layer range
        if not (
            0 <= start_layer < L and 0 < end_layer <= L and start_layer < end_layer
        ):
            raise ValueError(f"Invalid layer range {layer_range} for {L} layers")

    # Select the layers to plot
    arr_subset = arr[start_layer:end_layer]
    L_subset = end_layer - start_layer

    # Convert single token index to list for uniform processing
    if isinstance(token_indices, (int, np.integer)):
        token_indices = [token_indices]

    # Validate and normalize token indices
    norm_token_indices = []
    for idx in token_indices:
        tok = idx if idx >= 0 else T + idx
        if not (0 <= tok < T):
            raise IndexError(f"Token index {idx} out of range for T={T}")
        norm_token_indices.append(tok)

    # Validate token labels if provided
    if token_labels is not None and len(token_labels) != len(norm_token_indices):
        raise ValueError(
            f"Expected {len(norm_token_indices)} token labels, "
            f"got {len(token_labels)}"
        )

    # Validate layer labels if provided
    if layer_labels is not None:
        if len(layer_labels) != L_subset:
            raise ValueError(
                f"Expected {L_subset} layer labels for the selected range, "
                f"got {len(layer_labels)}"
            )

    # Create figure with specified size
    if figsize is None:
        figsize = (10, 6)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    # Set up colors
    if colors is None:
        # Default color cycle for multiple tokens
        if len(norm_token_indices) > 1:
            colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        else:
            colors = ["blue"]  # Default single color

    # Ensure colors is a list matching the number of tokens
    if isinstance(colors, str):
        colors = [colors] * len(norm_token_indices)
    elif len(colors) < len(norm_token_indices):
        # Cycle colors if not enough provided
        colors = (colors * (len(norm_token_indices) // len(colors) + 1))[
            : len(norm_token_indices)
        ]

    # Set up line styles
    if line_styles is None:
        line_styles = ["-"] * len(norm_token_indices)
    elif isinstance(line_styles, str):
        line_styles = [line_styles] * len(norm_token_indices)

    # Set up markers
    if markers is None:
        markers = [None] * len(norm_token_indices)
    elif isinstance(markers, str):
        markers = [markers] * len(norm_token_indices)

    # Set up marker sizes
    if isinstance(marker_size, (int, float)):
        marker_size = [marker_size] * len(norm_token_indices)

    # Set up line widths
    if isinstance(line_width, (int, float)):
        line_width = [line_width] * len(norm_token_indices)

    # X-coordinates for plotting
    x = np.arange(start_layer, end_layer)

    # Store lines for legend
    lines = []

    # Plot each token's series
    for i, token_idx in enumerate(norm_token_indices):
        # Extract y values for this token
        y = arr_subset[:, token_idx].copy()  # Copy to avoid modifying original data

        # Apply normalization if requested
        if normalize_y:
            y_min, y_max = np.min(y), np.max(y)
            if y_max > y_min:  # Avoid division by zero
                y = (y - y_min) / (y_max - y_min)

        # Apply smoothing if requested
        if smoothing is not None and smoothing > 1 and len(y) > smoothing:
            kernel = np.ones(smoothing) / smoothing
            # Use valid mode to avoid edge effects
            y_smooth = np.convolve(y, kernel, mode="valid")
            # Adjust x coordinates for the convolution
            x_smooth = (
                x[(smoothing - 1) // 2 : -(smoothing // 2)] if smoothing > 1 else x
            )
            # Plot both the original (faint) and smoothed lines
            ax.plot(x, y, alpha=0.3, color=colors[i], linestyle=":", linewidth=1)
            line = ax.plot(
                x_smooth,
                y_smooth,
                color=colors[i],
                linestyle=line_styles[i],
                marker=markers[i],
                markersize=marker_size[i],
                linewidth=line_width[i],
                alpha=alpha,
            )
        else:
            # Plot without smoothing
            line = ax.plot(
                x,
                y,
                color=colors[i],
                linestyle=line_styles[i],
                marker=markers[i],
                markersize=marker_size[i],
                linewidth=line_width[i],
                alpha=alpha,
            )

        lines.append(line[0])

        # Add token label to legend
        if token_labels is not None:
            lines[-1].set_label(token_labels[i])
        else:
            lines[-1].set_label(f"Token {token_idx}")

        # Show min/max points if requested
        if show_min_max:
            min_idx = np.argmin(y)
            max_idx = np.argmax(y)

            # Mark minimum point
            ax.plot(
                x[min_idx],
                y[min_idx],
                "v",
                color=colors[i],
                markersize=marker_size[i] * 1.5,
                alpha=alpha,
            )
            ax.text(
                x[min_idx],
                y[min_idx],
                f"min: {y[min_idx]:.3f}",
                ha="center",
                va="top",
                fontsize=8,
                color=colors[i],
            )

            # Mark maximum point
            ax.plot(
                x[max_idx],
                y[max_idx],
                "^",
                color=colors[i],
                markersize=marker_size[i] * 1.5,
                alpha=alpha,
            )
            ax.text(
                x[max_idx],
                y[max_idx],
                f"max: {y[max_idx]:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color=colors[i],
            )

        # Show average line if requested
        if show_average:
            avg = np.mean(y)
            ax.axhline(y=avg, color=colors[i], linestyle="--", alpha=0.5, linewidth=1)
            ax.text(
                x[-1],
                avg,
                f"avg: {avg:.3f}",
                ha="right",
                va="bottom",
                fontsize=8,
                color=colors[i],
            )

        # Show trend line if requested
        if show_trend and len(x) > 1:
            try:
                # Simple linear regression
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                trend_y = p(x)

                # Plot trend line
                ax.plot(x, trend_y, "--", color=colors[i], alpha=0.7, linewidth=1)

                # Add slope annotation
                slope = z[0]
                ax.text(
                    x[-1],
                    trend_y[-1],
                    f"slope: {slope:.3f}",
                    ha="right",
                    va="top",
                    fontsize=8,
                    color=colors[i],
                )
            except Exception as e:
                print(f"Could not calculate trend line: {e}")

    # Add comparison line if requested
    if comparison_value is not None:
        comp_line = ax.axhline(
            y=comparison_value,
            color=comparison_color,
            linestyle=comparison_style,
            linewidth=2,
            alpha=0.7,
        )
        if comparison_label:
            comp_line.set_label(comparison_label)

    # Highlight specific layers if requested
    if highlight_layers is not None:
        for layer_idx in highlight_layers:
            if start_layer <= layer_idx < end_layer:
                # Get relative index in the subset
                rel_idx = layer_idx - start_layer
                ax.axvspan(
                    x[rel_idx] - 0.5,
                    x[rel_idx] + 0.5,
                    color=highlight_color,
                    alpha=highlight_alpha,
                )

    # Add custom annotations if provided
    if add_annotations is not None:
        for layer_idx, annotation in add_annotations.items():
            if start_layer <= layer_idx < end_layer:
                # Get relative index in the subset
                rel_idx = layer_idx - start_layer
                # Find a good y position (above the maximum value at this x)
                y_values = [arr_subset[rel_idx, idx] for idx in norm_token_indices]
                y_pos = max(y_values) * 1.05
                ax.annotate(
                    annotation,
                    (x[rel_idx], y_pos),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    arrowprops={"arrowstyle": "->", "color": "gray"},
                )

    # Show statistics if requested
    if show_stats:
        stats_text = []
        for i, token_idx in enumerate(norm_token_indices):
            y = arr_subset[:, token_idx]
            token_name = (
                token_labels[i] if token_labels is not None else f"Token {token_idx}"
            )
            stats = f"{token_name}: "
            stats += f"min={np.min(y):.3f}, "
            stats += f"max={np.max(y):.3f}, "
            stats += f"mean={np.mean(y):.3f}, "
            stats += f"std={np.std(y):.3f}"
            stats_text.append(stats)

        # Add stats as text box
        ax.text(
            0.02,
            0.98,
            "\n".join(stats_text),
            transform=ax.transAxes,
            va="top",
            fontsize=8,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.7},
        )

    # Set axis labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Set custom x-ticks if layer_labels provided
    if layer_labels is not None:
        ax.set_xticks(x)
        ax.set_xticklabels(layer_labels, rotation=45, ha="right")

    # Set y-axis limits if provided
    if ylim is not None:
        ax.set_ylim(ylim)

    # Set log scale if requested
    if log_scale:
        ax.set_yscale("log")

    # Add grid
    ax.grid(True, linestyle="--", alpha=0.3)

    # Set title
    if title is None:
        if len(norm_token_indices) == 1:
            title = f"Layer series for token {token_indices[0]}"
        else:
            title = f"Layer series for {len(norm_token_indices)} tokens"

        # Add layer range info if specified
        if layer_range is not None:
            title += f" [layers {start_layer}–{end_layer-1}]"

    ax.set_title(title)

    # Add legend if requested and there are multiple tokens or a comparison line
    if show_legend and (len(norm_token_indices) > 1 or comparison_value is not None):
        ax.legend(loc=legend_loc)

    fig.tight_layout()
    return fig


def plot_layer_series_from_LT(
    lt: ArrayLike,
    layer_indices: Union[int, list[int], Sequence[int]],
    *,
    token_range: Optional[Tuple[int, int]] = None,
    title: Optional[str] = None,
    ylabel: str = "Value",
    xlabel: str = "Token index",
    layer_labels: Optional[Sequence[str]] = None,
    token_labels: Optional[Sequence[str]] = None,
    colors: Optional[Union[str, list[str]]] = None,
    line_styles: Optional[Union[str, list[str]]] = None,
    markers: Optional[Union[str, list[str]]] = "o",
    marker_size: Union[float, list[float]] = 6,
    line_width: Union[float, list[float]] = 1.5,
    alpha: float = 1.0,
    show_legend: bool = True,
    legend_loc: str = "best",
    highlight_tokens: Optional[list[int]] = None,
    highlight_color: str = "yellow",
    highlight_alpha: float = 0.3,
    show_stats: bool = False,
    show_min_max: bool = False,
    show_average: bool = False,
    show_trend: bool = False,
    figsize: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    normalize_y: bool = False,
    log_scale: bool = False,
    add_annotations: Optional[dict[int, str]] = None,
    comparison_value: Optional[float] = None,
    comparison_style: str = "--",
    comparison_color: str = "red",
    comparison_label: Optional[str] = None,
    smoothing: Optional[int] = None,
    skip_tokens: Optional[int] = None,
    token_tick_frequency: Optional[int] = None,
    show_position_markers: bool = False,
) -> plt.Figure:
    """
    Plot one or more layers' series across tokens using an [L, T] map.

    This visualization shows how different layers process tokens across a sequence,
    revealing patterns in how activations vary by position. It can help identify
    position-dependent patterns, attention to specific tokens, or compare how
    different layers process the same sequence.

    Works for reduction="norm" (per-layer L2 magnitude), "mean" (per-layer mean activation),
    or any other 2D array with layers as rows and tokens as columns.

    Args:
        lt: Array/Tensor [L, T].
        layer_indices: Layer(s) to visualize. Can be:
                      - A single integer (0-based)
                      - A list/sequence of integers for multiple layers
        token_range: Optional (start, end) tuple to plot only a subset of tokens.
                    End is exclusive, following Python convention.
        title: Optional title; if None, a default is used.
        ylabel: Y-axis label, e.g., "L2 norm".
        xlabel: X-axis label, e.g., "Token index".
        layer_labels: Optional custom labels for layers in the legend.
        token_labels: Optional custom labels for tokens (x-axis ticks).
        colors: Color(s) for the plotted line(s). Can be a single color or list of colors.
        line_styles: Line style(s) for the plotted line(s). Can be a single style or list.
        markers: Marker style(s) for points. Can be a single marker code or list.
        marker_size: Size(s) of markers. Can be a single value or list.
        line_width: Width(s) of lines. Can be a single value or list.
        alpha: Transparency of lines and markers (0-1).
        show_legend: Whether to show a legend (when plotting multiple layers).
        legend_loc: Location for the legend.
        highlight_tokens: Optional list of token indices to highlight with background color.
        highlight_color: Color for highlighted tokens.
        highlight_alpha: Alpha (transparency) for highlighted regions.
        show_stats: If True, add statistics annotations to the plot.
        show_min_max: If True, mark the minimum and maximum points on each line.
        show_average: If True, add a horizontal line showing the average value.
        show_trend: If True, add a trend line (linear regression) for each layer.
        figsize: Optional figure size as (width, height) in inches.
        ylim: Optional y-axis limits as (min, max).
        normalize_y: If True, normalize y values to [0,1] range for each layer.
        log_scale: If True, use logarithmic scale for y-axis.
        add_annotations: Optional dictionary mapping token indices to annotation strings.
        comparison_value: Optional horizontal line value to add for comparison.
        comparison_style: Line style for comparison line.
        comparison_color: Color for comparison line.
        comparison_label: Label for comparison line in legend.
        smoothing: Optional window size for moving average smoothing.
        skip_tokens: If set, only plot every nth token for cleaner visualization.
        token_tick_frequency: How often to show token ticks on the x-axis.
        show_position_markers: If True, add vertical lines at regular intervals.

    Returns:
        Matplotlib Figure object for further customization or saving.

    Examples:
        # Basic plot for a single layer
        fig = plot_layer_series_from_LT(layer_token_data, layer_index=5)

        # Compare multiple layers with custom styling
        fig = plot_layer_series_from_LT(
            layer_token_data,
            layer_indices=[0, 5, 10],
            colors=["blue", "red", "green"],
            markers=["o", "s", "^"],
            layer_labels=["Embedding", "Middle", "Final"]
        )

        # Focus on specific token range with statistics
        fig = plot_layer_series_from_LT(
            layer_token_data,
            layer_indices=0,
            token_range=(4, 12),
            highlight_tokens=[6, 7],
            show_min_max=True,
            show_average=True
        )

        # Add trend analysis and annotations
        fig = plot_layer_series_from_LT(
            layer_token_data,
            layer_indices=[0, -1],
            show_trend=True,
            add_annotations={5: "Subject", 10: "Verb", 15: "Object"}
        )
    """
    arr = _to_numpy(lt)
    _assert_shape(arr, ndim=2, name="lt")
    L, T = arr.shape

    # Handle token range selection
    start_token, end_token = 0, T
    if token_range is not None:
        start_token, end_token = token_range
        # Validate token range
        if not (
            0 <= start_token < T and 0 < end_token <= T and start_token < end_token
        ):
            raise ValueError(f"Invalid token range {token_range} for {T} tokens")

    # Select the tokens to plot
    if skip_tokens is not None and skip_tokens > 1:
        token_indices = list(range(start_token, end_token, skip_tokens))
    else:
        token_indices = list(range(start_token, end_token))

    # Convert single layer index to list for uniform processing
    if isinstance(layer_indices, (int, np.integer)):
        layer_indices = [layer_indices]

    # Validate and normalize layer indices
    norm_layer_indices = []
    for idx in layer_indices:
        layer_idx = idx if idx >= 0 else L + idx
        if not (0 <= layer_idx < L):
            raise IndexError(f"Layer index {idx} out of range for L={L}")
        norm_layer_indices.append(layer_idx)

    # Validate layer labels if provided
    if layer_labels is not None and len(layer_labels) != len(norm_layer_indices):
        raise ValueError(
            f"Expected {len(norm_layer_indices)} layer labels, "
            f"got {len(layer_labels)}"
        )

    # Validate token labels if provided
    if token_labels is not None:
        if len(token_labels) != len(token_indices):
            raise ValueError(
                f"Expected {len(token_indices)} token labels for the selected range, "
                f"got {len(token_labels)}"
            )

    # Create figure with specified size
    if figsize is None:
        figsize = (10, 6)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    # Set up colors
    if colors is None:
        # Default color cycle for multiple layers
        if len(norm_layer_indices) > 1:
            colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        else:
            colors = ["blue"]  # Default single color

    # Ensure colors is a list matching the number of layers
    if isinstance(colors, str):
        colors = [colors] * len(norm_layer_indices)
    elif len(colors) < len(norm_layer_indices):
        # Cycle colors if not enough provided
        colors = (colors * (len(norm_layer_indices) // len(colors) + 1))[
            : len(norm_layer_indices)
        ]

    # Set up line styles
    if line_styles is None:
        line_styles = ["-"] * len(norm_layer_indices)
    elif isinstance(line_styles, str):
        line_styles = [line_styles] * len(norm_layer_indices)

    # Set up markers
    if markers is None:
        markers = [None] * len(norm_layer_indices)
    elif isinstance(markers, str):
        markers = [markers] * len(norm_layer_indices)

    # Set up marker sizes
    if isinstance(marker_size, (int, float)):
        marker_size = [marker_size] * len(norm_layer_indices)

    # Set up line widths
    if isinstance(line_width, (int, float)):
        line_width = [line_width] * len(norm_layer_indices)

    # X-coordinates for plotting
    x = np.array(token_indices)

    # Store lines for legend
    lines = []

    # Plot each layer's series
    for i, layer_idx in enumerate(norm_layer_indices):
        # Extract y values for this layer
        y = arr[
            layer_idx, token_indices
        ].copy()  # Copy to avoid modifying original data

        # Apply normalization if requested
        if normalize_y:
            y_min, y_max = np.min(y), np.max(y)
            if y_max > y_min:  # Avoid division by zero
                y = (y - y_min) / (y_max - y_min)

        # Apply smoothing if requested
        if smoothing is not None and smoothing > 1 and len(y) > smoothing:
            kernel = np.ones(smoothing) / smoothing
            # Use valid mode to avoid edge effects
            y_smooth = np.convolve(y, kernel, mode="valid")
            # Adjust x coordinates for the convolution
            x_smooth = (
                x[(smoothing - 1) // 2 : -(smoothing // 2)] if smoothing > 1 else x
            )
            # Plot both the original (faint) and smoothed lines
            ax.plot(x, y, alpha=0.3, color=colors[i], linestyle=":", linewidth=1)
            line = ax.plot(
                x_smooth,
                y_smooth,
                color=colors[i],
                linestyle=line_styles[i],
                marker=markers[i],
                markersize=marker_size[i],
                linewidth=line_width[i],
                alpha=alpha,
            )
        else:
            # Plot without smoothing
            line = ax.plot(
                x,
                y,
                color=colors[i],
                linestyle=line_styles[i],
                marker=markers[i],
                markersize=marker_size[i],
                linewidth=line_width[i],
                alpha=alpha,
            )

        lines.append(line[0])

        # Add layer label to legend
        if layer_labels is not None:
            lines[-1].set_label(layer_labels[i])
        else:
            lines[-1].set_label(f"Layer {layer_idx}")

        # Show min/max points if requested
        if show_min_max:
            min_idx = np.argmin(y)
            max_idx = np.argmax(y)

            # Mark minimum point
            ax.plot(
                x[min_idx],
                y[min_idx],
                "v",
                color=colors[i],
                markersize=marker_size[i] * 1.5,
                alpha=alpha,
            )
            ax.text(
                x[min_idx],
                y[min_idx],
                f"min: {y[min_idx]:.3f}",
                ha="center",
                va="top",
                fontsize=8,
                color=colors[i],
            )

            # Mark maximum point
            ax.plot(
                x[max_idx],
                y[max_idx],
                "^",
                color=colors[i],
                markersize=marker_size[i] * 1.5,
                alpha=alpha,
            )
            ax.text(
                x[max_idx],
                y[max_idx],
                f"max: {y[max_idx]:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color=colors[i],
            )

        # Show average line if requested
        if show_average:
            avg = np.mean(y)
            ax.axhline(y=avg, color=colors[i], linestyle="--", alpha=0.5, linewidth=1)
            ax.text(
                x[-1],
                avg,
                f"avg: {avg:.3f}",
                ha="right",
                va="bottom",
                fontsize=8,
                color=colors[i],
            )

        # Show trend line if requested
        if show_trend and len(x) > 1:
            try:
                # Simple linear regression
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                trend_y = p(x)

                # Plot trend line
                ax.plot(x, trend_y, "--", color=colors[i], alpha=0.7, linewidth=1)

                # Add slope annotation
                slope = z[0]
                ax.text(
                    x[-1],
                    trend_y[-1],
                    f"slope: {slope:.3f}",
                    ha="right",
                    va="top",
                    fontsize=8,
                    color=colors[i],
                )
            except Exception as e:
                print(f"Could not calculate trend line: {e}")

    # Add comparison line if requested
    if comparison_value is not None:
        comp_line = ax.axhline(
            y=comparison_value,
            color=comparison_color,
            linestyle=comparison_style,
            linewidth=2,
            alpha=0.7,
        )
        if comparison_label:
            comp_line.set_label(comparison_label)

    # Highlight specific tokens if requested
    if highlight_tokens is not None:
        for token_idx in highlight_tokens:
            if start_token <= token_idx < end_token:
                # Only highlight if token is in the selected range
                if skip_tokens is not None:
                    # Check if this token is included in our skipped sequence
                    if (token_idx - start_token) % skip_tokens != 0:
                        continue

                # Get position in the x array
                try:
                    pos = token_indices.index(token_idx)
                    ax.axvspan(
                        x[pos] - 0.5,
                        x[pos] + 0.5,
                        color=highlight_color,
                        alpha=highlight_alpha,
                    )
                except ValueError:
                    # Token not in our indices
                    pass

    # Add custom annotations if provided
    if add_annotations is not None:
        for token_idx, annotation in add_annotations.items():
            if start_token <= token_idx < end_token:
                # Only annotate if token is in the selected range
                if skip_tokens is not None:
                    # Check if this token is included in our skipped sequence
                    if (token_idx - start_token) % skip_tokens != 0:
                        continue

                # Get position in the x array
                try:
                    pos = token_indices.index(token_idx)
                    # Find a good y position (above the maximum value at this x)
                    y_values = [
                        arr[layer_idx, token_idx] for layer_idx in norm_layer_indices
                    ]
                    y_pos = max(y_values) * 1.05
                    ax.annotate(
                        annotation,
                        (x[pos], y_pos),
                        ha="center",
                        va="bottom",
                        fontsize=9,
                        arrowprops={"arrowstyle": "->", "color": "gray"},
                    )
                except ValueError:
                    # Token not in our indices
                    pass

    # Add position markers if requested
    if show_position_markers:
        # Determine a reasonable interval for position markers
        interval = max(1, len(token_indices) // 10)
        for i in range(0, len(token_indices), interval):
            ax.axvline(x=x[i], color="gray", linestyle=":", alpha=0.3)

    # Show statistics if requested
    if show_stats:
        stats_text = []
        for i, layer_idx in enumerate(norm_layer_indices):
            y = arr[layer_idx, token_indices]
            layer_name = (
                layer_labels[i] if layer_labels is not None else f"Layer {layer_idx}"
            )
            stats = f"{layer_name}: "
            stats += f"min={np.min(y):.3f}, "
            stats += f"max={np.max(y):.3f}, "
            stats += f"mean={np.mean(y):.3f}, "
            stats += f"std={np.std(y):.3f}"
            stats_text.append(stats)

        # Add stats as text box
        ax.text(
            0.02,
            0.98,
            "\n".join(stats_text),
            transform=ax.transAxes,
            va="top",
            fontsize=8,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.7},
        )

    # Set axis labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Set custom x-ticks if token_labels provided or token_tick_frequency specified
    if token_labels is not None:
        ax.set_xticks(x)
        ax.set_xticklabels(token_labels, rotation=45, ha="right")
    elif token_tick_frequency is not None:
        # Show only a subset of token indices for readability
        tick_indices = list(range(0, len(token_indices), token_tick_frequency))
        tick_positions = [x[i] for i in tick_indices]
        tick_labels = [str(token_indices[i]) for i in tick_indices]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)

    # Set y-axis limits if provided
    if ylim is not None:
        ax.set_ylim(ylim)

    # Set log scale if requested
    if log_scale:
        ax.set_yscale("log")

    # Add grid
    ax.grid(True, linestyle="--", alpha=0.3)

    # Set title
    if title is None:
        if len(norm_layer_indices) == 1:
            title = f"Token series for layer {layer_indices[0]}"
        else:
            title = f"Token series for {len(norm_layer_indices)} layers"

        # Add token range info if specified
        if token_range is not None:
            title += f" [tokens {start_token}–{end_token-1}]"

    ax.set_title(title)

    # Add legend if requested and there are multiple layers or a comparison line
    if show_legend and (len(norm_layer_indices) > 1 or comparison_value is not None):
        ax.legend(loc=legend_loc)

    fig.tight_layout()
    return fig


# ---------------------- Token contribution bars from [L, T] ---------------------- #


def plot_token_contributions_from_LT(
    lt: ArrayLike,
    token_indices: Union[int, list[int], Sequence[int]],
    *,
    layer_range: Optional[Tuple[int, int]] = None,
    title: Optional[str] = None,
    ylabel: str = "Value",
    xlabel: str = "Layer",
    token_labels: Optional[Sequence[str]] = None,
    layer_labels: Optional[Sequence[str]] = None,
    colors: Optional[Union[str, list[str]]] = None,
    bar_width: float = 0.8,
    alpha: float = 0.8,
    show_legend: bool = True,
    legend_loc: str = "best",
    highlight_layers: Optional[list[int]] = None,
    highlight_color: str = "yellow",
    highlight_alpha: float = 0.3,
    show_values: bool = False,
    value_format: str = ".3f",
    value_fontsize: float = 8,
    show_stats: bool = False,
    show_threshold: Optional[float] = None,
    threshold_color: str = "red",
    threshold_linestyle: str = "--",
    threshold_label: Optional[str] = None,
    sort_by_value: bool = False,
    normalize: bool = False,
    horizontal: bool = False,
    figsize: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    log_scale: bool = False,
    bar_style: str = "grouped",
    show_total: bool = False,
    add_annotations: Optional[dict[int, str]] = None,
    group_layers: Optional[list[Tuple[int, int, str]]] = None,
) -> plt.Figure:
    """
    Bar chart showing per-layer contributions for one or more tokens using an [L, T] map.

    This visualization helps identify which layers contribute most to specific tokens,
    revealing where in the network important processing happens. It's especially useful
    when using reduction="norm" (L2 magnitude) to see activation strength by layer.

    Args:
        lt: Array/Tensor [L, T].
        token_indices: Token(s) to visualize. Can be:
                      - A single integer (supports negative indexing)
                      - A list/sequence of integers for multiple tokens
        layer_range: Optional (start, end) tuple to plot only a subset of layers.
                    End is exclusive, following Python convention.
        title: Optional title; if None, a default is used.
        ylabel: Y-axis label (e.g., "L2 norm").
        xlabel: X-axis label (e.g., "Layer").
        token_labels: Optional custom labels for tokens in the legend.
        layer_labels: Optional custom labels for layers (x-axis ticks).
        colors: Color(s) for the bars. Can be a single color or list of colors.
        bar_width: Width of bars (between 0 and 1).
        alpha: Transparency of bars (0-1).
        show_legend: Whether to show a legend (when plotting multiple tokens).
        legend_loc: Location for the legend.
        highlight_layers: Optional list of layer indices to highlight with background color.
        highlight_color: Color for highlighted layers.
        highlight_alpha: Alpha (transparency) for highlighted regions.
        show_values: If True, display values on top of each bar.
        value_format: Format string for bar values if show_values=True.
        value_fontsize: Font size for value labels.
        show_stats: If True, add statistics annotations to the plot.
        show_threshold: Optional threshold value to show as a horizontal line.
        threshold_color: Color for threshold line.
        threshold_linestyle: Line style for threshold line.
        threshold_label: Label for threshold line in legend.
        sort_by_value: If True, sort layers by their values (descending).
        normalize: If True, normalize values to sum to 1 (percentage contribution).
        horizontal: If True, create a horizontal bar chart.
        figsize: Optional figure size as (width, height) in inches.
        ylim: Optional y-axis limits as (min, max).
        log_scale: If True, use logarithmic scale for y-axis.
        bar_style: Bar style: "grouped" (side by side) or "stacked".
        show_total: If True, add a "Total" bar showing the sum across tokens.
        add_annotations: Optional dictionary mapping layer indices to annotation strings.
        group_layers: Optional list of (start, end, label) tuples to group layers.

    Returns:
        Matplotlib Figure object for further customization or saving.

    Examples:
        # Basic contribution plot for a single token
        fig = plot_token_contributions_from_LT(layer_token_data, token_index=5)

        # Compare multiple tokens with custom styling
        fig = plot_token_contributions_from_LT(
            layer_token_data,
            token_indices=[0, 5, 10],
            colors=["blue", "red", "green"],
            token_labels=["First", "Middle", "Last"],
            show_values=True
        )

        # Focus on specific layers with normalization
        fig = plot_token_contributions_from_LT(
            layer_token_data,
            token_indices=0,
            layer_range=(4, 12),
            normalize=True,
            highlight_layers=[6, 7],
            show_threshold=0.1
        )

        # Horizontal bar chart with layer grouping
        fig = plot_token_contributions_from_LT(
            layer_token_data,
            token_indices=[0, -1],
            horizontal=True,
            group_layers=[(0, 4, "Input"), (4, 8, "Middle"), (8, 12, "Output")]
        )
    """
    arr = _to_numpy(lt)
    _assert_shape(arr, ndim=2, name="lt")
    L, T = arr.shape

    # Handle layer range selection
    start_layer, end_layer = 0, L
    if layer_range is not None:
        start_layer, end_layer = layer_range
        # Validate layer range
        if not (
            0 <= start_layer < L and 0 < end_layer <= L and start_layer < end_layer
        ):
            raise ValueError(f"Invalid layer range {layer_range} for {L} layers")

    # Select the layers to plot
    arr_subset = arr[start_layer:end_layer]
    L_subset = end_layer - start_layer

    # Convert single token index to list for uniform processing
    if isinstance(token_indices, (int, np.integer)):
        token_indices = [token_indices]

    # Validate and normalize token indices
    norm_token_indices = []
    for idx in token_indices:
        tok = idx if idx >= 0 else T + idx
        if not (0 <= tok < T):
            raise IndexError(f"Token index {idx} out of range for T={T}")
        norm_token_indices.append(tok)

    # Validate token labels if provided
    if token_labels is not None and len(token_labels) != len(norm_token_indices):
        raise ValueError(
            f"Expected {len(norm_token_indices)} token labels, "
            f"got {len(token_labels)}"
        )

    # Validate layer labels if provided
    if layer_labels is not None:
        if len(layer_labels) != L_subset:
            raise ValueError(
                f"Expected {L_subset} layer labels for the selected range, "
                f"got {len(layer_labels)}"
            )

    # Extract values for each token
    values = []
    for tok_idx in norm_token_indices:
        values.append(arr_subset[:, tok_idx].copy())

    # Add total if requested
    if show_total and len(norm_token_indices) > 1:
        total_values = np.sum(values, axis=0)
        values.append(total_values)
        if token_labels is not None:
            token_labels = list(token_labels) + ["Total"]
        norm_token_indices = list(norm_token_indices) + ["Total"]

    # Apply normalization if requested
    if normalize:
        for i in range(len(values)):
            total = np.sum(values[i])
            if total > 0:  # Avoid division by zero
                values[i] = values[i] / total

    # Sort layers by value if requested
    if sort_by_value:
        # Use the first token's values for sorting
        sort_indices = np.argsort(values[0])[::-1]  # Descending
        for i in range(len(values)):
            values[i] = values[i][sort_indices]

        # Adjust layer labels if provided
        if layer_labels is not None:
            layer_labels = [layer_labels[i] for i in sort_indices]

        # Adjust highlight layers if provided
        if highlight_layers is not None:
            # This is more complex with sorting, so we'll disable it
            highlight_layers = None
            print("Warning: highlight_layers is not supported with sort_by_value=True")

    # Create figure with specified size
    if figsize is None:
        if horizontal:
            # Make horizontal plots taller for better readability
            figsize = (10, max(6, L_subset * 0.3 + 2))
        else:
            figsize = (max(8, L_subset * 0.3 + 2), 6)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    # Set up colors
    if colors is None:
        # Default color cycle for multiple tokens
        if len(norm_token_indices) > 1:
            colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        else:
            colors = ["blue"]  # Default single color

    # Ensure colors is a list matching the number of tokens
    if isinstance(colors, str):
        colors = [colors] * len(norm_token_indices)
    elif len(colors) < len(norm_token_indices):
        # Cycle colors if not enough provided
        colors = (colors * (len(norm_token_indices) // len(colors) + 1))[
            : len(norm_token_indices)
        ]

    # X-coordinates for plotting
    x = np.arange(L_subset)

    # Adjust bar width for grouped bars
    if bar_style == "grouped" and len(norm_token_indices) > 1:
        width = bar_width / len(norm_token_indices)
        offsets = np.linspace(
            -bar_width / 2 + width / 2,
            bar_width / 2 - width / 2,
            len(norm_token_indices),
        )
    else:
        width = bar_width
        offsets = [0] * len(norm_token_indices)

    # Store bars for legend
    bars = []

    # Plot bars for each token
    if bar_style == "stacked" and len(norm_token_indices) > 1:
        # Stacked bars
        bottom = np.zeros(L_subset)
        for i, vals in enumerate(values):
            if horizontal:
                bar = ax.barh(
                    x, vals, height=width, left=bottom, color=colors[i], alpha=alpha
                )
            else:
                bar = ax.bar(
                    x, vals, width=width, bottom=bottom, color=colors[i], alpha=alpha
                )
            bottom += vals
            bars.append(bar)

            # Add token label to legend
            if token_labels is not None:
                bars[-1].set_label(token_labels[i])
            else:
                label = (
                    "Total"
                    if i == len(values) - 1 and show_total
                    else f"Token {norm_token_indices[i]}"
                )
                bars[-1].set_label(label)

            # Show values if requested
            if show_values:
                for j, v in enumerate(vals):
                    if v > 0:  # Only show positive values
                        if horizontal:
                            text_x = bottom[j] - v / 2
                            text_y = x[j]
                            ha, va = "center", "center"
                        else:
                            text_x = x[j]
                            text_y = bottom[j] - v / 2
                            ha, va = "center", "center"

                        ax.text(
                            text_x,
                            text_y,
                            f"{v:{value_format}}",
                            ha=ha,
                            va=va,
                            fontsize=value_fontsize,
                            color="white",
                            fontweight="bold",
                        )
    else:
        # Grouped bars or single token
        for i, vals in enumerate(values):
            if horizontal:
                bar = ax.barh(
                    x + offsets[i], vals, height=width, color=colors[i], alpha=alpha
                )
            else:
                bar = ax.bar(
                    x + offsets[i], vals, width=width, color=colors[i], alpha=alpha
                )
            bars.append(bar)

            # Add token label to legend
            if token_labels is not None:
                bars[-1].set_label(token_labels[i])
            else:
                label = (
                    "Total"
                    if i == len(values) - 1 and show_total
                    else f"Token {norm_token_indices[i]}"
                )
                bars[-1].set_label(label)

            # Show values if requested
            if show_values:
                for j, v in enumerate(vals):
                    if horizontal:
                        text_x = v + (v * 0.02)  # Slight offset
                        text_y = x[j] + offsets[i]
                        ha, va = "left", "center"
                    else:
                        text_x = x[j] + offsets[i]
                        text_y = v + (v * 0.02)  # Slight offset
                        ha, va = "center", "bottom"

                    ax.text(
                        text_x,
                        text_y,
                        f"{v:{value_format}}",
                        ha=ha,
                        va=va,
                        fontsize=value_fontsize,
                        color="black",
                    )

    # Add threshold line if requested
    if show_threshold is not None:
        if horizontal:
            line = ax.axvline(
                x=show_threshold,
                color=threshold_color,
                linestyle=threshold_linestyle,
                linewidth=2,
                alpha=0.7,
            )
        else:
            line = ax.axhline(
                y=show_threshold,
                color=threshold_color,
                linestyle=threshold_linestyle,
                linewidth=2,
                alpha=0.7,
            )
        if threshold_label:
            line.set_label(threshold_label)

    # Highlight specific layers if requested
    if highlight_layers is not None:
        for layer_idx in highlight_layers:
            if start_layer <= layer_idx < end_layer:
                # Get relative index in the subset
                rel_idx = layer_idx - start_layer
                if horizontal:
                    ax.axhspan(
                        rel_idx - 0.5,
                        rel_idx + 0.5,
                        color=highlight_color,
                        alpha=highlight_alpha,
                    )
                else:
                    ax.axvspan(
                        rel_idx - 0.5,
                        rel_idx + 0.5,
                        color=highlight_color,
                        alpha=highlight_alpha,
                    )

    # Add layer group backgrounds if requested
    if group_layers is not None:
        for start_idx, end_idx, group_label in group_layers:
            # Adjust indices for layer_range
            rel_start = max(0, start_idx - start_layer)
            rel_end = min(L_subset, end_idx - start_layer)

            if rel_start < rel_end:  # Only if group is visible in current range
                if horizontal:
                    # rect = ax.axhspan(
                    #     rel_start - 0.5, rel_end - 0.5, color="gray", alpha=0.1
                    # )  # Currently unused
                    ax.axhspan(rel_start - 0.5, rel_end - 0.5, color="gray", alpha=0.1)
                    # Add group label
                    ax.text(
                        -0.01,
                        (rel_start + rel_end) / 2,
                        group_label,
                        ha="right",
                        va="center",
                        fontsize=10,
                        rotation=90,
                        transform=ax.get_yaxis_transform(),
                    )
                else:
                    # rect = ax.axvspan(
                    #     rel_start - 0.5, rel_end - 0.5, color="gray", alpha=0.1
                    # )  # Currently unused
                    ax.axvspan(rel_start - 0.5, rel_end - 0.5, color="gray", alpha=0.1)
                    # Add group label
                    ax.text(
                        (rel_start + rel_end) / 2,
                        -0.01,
                        group_label,
                        ha="center",
                        va="top",
                        fontsize=10,
                        transform=ax.get_xaxis_transform(),
                    )

    # Add custom annotations if provided
    if add_annotations is not None:
        for layer_idx, annotation in add_annotations.items():
            if start_layer <= layer_idx < end_layer:
                # Get relative index in the subset
                rel_idx = layer_idx - start_layer

                # Find a good position for the annotation
                if horizontal:
                    max_val = max(vals[rel_idx] for vals in values)
                    ax.annotate(
                        annotation,
                        (max_val * 1.05, rel_idx),
                        ha="left",
                        va="center",
                        fontsize=9,
                    )
                else:
                    max_val = max(vals[rel_idx] for vals in values)
                    ax.annotate(
                        annotation,
                        (rel_idx, max_val * 1.05),
                        ha="center",
                        va="bottom",
                        fontsize=9,
                    )

    # Show statistics if requested
    if show_stats:
        stats_text = []
        for i, tok_idx in enumerate(norm_token_indices):
            if tok_idx == "Total" and show_total:
                token_name = "Total"
            else:
                token_name = (
                    token_labels[i] if token_labels is not None else f"Token {tok_idx}"
                )

            vals = values[i]
            stats = f"{token_name}: "
            stats += f"sum={np.sum(vals):.3f}, "
            stats += f"max={np.max(vals):.3f} (layer {np.argmax(vals) + start_layer}), "
            stats += f"mean={np.mean(vals):.3f}"
            stats_text.append(stats)

        # Add stats as text box
        ax.text(
            0.02,
            0.98,
            "\n".join(stats_text),
            transform=ax.transAxes,
            va="top",
            fontsize=8,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.7},
        )

    # Set axis labels
    if horizontal:
        ax.set_ylabel(xlabel)
        ax.set_xlabel(ylabel)
    else:
        ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Set custom ticks if layer_labels provided
    if horizontal:
        ax.set_yticks(x)
        if layer_labels is not None:
            ax.set_yticklabels(layer_labels)
        else:
            # Default layer labels (adjusted for layer_range)
            ax.set_yticklabels([str(i + start_layer) for i in range(L_subset)])
    else:
        ax.set_xticks(x)
        if layer_labels is not None:
            ax.set_xticklabels(layer_labels, rotation=45, ha="right")
        else:
            # Default layer labels (adjusted for layer_range)
            ax.set_xticklabels([str(i + start_layer) for i in range(L_subset)])

    # Set axis limits if provided or adjust for horizontal bars
    if horizontal:
        if ylim is not None:
            ax.set_xlim(ylim)  # Note: ylim applies to x-axis in horizontal mode
    else:
        if ylim is not None:
            ax.set_ylim(ylim)

    # Set log scale if requested
    if log_scale:
        if horizontal:
            ax.set_xscale("log")
        else:
            ax.set_yscale("log")

    # Add grid for better readability
    ax.grid(True, linestyle="--", alpha=0.3, axis="both")

    # Set title
    if title is None:
        if len(norm_token_indices) == 1:
            title = f"Layer contributions for token {token_indices[0]}"
        else:
            title = f"Layer contributions for {len(values) - (1 if show_total else 0)} tokens"
            if show_total:
                title += " (with total)"

        # Add layer range info if specified
        if layer_range is not None:
            title += f" [layers {start_layer}–{end_layer-1}]"

        # Add normalization info if specified
        if normalize:
            title += " (normalized)"

    ax.set_title(title)

    # Add legend if requested and there are multiple tokens or a threshold
    if show_legend and (len(norm_token_indices) > 1 or show_threshold is not None):
        ax.legend(loc=legend_loc)

    fig.tight_layout()
    return fig


# ---------------------------- Δ heatmap between maps ---------------------------- #


def plot_delta_heatmap_LT(
    lt_a: ArrayLike,
    lt_b: ArrayLike,
    *,
    title: str = "Δ Map (B − A)",
    subtitle: Optional[str] = None,
    xlabel: str = "Token index",
    ylabel: str = "Layer",
    condition_a_name: str = "A",
    condition_b_name: str = "B",
    layer_range: Optional[Tuple[int, int]] = None,
    token_range: Optional[Tuple[int, int]] = None,
    layer_labels: Optional[Sequence[str]] = None,
    token_labels: Optional[Sequence[str]] = None,
    show_colorbar: bool = True,
    colorbar_label: Optional[str] = "Δ Value",
    cmap: Union[str, Colormap] = "RdBu_r",
    center_colormap: bool = True,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    show_values: bool = False,
    value_format: str = ".2f",
    value_fontsize: float = 7,
    highlight_threshold: Optional[float] = None,
    highlight_color: str = "black",
    highlight_alpha: float = 0.3,
    show_stats: bool = False,
    show_side_plots: bool = False,
    side_plot_width: float = 0.2,
    figsize: Optional[Tuple[float, float]] = None,
    log_scale: bool = False,
    absolute_diff: bool = False,
    relative_diff: bool = False,
    percent_diff: bool = False,
    mask_zeros: bool = False,
    mask_nan: bool = True,
    add_annotations: Optional[dict[Tuple[int, int], str]] = None,
    highlight_regions: Optional[list[Tuple[int, int, int, int]]] = None,
    show_original_heatmaps: bool = False,
) -> plt.Figure:
    """
    Create a heatmap showing the difference between two L×T maps, useful for comparing
    activations across different conditions, prompts, or interventions.

    This visualization helps identify where and how much two conditions differ in their
    layer-token activation patterns. It's especially useful for analyzing the effect of
    interventions or different inputs on model activations.

    Args:
        lt_a: L×T array/tensor for condition A.
        lt_b: L×T array/tensor for condition B. Must match shape of lt_a.
        title: Plot title.
        subtitle: Optional subtitle for additional context.
        xlabel: X-axis label (tokens).
        ylabel: Y-axis label (layers).
        condition_a_name: Name/label for condition A (for legend and stats).
        condition_b_name: Name/label for condition B (for legend and stats).
        layer_range: Optional (start, end) tuple to plot only a subset of layers.
                    End is exclusive, following Python convention.
        token_range: Optional (start, end) tuple to plot only a subset of tokens.
                    End is exclusive, following Python convention.
        layer_labels: Optional custom labels for layers (y-axis ticks).
        token_labels: Optional custom labels for tokens (x-axis ticks).
        show_colorbar: Whether to draw a colorbar.
        colorbar_label: Label for the colorbar.
        cmap: Colormap name or matplotlib colormap object. Diverging colormaps
              like "RdBu_r" are recommended for delta heatmaps.
        center_colormap: If True, center the colormap at zero for balanced visualization
                        of positive and negative differences.
        vmin: Minimum value for colormap scaling. If None, uses data minimum.
        vmax: Maximum value for colormap scaling. If None, uses data maximum.
        show_values: If True, display difference values in each cell.
        value_format: Format string for cell values if show_values=True.
        value_fontsize: Font size for value labels.
        highlight_threshold: If provided, highlight cells with absolute difference
                           above this threshold.
        highlight_color: Color for highlighted cells.
        highlight_alpha: Alpha (transparency) for highlighted cells.
        show_stats: If True, add statistics annotations to the plot.
        show_side_plots: If True, add side plots showing mean differences by layer and token.
        side_plot_width: Width of side plots as a fraction of the main plot.
        figsize: Optional figure size as (width, height) in inches.
        log_scale: If True, apply log(abs(x)) * sign(x) transform to differences.
        absolute_diff: If True, show absolute differences |B-A| instead of B-A.
        relative_diff: If True, show relative differences (B-A)/A where A≠0.
        percent_diff: If True, show percent differences ((B-A)/A)*100 where A≠0.
        mask_zeros: If True, mask out cells where both A and B are zero.
        mask_nan: If True, mask out cells where either A or B is NaN.
        add_annotations: Optional dictionary mapping (layer_idx, token_idx) to annotation strings.
        highlight_regions: Optional list of (layer_start, token_start, layer_end, token_end)
                         tuples defining rectangular regions to highlight.
        show_original_heatmaps: If True, include small heatmaps of the original A and B data.

    Returns:
        Matplotlib Figure object for further customization or saving.

    Examples:
        # Basic difference heatmap
        fig = plot_delta_heatmap_LT(data_a, data_b)

        # Customized difference heatmap with condition names and centered colormap
        fig = plot_delta_heatmap_LT(
            data_a, data_b,
            condition_a_name="Baseline",
            condition_b_name="Intervention",
            center_colormap=True,
            show_stats=True
        )

        # Focus on specific regions with highlighting
        fig = plot_delta_heatmap_LT(
            data_a, data_b,
            layer_range=(4, 12),
            token_range=(10, 30),
            highlight_threshold=0.5,
            show_values=True
        )

        # Show percent difference with side plots
        fig = plot_delta_heatmap_LT(
            data_a, data_b,
            percent_diff=True,
            show_side_plots=True,
            colorbar_label="% Difference"
        )
    """
    A = _to_numpy(lt_a)
    B = _to_numpy(lt_b)
    _assert_shape(A, ndim=2, name="lt_a")
    _assert_shape(B, ndim=2, name="lt_b")
    if A.shape != B.shape:
        raise ValueError(f"Shape mismatch: lt_a {A.shape} vs lt_b {B.shape}")

    L, T = A.shape

    # Handle layer and token range selection
    start_layer, end_layer = 0, L
    if layer_range is not None:
        start_layer, end_layer = layer_range
        # Validate layer range
        if not (
            0 <= start_layer < L and 0 < end_layer <= L and start_layer < end_layer
        ):
            raise ValueError(f"Invalid layer range {layer_range} for {L} layers")

    start_token, end_token = 0, T
    if token_range is not None:
        start_token, end_token = token_range
        # Validate token range
        if not (
            0 <= start_token < T and 0 < end_token <= T and start_token < end_token
        ):
            raise ValueError(f"Invalid token range {token_range} for {T} tokens")

    # Select the subset to plot
    A_subset = A[start_layer:end_layer, start_token:end_token]
    B_subset = B[start_layer:end_layer, start_token:end_token]
    L_subset, T_subset = A_subset.shape

    # Validate layer and token labels if provided
    if layer_labels is not None and len(layer_labels) != L_subset:
        raise ValueError(
            f"Expected {L_subset} layer labels for the selected range, "
            f"got {len(layer_labels)}"
        )

    if token_labels is not None and len(token_labels) != T_subset:
        raise ValueError(
            f"Expected {T_subset} token labels for the selected range, "
            f"got {len(token_labels)}"
        )

    # Calculate difference based on selected mode
    if absolute_diff:
        delta = np.abs(B_subset - A_subset)
        diff_type = "absolute"
    elif relative_diff:
        # Avoid division by zero
        mask = A_subset != 0
        delta = np.zeros_like(A_subset)
        delta[mask] = (B_subset[mask] - A_subset[mask]) / A_subset[mask]
        diff_type = "relative"
    elif percent_diff:
        # Avoid division by zero
        mask = A_subset != 0
        delta = np.zeros_like(A_subset)
        delta[mask] = ((B_subset[mask] - A_subset[mask]) / A_subset[mask]) * 100
        diff_type = "percent"
    else:
        delta = B_subset - A_subset
        diff_type = "raw"

    # Apply masking if requested
    if mask_zeros:
        # Mask where both A and B are zero (no difference)
        zero_mask = (A_subset == 0) & (B_subset == 0)
        delta = np.ma.masked_array(delta, mask=zero_mask)

    if mask_nan:
        # Mask NaN values
        nan_mask = np.isnan(A_subset) | np.isnan(B_subset)
        if isinstance(delta, np.ma.MaskedArray):
            delta.mask = delta.mask | nan_mask
        else:
            delta = np.ma.masked_array(delta, mask=nan_mask)

    # Apply log transform if requested
    if log_scale:
        # Sign-preserving log transform: log(abs(x)) * sign(x)
        with np.errstate(divide="ignore", invalid="ignore"):
            sign = np.sign(delta)
            log_delta = np.log1p(np.abs(delta)) * sign
            delta = log_delta

    # Create figure with appropriate size
    if show_original_heatmaps:
        # Make room for the original heatmaps
        if figsize is None:
            figsize = (14, 6)
        fig = plt.figure(figsize=figsize)
        gs = plt.GridSpec(1, 3, width_ratios=[1, 1, 1.2])
        ax_a = fig.add_subplot(gs[0])
        ax_b = fig.add_subplot(gs[1])
        ax = fig.add_subplot(gs[2])
    elif show_side_plots:
        # Make room for side plots
        if figsize is None:
            figsize = (10, 8)
        fig = plt.figure(figsize=figsize)
        gs = plt.GridSpec(
            2,
            2,
            width_ratios=[1 - side_plot_width, side_plot_width],
            height_ratios=[1 - side_plot_width, side_plot_width],
        )
        ax = fig.add_subplot(gs[0, 0])
        ax_right = fig.add_subplot(gs[0, 1], sharey=ax)
        ax_bottom = fig.add_subplot(gs[1, 0], sharex=ax)
    else:
        # Standard single plot
        if figsize is None:
            figsize = (10, 6)
        fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    # Set up colormap and limits
    cmap_obj = plt.cm.get_cmap(cmap) if isinstance(cmap, str) else cmap

    # Set up colormap limits
    if center_colormap and not absolute_diff:
        # Center colormap at zero for balanced visualization
        if vmin is None or vmax is None:
            abs_max = np.nanmax(np.abs(delta))
            if vmin is None:
                vmin = -abs_max
            if vmax is None:
                vmax = abs_max
    else:
        # Use data limits if not specified
        if vmin is None:
            vmin = np.nanmin(delta)
        if vmax is None:
            vmax = np.nanmax(delta)

    # Create the heatmap
    im = ax.imshow(delta, aspect="auto", cmap=cmap_obj, vmin=vmin, vmax=vmax)

    # Add title and subtitle
    if subtitle:
        ax.set_title(f"{title}\n{subtitle}")
    else:
        ax.set_title(title)

    # Set axis labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Set custom tick labels if provided
    if layer_labels is not None:
        ax.set_yticks(range(len(layer_labels)))
        ax.set_yticklabels(layer_labels)
    else:
        # Default layer labels (adjusted for layer_range)
        if L_subset <= 20:  # Only show all labels if not too many
            ax.set_yticks(range(L_subset))
            ax.set_yticklabels([str(i + start_layer) for i in range(L_subset)])

    if token_labels is not None:
        # For many tokens, show a subset of ticks to avoid overcrowding
        if len(token_labels) > 20:
            step = max(1, len(token_labels) // 10)
            tick_positions = range(0, len(token_labels), step)
            tick_labels = [token_labels[i] for i in tick_positions]
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, rotation=45, ha="right")
        else:
            ax.set_xticks(range(len(token_labels)))
            ax.set_xticklabels(token_labels, rotation=45, ha="right")
    elif T_subset <= 20:  # Only show all labels if not too many
        ax.set_xticks(range(T_subset))
        ax.set_xticklabels([str(i + start_token) for i in range(T_subset)])

    # Show values if requested
    if show_values:
        for i in range(L_subset):
            for j in range(T_subset):
                val = delta[i, j]
                if np.ma.is_masked(val) or np.isnan(val):
                    continue

                # Format the value
                text = f"{val:{value_format}}"

                # Determine text color based on cell brightness for better contrast
                if cmap_obj is not None:
                    # Normalize value to [0, 1] range for colormap
                    if vmin is not None and vmax is not None and vmax > vmin:
                        norm_val = (val - vmin) / (vmax - vmin)
                        # Clip to [0, 1] range
                        norm_val = max(0, min(1, norm_val))

                        # Get RGB color at this value
                        rgb = cmap_obj(norm_val)[:3]  # Exclude alpha

                        # Calculate perceived brightness
                        brightness = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]

                        # Use white text on dark backgrounds, black text on light backgrounds
                        text_color = "white" if brightness < 0.6 else "black"
                    else:
                        text_color = "black"
                else:
                    text_color = "black"

                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=value_fontsize,
                )

    # Highlight cells above threshold if requested
    if highlight_threshold is not None:
        # Create a mask for values above threshold (in absolute value)
        threshold_mask = np.abs(delta) > highlight_threshold

        # Apply the mask to create a highlight overlay
        highlight = np.zeros_like(delta)
        highlight[threshold_mask] = 1

        # Plot the highlight mask with transparency
        ax.imshow(
            highlight,
            cmap=plt.cm.colors.ListedColormap([highlight_color]),
            alpha=highlight_alpha * highlight,
            aspect="auto",
        )

    # Highlight specific regions if requested
    if highlight_regions is not None:
        for layer_start, token_start, layer_end, token_end in highlight_regions:
            # Adjust indices for layer_range and token_range
            rel_layer_start = max(0, layer_start - start_layer)
            rel_layer_end = min(L_subset, layer_end - start_layer)
            rel_token_start = max(0, token_start - start_token)
            rel_token_end = min(T_subset, token_end - start_token)

            if rel_layer_start < rel_layer_end and rel_token_start < rel_token_end:
                # Create rectangle patch
                rect = plt.Rectangle(
                    (rel_token_start - 0.5, rel_layer_start - 0.5),
                    rel_token_end - rel_token_start,
                    rel_layer_end - rel_layer_start,
                    linewidth=2,
                    edgecolor=highlight_color,
                    facecolor="none",
                    linestyle="-",
                    alpha=0.8,
                )
                ax.add_patch(rect)

    # Add custom annotations if provided
    if add_annotations is not None:
        for (layer_idx, token_idx), annotation in add_annotations.items():
            # Check if the position is within the selected range
            if (
                start_layer <= layer_idx < end_layer
                and start_token <= token_idx < end_token
            ):
                # Get relative indices
                rel_layer = layer_idx - start_layer
                rel_token = token_idx - start_token

                # Add annotation
                ax.annotate(
                    annotation,
                    (rel_token, rel_layer),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                    bbox={"boxstyle": "round,pad=0.3", "fc": "white", "alpha": 0.8},
                )

    # Add colorbar
    if show_colorbar:
        cbar = fig.colorbar(im, ax=ax)
        if colorbar_label:
            cbar.set_label(colorbar_label)

    # Show statistics if requested
    if show_stats:
        stats_text = []

        # Calculate basic statistics
        mean_diff = np.nanmean(delta)
        median_diff = np.nanmedian(delta)
        max_diff = np.nanmax(delta)
        min_diff = np.nanmin(delta)
        std_diff = np.nanstd(delta)

        # Add statistics to text box
        stats_text.append(f"Mean Δ: {mean_diff:.3f}")
        stats_text.append(f"Median Δ: {median_diff:.3f}")
        stats_text.append(f"Min Δ: {min_diff:.3f}")
        stats_text.append(f"Max Δ: {max_diff:.3f}")
        stats_text.append(f"Std Δ: {std_diff:.3f}")

        # Add percent of cells with positive/negative differences
        n_positive = np.sum(delta > 0)
        n_negative = np.sum(delta < 0)
        n_total = np.sum(~np.isnan(delta))
        if n_total > 0:
            pct_positive = 100 * n_positive / n_total
            pct_negative = 100 * n_negative / n_total
            stats_text.append(f"% Positive: {pct_positive:.1f}%")
            stats_text.append(f"% Negative: {pct_negative:.1f}%")

        # Add text box with statistics
        ax.text(
            1.02,
            0.98,
            "\n".join(stats_text),
            transform=ax.transAxes,
            va="top",
            fontsize=8,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.7},
        )

    # Add side plots if requested
    if show_side_plots:
        # Calculate mean differences by layer and token
        layer_means = np.nanmean(delta, axis=1)
        token_means = np.nanmean(delta, axis=0)

        # Right side plot (mean diff by layer)
        ax_right.barh(range(L_subset), layer_means, color="gray", alpha=0.7)
        ax_right.axvline(0, color="black", linestyle="--", alpha=0.5)
        ax_right.set_title("Mean Δ by Layer")

        # Bottom side plot (mean diff by token)
        ax_bottom.bar(range(T_subset), token_means, color="gray", alpha=0.7)
        ax_bottom.axhline(0, color="black", linestyle="--", alpha=0.5)
        ax_bottom.set_title("Mean Δ by Token")

        # Hide tick labels on side plots
        ax_right.set_yticks([])
        ax_bottom.set_xticks([])

        # Add grid to side plots
        ax_right.grid(True, linestyle="--", alpha=0.3)
        ax_bottom.grid(True, linestyle="--", alpha=0.3)

    # Show original heatmaps if requested
    if show_original_heatmaps:
        # Plot condition A
        im_a = ax_a.imshow(A_subset, aspect="auto")
        ax_a.set_title(f"Condition {condition_a_name}")
        ax_a.set_xlabel(xlabel)
        ax_a.set_ylabel(ylabel)
        fig.colorbar(im_a, ax=ax_a)

        # Plot condition B
        im_b = ax_b.imshow(B_subset, aspect="auto")
        ax_b.set_title(f"Condition {condition_b_name}")
        ax_b.set_xlabel(xlabel)
        ax_b.set_ylabel(ylabel)
        fig.colorbar(im_b, ax=ax_b)

    # Update title with diff type info if not custom title
    if title == "Δ Map (B − A)":
        diff_label = ""
        if diff_type == "absolute":
            diff_label = "|B − A|"
        elif diff_type == "relative":
            diff_label = "(B − A)/A"
        elif diff_type == "percent":
            diff_label = "((B − A)/A) × 100%"
        else:
            diff_label = "B − A"

        # Add log scale indicator
        if log_scale:
            diff_label = f"log({diff_label})"

        ax.set_title(f"Δ Map: {diff_label}")

    fig.tight_layout()
    return fig
