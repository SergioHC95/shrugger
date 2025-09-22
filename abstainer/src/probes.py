"""
Functions for extracting and analyzing the internal states of transformer models.
"""

from typing import Callable, Optional

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer


class ResidualStreamHook:
    """
    Hook class to capture residual stream vectors from transformer models.
    """

    def __init__(self):
        self.activations = {}
        self.handles = []

    def _hook_fn(self, name: str) -> Callable:
        """
        Create a hook function for a specific layer.

        Args:
            name: Name identifier for the layer

        Returns:
            Hook function that stores activations
        """

        def hook(module, input, output):
            # Handle different output formats
            if isinstance(output, tuple):
                # Some models return tuples, use the first element (usually hidden states)
                self.activations[name] = output[0].detach()
            else:
                # Store the residual stream (hidden states)
                self.activations[name] = output.detach()

        return hook

    def attach_hooks(self, model: PreTrainedModel) -> None:
        """
        Attach hooks to all transformer layers in the model.

        Args:
            model: HuggingFace model to attach hooks to
        """
        # Clear any existing hooks and activations
        self.remove_hooks()
        self.activations = {}

        # Different models have different architectures, so we need to handle them differently
        if "Gemma3" in type(model).__name__:
            # Special handling for Gemma3 models
            base = model.model
            prefix = "model"
            print(f"Detected Gemma3 model: {type(model).__name__}")
        elif hasattr(model, "transformer"):
            # For models like GPT-2, Gemma, etc.
            base = model.transformer
            prefix = "transformer"
        elif hasattr(model, "model"):
            # For models like T5, BART, etc.
            base = model.model
            prefix = "model"
        else:
            # Fallback
            base = model
            prefix = ""

        # Try to find the layers based on common model architectures
        if hasattr(base, "h"):
            # GPT-2 style
            layers = base.h
            layer_name = "h"
        elif hasattr(base, "layers"):
            # BERT style
            layers = base.layers
            layer_name = "layers"
        elif hasattr(base, "encoder") and hasattr(base.encoder, "layers"):
            # Some encoder-decoder models
            layers = base.encoder.layers
            layer_name = "encoder.layers"
        elif hasattr(model, "model") and hasattr(model.model, "layers"):
            # Gemma3 style
            layers = model.model.layers
            layer_name = "model.layers"
        elif (
            hasattr(model, "model")
            and hasattr(model.model, "decoder")
            and hasattr(model.model.decoder, "layers")
        ):
            # Some decoder-only models
            layers = model.model.decoder.layers
            layer_name = "model.decoder.layers"
        else:
            # Last resort: try to find any attribute that might be layers
            for attr_name in dir(model):
                if attr_name.endswith("layers") and hasattr(model, attr_name):
                    layers = getattr(model, attr_name)
                    layer_name = attr_name
                    break
            else:
                # If we still can't find layers, print the model structure to help debug
                print(f"Model structure for {type(model).__name__}:")
                for attr in dir(model):
                    if not attr.startswith("_") and not callable(getattr(model, attr)):
                        print(f"- {attr}")
                raise ValueError(
                    f"Could not identify layers in model architecture: {type(model).__name__}"
                )

        # Attach hooks to each layer's output
        for i, layer in enumerate(layers):
            # Try different possible locations for the residual stream
            if hasattr(layer, "output"):
                # BERT style
                handle = layer.output.register_forward_hook(
                    self._hook_fn(f"{prefix}.{layer_name}.{i}.output")
                )
            elif hasattr(layer, "mlp"):
                # GPT style
                handle = layer.mlp.register_forward_hook(
                    self._hook_fn(f"{prefix}.{layer_name}.{i}.mlp")
                )
            elif hasattr(layer, "feed_forward"):
                # Some other architectures
                handle = layer.feed_forward.register_forward_hook(
                    self._hook_fn(f"{prefix}.{layer_name}.{i}.feed_forward")
                )
            elif hasattr(layer, "block_sparse_moe"):
                # Gemma3 style with MoE
                handle = layer.block_sparse_moe.register_forward_hook(
                    self._hook_fn(f"{prefix}.{layer_name}.{i}.block_sparse_moe")
                )
            elif hasattr(layer, "ffn"):
                # Some models use ffn
                handle = layer.ffn.register_forward_hook(
                    self._hook_fn(f"{prefix}.{layer_name}.{i}.ffn")
                )
            else:
                # Fallback: just hook the layer itself
                handle = layer.register_forward_hook(
                    self._hook_fn(f"{prefix}.{layer_name}.{i}")
                )

            self.handles.append(handle)

    def remove_hooks(self) -> None:
        """Remove all attached hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def get_activations(self) -> dict[str, torch.Tensor]:
        """
        Get all captured activations.

        Returns:
            Dictionary mapping layer names to activation tensors
        """
        return self.activations


def get_residual_stream(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    text: str,
    layer_indices: Optional[list[int]] = None,
    token_index: int = -1,
) -> dict[str, torch.Tensor]:
    """
    Get residual stream vectors for a specific text input using the built-in
    output_hidden_states parameter.

    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        text: Input text
        layer_indices: Optional list of layer indices to extract (None = all layers)
        token_index: Index of token to extract activations for (-1 = last token)

    Returns:
        Dictionary mapping layer indices to residual stream vectors
    """
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Run forward pass with output_hidden_states=True
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # Get hidden states (includes embedding layer and all transformer layers)
    hidden_states = outputs.hidden_states

    if hidden_states is None:
        raise ValueError(
            "Model did not return hidden states. Make sure the model supports output_hidden_states=True"
        )

    # Extract specific token from each layer
    result = {}

    # Filter by layer indices if specified
    layer_range = range(len(hidden_states))
    if layer_indices is not None:
        layer_range = [i for i in layer_range if i in layer_indices]

    # Extract the specified token from each layer
    for i in layer_range:
        layer_name = f"layer_{i}"
        # Each hidden state has shape [batch, seq_len, hidden_dim]
        result[layer_name] = hidden_states[i][0, token_index, :]

    return result


def get_all_residual_streams(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizer, text: str
) -> torch.Tensor:
    """
    Get residual stream vectors for all layers and all tokens using the built-in
    output_hidden_states parameter.

    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        text: Input text

    Returns:
        Tensor of shape [num_layers, seq_len, hidden_dim]
    """
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Run forward pass with output_hidden_states=True
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # Get hidden states (includes embedding layer and all transformer layers)
    # Format is typically: tuple of tensors with shape [batch, seq_len, hidden_dim]
    hidden_states = outputs.hidden_states

    if hidden_states is None:
        raise ValueError(
            "Model did not return hidden states. Make sure the model supports output_hidden_states=True"
        )

    # Stack along layer dimension and remove batch dimension
    all_layers = torch.stack([state[0] for state in hidden_states], dim=0)

    return all_layers


def get_last_token_hidden_states(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizer, text: str
) -> torch.Tensor:
    """
    Get hidden state representations across all layers for the last token only.

    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        text: Input text

    Returns:
        Tensor of shape [num_layers, d_model] containing hidden states for the last token
    """
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Run forward pass with output_hidden_states=True
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # Get hidden states (includes embedding layer and all transformer layers)
    hidden_states = outputs.hidden_states

    if hidden_states is None:
        raise ValueError(
            "Model did not return hidden states. Make sure the model supports output_hidden_states=True"
        )

    # Extract the last token from each layer and convert to float32 for compatibility
    # Each hidden state has shape [batch, seq_len, hidden_dim]
    last_token_states = torch.stack(
        [state[0, -1, :].to(torch.float32) for state in hidden_states]
    )

    return last_token_states


def analyze_residual_stream(
    residual_stream: torch.Tensor,
    layer_idx: Optional[int] = None,
    token_idx: Optional[int] = None,
    reduction: str = "none",
) -> torch.Tensor:
    """
    Analyze residual stream vectors.

    Args:
        residual_stream: Tensor from get_all_residual_streams [num_layers, seq_len, hidden_dim]
        layer_idx: Optional layer index to analyze (None = all layers)
        token_idx: Optional token index to analyze (None = all tokens)
        reduction: Reduction method ('none', 'mean', 'norm', 'pca')

    Returns:
        Processed tensor based on the specified reduction
    """
    # Convert to float32 if needed for numerical stability
    if residual_stream.dtype in [torch.float16, torch.bfloat16]:
        residual_stream = residual_stream.to(torch.float32)

    # Extract specific layer/token if requested
    if layer_idx is not None:
        residual_stream = residual_stream[layer_idx : layer_idx + 1]

    if token_idx is not None:
        residual_stream = residual_stream[:, token_idx : token_idx + 1, :]

    # Apply reduction
    if reduction == "none":
        return residual_stream
    elif reduction == "mean":
        # Mean across hidden dimension
        return torch.mean(residual_stream, dim=-1)
    elif reduction == "norm":
        # L2 norm of each vector
        return torch.norm(residual_stream, dim=-1)
    elif reduction == "pca":
        # Simple PCA-like dimensionality reduction
        # Flatten and center the data
        shape = residual_stream.shape
        flattened = residual_stream.reshape(-1, shape[-1])
        centered = flattened - flattened.mean(dim=0, keepdim=True)

        # Compute covariance matrix and its eigenvectors
        cov = torch.mm(centered.T, centered) / (centered.shape[0] - 1)
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)

        # Sort by eigenvalues in descending order
        idx = torch.argsort(eigenvalues, descending=True)
        eigenvectors = eigenvectors[:, idx]

        # Project data onto top 3 principal components
        components = torch.mm(centered, eigenvectors[:, :3])

        # Reshape back to original dimensions but with reduced hidden dim
        return components.reshape(shape[0], shape[1], 3)
    else:
        raise ValueError(f"Unknown reduction method: {reduction}")
