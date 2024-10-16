import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_per_token_dot_product(dot_products, token_ids, tokenizer):
    # Convert token IDs to tokens
    tokens = [tokenizer.decode([id]) for id in token_ids]

    # Create a list of layer names
    layers = list(dot_products.keys())

    # Create a 2D array of dot products
    dot_product_array = np.array([dot_products[layer].cpu().to(float).numpy() for layer in layers])

    # Create the heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=dot_product_array,
            x=tokens,
            y=layers,
            colorscale="RdBu",
            zmid=0,
            colorbar=dict(title="Dot Product"),
        )
    )

    # Update layout
    fig.update_layout(
        title="Per-Token Dot Product Across Layers",
        # xaxis_title='Tokens',
        yaxis_title="Layers",
        xaxis_tickangle=-45,
    )

    # Show the plot
    fig.show()


# Example usage:
# plot_per_token_dot_product(dot_products, input_ids_harmful, tokenizer)

from functools import partial


def plot_per_token_dot_product_plt(
    dot_products, token_ids, tokenizer, reduction: callable = partial(np.mean, axis=0)
):
    # Convert token IDs to tokens
    # tokens = [tokenizer.decode([id]) for id in token_ids]

    # Create a list of layer names
    layers = list(dot_products.keys())

    # Create a 2D array of dot products
    dot_product_array = np.array(
        [reduction(dot_products[layer].cpu().to(float).numpy()) for layer in layers]
    )

    # Create the heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(
        dot_product_array,
        aspect="auto",
        cmap="RdBu",
        vmin=np.min(dot_product_array),
        vmax=np.max(dot_product_array),
    )

    # Add colorbar
    plt.colorbar(label="Dot Product")

    # Set axis labels
    # plt.xticks(ticks=np.arange(len(tokens)), labels=tokens, rotation=45, ha='right')
    plt.yticks(ticks=np.arange(len(layers)), labels=layers)

    # Set titles
    plt.title("Per-Token Dot Product Across Layers")
    plt.xlabel("Tokens")
    plt.ylabel("Layers")

    # Show the plot
    plt.tight_layout()
    plt.show()


def plot_histogram(dot_products_og: torch.Tensor, dot_products_harmful: torch.Tensor):
    # Flatten the dot products
    dot_products_og_flat = dot_products_og.cpu().to(float).numpy().flatten()
    dot_products_harmful_flat = dot_products_harmful.cpu().to(float).numpy().flatten()

    # plot histogram for both
    plt.figure(figsize=(10, 6))
    plt.hist(dot_products_og_flat, bins=50, edgecolor="black", label="Original Prompt")
    plt.hist(
        dot_products_harmful_flat, bins=50, edgecolor="red", alpha=0.75, label="Harmful Prompt"
    )
    plt.title("Histogram of Dot Products")
    plt.xlabel("Dot Product Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.show()
