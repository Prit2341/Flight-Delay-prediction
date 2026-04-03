import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np

def draw_network():
    fig, ax = plt.subplots(1, 1, figsize=(22, 14))
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 14)
    ax.axis('off')
    fig.patch.set_facecolor('#0f0f1a')
    ax.set_facecolor('#0f0f1a')

    # --- Layer definitions ---
    # (x_position, num_neurons_to_draw, label, real_size, color, extra_info)
    layers = [
        (1.5,  12, "INPUT",   12,    '#4fc3f7', "12 features"),
        (4.5,   8, "FC1",     1024,  '#7986cb', "1024 neurons\nBatchNorm\nReLU\nDropout 0.30"),
        (7.5,   7, "FC2",     512,   '#9575cd', "512 neurons\nBatchNorm\nReLU\nDropout 0.30"),
        (10.5,  6, "FC3",     256,   '#ba68c8', "256 neurons\nBatchNorm\nReLU\nDropout 0.25"),
        (13.5,  5, "FC4",     128,   '#f06292', "128 neurons\nBatchNorm\nReLU\nDropout 0.20"),
        (16.5,  4, "FC5",     64,    '#ff8a65', "64 neurons\nBatchNorm\nReLU\nDropout 0.20"),
        (18.5,  3, "FC6",     32,    '#ffb74d', "32 neurons\nBatchNorm\nReLU"),
        (20.5,  1, "OUTPUT",  1,     '#81c784', "1 neuron\nSigmoid\n→ P(delay)"),
    ]

    neuron_radius = 0.28
    layer_neuron_positions = []

    for (x, n_draw, label, real_size, color, info) in layers:
        # Center neurons vertically
        total_height = (n_draw - 1) * 1.1
        y_start = 7 - total_height / 2
        positions = [y_start + i * 1.1 for i in range(n_draw)]
        layer_neuron_positions.append(positions)

        # Draw neurons
        for y in positions:
            circle = plt.Circle((x, y), neuron_radius,
                                 color=color, zorder=5, alpha=0.9)
            ax.add_patch(circle)
            edge = plt.Circle((x, y), neuron_radius,
                               color='white', fill=False, linewidth=0.8,
                               zorder=6, alpha=0.4)
            ax.add_patch(edge)

        # Draw "..." if real size > drawn size
        if real_size > n_draw:
            mid_y = 7
            ax.text(x, mid_y - 1.85, '...', color=color, fontsize=16,
                    ha='center', va='center', zorder=7, fontweight='bold')

        # Layer label (top)
        ax.text(x, 13.2, label, color='white', fontsize=10,
                ha='center', va='center', fontweight='bold', zorder=7)

        # Real size below label
        size_str = f"({real_size})" if real_size > 1 else "(1)"
        ax.text(x, 12.7, size_str, color=color, fontsize=8.5,
                ha='center', va='center', zorder=7)

        # Info box below network
        ax.text(x, 0.5, info, color='#cccccc', fontsize=7.5,
                ha='center', va='center', zorder=7,
                linespacing=1.5,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#1e1e2e',
                          edgecolor=color, linewidth=1.2, alpha=0.85))

    # --- Draw connections between layers ---
    for i in range(len(layers) - 1):
        x1 = layers[i][0]
        x2 = layers[i + 1][0]
        pos1 = layer_neuron_positions[i]
        pos2 = layer_neuron_positions[i + 1]
        color = layers[i][4]

        for y1 in pos1:
            for y2 in pos2:
                ax.plot([x1 + neuron_radius, x2 - neuron_radius],
                        [y1, y2],
                        color=color, alpha=0.08, linewidth=0.6, zorder=2)

    # --- Feature labels on input neurons ---
    features = [
        "MONTH", "DAY_OF_WEEK", "DEP_TIME_CAT", "CRS_DEP_TIME",
        "CRS_ARR_TIME", "DISTANCE", "TAXI_OUT",
        "TEMPERATURE", "PRECIP_PROB", "WIND_SPEED", "VISIBILITY", "WEATHER_BAD"
    ]
    input_positions = layer_neuron_positions[0]
    for y, feat in zip(input_positions, features):
        ax.text(0.1, y, feat, color='#4fc3f7', fontsize=7,
                ha='left', va='center', zorder=7,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#0f0f1a',
                          edgecolor='none', alpha=0.7))
        ax.plot([0.85, 1.5 - neuron_radius], [y, y],
                color='#4fc3f7', alpha=0.4, linewidth=0.8, zorder=3)

    # --- Output label ---
    out_x = layers[-1][0]
    out_y = layer_neuron_positions[-1][0]
    ax.annotate("P(Delayed > 15 min)\n0 = On Time  |  1 = Delayed",
                xy=(out_x + neuron_radius, out_y),
                xytext=(out_x + 0.6, out_y),
                color='#81c784', fontsize=8.5,
                va='center', ha='left', zorder=7,
                arrowprops=dict(arrowstyle='->', color='#81c784', lw=1.2),
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#1e1e2e',
                          edgecolor='#81c784', linewidth=1.2))

    # --- Title ---
    ax.text(11, 13.8, "FlightDelayNN — Architecture",
            color='white', fontsize=16, ha='center', va='center',
            fontweight='bold', zorder=7)
    ax.text(11, 13.4, "7-Layer Feedforward Neural Network  |  Binary Classification  |  PyTorch",
            color='#888888', fontsize=9, ha='center', va='center', zorder=7)

    # --- Legend ---
    legend_items = [
        mpatches.Patch(color='#7986cb', label='Linear (FC) Layer'),
        mpatches.Patch(color='#aaaaaa', label='BatchNorm → stabilizes inputs'),
        mpatches.Patch(color='#ff8a65', label='ReLU → kills negatives, prevents vanishing gradient'),
        mpatches.Patch(color='#f48fb1', label='Dropout → randomly turns off neurons (prevents overfitting)'),
        mpatches.Patch(color='#81c784', label='Sigmoid → squashes output to 0–1 probability'),
    ]
    legend = ax.legend(handles=legend_items, loc='lower center',
                       bbox_to_anchor=(0.5, -0.04),
                       ncol=3, fontsize=8,
                       facecolor='#1e1e2e', edgecolor='#444444',
                       labelcolor='white', framealpha=0.9)

    plt.tight_layout()
    plt.savefig('network_architecture.png', dpi=180, bbox_inches='tight',
                facecolor='#0f0f1a')
    print("Saved: network_architecture.png")
    plt.show()


if __name__ == "__main__":
    draw_network()
