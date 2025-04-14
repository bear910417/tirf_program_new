import os
import matplotlib.pyplot as plt

def plot_fret_trace(time_array, fret_array, trace_index, base_path="."):
    """
    Plot a clean FRET vs. time figure and save it in a subfolder 'traces' under base_path.
    """
    fig, ax = plt.subplots(figsize=(6, 2.5), dpi=100)
    ax.plot(time_array, fret_array, color='green', linewidth=1.5)
    ax.set_xlabel('time (s)', fontsize=12)
    ax.set_ylabel('FRET', fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, None)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction='out', length=4, width=1)
    plt.tight_layout()
    traces_folder = os.path.join(base_path, "traces")
    if not os.path.exists(traces_folder):
        os.makedirs(traces_folder)
    file_path = os.path.join(traces_folder, f"fret_g_trace_{trace_index}.png")
    plt.savefig(file_path, bbox_inches='tight', dpi=100)
    plt.close(fig)
    return file_path
