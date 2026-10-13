"""Figure output with an explicit destination directory."""
from pathlib import Path
import matplotlib.pyplot as plt

def save_figure(fig, filename, figures_dir, show=True):
    """Save a 300-dpi figure, optionally display it, then close it."""
    destination = Path(figures_dir) / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(destination, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)
    return destination
