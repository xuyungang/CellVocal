"""Shared font selection without changing the caller's global plotting style."""
from pathlib import Path

from matplotlib import font_manager
from matplotlib.text import Text


def style_communication_figure(fig):
    """Use Arial when installed (including the local user font directory)."""
    available = {font.name for font in font_manager.fontManager.ttflist}
    if "Arial" not in available:
        for path in (Path.home() / ".local/share/fonts/arial").glob("*"):
            if path.suffix.lower() in {".ttf", ".otf"}:
                font_manager.fontManager.addfont(str(path))
        available = {font.name for font in font_manager.fontManager.ttflist}
    family = "Arial" if "Arial" in available else "DejaVu Sans"
    for text in fig.findobj(Text):
        text.set_fontfamily(family)
    return fig
