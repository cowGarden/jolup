from __future__ import annotations

import matplotlib.pyplot as plt


def plot_timeseries(df, x, y, title, ax=None):
    ax = ax or plt.gca()
    ax.plot(df[x], df[y], lw=1.2)
    ax.set_title(title)
    ax.grid(alpha=0.2)
    return ax
