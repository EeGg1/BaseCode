from typing import Optional, List

import numpy as np
import matplotlib.pyplot as plt


def vis_1d(data: np.ndarray, title: str = "", save_path: str = "", show: bool = False):
    assert data.ndim == 1
    
    plt.clf()
    plt.plot(data)
    
    if title:
        plt.title(title)
    
    if save_path:
        plt.savefig(save_path)

    if show:
        plt.show()


def vis_input(
    data: np.ndarray, 
    var_names: Optional[List[str]] = None,
    title: str = "",
    show: bool = False,
    save_path: str = ""
    ):
    assert data.ndim == 2
    window_size, n_var = data.shape

    if var_names is not None:
        assert n_var == len(var_names)
    
    fig, axes = plt.subplots(nrows=n_var)
    for idx, ax in enumerate(axes):
        var_name = var_names[idx] if var_names is not None else None
        ax.plot(data[:, idx], label=var_name)
        if var_name is not None:
            ax.legend(loc='upper right')
    
    if title:
        fig.suptitle(title)
    
    if save_path:
        plt.savefig(save_path)

    if show:
        plt.show()
