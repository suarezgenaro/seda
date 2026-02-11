from typing import Tuple, Literal, Optional, Union, Dict, List, Mapping
from numpy.typing import ArrayLike
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator
from sys import exit
from .utils import *
from .utils import normalize_flux
from .spectral_indices.spectral_indices import user_index_integral

# ==========================
# Step 1 -- Measure all the indices defined in Oliveros-Gomez, et. al. 2022 to T dwarfs and in Oliveros-Gomez, et. al. 2024 to L dwarfs
# ==========================

SpectralType = Literal["L", "T"]

# ==========================
# NIR index definitions
# ==========================

_T_INDEX_DEFS = [
    {"name": "J",      "num_range": (1.15, 1.18), "den_range": (1.22, 1.25), "mode": "ratio"},
    {"name": "H",      "num_range": (1.64, 1.67), "den_range": (1.59, 1.62), "mode": "ratio"},
    {"name": "HJ",     "num_range": (1.51, 1.62), "den_range": (1.205, 1.315), "mode": "ratio"},
    {"name": "J_H",    "num_range": (1.51, 1.62), "den_range": (1.205, 1.315), "mode": "difference"},
    {"name": "Jslope", "num_range": (1.30, 1.33), "den_range": (1.27, 1.30), "mode": "ratio"},
    {"name": "Jcurve", "num_range": (1.14, 1.17), "den_range": (1.26, 1.29), "mode": "ratio"},
]

_L_INDEX_DEFS = [
    {"name": "mostH",  "num_range": (1.219, 1.237), "den_range": (1.376, 1.394), "mode": "ratio"},
    {"name": "mostJ",  "num_range": (1.37, 1.42),   "den_range": (1.57, 1.67),   "mode": "ratio"},
    {"name": "less",   "num_range": (1.670, 1.688), "den_range": (1.347, 1.365), "mode": "ratio"},
    {"name": "Jcurve", "num_range": (1.26, 1.29),   "den_range": (1.14, 1.17),   "mode": "ratio"},
    {"name": "H2OJ",   "num_range": (1.14, 1.165),  "den_range": (1.26, 1.285),  "mode": "ratio"},
    {"name": "CH4J",   "num_range": (1.315, 1.335), "den_range": (1.26, 1.285),  "mode": "ratio"},
]

#########

def nir_indices(
    wavelength,
    flux,
    spectral_type: str,
    *,
    normalize: bool = False,
    plot: bool = False,
    plot_save: Union[bool, str] = False,
) -> Dict[str, float]:
    #Compute NIR spectral indices for L and T brown dwarfs, defined by Oliveros-Gomez et. al. 2022, 2024,
    #to focused in found variable candidate objects.

    #Parameters: wavelength, flux (array-like), spectral_type : {"L", "T"}, normalize : bool, default False,
    #plot : bool, default False, plot_save : bool or str, default False

    #Returns: indices : dict -- Dictionary of index_name -> value.

    wave = np.asarray(wavelength, dtype=float)
    flx = np.asarray(flux, dtype=float)

    if wave.shape != flx.shape:
        raise ValueError("`wavelength` and `flux` must have the same shape.")

    if normalize:
        flx = _normalize_flux(flx)

    spt = spectral_type.strip().upper()
    indices: Dict[str, float] = {}

    if spt.startswith("T"):
        index_defs = _T_INDEX_DEFS
    elif spt.startswith("L"):
        index_defs = _L_INDEX_DEFS
    else:
        raise ValueError(
            f"Unsupported spectral_type={spectral_type!r}. Use 'L' or 'T'."
        )

    # Calcular todos los índices a partir de las definiciones
    for idx_def in index_defs:
        name = idx_def["name"]
        num_range = idx_def["num_range"]
        den_range = idx_def["den_range"]
        mode = idx_def["mode"]
        indices[name] = user_index_integral(wave, flx, num_range, den_range, mode=mode)

    # Plot opcional de las ventanas de índice
    if plot:
        # decidir nombre del archivo si plot_save=True o str
        savepath: str | None
        if isinstance(plot_save, str):
            savepath = plot_save
        elif plot_save is True:
            savepath = f"nir_indices_{spt}.pdf"
        else:
            savepath = None

        _plot_user_index_integral_windows(wave, flx, spt, index_defs, savepath=savepath)

    return indices

######

def _plot_user_index_integral_windows(
    wavelength: ArrayLike,
    flux: ArrayLike,
    spectral_type: str,
    index_defs,
    *,
    figsize=None,
    savepath: Optional[str] = None,
):
    #Internal helper to plot NIR indices bandpasses over the input spectrum.
    #One panel per index, similar to the user_index plot.

    wave = np.asarray(wavelength, dtype=float)
    flx = np.asarray(flux, dtype=float)

    n = len(index_defs)
    # 6 índices -> 2x3; si algún día cambian, ajusta aquí.
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    if figsize is None:
        figsize = (12, 6) if nrows == 2 else (12, 8)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes = np.array(axes).ravel()

    for i, idx_def in enumerate(index_defs):
        ax = axes[i]
        name = idx_def["name"]
        num_range = idx_def["num_range"]
        den_range = idx_def["den_range"]

        ax.plot(wave, flx, lw=1.2)

        # Denominador in blue
        ax.axvspan(den_range[0], den_range[1], alpha=0.25, label="denominator")
        # Numerador in red (the user can change the color)
        ax.axvspan(num_range[0], num_range[1], alpha=0.25, color='red')

        ax.set_title(f"{name} index")
        ax.set_xlabel("Wavelength [µm]")
        if i % ncols == 0:
            ax.set_ylabel("Normalized Flux")

        ax.set_xlim(wave.min(), wave.max())

    # Ocultar subplots vacíos si los hay
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"NIR {spectral_type}-dwarf indices bandpasses", fontsize=14)

    if savepath is not None:
        fig.savefig(savepath, bbox_inches="tight")

    plt.show()
    return fig, axes


# ==========================
# Step 2 -- Define the variability areas defined in Oliveros-Gomez, et. al., 2022 and 2024
# ==========================

RegionDef = Dict[str, object]  # para no complicar tipos

def _build_path(verts: List[Tuple[float, float]]) -> Path:
    #Build a Matplotlib Path from a list of vertices, mimicking the original
    #implementation where different code arrays (codes_3v, codes_4v, codes_5v)
    #were used with CLOSEPOLY at the end.

    #Behavior:
    #- First vertex: MOVETO
    #- Intermediate vertices: LINETO
    #- Last vertex: CLOSEPOLY

    #This works whether the last vertex is a repetition of the first or not

    verts = list(verts)

    # close the poligone
    if verts[0] != verts[-1]:
        verts.append(verts[0])

    n = len(verts)
    if n < 4:
        # 3 real verts + close
        raise ValueError("A polygon needs at least 3 vertices.")

    # MOVETO for first point, LINETO for all intermediate, CLOSEPOLY for last
    if n == 4:
        codes = [Path.MOVETO, Path.LINETO, Path.LINETO]
    else:
        # general: MOVETO + LINETO * (n-2) + CLOSEPOLY
        codes = [Path.MOVETO] + [Path.LINETO] * (n - 2)

    # adding the CLOSEPOLY at the last point
    codes.append(Path.CLOSEPOLY)

    return Path(verts, codes)


# ==========================
# T-DWARF REGIONS (Oliveros-Gomez et. al., 2022)
# ==========================

_T_REGIONS: List[RegionDef] = [
    # ax1: H vs J
    # ylim = J_temp ± 0.15 ; xlim = H_temp ± 0.25
    {
        "name": "00",
        "x_index": "H",
        "y_index": "J",
        "x_center": "H",
        "x_offset": (-0.25, 0.25),
        "y_center": "J",
        "y_offset": (-0.15, 0.15),
        "verts": [
            (0.2, 0.37),
            (0.39, 0.37),
            (0.463, 0.2010),
            (0.65, 0.05),
            (0.2, 0.05),
            (0.2, 0.37),
        ],
    },
    # ax2: H vs HJ
    # ylim = HJ_temp ± 0.12 ; xlim = H_temp ± 0.25
    {
        "name": "01",
        "x_index": "H",
        "y_index": "HJ",
        "x_center": "H",
        "x_offset": (-0.25, 0.25),
        "y_center": "HJ",
        "y_offset": (-0.12, 0.12),
        "verts": [
            (0.2, 0.55),
            (0.415, 0.55),
            (0.462, 0.4167),
            (0.4, 0.28),
            (0.2, 0.28),
            (0.2, 0.55),
        ],
    },
    # ax3: J_H vs HJ
    # ylim = HJ_temp ± 0.12 ; xlim = J_H_temp ± 0.02
    {
        "name": "02",
        "x_index": "J_H",
        "y_index": "HJ",
        "x_center": "J_H",
        "x_offset": (-0.02, 0.02),
        "y_center": "HJ",
        "y_offset": (-0.12, 0.12),
        "verts": [
            (0.038, 0.55),
            (0.065, 0.55),
            (0.065, 0.28),
            (0.057, 0.28),
            (0.038, 0.55),
        ],
    },
    # ax4: J_H vs H
    # ylim = H_temp ± 0.25 ; xlim = J_H_temp ± 0.02
    {
        "name": "03",
        "x_index": "J_H",
        "y_index": "H",
        "x_center": "J_H",
        "x_offset": (-0.02, 0.02),
        "y_center": "H",
        "y_offset": (-0.25, 0.25),
        "verts": [
            (0.067, 0.68),
            (0.067, 0.2),
            (0.0325, 0.2),
            (0.067, 0.68),
        ],
    },

    # ax5: J_H vs J
    # ylim = J_temp ± 0.15 ; xlim = J_H_temp ± 0.02
    {
        "name": "10",
        "x_index": "J_H",
        "y_index": "J",
        "x_center": "J_H",
        "x_offset": (-0.02, 0.02),
        "y_center": "J",
        "y_offset": (-0.15, 0.15),
        "verts": [
            (0.038, 0.37),
            (0.067, 0.37),
            (0.067, 0.05),
            (0.0352, 0.2005),
            (0.038, 0.37),
        ],
    },
    # ax6: Jslope vs J
    # ylim = J_temp ± 0.15 ; xlim = Jslope_temp ± 0.2
    {
        "name": "11",
        "x_index": "Jslope",
        "y_index": "J",
        "x_center": "Jslope",
        "x_offset": (-0.2, 0.2),
        "y_center": "J",
        "y_offset": (-0.15, 0.15),
        "verts": [
            (0.43, 0.37),
            (0.63, 0.37),
            (0.65, 0.201),
            (0.63, 0.05),
            (0.43, 0.05),
            (0.43, 0.37),
        ],
    },
    # ax7: Jslope vs H
    # ylim = H_temp ± 0.25 ; xlim = Jslope_temp ± 0.2
    {
        "name": "12",
        "x_index": "Jslope",
        "y_index": "H",
        "x_center": "Jslope",
        "x_offset": (-0.2, 0.2),
        "y_center": "H",
        "y_offset": (-0.25, 0.25),
        "verts": [
            (0.43, 0.515),
            (0.646, 0.462),
            (0.75, 0.2),
            (0.43, 0.2),
            (0.43, 0.515),
        ],
    },
    # ax8: Jslope vs HJ
    # ylim = HJ_temp ± 0.1 ; xlim = Jslope_temp ± 0.2
    {
        "name": "13",
        "x_index": "Jslope",
        "y_index": "HJ",
        "x_center": "Jslope",
        "x_offset": (-0.2, 0.2),
        "y_center": "HJ",
        "y_offset": (-0.10, 0.10),
        "verts": [
            (0.43, 0.52),
            (0.63, 0.52),
            (0.65, 0.4169),
            (0.63, 0.28),
            (0.43, 0.28),
            (0.43, 0.52),
        ],
    },

    # ax9: Jslope vs J_H
    # ylim = J_H_temp ± 0.02 ; xlim = Jslope_temp ± 0.2
    {
        "name": "20",
        "x_index": "Jslope",
        "y_index": "J_H",
        "x_center": "Jslope",
        "x_offset": (-0.2, 0.2),
        "y_center": "J_H",
        "y_offset": (-0.02, 0.02),
        "verts": [
            (0.43, 0.065),
            (0.85, 0.065),
            (0.85, 0.037),
            (0.43, 0.046),
            (0.43, 0.065),
        ],
    },
    # ax10: Jslope vs Jcurve
    # ylim = Jcurve_temp ± 0.22 ; xlim = Jslope_temp ± 0.2
    {
        "name": "21",
        "x_index": "Jslope",
        "y_index": "Jcurve",
        "x_center": "Jslope",
        "x_offset": (-0.2, 0.2),
        "y_center": "Jcurve",
        "y_offset": (-0.22, 0.22),
        "verts": [
            (0.43, 0.35),
            (0.85, -0.07),
            (0.43, -0.07),
            (0.43, 0.35),
        ],
    },
    # ax11: H vs Jcurve
    # ylim = Jcurve_temp ± 0.22 ; xlim = H_temp ± 0.25
    {
        "name": "22",
        "x_index": "H",
        "y_index": "Jcurve",
        "x_center": "H",
        "x_offset": (-0.25, 0.25),
        "y_center": "Jcurve",
        "y_offset": (-0.22, 0.22),
        "verts": [
            (0.2, 0.35),
            (0.68, -0.07),
            (0.2, -0.07),
            (0.2, 0.35),
        ],
    },
    # ax12: J_H vs Jcurve
    # ylim = Jcurve_temp ± 0.22 ; xlim = J_H_temp ± 0.02
    {
        "name": "23",
        "x_index": "J_H",
        "y_index": "Jcurve",
        "x_center": "J_H",
        "x_offset": (-0.02, 0.02),
        "y_center": "Jcurve",
        "y_offset": (-0.22, 0.22),
        "verts": [
            (0.065, 0.34),
            (0.065, -0.07),
            (0.025, -0.07),
            (0.065, 0.34),
        ],
    },
]


_T_N_TOTAL = 12       # número total de regiones (ajusta si cambias algo)
_T_THRESHOLD = 10     # criterio original: > 10 de 12


# ==========================
# L-DWARF REGIONS (Oliveros-Gomez, et. al., 2024 )
# ==========================

_L_REGIONS: List[RegionDef] = [
    # ax1: mostH vs less
    # ylim = less ± 1.5 ; xlim = mostH ± 1
    {
        "name": "00",
        "x_index": "mostH",
        "y_index": "less",
        "x_center": "mostH",
        "x_offset": (-1.0, 1.0),
        "y_center": "less",
        "y_offset": (-1.5, 1.5),
        "verts": [
            (3, 5.3),
            (3, 0.78),
            (0.9, -0.5),
            (3, 5.3),
        ],
    },
    # ax2: mostJ vs less
    # ylim = less ± 1.5 ; xlim = mostJ ± 0.2
    {
        "name": "01",
        "x_index": "mostJ",
        "y_index": "less",
        "x_center": "mostJ",
        "x_offset": (-0.2, 0.2),
        "y_center": "less",
        "y_offset": (-1.5, 1.5),
        "verts": [
            (-0.55, 2.262),
            (0.8, 2.262),
            (0.8, 0.75),
            (-0.55, 0.75),
            (-0.55, 2.32),
        ],
    },
    # ax3: CH4J vs less
    # ylim = less ± 1.5 ; xlim = CH4J ± 0.7
    {
        "name": "02",
        "x_index": "CH4J",
        "y_index": "less",
        "x_center": "CH4J",
        "x_offset": (-0.7, 0.7),
        "y_center": "less",
        "y_offset": (-1.5, 1.5),
        "verts": [
            (0, 2.72),
            (1.5, 1.75),
            (1.8, 0.75),
            (0, 0.75),
            (0, 2.54),
        ],
    },
    # ax4: Jcurve vs less
    # ylim = less ± 1.5 ; xlim = Jcurve ± 1
    {
        "name": "03",
        "x_index": "Jcurve",
        "y_index": "less",
        "x_center": "Jcurve",
        "x_offset": (-1.0, 1.0),
        "y_center": "less",
        "y_offset": (-1.5, 1.5),
        "verts": [
            (0.3, 3.5),
            (2.6, 0.75),
            (0.3, 0.75),
            (0.3, 3.5),
        ],
    },

    # ax5: Jcurve vs CH4J
    # ylim = CH4J ± 0.7 ; xlim = Jcurve ± 1
    {
        "name": "10",
        "x_index": "Jcurve",
        "y_index": "CH4J",
        "x_center": "Jcurve",
        "x_offset": (-1.0, 1.0),
        "y_center": "CH4J",
        "y_offset": (-0.7, 0.7),
        "verts": [
            (0.28, 2.92),
            (2.62, -1.8),
            (0.28, -1.3),
            (0.28, 3.2),
        ],
    },
    # ax6: Jcurve vs mostJ
    # ylim = mostJ ± 0.2 ; xlim = Jcurve ± 1
    {
        "name": "11",
        "x_index": "Jcurve",
        "y_index": "mostJ",
        "x_center": "Jcurve",
        "x_offset": (-1.0, 1.0),
        "y_center": "mostJ",
        "y_offset": (-0.2, 0.2),
        "verts": [
            (0.31, 0.8),
            (1.37, 0.8),
            (1.37, -0.25),
            (0.3, -0.25),
            (0.4, 0.8),
        ],
    },
    # ax7: Jcurve vs mostH
    # ylim = mostH ± 1 ; xlim = Jcurve ± 1
    {
        "name": "12",
        "x_index": "Jcurve",
        "y_index": "mostH",
        "x_center": "Jcurve",
        "x_offset": (-1.0, 1.0),
        "y_center": "mostH",
        "y_offset": (-1.0, 1.0),
        "verts": [
            (0.3, 3.05),
            (2.38, 3.05),
            (2.38, 2.15),
            (0.3, 1.65),
            (0.3, 3.05),
        ],
    },
    # ax8: CH4J vs mostH
    # ylim = mostH ± 1 ; xlim = CH4J ± 0.7
    {
        "name": "13",
        "x_index": "CH4J",
        "y_index": "mostH",
        "x_center": "CH4J",
        "x_offset": (-0.7, 0.7),
        "y_center": "mostH",
        "y_offset": (-1.0, 1.0),
        "verts": [
            (-0.25, 1.905),
            (1.9, 1.905),
            (1.9, 3),
            (-0.25, 3),
            (-0.25, 1.905),
        ],
    },

    # ax9: mostJ vs mostH
    # ylim = mostH ± 1 ; xlim = mostJ ± 0.2
    {
        "name": "20",
        "x_index": "mostJ",
        "y_index": "mostH",
        "x_center": "mostJ",
        "x_offset": (-0.2, 0.2),
        "y_center": "mostH",
        "y_offset": (-1.0, 1.0),
        "verts": [
            (-0.5, 2.41),
            (0.8, 1.51),
            (0.8, 3),
            (-0.5, 3),
            (-0.5, 2.41),
        ],
    },
    # ax10: CH4J vs mostJ
    # ylim = mostJ ± 0.2 ; xlim = CH4J ± 0.7
    {
        "name": "21",
        "x_index": "CH4J",
        "y_index": "mostJ",
        "x_center": "CH4J",
        "x_offset": (-0.7, 0.7),
        "y_center": "mostJ",
        "y_offset": (-0.2, 0.2),
        "verts": [
            (0.739, 0.8),
            (-0.25, 0.8),
            (-0.25, -0.25),
            (0.742, -0.25),
            (0.739, 0.8),
        ],
    },
    # ax11: H2OJ vs mostJ
    # ylim = mostJ ± 0.2 ; xlim = H2OJ ± 0.3
    {
        "name": "22",
        "x_index": "H2OJ",
        "y_index": "mostJ",
        "x_center": "H2OJ",
        "x_offset": (-0.3, 0.3),
        "y_center": "mostJ",
        "y_offset": (-0.2, 0.2),
        "verts": [
            (0.739, 0.8),
            (1.4, 0.8),
            (1.4, -0.25),
            (0.742, -0.25),
            (0.739, 0.8),
        ],
    },
    # ax12: H2OJ vs less
    # ylim = less ± 1.5 ; xlim = H2OJ ± 0.3
    {
        "name": "23",
        "x_index": "H2OJ",
        "y_index": "less",
        "x_center": "H2OJ",
        "x_offset": (-0.3, 0.3),
        "y_center": "less",
        "y_offset": (-1.5, 1.5),
        "verts": [
            (0.2, 2.24),
            (1.26, 2.29),
            (1.26, 0.75),
            (0.2, 0.75),
            (0.2, 2.24),
        ],
    },

    # ax13: H2OJ vs mostH
    # ylim = mostH ± 1 ; xlim = H2OJ ± 0.3
    {
        "name": "30",
        "x_index": "H2OJ",
        "y_index": "mostH",
        "x_center": "H2OJ",
        "x_offset": (-0.3, 0.3),
        "y_center": "mostH",
        "y_offset": (-1.0, 1.0),
        "verts": [
            (0.18, 2.12),
            (1.26, 1.74),
            (1.26, 3),
            (0.18, 3),
            (0.18, 2.12),
        ],
    },
    # ax14: CH4J vs H2OJ
    # ylim = H2OJ ± 0.3 ; xlim = CH4J ± 0.7
    {
        "name": "31",
        "x_index": "CH4J",
        "y_index": "H2OJ",
        "x_center": "CH4J",
        "x_offset": (-0.7, 0.7),
        "y_center": "H2OJ",
        "y_offset": (-0.3, 0.3),
        "verts": [
            (-0.18, 0.125),
            (1.8, 1.445),
            (-0.18, 1.725),
            (-0.18, 0.125),
        ],
    },
    # ax15: H2OJ vs Jcurve
    # ylim = Jcurve ± 1 ; xlim = H2OJ ± 0.3
    {
        "name": "32",
        "x_index": "H2OJ",
        "y_index": "Jcurve",
        "x_center": "H2OJ",
        "x_offset": (-0.3, 0.3),
        "y_center": "Jcurve",
        "y_offset": (-1.0, 1.0),
        "verts": [
            (-0.18, 1.36),
            (1.8, 1.36),
            (1.8, 0.36),
            (-0.18, 0.36),
            (-0.18, 1.36),
        ],
    },
]


_L_N_TOTAL = 15       # número total de regiones L
_L_THRESHOLD = 8     # criterio original: > 8 de 15


# ==========================
# Step 3 -- Count the number of indides that fall in the variabile areas
# ==========================


def _count_regions_triggered(
    indices: Mapping[str, float],
    regions: List[RegionDef],
) -> Tuple[int, List[str]]:
    #Given a set of indices and region definitions, count how many
    #index–index points fall inside the variable regions.

    #Parameters: indices : mapping, regions : list of dict

    #Returns: n_triggered : int,    region_names : list of str
    triggered: List[str] = []

    for reg in regions:
        name: str = reg["name"]  # type: ignore
        x_name: str = reg["x_index"]  # type: ignore
        y_name: str = reg["y_index"]  # type: ignore
        verts: List[Tuple[float, float]] = reg["verts"]  # type: ignore

        if x_name not in indices or y_name not in indices:
            # Si falta algún índice, simplemente saltamos esta región
            continue

        x_val = float(indices[x_name])
        y_val = float(indices[y_name])

        path = _build_path(verts)
        # contains_points espera una matriz (N, 2)
        inside = path.contains_points(np.array([[x_val, y_val]]))[0]

        if inside:
            triggered.append(name)

    return len(triggered), triggered

# ==========================
# Step 4 -- Classify the variable and non-variable objects
# ==========================

@dataclass
class VariabilityResult:
    spectral_type: str
    scheme: str
    is_candidate_variable: bool
    n_regions_triggered: int
    n_regions_total: int
    threshold: int
    indices: Dict[str, float]
    regions_triggered: List[str]
    normalize: Optional[bool] = None

    def summary(self) -> str:
        tag = "candidate VARIABLE" if self.is_candidate_variable else "candidate non-variable"
        return (
            f"Scheme: {self.scheme}\n"
            f"Spectral type: {self.spectral_type}\n"
            + (f"Normalize: {self.normalize}\n" if self.normalize is not None else "")
            + f"Triggered regions: {self.n_regions_triggered}/"
              f"{self.n_regions_total} (threshold ≥ {self.threshold})\n"
            + f"Classification: {tag}"
        )


def _classify_T_variability(
    wavelength,
    flux,
    *,
    normalize: bool = True,
    scheme: str = "Oliveros-Gomez+2022",
) -> VariabilityResult:

    #Classify T-type brown dwarf variability using NIR spectral indices.

    #This wraps `seda.spectral_indices.nir_indices` and the original
    #polygon-based criterion (12 regions, threshold ~11).

    indices = nir_indices(wavelength, flux, spectral_type="T", normalize=normalize)
    n_trig, names = _count_regions_triggered(indices, _T_REGIONS)

    is_var = n_trig >= _T_THRESHOLD

    return VariabilityResult(
        spectral_type="T",
        scheme=scheme,
        is_candidate_variable=is_var,
        n_regions_triggered=n_trig,
        n_regions_total=_T_N_TOTAL,
        threshold=_T_THRESHOLD,
        indices=indices,
        regions_triggered=names,
    )


def _classify_L_variability(
    wavelength,
    flux,
    *,
    normalize: bool = True,
    scheme: str = "Oliveros-Gomez+2024",
) -> VariabilityResult:

    #Classify L-type brown dwarf variability using NIR spectral indices.

   # This wraps `seda.spectral_indices.nir_indices` and the original
    #polygon-based criterion (15 regions, threshold ~9).
    
    indices = nir_indices(wavelength, flux, spectral_type="L", normalize=normalize)
    n_trig, names = _count_regions_triggered(indices, _L_REGIONS)

    is_var = n_trig >= _L_THRESHOLD

    return VariabilityResult(
        spectral_type="L",
        scheme=scheme,
        is_candidate_variable=is_var,
        n_regions_triggered=n_trig,
        n_regions_total=_L_N_TOTAL,
        threshold=_L_THRESHOLD,
        indices=indices,
        regions_triggered=names,
    )

def classify_variability(
    wavelength,
    flux,
    *,
    spectral_type: str,
    normalize: bool = True,
    scheme: Optional[str] = None,
    plot_diagrams: bool = False,
    plot_index_windows: bool = False,
    plot_save: Union[bool, str] = False,
    show: bool = True,
) -> VariabilityResult:
    """
    Description:
    ------------
    Classify a brown dwarf as candidate variable or non-variable using
    NIR spectral index–index variability regions.

    This is the main public interface for variability classification in SEDA.
    It computes the required spectral indices, evaluates the polygon-based
    variability criteria, and optionally produces diagnostic plots.

    Parameters:
    -----------
    - wavelength : array-like
        Wavelength array (microns).
    - flux : array-like
        Flux array corresponding to `wavelength`.
    - spectral_type : {"L", "T"}
        Spectral type scheme to use for the classification.
    - normalize : bool, default True
        If True, the flux is median-normalized before computing indices.
    - scheme : str or None, optional
        Name or reference of the variability scheme. If None, the default
        scheme for the selected spectral type is used.
    - plot_diagrams : bool, default False
        If True, generate the index–index variability diagrams.
    - plot_index_windows : bool, default False
        If True, plot the spectrum with the numerator/denominator
        windows used to compute the NIR indices.
    - plot_save : bool or str, default False
        If True, save plots to default filenames.
        If a string, use it as the base path or filename.
    - show : bool, default True
        If True, display the plots using `plt.show()`.

    Returns:
    --------
    - result : VariabilityResult
        Structured classification result with the following attributes:
           - spectral_type : str
             Spectral type scheme used for the classification ("L" or "T").
           - scheme : str
             Name or reference of the variability scheme (e.g., "Oliveros-Gomez+2022", "Oliveros-Gomez+2024").
           - is_candidate_variable : bool
             Final classification flag. True if the number of triggered index–index regions meets or exceeds the adopted threshold.
           - n_regions_triggered : int
             Number of variability regions in which the target falls.
           - n_regions_total : int
             Total number of regions evaluated for the selected spectral type.
           - threshold : int
             Minimum number of triggered regions required to be classified as a candidate variable.
           - indices : dict[str, float]
             Dictionary of computed NIR spectral indices. Keys correspond to the physical index names (e.g., "J", "H", "Jslope", "Jcurve",
             "H2OJ", "CH4J"), and values are the numerical index values.
           - regions_triggered : list[str]
             List of region identifiers (or names) for which the target falls inside the corresponding variability polygon.
           - normalize : bool
             Whether a median flux normalization was applied to the input spectrum prior to computing the indices.
           - summary() : str
             Returns a human-readable, multi-line summary of the classification outcome. The result object also provides a convenience method:

    Examples
    --------
    >>> result = classify_variability(wave, flux, spectral_type="T", normalize=False)
    >>> print(result.is_candidate_variable)
    True
    >>> print(result.n_regions_triggered, "/", result.n_regions_total)
    11 / 12
    >>> print(result.indices["Jslope"], result.indices["Jcurve"])
    0.63 0.15
    >>> print(result.summary())
    Scheme: Oliveros-Gomez+2022
    Spectral type: T
    Normalize: False
    Triggered regions: 11/12 (threshold >= 11)
    Classification: candidate VARIABLE

    Author:
    -------
    Natalia Oliveros-Gomez
    """

    spt = str(spectral_type).upper()

    # ----------------------------
    # 1) Run classification
    # ----------------------------
    if spt.startswith("T"):
        result = _classify_T_variability(
            wavelength,
            flux,
            normalize=normalize,
            scheme=scheme or "Oliveros-Gomez+2022",
        )
    elif spt.startswith("L"):
        result = _classify_L_variability(
            wavelength,
            flux,
            normalize=normalize,
            scheme=scheme or "Oliveros-Gomez+2024",
        )
    else:
        raise ValueError(
            f"Unsupported spectral_type={spectral_type!r}. Use 'L' or 'T'."
        )

    # ----------------------------
    # 2) Plot index windows
    # ----------------------------
    if plot_index_windows:
        if isinstance(plot_save, str):
            savepath = plot_save
        elif plot_save is True:
            savepath = f"indices_windows_{spt}.pdf"
        else:
            savepath = None

        index_defs = _T_INDEX_DEFS if spt.startswith("T") else _L_INDEX_DEFS
        
        _plot_user_index_integral_windows(
            wavelength,
            flux,
            spectral_type=spt,
            index_defs=index_defs,
            savepath=savepath,
        )

    # ----------------------------
    # 3) Plot index–index diagrams
    # ----------------------------
    if plot_diagrams:
        if isinstance(plot_save, str):
            savepath = plot_save
        elif plot_save is True:
            savepath = f"indices_{spt}.pdf"
        else:
            savepath = None

        _plot_variability_diagrams(
            result,
            show=show,
            savepath=savepath,
        )

    return result


# ==========================
# Step 5 -- Plot the index-index variability diagrams
# ==========================


# TEMPLATE INDICES

# T-dwarf template: 2MASS J22282889–4310262 (Values obtained in Oliveros-Gomez et. al. 2022)
_T_TEMPLATE_INDICES: Dict[str, float] = {
    "J": 0.205,
    "H": 0.463,
    "HJ": 0.4145,
    "J_H": 0.0451,
    "Jslope": 0.633,
    "Jcurve": 0.154,
}

# L-dwarf template: LP 261-75B (Values obtained in Oliveros-Gomez et. al. 2024)
_L_TEMPLATE_INDICES: Dict[str, float] = {
    "mostH": 1.913080124768479,
    "mostJ": 0.23732289733907183,
    "less": 2.2632193763577337,
    "Jcurve": 1.363828699841246,
    "H2OJ": 0.7366212163364357,
    "CH4J": 0.7414549450272985,
}

#Labels definition for the plots
_T_LABELS = {
    "J": "J-index",
    "H": "H-index",
    "HJ": "H/J-index",
    "J_H": "J–H index",
    "Jslope": "J-slope index",
    "Jcurve": "J-curve index",
}

_L_LABELS = {
    "mostH": "max-var (H)",
    "mostJ": "max-var (J)",
    "less": "max-var",
    "Jcurve": "J-curve",
    "H2OJ": "H$_2$O-J",
    "CH4J": "CH$_4$-J",
}


# Creating the plot
def _plot_variability_diagrams(
    result: VariabilityResult,
    *,
    figsize=None,
    show: bool = True,
    savepath: Optional[str] = None,
):
    #Plot index–index diagrams with variability regions for L/T brown dwarfs.
    #This function generates a grid of index–index panels corresponding to the
    #adopted variability scheme for the given spectral type. Each panel shows:
    #(i) the polygonal variability region, (ii) the reference/template object,
    #and (iii) the measured target indices.
    
    #Parameters: result : VariabilityResult, show : bool, default True, savepath : str or None, default None

    #Returns: fig : matplotlib.figure.Figure, axes : ndarray of matplotlib.axes.Axes
    #Notes: The variability regions are defined as polygonal areas in index–index
    #space following the original scheme described in Oliveros-Gomez et al.
    #(2022, T dwarfs) and Oliveros-Gomez et al. (2024, L dwarfs). A target is
    #classified as a candidate variable if the number of panels in which its
    #indices fall inside the corresponding polygon meets or exceeds the
    #threshold stored in the input `result`.

    #The reference/template object is shown as a black star marker, while the
    #target object is shown as a blue star marker. Shaded red regions indicate
    #the variability polygons.

    spt = result.spectral_type.upper()

    if spt == "T":
        regions = _T_REGIONS
        template = _T_TEMPLATE_INDICES
        labels = _T_LABELS
        n_total = _T_N_TOTAL
        nrows, ncols = 3, 4  # 12 panels
        if figsize is None:
            figsize = (22, 13.5)
        template_label = "Template 2M2228"
        bd_label = "Target brown dwarf"
    elif spt == "L":
        regions = _L_REGIONS
        template = _L_TEMPLATE_INDICES
        labels = _L_LABELS
        n_total = _L_N_TOTAL
        nrows, ncols = 3, 5  # 15 panels
        if figsize is None:
            figsize = (26, 14)
        template_label = "Template LP 261-75B"
        bd_label = "Target brown dwarf"
    else:
        raise ValueError(f"Unsupported spectral_type={result.spectral_type!r}")

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        constrained_layout=True,
    )

    axes_flat = np.array(axes).ravel()
    triggered_set = set(result.regions_triggered)

    for i, reg in enumerate(regions):
        if i >= len(axes_flat):
            break

        ax = axes_flat[i]

        name: str = reg["name"]          # type: ignore
        x_name: str = reg["x_index"]     # type: ignore
        y_name: str = reg["y_index"]     # type: ignore
        verts = reg["verts"]             # type: ignore

        # Si faltan índices, apagamos el panel
        if x_name not in result.indices or y_name not in result.indices:
            ax.set_visible(False)
            continue

        x_val = float(result.indices[x_name])
        y_val = float(result.indices[y_name])

        # Valores de la plantilla (si existen)
        x_temp = template.get(x_name, None)
        y_temp = template.get(y_name, None)

        # Polígono de región
        path = _build_path(verts)
        patch = PathPatch(
            path,
            facecolor="red",
            alpha=0.35 if name in triggered_set else 0.15,
            lw=1,
        )
        ax.add_patch(patch)

        # Template (estrella negra)
        if x_temp is not None and y_temp is not None:
            ax.plot(x_temp, y_temp, marker="*", markersize=16, color="black", zorder=3)

        # Brown dwarf (estrella azul)
        ax.scatter(x_val, y_val, marker="*", s=150, color="blue", zorder=4)

        # Labels de ejes
        ax.set_xlabel(labels.get(x_name, x_name))
        ax.set_ylabel(labels.get(y_name, y_name))

        # ---- Límites de ejes centrados en el template ----
        x_center_key = reg.get("x_center", x_name)
        y_center_key = reg.get("y_center", y_name)
        x_off = reg.get("x_offset", None)
        y_off = reg.get("y_offset", None)

        # Eje X
        if x_off is not None:
            x_center_val = template.get(x_center_key)
            if x_center_val is not None:
                ax.set_xlim(
                    x_center_val + x_off[0],
                    x_center_val + x_off[1],
                )

        # Eje Y
        if y_off is not None:
            y_center_val = template.get(y_center_key)
            if y_center_val is not None:
                ax.set_ylim(
                    y_center_val + y_off[0],
                    y_center_val + y_off[1],
                )

        # Título: nombre de región + flag simple
        flag = "VAR" if name in triggered_set else "non-VAR"
        ax.set_title(f"Region {name} ({flag})", fontsize=10)

        ax.grid(alpha=0.3, linestyle="--")

    # Apagar axes sobrantes
    for j in range(len(regions), len(axes_flat)):
        axes_flat[j].set_visible(False)

    # Título global
    suptitle_flag = (
        "candidate VARIABLE" if result.is_candidate_variable else "candidate non-variable"
    )
    fig.suptitle(
        f"{spt}-dwarf variability scheme ({result.scheme}) — "
        f"{suptitle_flag} ({result.n_regions_triggered}/{n_total} regions)",
        fontsize=16,
    )

    # Leyenda global
    handles = []
    if template:
        handles.append(
            plt.Line2D([], [], marker="*", color="black", linestyle="None", label=template_label)
        )
    handles.append(
        plt.Line2D([], [], marker="*", color="blue", linestyle="None", label=bd_label)
    )
    fig.legend(handles=handles, loc="upper right", fontsize=10)

    if savepath is not None:
        fig.savefig(savepath, bbox_inches="tight")

    if show:
        plt.show()

    return fig, axes
