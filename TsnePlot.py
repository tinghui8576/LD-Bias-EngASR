import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colorbar import ColorbarBase

# ==========================================================
# CONFIG
# ==========================================================
model_layers = {
    'CanaryV2': 32,
    'ParakeetV3': 24,
    'Whisper-Large': 33,
}
TSNE_DIR_TEMPLATE = "Embeddings/tsne0/{model}/tsne_layer_{layer}.h5"
feature_names = ["LD", "WER", "L1"]  # 3 active rows per block
cmap_master = plt.cm.plasma_r
cmap_discrete = plt.cm.gist_ncar

LAYERS_PER_ROW = 10  # max number of layer-columns per block/row-group

os.makedirs("Plot/Emb_combined", exist_ok=True)


def load_layer(model, layer_idx):
    h5_path = TSNE_DIR_TEMPLATE.format(model=model, layer=layer_idx)
    try:
        with h5py.File(h5_path, "r") as f:
            emb_2d = f["tsne_coords"][:]
            dists = f["dists"][:]
            wer_arr = f["wer"][:]
            accents_raw = f["accents"][:]
            accents = [a.decode("utf-8") if isinstance(a, bytes) else a
                       for a in accents_raw]
    except (OSError, FileNotFoundError, KeyError) as e:
        print(f"  Layer {layer_idx}: skipping ({e})")
        return None

    lengths = {len(emb_2d), len(dists), len(wer_arr), len(accents)}
    if len(lengths) != 1:
        raise ValueError(
            f"Layer {layer_idx}: mismatched array lengths {lengths} - "
            f"'wer' is probably unfiltered. Use the tsne0_fixed files."
        )

    return dict(
        emb_2d=emb_2d, dists=dists,
        wer=wer_arr[:, 0], wil=wer_arr[:, 1], semdist=wer_arr[:, 2],
        accents=accents,
    )


def build_global_lang_to_id(model, all_layer_indices):
    """Accents are identical across layers for a given model, so reading
    just one layer is enough to build a consistent color mapping."""
    for layer_idx in all_layer_indices:
        data = load_layer(model, layer_idx)
        if data is not None:
            langs = sorted(set(data["accents"]))
            return {lang: idx for idx, lang in enumerate(langs)}
    return {}


# ==========================================================
# MAIN LOOP: ONE figure per model, layers wrapped into blocks
# ==========================================================
for model, total_layers in model_layers.items():
    print(f"\n===== {model} =====")
    all_layer_indices = list(range(total_layers))

    lang_to_id = build_global_lang_to_id(model, all_layer_indices)
    num_languages = len(lang_to_id)
    if num_languages == 0:
        print(f"  No readable layers found for {model}, skipping entirely.")
        continue

    vmax_mapping = {0: 100, 1: 1}  # adjust per metric as needed

    # Preload ALL layers once (needed regardless of layout)
    cached = {}
    for layer_idx in all_layer_indices:
        data = load_layer(model, layer_idx)
        if data is not None:
            cached[layer_idx] = data
    if not cached:
        print(f"  No usable layers for {model}, skipping.")
        continue

    n_feature_rows = len(feature_names)  # 3
    blocks = [
        all_layer_indices[i:i + LAYERS_PER_ROW]
        for i in range(0, len(all_layer_indices), LAYERS_PER_ROW)
    ]
    n_blocks = len(blocks)
    total_fig_rows = n_feature_rows * n_blocks
    n_cols = LAYERS_PER_ROW  # fixed column count; short last block leaves blanks

    fig, axes = plt.subplots(
        total_fig_rows, n_cols,
        figsize=(3.5 * n_cols, 3 * total_fig_rows),
        squeeze=False
    )
    row_scatters = {}  # (block_idx, row_offset) -> scatter artist, for colorbars

    for block_idx, batch_layers in enumerate(blocks):
        row_base = block_idx * n_feature_rows

        for col in range(n_cols):
            if col >= len(batch_layers) or batch_layers[col] not in cached:
                for row_offset in range(n_feature_rows):
                    axes[row_base + row_offset, col].axis("off")
                continue

            layer_idx = batch_layers[col]
            data = cached[layer_idx]
            emb_2d = data["emb_2d"]
            L1_encoded = np.array([lang_to_id[a] for a in data["accents"]])
            metrics_data = {0: data["dists"], 1: data["wer"], 2: L1_encoded}

            for row_offset in range(n_feature_rows):
                ax = axes[row_base + row_offset, col]

                if row_offset == 2:
                    sc = ax.scatter(emb_2d[:, 0], emb_2d[:, 1], c=metrics_data[row_offset],
                                     cmap=cmap_discrete, vmin=0, vmax=max(num_languages - 1, 1),
                                     s=4, alpha=0.6, rasterized=True)
                else:
                    sc = ax.scatter(emb_2d[:, 0], emb_2d[:, 1], c=metrics_data[row_offset],
                                     cmap=cmap_master, vmin=0, vmax=vmax_mapping[row_offset],
                                     s=4, alpha=0.6, rasterized=True)

                if col == len(batch_layers) - 1:  # last real column in this block
                    row_scatters[(block_idx, row_offset)] = sc

                if col == 0:
                    ax.set_ylabel(feature_names[row_offset], fontsize=35, fontweight="bold")
                if row_offset == 0:
                    ax.set_title(f"L{layer_idx}", fontsize=35)

                ax.set_xticks([]); ax.set_yticks([])

    fig.tight_layout(rect=[0, 0.15, 0.90, 0.99])

    # ------------------------------------------------------
    # Colorbars - one small set per block, anchored to that
    # block's row positions on the right margin
    # ------------------------------------------------------
    for (block_idx, row_offset), sc in row_scatters.items():
        abs_row = block_idx * n_feature_rows + row_offset
        pos = axes[abs_row, 0].get_position()

        if row_offset == 2:
            cax = fig.add_axes([0.905, pos.y0, 0.012, pos.height])
            cax.axis("off")
            if block_idx == 0:  # only draw the language legend once, at the top block
                legend_elements = [
                    plt.Line2D([0], [0], marker="o", color="w", label=lang.capitalize(),
                               markerfacecolor=cmap_discrete(idx / max(num_languages - 1, 1)),
                               markersize=10)
                    for lang, idx in lang_to_id.items()
                ]
                fig.legend(
                    handles=legend_elements,
                    loc="lower center",
                    bbox_to_anchor=(0.5, 0),  # (x, y) in FIGURE-fraction coords: 0.5 = horizontal center, 0.0 = bottom edge
                    title="Native Languages",
                    title_fontproperties={"weight": "bold", "size": 30},
                    fontsize=26, frameon=False, ncol=11, columnspacing=0.5, handletextpad=0.2
                )
        else:
            cax = fig.add_axes([0.905, pos.y0, 0.012, pos.height])
            norm = colors.Normalize(vmin=0, vmax=vmax_mapping[row_offset])
            cb = ColorbarBase(cax, cmap=cmap_master, norm=norm, orientation="vertical")
            cb.set_ticks(np.linspace(0, vmax_mapping[row_offset], 3) if vmax_mapping[row_offset] > 10
                         else [0.0, 0.5, 1.0])
            cb.ax.tick_params(labelsize=35)

    save_filename = f"Plot/Emb_combined/Tsne_Layer_{model}_wrapped.pdf"
    fig.savefig(save_filename, dpi=100)  
    print(f"  Saved: {save_filename}")
    plt.close(fig)



# ==========================================================
# CONFIG
# ==========================================================
model_layers = {
    'Whisper-Small': 13,
}

# ==========================================================
# MAIN LOOP (one figure per model)
# ==========================================================
for model, layer_indices in model_layers.items():
    # ----------------------------------------------------
    # PASS 1: load every layer file once, cache the arrays,
    # and build a GLOBAL language -> color-id mapping so
    # colors are consistent across all layer panels.
    # ----------------------------------------------------
    cached = {}  
    all_langs = set()

    for i in range(layer_indices):
        h5_path = TSNE_DIR_TEMPLATE.format(model=model, layer=i)
        try:
            with h5py.File(h5_path, "r") as f:
                emb_2d = f["tsne_coords"][:]
                dists = f["dists"][:]
                wer_arr = f["wer"][:]          
                
                accents_raw = f["accents"][:]
                accents = [a.decode("utf-8") if isinstance(a, bytes) else a
                           for a in accents_raw]
        except (OSError, FileNotFoundError) as e:
            print(f"Skipping layer {i} for {model}: {e}")
            continue

        # sanity check the alignment bug is actually fixed
        lengths = {len(emb_2d), len(dists), len(wer_arr), len(accents)}
        if len(lengths) != 1:
            raise ValueError(
                f"Layer {i}: mismatched array lengths {lengths} - "
                f"the 'wer' array is probably unfiltered. Re-run the "
                f"generation script with the fix applied."
            )

        cached[i] = dict(
            emb_2d=emb_2d,
            dists=dists,
            wer=wer_arr[:, 0],
            wil=wer_arr[:, 1],
            semdist=wer_arr[:, 2],
            accents=accents,
        )
        all_langs.update(accents)

    if not cached:
        print(f"No usable layer files found for {model}, skipping.")
        continue

    lang_to_id = {lang: idx for idx, lang in enumerate(sorted(all_langs))}
    num_languages = len(lang_to_id)

    # vmax per metric row - computed globally across all cached layers
    # so the colorbar scale is consistent across panels
    vmax_mapping = {
        0: 100,
        1: 1,  
    }

    # ----------------------------------------------------
    # PASS 2: build the grid figure
    # ----------------------------------------------------
    fig, axes = plt.subplots(
        3, layer_indices,
        figsize=(6 * layer_indices, 3 * 5),
        squeeze=False
    )
    row_scatters = {}

    for col, layer_idx in enumerate(range(layer_indices)):
        if layer_idx not in cached:
            for row_idx in range(5):
                axes[row_idx, col].axis("off")
            continue

        data = cached[layer_idx]
        emb_2d = data["emb_2d"]
        L1_encoded = np.array([lang_to_id[a] for a in data["accents"]])

        metrics_data = {
            0: data["dists"],
            1: data["wer"],
            2: L1_encoded,
        }

        for row_idx in range(3):
            ax = axes[row_idx, col]

            if row_idx == 2:
                sc = ax.scatter(
                    emb_2d[:, 0], emb_2d[:, 1],
                    c=metrics_data[row_idx],
                    cmap=cmap_discrete,
                    vmin=0, vmax=max(num_languages - 1, 1),
                    s=4, alpha=0.6, rasterized=True
                )
            else:
                sc = ax.scatter(
                    emb_2d[:, 0], emb_2d[:, 1],
                    c=metrics_data[row_idx],
                    cmap=cmap_master,
                    vmin=0, vmax=vmax_mapping[row_idx],
                    s=4, alpha=0.6, rasterized=True
                )

            if col == layer_indices - 1:
                row_scatters[row_idx] = sc

            if col == 0:
                ax.set_ylabel(feature_names[row_idx], fontsize=50, fontweight="bold")
            if row_idx == 0:
                ax.set_title(f"L{layer_idx}", fontsize=50)

            ax.set_xticks([])
            ax.set_yticks([])

    fig.tight_layout(rect=[0, 0.24, 0.88, 0.99])

    # ----------------------------------------------------
    # COLORBARS / LEGEND
    # ----------------------------------------------------
    for row_idx in range(3):
        if row_idx not in row_scatters:
            continue

        if row_idx == 2:
            pos = axes[row_idx, 0].get_position()
            cax = fig.add_axes([0.89, pos.y0, 0.015, pos.height])
            cax.axis("off")
            legend_elements = [
                plt.Line2D([0], [0], marker="o", color="w",
                           label=lang.capitalize(),
                           markerfacecolor=cmap_discrete(idx / max(num_languages - 1, 1)),
                           markersize=20)
                for lang, idx in lang_to_id.items()
            ]
            fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.45, 0),
                title="Native Languages", title_fontproperties={'weight':'bold', 'size': 35},
                fontsize=35, frameon=False, ncol=17, columnspacing=0.6, handletextpad=0.2)
        else:

            if row_idx == 0:
                y_offset = 0.04   # Nudge row 0 colorbar slightly UP
            elif row_idx == 1:
                y_offset =  0.015 # Nudge row 1 colorbar slightly DOWN

            cax = fig.add_axes([0.89, axes[row_idx, 0].get_position().y0 + y_offset, 0.015, axes[row_idx, 0].get_position().height])
            norm = colors.Normalize(vmin=0, vmax=vmax_mapping[row_idx])
            
            cb = ColorbarBase(
                cax,
                cmap=cmap_master,
                norm=norm,
                orientation='vertical'
            )
            
            if vmax_mapping[row_idx] > 10:
                cb.set_ticks(np.linspace(0, vmax_mapping[row_idx], 3))
            else:
                cb.set_ticks([0.0,  0.5,  1.0])
            cb.ax.tick_params(labelsize=30)
            cb.ax.legend(bbox_to_anchor=(0, row_idx * 0.1))
            cb.ax.set_title("", fontsize=30, fontweight='bold')

    save_filename = f"Plot/Emb_combined/Tsne_Layer_{model}_new.pdf"
    fig.savefig(save_filename, dpi=100, bbox_inches="tight")
    print(f"Saved: {save_filename}")
    plt.close(fig) 