import matplotlib.pyplot as plt
import numpy as np


def plot_scene(red, green, nir, title, save_path):
    # tworzenie obrazu RGB (kompozycja kanałów)
    rgb = np.dstack((nir, red, green))
    # zamiana NaN na 0 (żeby obraz się poprawnie wyświetlał)
    rgb = np.nan_to_num(rgb)
    # normalizacja do zakresu 0–1 (ucięcie wartości odstających)
    p = np.percentile(rgb, 98)
    if p > 0:
        rgb = np.clip(rgb, 0, p)
        rgb = rgb / p
    plt.figure(figsize=(8, 8))
    plt.imshow(rgb)
    plt.title(title)
    plt.xlabel("X (pixels)")
    plt.ylabel("Y (pixels)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_index(index, title, save_path, cmap="RdYlGn"):
    # wizualizacja indeksu (np. NDVI, NDWI)
    plt.figure(figsize=(8, 8))
    # NaN jako transparentne
    masked = np.ma.masked_invalid(index)
    im = plt.imshow(masked, cmap=cmap)
    plt.colorbar(im, label="Index value")
    plt.title(title)
    plt.xlabel("X (pixels)")
    plt.ylabel("Y (pixels)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_clusters(cluster_map, title, save_path):
    # wizualizacja klastrów jako mapa kolorów
    plt.figure(figsize=(8, 8))
    masked = np.ma.masked_invalid(cluster_map)
    im = plt.imshow(masked, cmap="tab20")
    plt.colorbar(im, label="Cluster ID")
    plt.title(title)
    plt.xlabel("X (pixels)")
    plt.ylabel("Y (pixels)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_clusters_overlay(base, clusters, title, save_path):
    # nałożenie klastrów na obraz bazowy (np. NDVI)
    plt.figure(figsize=(8, 8))
    # obraz bazowy w odcieniach szarości
    base_masked = np.ma.masked_invalid(base)
    plt.imshow(base_masked, cmap="gray")
    # maskujemy NaN i noise (-1)
    clusters = np.array(clusters)
    masked_clusters = np.ma.masked_where(
        np.isnan(clusters) | (clusters == -1),
        clusters
    )
    # półprzezroczyste klastry
    plt.imshow(masked_clusters, cmap="tab20", alpha=0.6)
    plt.title(title)
    plt.xlabel("X (pixels)")
    plt.ylabel("Y (pixels)")
    # legenda klastrów (tylko istniejące etykiety)
    unique = np.unique(clusters)
    unique = unique[~np.isnan(unique)]
    unique = unique[unique != -1]

    if len(unique) > 0:
        handles = []
        for c in unique:
            try:
                color = plt.cm.tab20(int(float(c)) % 20)
                handles.append(
                    plt.Line2D(
                        [0], [0],
                        marker='o',
                        color='w',
                        label=f'Cluster {int(c)}',
                        markerfacecolor=color,
                        markersize=8
                    )
                )
            except:
                continue
        if handles:
            plt.legend(
                handles=handles,
                bbox_to_anchor=(1.05, 1),
                loc='upper left'
            )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()