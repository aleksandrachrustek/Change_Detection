import numpy as np

def compute_cluster_stats(cluster_map, ndvi, ndwi):
    """
    Liczy średnie wartości NDVI i NDWI dla każdego klastra.
    """

    stats = {}
    # tylko piksele należące do klastrów
    valid = ~np.isnan(cluster_map)
    # unikalne identyfikatory klastrów
    labels = np.unique(cluster_map[valid]).astype(int)

    for cid in labels:
        # maska dla danego klastra
        mask = cluster_map == cid
        # średnie wartości indeksów w klastrze
        stats[cid] = {
            "ndvi_mean": float(np.nanmean(ndvi[mask])),
            "ndwi_mean": float(np.nanmean(ndwi[mask]))
        }
    return stats


def classify_cluster(ndvi, ndwi):
    """
    Przypisuje klasę semantyczną na podstawie średnich wartości NDVI i NDWI.
    """

    # woda – wysoki NDWI
    if ndwi > 0.2:
        return 1
    # roślinność – wysoki NDVI
    elif ndvi > 0.4:
        return 2
    # pozostałe (np. gleba, zabudowa)
    else:
        return 3


def build_semantic_map(cluster_map, stats):
    """
    Zamienia mapę klastrów na mapę klas semantycznych.
    """

    # mapa wynikowa (NaN poza obszarem danych)
    semantic_map = np.full_like(cluster_map, np.nan)
    for cid, s in stats.items():
        # przypisanie klasy na podstawie średnich wartości klastra
        label = classify_cluster(
            s["ndvi_mean"],
            s["ndwi_mean"]
        )
        # przypisanie etykiety wszystkim pikselom klastra
        semantic_map[cluster_map == cid] = label

    return semantic_map