import numpy as np
from sklearn.cluster import DBSCAN
import hdbscan
from dbscan_tuning import tune_dbscan
from hdbscan_tuning import tune_hdbscan

def run_clustering(features, env_name, algorithm, params=None):
    """
    Uruchamia klasteryzację (DBSCAN lub HDBSCAN).
    Jeśli brak parametrów, wykonywany jest tuning.
    """

    # DBSCAN
    if algorithm == "DBSCAN":
        # jeśli brak parametrów → szukamy najlepszych
        if params is None:
            return tune_dbscan(features, env_name)
        # tworzenie modelu z zadanymi parametrami
        model = DBSCAN(
            eps=params["eps"],
            min_samples=params["min_samples"]
        )

        labels = model.fit_predict(features)
        
        # liczba klastrów (bez noise = -1)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        # udział punktów uznanych za noise
        noise_ratio = np.mean(labels == -1)

        return {
            "eps": params["eps"],
            "min_samples": params["min_samples"],
            "labels": labels,
            "metrics": {
                "clusters": n_clusters,
                "noise_ratio": noise_ratio
            }
        }

    # HDBSCAN
    elif algorithm == "HDBSCAN":
        # jeśli brak parametrów → tuning
        if params is None:
            return tune_hdbscan(features, env_name)
        # model HDBSCAN
        model = hdbscan.HDBSCAN(
            min_cluster_size=params["min_cluster_size"],
            min_samples=params["min_samples"]
        )

        labels = model.fit_predict(features)

        # liczba klastrów bez noise
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        # udział noise
        noise_ratio = np.mean(labels == -1)

        return {
            "min_cluster_size": params["min_cluster_size"],
            "min_samples": params["min_samples"],
            "labels": labels,
            "metrics": {
                "clusters": n_clusters,
                "noise_ratio": noise_ratio
            }
        }

    else:
        raise ValueError("Unknown algorithm")


def labels_to_image(labels, mask, shape):
    """
    Zamienia wektor etykiet na obraz (mapę klastrów).
    Piksele poza maską ustawiane jako NaN.
    """

    # tworzymy pusty obraz
    out = np.full(shape, np.nan)
    # przypisujemy etykiety tylko dla poprawnych pikseli
    out[mask] = labels

    return out