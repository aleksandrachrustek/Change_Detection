import numpy as np
import hdbscan
from sklearn.metrics import silhouette_score, davies_bouldin_score

def evaluate_hdbscan(features, labels):
    """
    Liczy podstawowe metryki jakości dla HDBSCAN.
    Punkty noise (-1) są pomijane przy metrykach.
    """

    # liczba klastrów bez noise
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    # udział punktów oznaczonych jako noise
    noise_ratio = np.sum(labels == -1) / len(labels)

    result = {
        "clusters": n_clusters,
        "noise_ratio": noise_ratio,
        "silhouette": None,
        "davies_bouldin": None
    }

    # bierzemy tylko punkty niebędące noise
    valid = labels != -1

    # metryki mają sens tylko przy >1 klastrze i wystarczającej liczbie punktów
    if n_clusters > 1 and np.sum(valid) > 10:

        try:
            result["silhouette"] = silhouette_score(
                features[valid],
                labels[valid]
            )
            result["davies_bouldin"] = davies_bouldin_score(
                features[valid],
                labels[valid]
            )

        except:
            pass

    return result


def tune_hdbscan(features, env_name):
    """
    Dobiera parametry HDBSCAN poprzez przeszukiwanie siatki.
    Zakres parametrów zależy od typu środowiska.
    """

    # różne zakresy dla różnych środowisk
    if env_name == "lasy":
        cluster_sizes = [50, 80, 120]
        min_samples_values = [10, 15, 20]

    elif env_name == "rzeki":
        cluster_sizes = [30, 60, 100]
        min_samples_values = [5, 10, 15]

    else:  # wody
        cluster_sizes = [80, 120, 180]
        min_samples_values = [10, 20, 30]

    best_score = -999
    best_result = None

    # testujemy wszystkie kombinacje parametrów
    for cs in cluster_sizes:
        for ms in min_samples_values:
            model = hdbscan.HDBSCAN(
                min_cluster_size=cs,
                min_samples=ms
            )
            labels = model.fit_predict(features)
            metrics = evaluate_hdbscan(features, labels)
            score = -999
            # score oparty na jakości klastrów i ilości noise
            if metrics["silhouette"] is not None:
                score = metrics["silhouette"] - metrics["noise_ratio"]
            # kara za zbyt dużą liczbę klastrów (niestabilne wyniki)
            if metrics["clusters"] > 15:
                score -= 0.3
            # zapis najlepszego wyniku
            if score > best_score:
                best_score = score
                best_result = {
                    "min_cluster_size": cs,
                    "min_samples": ms,
                    "labels": labels,
                    "metrics": metrics
                }

    # fallback jeśli tuning nie znalazł sensownego rozwiązania
    if best_result is None:
        model = hdbscan.HDBSCAN(
            min_cluster_size=50,
            min_samples=10
        )
        labels = model.fit_predict(features)
        best_result = {
            "min_cluster_size": 50,
            "min_samples": 10,
            "labels": labels,
            "metrics": evaluate_hdbscan(features, labels)
        }

    return best_result