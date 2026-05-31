import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

def estimate_eps(features, min_samples=5):
    """
    Szacuje wartość eps na podstawie k-distance (punkt największego załamania).
    """

    neigh = NearestNeighbors(n_neighbors=min_samples)
    neigh.fit(features)
    distances, _ = neigh.kneighbors(features)

    # bierzemy odległość do k-tego sąsiada i sortujemy
    k_distances = np.sort(distances[:, -1])

    # szukamy największej zmiany (tzw. "kolano")
    gradient = np.gradient(k_distances)
    idx = np.argmax(gradient)
    eps = k_distances[idx]

    return float(eps)


def evaluate_clustering(features, labels):
    """
    Liczy metryki jakości klasteryzacji.
    Noise (-1) jest pomijany przy metrykach typu silhouette.
    """

    unique = set(labels)
    # liczba klastrów bez punktów szumowych
    n_clusters = len(unique) - (1 if -1 in labels else 0)
    # udział punktów uznanych za noise
    noise_ratio = np.sum(labels == -1) / len(labels)

    result = {
        "clusters": n_clusters,
        "noise_ratio": noise_ratio,
        "silhouette": None,
        "davies_bouldin": None,
        "calinski_harabasz": None
    }

    # bierzemy tylko punkty niebędące noise
    valid = labels != -1

    # metryki mają sens tylko przy więcej niż 1 klastrze
    if len(set(labels[valid])) > 1:

        try:
            result["silhouette"] = silhouette_score(
                features[valid],
                labels[valid]
            )
            result["davies_bouldin"] = davies_bouldin_score(
                features[valid],
                labels[valid]
            )
            result["calinski_harabasz"] = calinski_harabasz_score(
                features[valid],
                labels[valid]
            )

        except:
            pass

    return result


def tune_dbscan(features, env_name=None, eps_values=None, min_samples_values=None):
    """
    Dobiera parametry DBSCAN na podstawie metryk jakości.
    Uwzględnia też proste heurystyki (np. kara za zbyt dużo klastrów).
    """

    # jeśli nie podano eps, wyznaczamy zakres wokół wartości automatycznej
    if eps_values is None:
        auto_eps = estimate_eps(features)
        eps_values = np.linspace(auto_eps * 0.5, auto_eps * 1.5, 6)

    # domyślne wartości min_samples
    if min_samples_values is None:
        min_samples_values = [5, 8, 10, 15]

    best_score = -999
    best_result = None

    # sprawdzamy wszystkie kombinacje parametrów
    for eps in eps_values:
        for ms in min_samples_values:

            model = DBSCAN(
                eps=float(eps),
                min_samples=int(ms)
            )
            labels = model.fit_predict(features)
            metrics = evaluate_clustering(features, labels)
            score = -999

            # liczymy score tylko jeśli silhouette istnieje
            if metrics["silhouette"] is not None:

                # baza: dobra separacja - kara za noise
                score = metrics["silhouette"] - metrics["noise_ratio"]
                n_clusters = metrics["clusters"]
                # kara za zbyt dużą liczbę klastrów (przeuczenie)
                if n_clusters > 10:
                    score -= 0.25
                if n_clusters > 15:
                    score -= 0.50
                # dodatkowe ograniczenia dla środowiska wodnego
                if env_name == "wody":
                    if n_clusters > 6:
                        score -= 0.60
                    if n_clusters > 10:
                        score -= 1.00
                    if metrics["noise_ratio"] > 0.10:
                        score -= 0.50

            # zapis najlepszego wyniku
            if score > best_score:
                best_score = score
                best_result = {
                    "eps": float(eps),
                    "min_samples": int(ms),
                    "labels": labels,
                    "metrics": metrics
                }

    # fallback jeśli nic sensownego nie znaleziono
    if best_result is None:
        model = DBSCAN(
            eps=0.05,
            min_samples=10
        )
        labels = model.fit_predict(features)
        best_result = {
            "eps": 0.05,
            "min_samples": 10,
            "labels": labels,
            "metrics": evaluate_clustering(features, labels)
        }

    return best_result