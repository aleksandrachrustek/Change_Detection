import os
import csv
import pandas as pd
import numpy as np

RESULTS_DIR = "results"

# tworzenie katalogu na wyniki
os.makedirs(RESULTS_DIR, exist_ok=True)

def mean_index(index_img):
    # średnia wartość indeksu (bez NaN)
    valid = index_img[~np.isnan(index_img)]
    if len(valid) == 0:
        return np.nan

    return float(np.mean(valid))


def cluster_percentages(cluster_map):
    # udział procentowy klastrów w obrazie
    valid = cluster_map[~np.isnan(cluster_map)]
    if len(valid) == 0:
        return {}
    valid = valid.astype(int)
    labels, counts = np.unique(valid, return_counts=True)
    total = counts.sum()

    out = {}
    for lab, cnt in zip(labels, counts):
        out[int(lab)] = round(cnt / total * 100, 2)

    return out


def save_result(env_name, year, tuning, cluster_map, ndvi, ndwi, algorithm):
    # zapis podsumowania wyników do CSV
    file_path = os.path.join(RESULTS_DIR, "results_summary.csv")
    file_exists = os.path.isfile(file_path)
    metrics = tuning["metrics"]
    row = {
        "environment": env_name,
        "year": year,
        "algorithm": algorithm,
        "eps": tuning.get("eps", None),
        "min_samples": tuning.get("min_samples", None),
        "clusters": metrics["clusters"],
        "noise_ratio": round(metrics["noise_ratio"], 4),
        "silhouette": metrics.get("silhouette", None),
        "davies_bouldin": metrics.get("davies_bouldin", None),
        "calinski_harabasz": metrics.get("calinski_harabasz", None),
        "mean_ndvi": round(mean_index(ndvi), 4),
        "mean_ndwi": round(mean_index(ndwi), 4)
    }

    with open(file_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    # zapis rozkładu klastrów
    save_cluster_stats(env_name, year, algorithm, cluster_map)


def save_cluster_stats(env_name, year, algorithm, cluster_map):
    # zapis udziałów klastrów (%)

    file_path = os.path.join(RESULTS_DIR, "cluster_stats.csv")
    file_exists = os.path.isfile(file_path)
    perc = cluster_percentages(cluster_map)

    with open(file_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "environment",
                "year",
                "algorithm",
                "cluster_id",
                "percentage"
            ])
        for cid, pct in perc.items():
            writer.writerow([
                env_name,
                year,
                algorithm,
                cid,
                pct
            ])


def compare_years():
    # porównanie kolejnych lat (różnice metryk)
    src = os.path.join(RESULTS_DIR, "results_summary.csv")
    if not os.path.isfile(src):
        return
    df = pd.read_csv(src)
    rows = []

    for alg in df["algorithm"].unique():
        dfa = df[df["algorithm"] == alg]

        for env in dfa["environment"].unique():
            sub = dfa[dfa["environment"] == env].sort_values("year")
            years = sub["year"].tolist()

            for i in range(len(years) - 1):
                a = sub.iloc[i]
                b = sub.iloc[i + 1]
                rows.append({
                    "algorithm": alg,
                    "environment": env,
                    "from_year": a["year"],
                    "to_year": b["year"],
                    "delta_clusters": b["clusters"] - a["clusters"],
                    "delta_noise": round(b["noise_ratio"] - a["noise_ratio"], 4),
                    "delta_ndvi": round(b["mean_ndvi"] - a["mean_ndvi"], 4),
                    "delta_ndwi": round(b["mean_ndwi"] - a["mean_ndwi"], 4)
                })

    out = pd.DataFrame(rows)
    out.to_csv(
        os.path.join(RESULTS_DIR, "comparison.csv"),
        index=False
    )


def clear_results():
    # usuwa zapisane pliki wyników
    files = [
        "results_summary.csv",
        "cluster_stats.csv",
        "comparison.csv"
    ]

    for f in files:
        path = os.path.join(RESULTS_DIR, f)
        if os.path.isfile(path):
            os.remove(path)


def save_change_types(env_name, from_year, to_year, algorithm, change_types_map):
    # zapis typów zmian (liczba pikseli i %)
    file_path = os.path.join(RESULTS_DIR, "change_types.csv")
    file_exists = os.path.isfile(file_path)
    valid = change_types_map[~np.isnan(change_types_map)].astype(int)
    labels, counts = np.unique(valid, return_counts=True)
    total = counts.sum()

    with open(file_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "environment",
                "from_year",
                "to_year",
                "algorithm",
                "change_type",
                "pixels",
                "percent"
            ])
        for lab, cnt in zip(labels, counts):
            writer.writerow([
                env_name,
                from_year,
                to_year,
                algorithm,
                int(lab),
                int(cnt),
                round(cnt / total * 100, 2)
            ])