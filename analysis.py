import os
import numpy as np
from io_utils import find_band, read_window, align_window
from preprocessing import mask_invalid, mask_clouds, denoise, compute_global_stats, normalize_with_stats
from indices import compute_ndvi, compute_ndwi
from features import create_feature_vector
from visualisation import plot_clusters, plot_scene, plot_index, plot_clusters_overlay
from clustering import run_clustering, labels_to_image
from sklearn.neighbors import NearestNeighbors
from results import save_result, save_change_types
from semantic import compute_cluster_stats, build_semantic_map
from change_detection import detect_changes, change_statistics, change_area_stats
from change_types import classify_change


def load_bands_aligned_window(folder_path, ref_folder, crop_coords):
    """
    Wczytuje pasma (R, NIR, G) i wyrównuje je do sceny referencyjnej (2016).
    """

    files = os.listdir(folder_path)
    ref_files = os.listdir(ref_folder)

    # współrzędne wycinka obrazu
    i, j = crop_coords

    # wczytanie pasma referencyjnego (2016), do którego dopasujemy resztę
    ref_red = find_band(ref_files, "B04")
    ref_path = os.path.join(ref_folder, ref_red)
    ref_data, ref_transform, ref_crs = read_window(ref_path, i, j, 1000)

    # znalezienie odpowiednich pasm w aktualnym obrazie
    red_file = find_band(files, "B04")
    nir_file = find_band(files, "B08")
    green_file = find_band(files, "B03")

    # wczytanie fragmentów pasm
    red_data, red_tr, red_crs = read_window(os.path.join(folder_path, red_file), i, j, 1000)
    nir_data, nir_tr, nir_crs = read_window(os.path.join(folder_path, nir_file), i, j, 1000)
    green_data, green_tr, green_crs = read_window(os.path.join(folder_path, green_file), i, j, 1000)

    # dopasowanie wszystkich pasm do tej samej siatki (referencji)
    red = align_window(red_data, red_tr, red_crs, ref_transform, ref_crs, ref_data.shape)
    nir = align_window(nir_data, nir_tr, nir_crs, ref_transform, ref_crs, ref_data.shape)
    green = align_window(green_data, green_tr, green_crs, ref_transform, ref_crs, ref_data.shape)

    return red, nir, green


def process_year(folder_path, ref_folder, crop_coords, env_name, save_dir=None, norm_stats=None, algorithm="DBSCAN"):
    """
    Przetwarza dane dla jednego roku:
    przygotowanie danych, indeksy, klasteryzacja i zapis wyników.
    """

    # wczytanie danych wejściowych
    red, nir, green = load_bands_aligned_window(folder_path, ref_folder, crop_coords)

    # nazwa folderu traktowana jako etykieta (rok)
    label = os.path.basename(folder_path)

    # tworzenie katalogu na wyniki
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # usunięcie błędnych wartości (np. brak danych)
    red = mask_invalid(red)
    nir = mask_invalid(nir)
    green = mask_invalid(green)

    # redukcja szumu (lepsza jakość indeksów)
    red = denoise(red)
    nir = denoise(nir)
    green = denoise(green)

    # maskowanie chmur
    red, green, nir = mask_clouds(red, green, nir)

    # obliczenie indeksów spektralnych
    ndvi = compute_ndvi(nir, red)     # roślinność
    ndwi = compute_ndwi(green, nir)   # woda

    # wyznaczenie zakresów do normalizacji (tylko dla pierwszego roku)
    if norm_stats is None:
        ndvi_low, ndvi_high = compute_global_stats(ndvi)
        ndwi_low, ndwi_high = compute_global_stats(ndwi)
    else:
        # dla kolejnych lat używamy tych samych zakresów
        ndvi_low, ndvi_high, ndwi_low, ndwi_high = norm_stats

    # normalizacja danych (żeby były porównywalne między latami)
    ndvi_norm = normalize_with_stats(ndvi, ndvi_low, ndvi_high)
    ndwi_norm = normalize_with_stats(ndwi, ndwi_low, ndwi_high)

    # tworzenie wektorów cech dla każdego piksela
    features, mask = create_feature_vector(ndvi_norm, ndwi_norm)

    # ograniczenie liczby punktów
    max_points = 5000
    if len(features) > max_points:
        idx = np.random.choice(len(features), max_points, replace=False)
        features_sampled = features[idx]
    else:
        features_sampled = features

    # uruchomienie klasteryzacji
    best = run_clustering(features_sampled, env_name, algorithm=algorithm)

    print(f"\n--- {algorithm} ---")
    print("params:", {k: v for k, v in best.items() if k != "labels"})
    print("metrics:", best["metrics"])

    labels_sampled = best["labels"]

    # przypisanie etykiet do wszystkich pikseli na podstawie najbliższego sąsiada
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(features_sampled)

    _, indices = nn.kneighbors(features)
    labels_full = labels_sampled[indices.flatten()]

    # zamiana etykiet na obraz (mapa klastrów)
    cluster_map = labels_to_image(labels_full, mask, ndvi.shape)

    # podstawowe statystyki klasteryzacji
    n_clusters = len(set(labels_full)) - (1 if -1 in labels_full else 0)
    noise = np.sum(labels_full == -1) / len(labels_full)

    print(f"{label} - klastry: {n_clusters}")
    print(f"{label} - noise: {noise:.2%}")

    # zapis wizualizacji
    if save_dir:
        plot_scene(
            red, green, nir,
            f"{algorithm} | {env_name} | {label} RGB",
            os.path.join(save_dir, f"{label}_01_rgb.png")
        )

        plot_index(
            ndvi,
            f"{algorithm} | {env_name} | {label} NDVI",
            os.path.join(save_dir, f"{label}_02_ndvi.png")
        )

        plot_index(
            ndwi,
            f"{algorithm} | {env_name} | {label} NDWI",
            os.path.join(save_dir, f"{label}_03_ndwi.png"),
            cmap="Blues"
        )

        plot_clusters_overlay(
            ndvi,
            cluster_map,
            f"{algorithm} | {env_name} | {label} NDVI + clusters",
            os.path.join(save_dir, f"{label}_04_clusters_ndvi.png")
        )

        plot_clusters_overlay(
            ndwi,
            cluster_map,
            f"{algorithm} | {env_name} | {label} NDWI + clusters",
            os.path.join(save_dir, f"{label}_05_clusters_ndwi.png")
        )

        plot_clusters(
            cluster_map,
            f"{algorithm} | {env_name} | {label} CLUSTERS ONLY",
            os.path.join(save_dir, f"{label}_06_clusters_only.png")
        )

    # zapis wyników do pliku
    save_result(env_name, label, best, cluster_map, ndvi, ndwi, algorithm)

    # statystyki klastrów
    cluster_stats = compute_cluster_stats(cluster_map, ndvi, ndwi)

    # przypisanie znaczenia klastrom
    semantic_map = build_semantic_map(cluster_map, cluster_stats)

    return ndvi, ndwi, features, mask, cluster_map, semantic_map, (ndvi_low, ndvi_high, ndwi_low, ndwi_high)


def run_analysis(datasets, algorithm):
    """
    Główna pętla analizy:
    najpierw rok referencyjny, potem kolejne lata i detekcja zmian.
    """

    for env_name, years in datasets.items():

        print(f"\n--- {algorithm} Analysis: {env_name} ---")

        # pobranie współrzędnych wycinka sceny
        try:
            from config import CROP_COORDS
            crop_coords = CROP_COORDS[env_name]
        except:
            raise ValueError("Brak CROP_COORDS")

        ref_folder = years["2016"]
        save_dir = os.path.join("output_scenes", algorithm, env_name)

        # przetwarzanie roku bazowego (2016)
        ndvi_2016, ndwi_2016, _, _, cluster_2016, sem_2016, stats = process_year(
            years["2016"],
            ref_folder,
            crop_coords,
            env_name,
            save_dir=save_dir,
            norm_stats=None,
            algorithm=algorithm
        )

        # kolejne lata do porównania
        for year in ["2021", "2026"]:

            ndvi_t, ndwi_t, _, _, cluster_t, sem_t, _ = process_year(
                years[year],
                ref_folder,
                crop_coords,
                env_name,
                save_dir=save_dir,
                norm_stats=stats,
                algorithm=algorithm
            )

            # wykrywanie zmian względem 2016
            change_map, delta = detect_changes(
                sem_2016,
                sem_t,
                ndvi_2016,
                ndvi_t,
                ndwi_2016,
                ndwi_t
            )

            # statystyki powierzchni zmian
            area_stats = change_area_stats(change_map)

            print(f"{env_name} {year}:")
            print(f"Zmiany [%]: {area_stats['changed_percent']}")
            print(f"Powierzchnia zmian [km2]: {area_stats['changed_area_km2']}")

            # klasyfikacja typów zmian
            change_types_map = classify_change(sem_2016, sem_t)

            unique, counts = np.unique(change_types_map, return_counts=True)

            print(f"\n{env_name} {year} - typy zmian:")
            for u, c in zip(unique, counts):
                print(f"typ {u}: {c} pikseli")

            stats_change = change_statistics(change_map)

            # zapis mapy typów zmian
            save_change_types(env_name, "2016", year, algorithm, change_types_map)

            print(f"{env_name} {year} zmiany [%]: {stats_change['change_percent']}")

            # wizualizacja zmian
            plot_index(
                change_map,
                f"{algorithm} | {env_name} | {year} CHANGE MAP",
                os.path.join(save_dir, f"{year}_07_change_map.png")
            )

            plot_index(
                change_types_map,
                f"{env_name} {year} CHANGE TYPES",
                os.path.join(save_dir, f"{year}_08_change_types.png")
            )