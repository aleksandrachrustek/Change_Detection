import numpy as np

def detect_changes(sem_t1, sem_t2, ndvi_t1, ndvi_t2, ndwi_t1=None, ndwi_t2=None):
    """
    Wykrywa zmiany między dwoma momentami w czasie.
    Uwzględnia zmianę klasy oraz różnice w NDVI (i opcjonalnie NDWI).
    """

    # mapa zmian (0 – brak zmiany, 1 – zmiana)
    change_map = np.zeros_like(sem_t1)

    # zmiana klasy (np. roślinność → woda)
    class_change = sem_t1 != sem_t2

    # różnica NDVI między latami
    delta_ndvi = ndvi_t2 - ndvi_t1

    # uznajemy zmianę tylko jeśli przekracza próg
    ndvi_change = np.abs(delta_ndvi) > 0.15

    # bierzemy pod uwagę NDWI (woda)
    if ndwi_t1 is not None and ndwi_t2 is not None:
        delta_ndwi = ndwi_t2 - ndwi_t1
        ndwi_change = np.abs(delta_ndwi) > 0.15
    else:
        # brak NDWI → nie wpływa na wynik
        ndwi_change = False

    # końcowa maska zmian (jeśli którykolwiek warunek spełniony)
    real_change = class_change | ndvi_change | ndwi_change

    # zapis zmian jako 1
    change_map[real_change] = 1

    return change_map, delta_ndvi


def change_statistics(change_map):
    """
    Oblicza podstawowe statystyki zmian (liczba i procent).
    """

    # maska poprawnych pikseli
    valid = ~np.isnan(change_map)

    total = np.sum(valid)
    changed = np.sum(change_map[valid] > 0)

    return {
        "changed_pixels": int(changed),
        "change_percent": round(changed / total * 100, 2)
    }


def change_area_stats(change_map, pixel_size=10):
    """
    Oblicza powierzchnię zmian.
    pixel_size = 10 m (Sentinel-2)
    """

    # tylko poprawne piksele
    valid = ~np.isnan(change_map)

    changed = np.sum(change_map[valid] == 1)
    total = np.sum(valid)

    # powierzchnia jednego piksela w m²
    pixel_area = pixel_size * pixel_size

    return {
        "changed_pixels": int(changed),
        "changed_percent": round(changed / total * 100, 2),
        "changed_area_km2": round(changed * pixel_area / 1e6, 3)
    }