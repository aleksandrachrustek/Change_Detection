import numpy as np
from scipy.ndimage import median_filter

def mask_invalid(image):
    """
    Zamienia niepoprawne wartości (<= 0) na NaN.
    """
    image = image.astype(np.float32)
    # piksele bez istotnej informacji traktujemy jako brak danych
    image[image <= 0] = np.nan

    return image

def mask_clouds(red, green, nir):
    """
    Wykrywa i maskuje chmury na podstawie jasności i NDVI.
    """

    # średnia jasność piksela
    brightness = (red + green + nir) / 3
    # próg – bierzemy najjaśniejsze ~2% pikseli
    threshold = np.nanpercentile(brightness, 98)
    # NDVI pomaga odróżnić chmury od roślinności
    ndvi = (nir - red) / (nir + red + 1e-6)
    # chmury: bardzo jasne + niskie NDVI
    cloud_mask = (brightness > threshold) & (ndvi < 0.3)
    # ustawiamy chmury jako brak danych
    red[cloud_mask] = np.nan
    green[cloud_mask] = np.nan
    nir[cloud_mask] = np.nan

    return red, green, nir


def denoise(image):
    """
    Redukuje szum filtrem medianowym (3x3).
    """

    # zapamiętujemy gdzie były NaN
    nan_mask = np.isnan(image)
    # zamieniamy NaN na 0, żeby filtr działał poprawnie
    filled = np.nan_to_num(image, nan=0.0)
    # filtr medianowy wygładza lokalne zakłócenia
    filtered = median_filter(filled, size=3)
    # przywracamy NaN
    filtered[nan_mask] = np.nan

    return filtered


def compute_global_stats(image):
    """
    Wyznacza percentyle (2 i 98) używane do normalizacji.
    """

    # tylko poprawne wartości
    valid = image[~np.isnan(image)]

    return np.percentile(valid, 2), np.percentile(valid, 98)


def normalize_with_stats(image, low, high):
    """
    Normalizuje obraz do zakresu 0–1 na podstawie podanych percentyli.
    """

    # zabezpieczenie przed dzieleniem przez zero
    if high - low < 1e-6:
        return np.zeros_like(image)
    image = (image - low) / (high - low)
    # obcinamy wartości spoza zakresu
    return np.clip(image, 0, 1)