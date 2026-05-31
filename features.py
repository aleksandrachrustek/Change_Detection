import numpy as np

def create_feature_vector(ndvi, ndwi):
    """
    Tworzy wektor cech dla każdego piksela na podstawie NDVI, NDWI i położenia.
    """

    # maska pikseli z poprawnymi wartościami (bez NaN)
    mask = ~np.isnan(ndvi) & ~np.isnan(ndwi)
    # współrzędne pikseli spełniających maskę
    y, x = np.where(mask)
    # wektor cech:
    # - NDVI (roślinność)
    # - NDWI (woda)
    # - x, y (znormalizowane współrzędne – dodają kontekst przestrzenny)
    features = np.stack([
        ndvi[mask],
        ndwi[mask],
        (x / ndvi.shape[1]) * 0.6,
        (y / ndvi.shape[0]) * 0.6
    ], axis=1)

    return features, mask