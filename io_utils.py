import rasterio
import os
import numpy as np
from rasterio.windows import Window
from rasterio.warp import reproject, Resampling

def find_band(files, band):
    """
    Zwraca nazwę pliku odpowiadającego danemu pasmu (np. B04).
    """
    matches = [f for f in files if band in f]
    if not matches:
        raise ValueError(f"Band {band} not found")

    return matches[0]

def read_window(path, i, j, size):
    """
    Wczytuje fragment obrazu (window) o zadanym rozmiarze.
    Zwraca dane, transformację oraz układ współrzędnych.
    """
    with rasterio.open(path) as src:
        # definiujemy wycinek obrazu
        window = Window(j, i, size, size)
        # wczytanie jednego kanału
        data = src.read(1, window=window).astype(np.float32)
        # transformacja przestrzenna dla wycinka
        transform = src.window_transform(window)
        # układ współrzędnych
        crs = src.crs

    return data, transform, crs

def align_window(src_data, src_transform, src_crs,
                 ref_transform, ref_crs, shape):
    """
    Dopasowuje obraz do układu referencyjnego (ta sama siatka pikseli).
    """

    # przygotowanie pustej tablicy wynikowej
    aligned = np.empty(shape, dtype=np.float32)
    # reprojekcja i resampling do układu referencyjnego
    reproject(
        source=src_data,
        destination=aligned,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=ref_transform,
        dst_crs=ref_crs,
        resampling=Resampling.bilinear
    )

    return aligned