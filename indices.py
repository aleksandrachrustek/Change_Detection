# indeksy spektralne

def compute_ndvi(nir, red):
    """
    Oblicza NDVI – wskaźnik roślinności.
    Wyższe wartości oznaczają większą ilość zielonej roślinności.
    """
    return (nir - red) / (nir + red + 1e-6)

def compute_ndwi(green, nir):
    """
    Oblicza NDWI – wskaźnik obecności wody.
    Wyższe wartości wskazują na obszary wodne.
    """
    return (green - nir) / (green + nir + 1e-6)