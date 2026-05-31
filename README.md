# Zastosowanie metod uczenia nienadzorowanego do identyfikacji zmian środowiskowych na podstawie danych satelitarnych

## Opis projektu
Projekt realizowany w ramach pracy magisterskiej, którego celem jest identyfikacja zmian środowiskowych na podstawie wieloczasowych danych satelitarnych Sentinel-2. W analizie wykorzystano wskaźniki teledetekcyjne (NDVI, NDWI) oraz algorytmy klasteryzacji DBSCAN i HDBSCAN do wykrywania obszarów o podobnych właściwościach i potencjalnych zmian środowiskowych.

## Technologie
* Python
* NumPy
* Rasterio
* Scikit-learn
* Matplotlib
* HDBSCAN
* Pandas

## Uruchomienie
```bash
pip install -r requirements.txt
python main.py
```

## Wyniki
Program generuje:
* obrazy RGB,
* mapy wskaźników NDVI i NDWI,
* wyniki klasteryzacji DBSCAN i HDBSCAN,
* wizualizacje wykrytych zmian środowiskowych i ich typu.
