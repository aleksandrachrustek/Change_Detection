import numpy as np

def classify_change(sem_t1, sem_t2):
    """
    Klasyfikuje typ zmiany na podstawie map semantycznych z dwóch momentów.
    Zwraca mapę z kodami zmian.
    """

    # mapa wynikowa (0 = brak zmiany)
    change_type = np.zeros_like(sem_t1)

    # oznaczenia klas:
    # 1 = woda
    # 2 = roślinność
    # 3 = inne (np. gleba, zabudowa)

    # oznaczenia zmian:
    # 2 = roślinność → inne (np. wylesienie)
    change_type[(sem_t1 == 2) & (sem_t2 == 3)] = 2

    # 3 = inne → roślinność (np. regeneracja)
    change_type[(sem_t1 == 3) & (sem_t2 == 2)] = 3

    # 4 = inne → woda (np. pojawienie się zbiornika)
    change_type[(sem_t1 == 3) & (sem_t2 == 1)] = 4

    # 5 = woda → inne (np. zanik wody)
    change_type[(sem_t1 == 1) & (sem_t2 == 3)] = 5

    return change_type