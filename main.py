from analysis import run_analysis
from results import clear_results, compare_years

# ścieżki do danych dla różnych środowisk i lat
DATASETS = {
    "lasy": {
        "2016": "dane/lasy/2016",
        "2021": "dane/lasy/2021",
        "2026": "dane/lasy/2026"
    },
    "rzeki": {
        "2016": "dane/rzeki/2016",
        "2021": "dane/rzeki/2021",
        "2026": "dane/rzeki/2026"
    },
    "wody": {
        "2016": "dane/wody/2016",
        "2021": "dane/wody/2021",
        "2026": "dane/wody/2026"
    }
}

def show_menu():
    print("\nMenu:\n")
    print("1 - Run DBSCAN")
    print("2 - Run HDBSCAN")
    print("3 - Compare results (CSV)")
    print("4 - Full pipeline")
    print("5 - Clear existing results")
    print("0 - Exit")


def main():
    while True:
        show_menu()
        choice = input("\nChoose option: ")

        # pełne uruchomienie analizy
        if choice == "1":
            run_analysis(
                DATASETS,
                "DBSCAN"
            )

        elif choice == "2":
            run_analysis(
                DATASETS,
                "HDBSCAN"
            )

        # porównanie wyników między latami
        elif choice == "3":
            compare_years()
            print("\nComparison CSV created.")

        # cały pipeline: tuning + analiza + porównanie + wykresy
        elif choice == "4":
            clear_results()
            run_analysis(DATASETS, "DBSCAN")
            run_analysis(DATASETS, "HDBSCAN")
            compare_years()
            print("\nFull pipeline completed.")

        # czyszczenie wyników
        elif choice == "5":
            clear_results()
            print("\nResults cleared.")

        # wyjście z programu
        elif choice == "0":
            print("\nBye.")
            break

        else:
            print("\nWrong option.")

if __name__ == "__main__":
    main()