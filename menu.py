from benchmark import SortBenchmark
from algos import BubbleSort, InsertionSort, SelectionSort, HeapSort, QuickSort, MergeSort
from colorama import init, Fore, Style

init(autoreset=True)

def afficher_menu_titre(titre):
    print(Fore.CYAN + Style.BRIGHT + "\n" + "="*50)
    print(Fore.CYAN + Style.BRIGHT + f"{titre.center(50)}")
    print(Fore.CYAN + Style.BRIGHT + "="*50 + "\n")

def choisir_algorithmes():
    afficher_menu_titre("Sélection des Algorithmes")

    algos_disponibles = [
        ('1', 'Bubble Sort', BubbleSort()),
        ('2', 'Insertion Sort', InsertionSort()),
        ('3', 'Selection Sort', SelectionSort()),
        ('4', 'Heap Sort', HeapSort()),
        ('5', 'Quick Sort', QuickSort()),
        ('6', 'Merge Sort', MergeSort())
    ]
    
    for code, name, _ in algos_disponibles:
        print(f"{Fore.GREEN}{code}: {name}")

    choix = input(Fore.YELLOW + "\nEntrez les numéros des algorithmes séparés par des virgules (ex: 1,2,3) : ")
    choix_split = choix.split(',')

    algorithmes_selectionnes = []
    for code, name, algo in algos_disponibles:
        if code in choix_split:
            algorithmes_selectionnes.append(algo)
    
    if not algorithmes_selectionnes:
        print(Fore.RED + "\nAucun algorithme sélectionné. Sélection par défaut : Bubble Sort.")
        algorithmes_selectionnes.append(BubbleSort())
    
    return algorithmes_selectionnes

def choisir_taille_tableaux():
    afficher_menu_titre("Paramètres des Tableaux")

    nlist = input(Fore.YELLOW + "Nombre de tableaux à générer (défaut 20) : ")
    nval = input(Fore.YELLOW + "Taille initiale des tableaux (défaut 100) : ")

    try:
        nlist = int(nlist)
    except ValueError:
        nlist = 20

    try:
        nval = int(nval)
    except ValueError:
        nval = 100

    print(Fore.GREEN + f"\nVous avez sélectionné {nlist} tableaux avec une taille initiale de {nval}.\n")
    return nlist, nval

def choisir_types_tri():
    afficher_menu_titre("Types de Tableaux à Trier")

    print("1: Tableaux aléatoires (random)")
    print("2: Tableaux triés (sorted)")
    print("3: Tableaux triés à l'envers (inverted)")

    choix = input(Fore.YELLOW + "\nEntrez les numéros des types à trier, séparés par des virgules (ex: 1,3) (défaut: 1,2,3) : ")
    choix_split = choix.split(',')

    types_possibles = {'1': 'random', '2': 'sorted', '3': 'inverted'}
    types_selectionnes = [types_possibles[num] for num in choix_split if num in types_possibles]

    if not types_selectionnes:
        types_selectionnes = ['random', 'sorted', 'inverted']

    print(Fore.GREEN + f"\nTypes de tableaux sélectionnés : {', '.join(types_selectionnes)}.\n")
    return types_selectionnes

def choisir_nb_repetitions():
    afficher_menu_titre("Nombre de Répétitions")

    n_runs = input(Fore.YELLOW + "Combien de répétitions pour chaque benchmark (défaut 1) : ")

    try:
        n_runs = int(n_runs)
    except ValueError:
        n_runs = 1

    print(Fore.GREEN + f"\nLe nombre de répétitions sera : {n_runs}.\n")
    return n_runs
    
def choisir_nom_fichier():
    nom_fichier = input(Fore.YELLOW + "\nNom du dossier pour enregistrer les graphiques (défaut : mon_benchmark) : ")
    if not nom_fichier:
        nom_fichier = 'mon_benchmark'

    print(Fore.GREEN + f"\nLe fichier sera sauvegardé sous : {nom_fichier}\n")
    return nom_fichier

def fin_et_rejouer():
    print(Fore.CYAN + Style.BRIGHT + "\n" + "="*50)
    print(Fore.CYAN + Style.BRIGHT + "TEST TERMINÉ".center(50))
    print(Fore.CYAN + Style.BRIGHT + "="*50 + "\n")

    choix = input(Fore.YELLOW + "Voulez-vous effectuer un autre test ? (O/N) : ").strip().lower()
    if choix == 'o':
        return True
    else:
        print(Fore.GREEN + "\nMerci d'avoir utilisé le programme. À bientôt !\n")
        return False

if __name__ == "__main__":
    continuer = True

    while continuer:
        algorithms = choisir_algorithmes()

        nlist, nval = choisir_taille_tableaux()

        types_to_sort = choisir_types_tri()
        n_runs = choisir_nb_repetitions()
        benchmark_img_path = choisir_nom_fichier()

        benchmark = SortBenchmark(algorithms, nlist=nlist, nval=nval, types_to_sort=types_to_sort, n_runs=n_runs, savepath=benchmark_img_path)
        benchmark.run()

        continuer = fin_et_rejouer()
