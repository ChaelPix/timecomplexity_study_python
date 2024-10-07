import copy
import random
import time
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import heapq

class SortAlgorithm(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def sort(self, data):
        pass

    def benchmark(self, data):
        start_time = time.time()
        sorted_data, operations = self.sort(copy.deepcopy(data))
        end_time = time.time()
        return end_time - start_time, operations

class BubbleSort(SortAlgorithm):
    def __init__(self):
        super().__init__('Bubble Sort')

    def sort(self, data):
        n = len(data)
        operations = 0
        for i in range(n):
            for j in range(0, n-i-1):
                operations += 1
                if data[j] > data[j+1]:
                    data[j], data[j+1] = data[j+1], data[j]
                    operations += 3
        return data, operations

class InsertionSort(SortAlgorithm):
    def __init__(self):
        super().__init__('Insertion Sort')

    def sort(self, data):
        operations = 0
        for i in range(1, len(data)):
            key = data[i]
            j = i-1
            operations += 1
            while j >= 0 and key < data[j]:
                data[j+1] = data[j]
                operations += 2
                j -= 1
            data[j+1] = key
            operations += 1
        return data, operations

class SelectionSort(SortAlgorithm):
    def __init__(self):
        super().__init__('Selection Sort')

    def sort(self, data):
        n = len(data)
        operations = 0
        for i in range(n):
            min_idx = i
            for j in range(i+1, n):
                operations += 1
                if data[j] < data[min_idx]:
                    min_idx = j
            data[i], data[min_idx] = data[min_idx], data[i]
            operations += 3
        return data, operations

class HeapSort(SortAlgorithm):
    def __init__(self):
        super().__init__('Heap Sort')

    def sort(self, data):
        operations = 0
        heapq.heapify(data)
        sorted_data = []
        while data:
            sorted_data.append(heapq.heappop(data))
            operations += 1
        return sorted_data, operations

class QuickSort(SortAlgorithm):
    def __init__(self):
        super().__init__('Quick Sort')

    def sort(self, data):
        operations = 0
        if len(data) <= 1:
            return data, operations
        pivot = data[len(data) // 2]
        left = [x for x in data if x < pivot]
        middle = [x for x in data if x == pivot]
        right = [x for x in data if x > pivot]
        operations += len(data)
        left_sorted, left_operations = self.sort(left)
        right_sorted, right_operations = self.sort(right)
        operations += left_operations + right_operations
        return left_sorted + middle + right_sorted, operations

class MergeSort(SortAlgorithm):
    def __init__(self):
        super().__init__('Merge Sort')

    def sort(self, data):
        operations = 0
        if len(data) <= 1:
            return data, operations
        
        def merge(left, right):
            sorted_list = []
            i = j = 0
            operations = 0
            while i < len(left) and j < len(right):
                operations += 1
                if left[i] < right[j]:
                    sorted_list.append(left[i])
                    i += 1
                else:
                    sorted_list.append(right[j])
                    j += 1
                operations += 1
            sorted_list.extend(left[i:])
            sorted_list.extend(right[j:])
            operations += len(left[i:]) + len(right[j:])
            return sorted_list, operations
        
        mid = len(data) // 2
        left_sorted, left_operations = self.sort(data[:mid])
        right_sorted, right_operations = self.sort(data[mid:])
        merged_data, merge_operations = merge(left_sorted, right_sorted)
        total_operations = left_operations + right_operations + merge_operations
        return merged_data, total_operations


import copy
import os
import random
import time
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import tracemalloc
import numpy as np
import gc
import matplotlib.ticker as mticker

class SortBenchmark:
    def __init__(self, algorithms, nlist=15, nval=200, types_to_sort=None, n_runs=1, savepath='benchmark_results', show_plots=False):
        if types_to_sort is None:
            types_to_sort = ['random', 'sorted', 'inverted']
        
        self.algorithms = algorithms
        self.nlist = nlist
        self.nval = nval
        self.types_to_sort = types_to_sort
        self.n_runs = n_runs
        self.savepath = savepath
        self.show_plots = show_plots
        self.axis, self.listDataRandom, self.listDataSorted, self.listDataInverted = self.create_data()

    def create_data(self):
        listDataRandom = []
        listDataSorted = []
        listDataInvertedSorted = []
        sizeArrays = []

        for i in range(1, self.nlist + 1):
            s = self.nval * i
            dataRandom = s * [0]
            dataSorted = s * [0]
            dataInverted = s * [0]

            for j in range(s):
                dataRandom[j] = j
                dataSorted[j] = j
                dataInverted[j] = j

            dataInverted.reverse()
            random.shuffle(dataRandom)

            listDataRandom.append(dataRandom)
            listDataSorted.append(dataSorted)
            listDataInvertedSorted.append(dataInverted)
            sizeArrays.append(s)

        return sizeArrays, listDataRandom, listDataSorted, listDataInvertedSorted

    def execute_sort(self, algo):
        """
        Exécute l'algorithme sur plusieurs tailles de tableaux et calcule la moyenne sur n_runs.
        Retourne les données des temps, opérations, et mémoire utilisée.
        """
        toplotRandom = []
        toplotSorted = []
        toplotInverted = []

        operationsRandom = []
        operationsSorted = []
        operationsInverted = []

        memoryRandom = []
        memorySorted = []
        memoryInverted = []

        for i in range(len(self.axis)):
            dataTestRandom = copy.deepcopy(self.listDataRandom[i])
            dataTestSorted = copy.deepcopy(self.listDataSorted[i])
            dataTestInverted = copy.deepcopy(self.listDataInverted[i])

            if 'random' in self.types_to_sort:
                tracemalloc.start()
                time1 = time.time()
                sorted_data, operations = algo.sort(dataTestRandom)
                time2 = time.time()
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                toplotRandom.append(time2 - time1)
                operationsRandom.append(operations)
                memoryRandom.append(peak)

            if 'sorted' in self.types_to_sort:
                tracemalloc.start()
                time3 = time.time()
                sorted_data, operations = algo.sort(dataTestSorted)
                time4 = time.time()
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                toplotSorted.append(time4 - time3)
                operationsSorted.append(operations)
                memorySorted.append(peak)

            if 'inverted' in self.types_to_sort:
                tracemalloc.start()
                time5 = time.time()
                sorted_data, operations = algo.sort(dataTestInverted)
                time6 = time.time()
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                toplotInverted.append(time6 - time5)
                operationsInverted.append(operations)
                memoryInverted.append(peak)

        return (toplotRandom, toplotSorted, toplotInverted,
                operationsRandom, operationsSorted, operationsInverted,
                memoryRandom, memorySorted, memoryInverted)

    def run(self):
        avg_times = {algo.name: {'random': [], 'sorted': [], 'inverted': []} for algo in self.algorithms}
        avg_operations = {algo.name: {'random': [], 'sorted': [], 'inverted': []} for algo in self.algorithms}
        avg_memory = {algo.name: {'random': [], 'sorted': [], 'inverted': []} for algo in self.algorithms}

        for benchmark_num in range(1, self.n_runs + 1):
            print(f"Benchmark {benchmark_num} / {self.n_runs}")

            for algo in self.algorithms:
                print(f"Exécution de {algo.name}")
                (toplotRandom, toplotSorted, toplotInverted,
                 operationsRandom, operationsSorted, operationsInverted,
                 memoryRandom, memorySorted, memoryInverted) = self.execute_sort(algo)

                if 'random' in self.types_to_sort:
                    avg_times[algo.name]['random'].append(toplotRandom)
                    avg_operations[algo.name]['random'].append(operationsRandom)
                    avg_memory[algo.name]['random'].append(memoryRandom)

                if 'sorted' in self.types_to_sort:
                    avg_times[algo.name]['sorted'].append(toplotSorted)
                    avg_operations[algo.name]['sorted'].append(operationsSorted)
                    avg_memory[algo.name]['sorted'].append(memorySorted)

                if 'inverted' in self.types_to_sort:
                    avg_times[algo.name]['inverted'].append(toplotInverted)
                    avg_operations[algo.name]['inverted'].append(operationsInverted)
                    avg_memory[algo.name]['inverted'].append(memoryInverted)

        for algo_name in avg_times.keys():
            for list_type in avg_times[algo_name].keys():
                avg_times[algo_name][list_type] = np.mean(avg_times[algo_name][list_type], axis=0)
                avg_operations[algo_name][list_type] = np.mean(avg_operations[algo_name][list_type], axis=0)
                avg_memory[algo_name][list_type] = np.mean(avg_memory[algo_name][list_type], axis=0)

        self.plot_results(avg_times, "moyenne_comparaison_time.png", 'Temps d\'exécution (s)')
        self.plot_results(avg_operations, "moyenne_comparaison_complexity.png", 'Nombre d\'opérations')
        self.plot_results(avg_memory, "moyenne_comparaison_memory.png", 'Mémoire utilisée (octets)')



    def plot_results(self, avg_results, filename, ylabel):
        """
        Générer et sauvegarder les graphiques des moyennes de tous les benchmarks.
        """
        plt.figure(figsize=(10, 6))

        for algo_name in avg_results.keys():
            for list_type in avg_results[algo_name].keys():
                label = f'{algo_name} ({list_type.capitalize()})' 
                plt.plot(self.axis, avg_results[algo_name][list_type], label=label)

        plt.xlabel('Taille des tableaux')
        plt.ylabel(ylabel)
        plt.title(f'Moyenne des algorithmes - {ylabel}')
        plt.legend(loc='best')

        ax = plt.gca() 
        formatter = mticker.ScalarFormatter(useMathText=False)
        formatter.set_scientific(False)
        ax.yaxis.set_major_formatter(formatter)

        plt.tight_layout()

        if self.show_plots:
            plt.show()  # Affiche dans le notebook
        else:
            save_dir = os.path.join("benchmarks", self.savepath)
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, filename))
        plt.close()







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
