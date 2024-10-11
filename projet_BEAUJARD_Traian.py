############################################## algos.py
import copy
import random
import time
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from sys import setrecursionlimit
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
            swapped = False
            for j in range(0, n-i-1):
                operations += 1  # La comparaison suivante est une opération.
                if data[j] > data[j+1]:
                    data[j], data[j+1] = data[j+1], data[j]
                    operations += 3  # Interversion, 3 opérations avec variable intermédiaire.
                    swapped = True
            if not swapped:  # Si aucun échange n'a été fait, la liste est déjà triée.
                break
        return data, operations

class InsertionSort(SortAlgorithm):
    def __init__(self):
        super().__init__('Insertion Sort')

    def sort(self, data):
        operations = 0
        for i in range(1, len(data)):
            key = data[i]
            j = i - 1
            operations += 1  # Initialisation de 'j'
            while j >= 0 and key < data[j]:
                operations += 2  # 2 Comparaisons
                data[j + 1] = data[j]
                operations += 1  # Déplacement de data
                j -= 1
                operations += 1  # Décrémentation de 'j'
            data[j + 1] = key
            operations += 1  # Insertion de 'key'
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

    def heapify(self, data, n, i):
        largest = i
        left = 2 * i + 1
        right = 2 * i + 2 
        operations = 0
        
        if left < n and data[left] > data[largest]:
            largest = left
            operations += 1  # comparaison

        if right < n and data[right] > data[largest]:
            largest = right
            operations += 1 # comparaison 

        if largest != i:
            data[i], data[largest] = data[largest], data[i]
            operations += 3   # 3 opérations pour l'échange

            ops_from_recursive_call = self.heapify(data, n, largest)
            operations += ops_from_recursive_call

        return operations

    def sort(self, data):
        n = len(data)
        operations = 0

        for i in range(n // 2 - 1, -1, -1):
            operations += self.heapify(data, n, i)

        for i in range(n - 1, 0, -1):
            data[i], data[0] = data[0], data[i]
            operations += 3  # 3 opérations pour l'échange
            operations += self.heapify(data, i, 0)

        return data, operations
    

class QuickSort(SortAlgorithm):
    def __init__(self):
        super().__init__('Quick Sort')

    def swap(self, arr, i, j):
        arr[i], arr[j] = arr[j], arr[i]

    def partition(self, arr, low, high):
        operations = 0

        mid = (low + high) // 2
        pivot = arr[mid]
        self.swap(arr, mid, high)
        operations += 3  # échange du pivot avec la fin
        
        i = low - 1
        
        for j in range(low, high):
            operations += 1  # Comparaison
            if arr[j] < pivot:
                i += 1
                self.swap(arr, i, j)
                operations += 3  # Échange = 3 opérations
        
        self.swap(arr, i + 1, high)
        operations += 3  # Échange du pivot
        
        return i + 1, operations

    def quicksort(self, arr, low, high):
        operations = 0
        if low < high:
            pi, partition_operations = self.partition(arr, low, high)
            operations += partition_operations

            operations += self.quicksort(arr, low, pi - 1)
            operations += self.quicksort(arr, pi + 1, high)

        return operations

    def sort(self, data):
        setrecursionlimit(50000)
        operations = self.quicksort(data, 0, len(data) - 1)
        return data, operations


class MergeSort(SortAlgorithm):
    def __init__(self):
        super().__init__('Merge Sort')

    def merge(self, arr, l, m, r):
        operations = 0

        n1 = m - l + 1
        n2 = r - m
        L = [0] * n1
        R = [0] * n2
        operations += 4

        for i in range(0, n1):
            L[i] = arr[l + i]
            operations += 1  # copier dans L[]

        for j in range(0, n2):
            R[j] = arr[m + 1 + j]
            operations += 1  # copier dans R[]

        i = 0
        j = 0 
        k = l
        
        while i < n1 and j < n2:
            operations += 2  # Comparaisons
            if L[i] <= R[j]:
                arr[k] = L[i]
                i += 1
            else:
                arr[k] = R[j]
                j += 1
            k += 1
            operations += 2

        while i < n1:
            arr[k] = L[i]
            i += 1
            k += 1
            operations += 3

        while j < n2:
            arr[k] = R[j]
            j += 1
            k += 1
            operations += 3

        return operations

    def merge_sort(self, arr, l, r):
        operations = 0
        if l < r:
            m = l + (r - l) // 2

            operations += self.merge_sort(arr, l, m)
            operations += self.merge_sort(arr, m + 1, r)

            operations += self.merge(arr, l, m, r)

        return operations

    def sort(self, data):
        operations = self.merge_sort(data, 0, len(data) - 1)
        return data, operations


############################################## benchmark.py
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

    def remove_outliers_and_interpolate(self, data, threshold=2.0):
        smoothed_data = np.copy(data)
        
        for i in range(data.shape[1]):
            col = data[:, i]
            median = np.median(col)
            deviation = np.abs(col - median)
            mad = np.median(deviation)
            outliers = deviation > threshold * mad
            
            smoothed_data[outliers, i] = np.interp(np.flatnonzero(outliers), np.flatnonzero(~outliers), col[~outliers])
        
        return smoothed_data

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

    def smooth_memory_data(self, memory_data):
        smoothed_data = np.copy(memory_data)

        if memory_data[0] > memory_data[1]:
            smoothed_data[0] = (memory_data[1] + memory_data[2]) / 2

        if memory_data[1] > memory_data[0] and memory_data[1] > memory_data[2]:
            smoothed_data[1] = (memory_data[0] + memory_data[2]) / 2

        for i in range(1, len(memory_data) - 1):
            if memory_data[i] > memory_data[i-1] and memory_data[i] > memory_data[i+1]:
                #print("-1: ", memory_data[i-1], " i: ", memory_data[i], " +1: ", memory_data[i+1], "\n")
                smoothed_data[i] = (memory_data[i-1] + memory_data[i+1]) / 2
                #print("NEW : ", "-1: ", smoothed_data[i-1], " i: ", smoothed_data[i], " +1: ", smoothed_data[i+1], "\n")


        return smoothed_data


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
                if avg_times[algo_name][list_type]:
                    times_data = np.array(avg_times[algo_name][list_type])
                    avg_times[algo_name][list_type] = np.mean(times_data, axis=0)
                    avg_times[algo_name][list_type] = self.smooth_memory_data(avg_times[algo_name][list_type])

                if avg_operations[algo_name][list_type]:
                    operations_data = np.array(avg_operations[algo_name][list_type])
                    avg_operations[algo_name][list_type] = np.mean(operations_data, axis=0)
                    avg_operations[algo_name][list_type] = self.smooth_memory_data(avg_operations[algo_name][list_type])

                if avg_memory[algo_name][list_type]:
                    if avg_memory[algo_name][list_type]:
                        memory_data = np.array(avg_memory[algo_name][list_type])
                        avg_temp = np.min(memory_data, axis=0)
                        avg_memory[algo_name][list_type] = self.smooth_memory_data(avg_temp)
                    # memory_array = np.array(avg_memory[algo_name][list_type])
                    # memory_array_smoothed = self.remove_outliers_and_interpolate(memory_array, threshold=2.0)
                    # avg_memory[algo_name][list_type] = np.mean(memory_array_smoothed, axis=0)


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
                if len(avg_results[algo_name][list_type]) > 0:
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
            plt.show()
        else:
            save_dir = os.path.join("benchmarks", self.savepath)
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, filename))
        plt.close()




############################################## menu.py
from colorama import init, Fore, Style

init(autoreset=True)

class Menu:
    def __init__(self):
        self.algorithmes_selectionnes = []
        self.nlist = 20
        self.nval = 100
        self.types_selectionnes = ['random', 'sorted', 'inverted']
        self.n_runs = 1
        self.benchmark_img_path = 'mon_benchmark'

    def afficher_menu_titre(self, titre):
        print(Fore.CYAN + Style.BRIGHT + "\n" + "="*50)
        print(Fore.CYAN + Style.BRIGHT + f"{titre.center(50)}")
        print(Fore.CYAN + Style.BRIGHT + "="*50 + "\n")

    def choisir_algorithmes(self):
        self.afficher_menu_titre("Sélection des Algorithmes")

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

        self.algorithmes_selectionnes = [algo for code, name, algo in algos_disponibles if code in choix_split]

        if not self.algorithmes_selectionnes:
            print(Fore.RED + "\nAucun algorithme sélectionné. Sélection par défaut : Bubble Sort.")
            self.algorithmes_selectionnes.append(BubbleSort())

    def choisir_taille_tableaux(self):
        self.afficher_menu_titre("Paramètres des Tableaux")

        nlist = input(Fore.YELLOW + "Nombre de tableaux à générer (défaut 20) : ")
        nval = input(Fore.YELLOW + "Taille initiale des tableaux (défaut 100) : ")

        try:
            self.nlist = int(nlist)
        except ValueError:
            self.nlist = 20

        try:
            self.nval = int(nval)
        except ValueError:
            self.nval = 100

        print(Fore.GREEN + f"\nVous avez sélectionné {self.nlist} tableaux avec une taille initiale de {self.nval}.\n")

    def choisir_types_tri(self):
        self.afficher_menu_titre("Types de Tableaux à Trier")

        print("1: Tableaux aléatoires (random)")
        print("2: Tableaux triés (sorted)")
        print("3: Tableaux triés à l'envers (inverted)")

        choix = input(Fore.YELLOW + "\nEntrez les numéros des types à trier, séparés par des virgules (ex: 1,3) (défaut: 1,2,3) : ")
        choix_split = choix.split(',')

        types_possibles = {'1': 'random', '2': 'sorted', '3': 'inverted'}
        self.types_selectionnes = [types_possibles[num] for num in choix_split if num in types_possibles]

        if not self.types_selectionnes:
            self.types_selectionnes = ['random', 'sorted', 'inverted']

        print(Fore.GREEN + f"\nTypes de tableaux sélectionnés : {', '.join(self.types_selectionnes)}.\n")

    def choisir_nb_repetitions(self):
        self.afficher_menu_titre("Nombre de Répétitions")

        n_runs = input(Fore.YELLOW + "Combien de répétitions pour chaque benchmark (défaut 1) : ")

        try:
            self.n_runs = int(n_runs)
        except ValueError:
            self.n_runs = 1

        print(Fore.GREEN + f"\nLe nombre de répétitions sera : {self.n_runs}.\n")

    def choisir_nom_fichier(self):
        self.benchmark_img_path = input(Fore.YELLOW + "\nNom du dossier pour enregistrer les graphiques (défaut : mon_benchmark) : ")
        if not self.benchmark_img_path:
            self.benchmark_img_path = 'mon_benchmark'

        print(Fore.GREEN + f"\nLe dossier sera sauvegardé sous : {self.benchmark_img_path}\n")

    def fin_et_rejouer(self):
        print(Fore.CYAN + Style.BRIGHT + "\n" + "="*50)
        print(Fore.CYAN + Style.BRIGHT + "TEST TERMINÉ".center(50))
        print(Fore.CYAN + Style.BRIGHT + "="*50 + "\n")

        choix = input(Fore.YELLOW + "Voulez-vous effectuer un autre test ? (O/N) : ").strip().lower()
        if choix == 'o':
            return True
        else:
            print(Fore.GREEN + "\nMerci d'avoir utilisé le programme. À bientôt !\n")
            return False

    def lancer(self):
        continuer = True
        while continuer:
            self.choisir_algorithmes()
            self.choisir_taille_tableaux()
            self.choisir_types_tri()
            self.choisir_nb_repetitions()
            self.choisir_nom_fichier()

            benchmark = SortBenchmark(self.algorithmes_selectionnes, nlist=self.nlist, nval=self.nval, 
                                      types_to_sort=self.types_selectionnes, n_runs=self.n_runs, 
                                      savepath=self.benchmark_img_path)
            benchmark.run()

            continuer = self.fin_et_rejouer()

if __name__ == "__main__":
    menu = Menu()
    menu.lancer()
