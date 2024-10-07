import copy
import os
import random
import time
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import tracemalloc
import numpy as np
import gc

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
    plt.tight_layout()

    if self.show_plots:
        plt.show() 
    else:
        save_dir = os.path.join("benchmarks", self.savepath)
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, filename))
    plt.close()
