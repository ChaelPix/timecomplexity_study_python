import copy
import os
import random
import time
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import matplotlib.ticker as mticker
import tracemalloc
import numpy as np 

class SortBenchmark:
    def __init__(self, algorithms, nlist=15, nval=200, types_to_sort=None, n_runs=1, savepath='benchmark_results'):
        if types_to_sort is None:
            types_to_sort = ['random', 'sorted', 'inverted']
        
        self.algorithms = algorithms
        self.nlist = nlist
        self.nval = nval
        self.types_to_sort = types_to_sort
        self.n_runs = n_runs
        self.savepath = savepath
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
        all_times_random = []
        all_times_sorted = []
        all_times_inverted = []

        all_ops_random = []
        all_ops_sorted = []
        all_ops_inverted = []

        all_mem_random = []
        all_mem_sorted = []
        all_mem_inverted = []

        for benchmark_num in range(1, self.n_runs + 1):
            benchmark_times_random = {}
            benchmark_times_sorted = {}
            benchmark_times_inverted = {}

            benchmark_ops_random = {}
            benchmark_ops_sorted = {}
            benchmark_ops_inverted = {}

            benchmark_mem_random = {}
            benchmark_mem_sorted = {}
            benchmark_mem_inverted = {}

            print(f"Benchmark {benchmark_num} / {self.n_runs}")

            for algo in self.algorithms:
                print(f"Exécution de {algo.name}")
                (toplotRandom, toplotSorted, toplotInverted,
                 operationsRandom, operationsSorted, operationsInverted,
                 memoryRandom, memorySorted, memoryInverted) = self.execute_sort(algo)

                if 'random' in self.types_to_sort:
                    benchmark_times_random[algo.name] = toplotRandom
                    benchmark_ops_random[algo.name] = operationsRandom
                    benchmark_mem_random[algo.name] = memoryRandom
                if 'sorted' in self.types_to_sort:
                    benchmark_times_sorted[algo.name] = toplotSorted
                    benchmark_ops_sorted[algo.name] = operationsSorted
                    benchmark_mem_sorted[algo.name] = memorySorted
                if 'inverted' in self.types_to_sort:
                    benchmark_times_inverted[algo.name] = toplotInverted
                    benchmark_ops_inverted[algo.name] = operationsInverted
                    benchmark_mem_inverted[algo.name] = memoryInverted

            save_dir_time = os.path.join("benchmarks", self.savepath, "time_graph")
            save_dir_complexity = os.path.join("benchmarks", self.savepath, "complexity_graph")
            save_dir_memory = os.path.join("benchmarks", self.savepath, "memory_graph")
            os.makedirs(save_dir_time, exist_ok=True)
            os.makedirs(save_dir_complexity, exist_ok=True)
            os.makedirs(save_dir_memory, exist_ok=True)

            self.plot_results(benchmark_times_random, benchmark_times_sorted, benchmark_times_inverted, f"comparaison_time_{benchmark_num}.png", save_dir_time, 'Temps d\'exécution (s)')
            self.plot_results(benchmark_ops_random, benchmark_ops_sorted, benchmark_ops_inverted, f"comparaison_complexity_{benchmark_num}.png", save_dir_complexity, 'Nombre d\'opérations')
            self.plot_results(benchmark_mem_random, benchmark_mem_sorted, benchmark_mem_inverted, f"comparaison_memory_{benchmark_num}.png", save_dir_memory, 'Mémoire utilisée (octets)')

            all_times_random.append(benchmark_times_random)
            all_times_sorted.append(benchmark_times_sorted)
            all_times_inverted.append(benchmark_times_inverted)

            all_ops_random.append(benchmark_ops_random)
            all_ops_sorted.append(benchmark_ops_sorted)
            all_ops_inverted.append(benchmark_ops_inverted)

            all_mem_random.append(benchmark_mem_random)
            all_mem_sorted.append(benchmark_mem_sorted)
            all_mem_inverted.append(benchmark_mem_inverted)

        self.plot_mean_results(all_times_random, all_times_sorted, all_times_inverted, "moyenne_time.png", save_dir_time, 'Temps d\'exécution (s)')
        self.plot_mean_results(all_ops_random, all_ops_sorted, all_ops_inverted, "moyenne_complexity.png", save_dir_complexity, 'Nombre d\'opérations')
        self.plot_mean_results(all_mem_random, all_mem_sorted, all_mem_inverted, "moyenne_memory.png", save_dir_memory, 'Mémoire utilisée (octets)')

    def plot_results(self, times_random, times_sorted, times_inverted, filename, save_dir, ylabel):
        """
        Générer et sauvegarder les graphiques des résultats d'un benchmark pour un type (temps, opérations, mémoire).
        """
        plt.figure()

        if 'random' in self.types_to_sort:
            for algo_name, times in times_random.items():
                plt.plot(self.axis, times, '-o', label=f'{algo_name} (Random)')
        if 'sorted' in self.types_to_sort:
            for algo_name, times in times_sorted.items():
                plt.plot(self.axis, times, '--s', label=f'{algo_name} (Sorted)')
        if 'inverted' in self.types_to_sort:
            for algo_name, times in times_inverted.items():
                plt.plot(self.axis, times, ':d', label=f'{algo_name} (Inverted)')
        plt.xlabel('Taille des tableaux')
        plt.ylabel(ylabel)
        plt.title('Comparaison des algorithmes')
        plt.legend()
        plt.savefig(os.path.join(save_dir, filename))
        plt.close()

    def plot_mean_results(self, all_times_random, all_times_sorted, all_times_inverted, filename, save_dir, ylabel):
        """
        Générer et sauvegarder les graphiques des moyennes de tous les benchmarks.
        """
        plt.figure()

        if 'random' in self.types_to_sort:
            for algo_name in all_times_random[0].keys():
                mean_times = np.mean([bench[algo_name] for bench in all_times_random], axis=0)
                plt.plot(self.axis, mean_times, '-o', label=f'{algo_name} (Random)')
        if 'sorted' in self.types_to_sort:
            for algo_name in all_times_sorted[0].keys():
                mean_times = np.mean([bench[algo_name] for bench in all_times_sorted], axis=0)
                plt.plot(self.axis, mean_times, '--s', label=f'{algo_name} (Sorted)')
        if 'inverted' in self.types_to_sort:
            for algo_name in all_times_inverted[0].keys():
                mean_times = np.mean([bench[algo_name] for bench in all_times_inverted], axis=0)
                plt.plot(self.axis, mean_times, ':d', label=f'{algo_name} (Inverted)')
        plt.xlabel('Taille des tableaux')
        plt.ylabel(ylabel)
        plt.title('Moyenne des algorithmes')
        plt.legend()
        plt.savefig(os.path.join(save_dir, filename))
        plt.close()
