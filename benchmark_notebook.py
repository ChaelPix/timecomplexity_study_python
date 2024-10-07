import copy
import os
import random
import time
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import tracemalloc
import numpy as np

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

    def run_benchmark(self):
        for benchmark_num in range(1, self.n_runs + 1):
            print(f"Benchmark {benchmark_num} / {self.n_runs}")

            for algo in self.algorithms:
                print(f"Exécution de {algo.name}")
                (toplotRandom, toplotSorted, toplotInverted,
                 operationsRandom, operationsSorted, operationsInverted,
                 memoryRandom, memorySorted, memoryInverted) = self.execute_sort(algo)

                self.plot_results(toplotRandom, toplotSorted, toplotInverted, f"comparaison_time_{benchmark_num}.png", 'Temps d\'exécution (s)', self.show_plots)
                self.plot_results(operationsRandom, operationsSorted, operationsInverted, f"comparaison_complexity_{benchmark_num}.png", 'Nombre d\'opérations', self.show_plots)
                self.plot_results(memoryRandom, memorySorted, memoryInverted, f"comparaison_memory_{benchmark_num}.png", 'Mémoire utilisée (octets)', self.show_plots)

    def plot_results(self, results_random, results_sorted, results_inverted, filename, ylabel, show_plot=False):
        plt.figure()

        if 'random' in self.types_to_sort:
            plt.plot(self.axis, results_random, '-o', label=f'Random')
        if 'sorted' in self.types_to_sort:
            plt.plot(self.axis, results_sorted, '--s', label=f'Sorted')
        if 'inverted' in self.types_to_sort:
            plt.plot(self.axis, results_inverted, ':d', label=f'Inverted')

        plt.xlabel('Taille des tableaux')
        plt.ylabel(ylabel)
        plt.title('Comparaison des algorithmes')
        plt.legend()

        if show_plot:
            plt.show()
        else:
            save_dir = os.path.join("benchmarks", self.savepath)
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, filename))
        plt.close()
