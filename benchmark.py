import copy
import random
import time
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class SortBenchmark:
    def __init__(self, algorithms, nlist=15, nval=200, types_to_sort=None):
        """
        :param algorithms: Liste des algorithmes à tester
        :param nlist: Nombre de tableaux à générer
        :param nval: Taille initiale des tableaux (taille croissante)
        :param types_to_sort: Liste des types de tableaux à trier ['random', 'sorted', 'inverted']
        """
        if types_to_sort is None:
            types_to_sort = ['random', 'sorted', 'inverted']

        self.algorithms = algorithms
        self.nlist = nlist
        self.nval = nval
        self.types_to_sort = types_to_sort
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

    def execute_sort(self, algo, surplace=True):
        toplotRandom = []
        toplotSorted = []
        toplotInverted = []

        dataTestRandom = copy.deepcopy(self.listDataRandom)
        dataTestSorted = copy.deepcopy(self.listDataSorted)
        dataTestInverted = copy.deepcopy(self.listDataInverted)

        for i in range(len(self.axis)):
            if 'random' in self.types_to_sort:
                time1 = time.time()
                if surplace:
                    algo.sort(dataTestRandom[i])
                else:
                    dataTestRandom[i] = algo.sort(dataTestRandom[i])
                time2 = time.time()
                toplotRandom.append(time2 - time1)

            if 'sorted' in self.types_to_sort:
                time3 = time.time()
                if surplace:
                    algo.sort(dataTestSorted[i])
                else:
                    dataTestSorted[i] = algo.sort(dataTestSorted[i])
                time4 = time.time()
                toplotSorted.append(time4 - time3)

            if 'inverted' in self.types_to_sort:
                time5 = time.time()
                if surplace:
                    algo.sort(dataTestInverted[i])
                else:
                    dataTestInverted[i] = algo.sort(dataTestInverted[i])
                time6 = time.time()
                toplotInverted.append(time6 - time5)

        if 'random' in self.types_to_sort:
            plt.plot(self.axis, toplotRandom, '-o', label=f'{algo.name} (Random)')
        if 'sorted' in self.types_to_sort:
            plt.plot(self.axis, toplotSorted, '--s', label=f'{algo.name} (Sorted)')
        if 'inverted' in self.types_to_sort:
            plt.plot(self.axis, toplotInverted, ':d', label=f'{algo.name} (Inverted)')

    def run(self, save_path='benchmark_results.png'):
        for algo in self.algorithms:
            print(f"Exécution de {algo.name}")
            self.execute_sort(algo)
        
        plt.xlabel('Taille des tableaux')
        plt.ylabel('Temps d\'exécution (s)')
        plt.title('Comparaison des algorithmes de tri')
        plt.legend()

        plt.savefig(save_path)
        print(f"Graphique sauvegardé sous {save_path}")
        plt.clf()