from benchmark import *
from algos import *

if __name__ == "__main__":

    algorithms = [
        BubbleSort(),
        InsertionSort(),
        SelectionSort(),
        HeapSort(),
        QuickSort(),
        MergeSort()
    ]

    nlist = 20  # nb tableaux
    nval = 100  # taille initiale
    types_to_sort = ['random', 'inverted', 'sorted'] # ['random', 'sorted', 'inverted']
    benchmark_img_path = 'mon_graphique3.png'

    benchmark = SortBenchmark(algorithms, nlist=nlist, nval=nval, types_to_sort=types_to_sort)
    benchmark.run(save_path=benchmark_img_path)
