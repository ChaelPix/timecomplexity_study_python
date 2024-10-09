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
