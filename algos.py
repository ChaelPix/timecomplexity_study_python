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
        sorted_data = self.sort(copy.deepcopy(data))
        end_time = time.time()
        return end_time - start_time

class BubbleSort(SortAlgorithm):
    def __init__(self):
        super().__init__('Bubble Sort')

    def sort(self, data):
        n = len(data)
        for i in range(n):
            for j in range(0, n-i-1):
                if data[j] > data[j+1]:
                    data[j], data[j+1] = data[j+1], data[j]
        return data

class InsertionSort(SortAlgorithm):
    def __init__(self):
        super().__init__('Insertion Sort')

    def sort(self, data):
        for i in range(1, len(data)):
            key = data[i]
            j = i-1
            while j >= 0 and key < data[j]:
                data[j+1] = data[j]
                j -= 1
            data[j+1] = key
        return data

class SelectionSort(SortAlgorithm):
    def __init__(self):
        super().__init__('Selection Sort')

    def sort(self, data):
        n = len(data)
        for i in range(n):
            min_idx = i
            for j in range(i+1, n):
                if data[j] < data[min_idx]:
                    min_idx = j
            data[i], data[min_idx] = data[min_idx], data[i]
        return data

class HeapSort(SortAlgorithm):
    def __init__(self):
        super().__init__('Heap Sort')

    def sort(self, data):
        heapq.heapify(data)
        sorted_data = [heapq.heappop(data) for _ in range(len(data))]
        return sorted_data

class QuickSort(SortAlgorithm):
    def __init__(self):
        super().__init__('Quick Sort')

    def sort(self, data):
        if len(data) <= 1:
            return data
        pivot = data[len(data) // 2]
        left = [x for x in data if x < pivot]
        middle = [x for x in data if x == pivot]
        right = [x for x in data if x > pivot]
        return self.sort(left) + middle + self.sort(right)

class MergeSort(SortAlgorithm):
    def __init__(self):
        super().__init__('Merge Sort')

    def sort(self, data):
        if len(data) <= 1:
            return data
        
        def merge(left, right):
            sorted_list = []
            i = j = 0
            while i < len(left) and j < len(right):
                if left[i] < right[j]:
                    sorted_list.append(left[i])
                    i += 1
                else:
                    sorted_list.append(right[j])
                    j += 1
            sorted_list.extend(left[i:])
            sorted_list.extend(right[j:])
            return sorted_list
        
        mid = len(data) // 2
        left = self.sort(data[:mid])
        right = self.sort(data[mid:])
        return merge(left, right)