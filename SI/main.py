from utils.yolo11n_extract import YOLOLayerAnalyzer
from src.handle_data import Data
from src.dijkstra import Dijkstra

import time

analyzer = YOLOLayerAnalyzer("yolo11n.pt")
flops_layers, output_size_layers = analyzer.analyze()

f_edge_set = {i: ((i + 1)%10+1) * 2e9 for i in range(10)}
f_server_set = {i: ((i + 1)%10+1) * 30e9 for i in range(10)}

bandwidth = 10

def measure(num):
    for i in range(len(f_edge_set)):
        cost = Data(flops_layers  , f_edge_set[i] , f_server_set[i] , output_size_layers , bandwidth).run()
        dijkstra = Dijkstra(cost , machine=['1', '2']).run()
        return dijkstra

# warm-up
for i in range(10):
    measure(10)

for num in range(0, 20 , 2):
    if num == 0  :
        num = 1
    n = 10
    lst_time = []
    for _ in range(n) :
        start = time.perf_counter_ns()
        measure(num)
        end = time.perf_counter_ns()
        lst_time.append(end - start)
    print(f'{num} : {sum(lst_time) / len(lst_time)}')

