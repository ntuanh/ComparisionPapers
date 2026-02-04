import time

from src.Compute import Compute
from src.Hungarian import Hungarian
from src.OPP import OPP
from src.PPR import PPR
from utils.yolo11n_extract import YOLOLayerAnalyzer, LayerProfile , DNNProfile

analyzer = YOLOLayerAnalyzer()
yolo11n = analyzer.analyze()

def measure(num) :
    ###############################
    compute = Compute()


    bandwidth = [1.0, 5.0, 10.0]
    curren_bandwidth = bandwidth[-1]    # use for OPP
    # Local computing resources
    # f_local_set = {
    #     0: 2e9
    # }
    #
    # # Edge computing resources
    # f_edge_set = {
    #     0: 30e9
    # }

    f_local_set = {i: ((i + 1)%10+1) * 2e9 for i in range(num)}
    f_edge_set = {i: ((i + 1) % 10+1 ) * 30e9 for i in range(num)}

    dnn_set = [yolo11n] * len(f_local_set)    # list[DNNProfile]
    RP = {}
    for i in range(len(f_local_set)):
        new_set = set()
        for j in range(len(f_edge_set)):
            subRP = PPR(
                dnn_set=[yolo11n] ,
                bandwidth_set= bandwidth,
                f_local=f_local_set[i],
                f_edge=f_edge_set[j]
            ).run()
            new_set = new_set.union(subRP)
        RP[i] = new_set

    # print(RP)


    opp = OPP(
        dnn_set=dnn_set,
        bandwidth_set=curren_bandwidth,
        f_local_set=f_local_set,
        f_edge_set=f_edge_set,
        RP_set=RP,
        compute=compute,
    )

    CM = opp.run()
    # print("##################")
    # print("CM:", CM)
    # print(f'[Type of CM{type(CM)}')
    # print(f'len(CM) : {len(CM)}')

    # Convert to matrix
    CM_val = []
    CM_pnt = []

    for m , dict_s in CM.items():
        vals = []
        pnts = []
        for _ , s in dict_s.items():
            vals.append(s[0])
            pnts.append(s[1])
        CM_val.append(vals)
        CM_pnt.append(pnts)


    # print(f'CM_val{CM_val}')
    # print(f'CM_pnt{CM_pnt}')

    best_matching = Hungarian(cost_matrix=CM_val)
    assign , best_cost = best_matching.solve()

    return best_cost

    # print(best_cost)
    # for pair in assign :
    #     md = pair[0]
    #     es = pair[1]
        # print(f'{md} -> {es}|| cut point : {CM_pnt[md][es]}|| cost : {CM_val[md][es]}')

# warm-up
for i in range(10):
    measure(10)

for num in range(0 , 20 , 2):
    if num == 0 :
        num = 1
    n = 10
    lst_time = []
    for i in range(n) :
        start = time.perf_counter_ns()
        measure(num)
        end = time.perf_counter_ns()
        lst_time.append(end - start)

    print(f'{num} : {sum(lst_time) / len(lst_time)}')

