from SpiPy.ModelProd import create_set
from SpiPy.ModelConfidenceSet import *
from SpiPy.Forecast import *
from typing import Callable
import pandas as pd
import time


class Thesis_Process:
    def __init__(self, score_function: Callable[[np.ndarray, np.ndarray], float],
                 aggreg: str, intervals: str):
        self.results = None
        self.performance_metric = score_function.__name__
        self.performance_function = score_function
        self.aggregation_level = aggreg
        self.time_intervals = intervals

    def run(self):
        train_set = create_set(geo_lev=self.aggregation_level, time_lev=self.time_intervals)
        print(f'Train set ready')
        performance_set = get_performance(input_set=train_set, geo_lev=self.aggregation_level,
                                          time_lev=self.time_intervals, func=self.performance_function)
        print(f'Performance set ready')
        alphas = [0.5, 0.2, 0.1, 0.05]
        for i in range(len(alphas)):
            print(ModelConfidenceSet(data=performance_set, alpha=alphas[i], B=3, w=1000).run().included)


def main():
    start_time = time.time()
    mun_level = "municipality"
    street_level = "street"
    hour_interval = "hours"
    day_interval = "day"

    perfomance_funcitons = [MAE, MSE, RMSE, MAPE]
    for function in perfomance_funcitons:
        Thesis_Process(score_function=function, aggreg=mun_level, intervals=hour_interval).run()
        Thesis_Process(score_function=function, aggreg=mun_level, intervals=day_interval).run()
        Thesis_Process(score_function=function, aggreg=street_level, intervals=hour_interval).run()
        Thesis_Process(score_function=function, aggreg=street_level, intervals=day_interval).run()

    end_time = time.time()
    print("Time taken: ", end_time - start_time, "seconds")
    return


if __name__ == "__main__":
    main()
