from SpiPy.ModelConfidenceSet import *
from SpiPy.ModelProd import *
from SpiPy.Backbone import *
import warnings
import time


class ThesisProcess:
    def __init__(self, aggregate: str, intervals: str, save_data: bool = False) -> None:
        self.results = {}
        self.aggregation_level = aggregate
        self.time_intervals = intervals
        self.trained_set = None
        self.performance_set = None
        self.database = None
        self.save_data = save_data

    def run(self) -> None:
        self.database = RunFlow(save=self.save_data)
        self.database.run(geo_lev=self.aggregation_level,
                          time_lev=self.time_intervals)

        self.trained_set = ModelSet(database=self.database)
        self.trained_set.run()
        print("Train set ready")

        self.performance_set = self.trained_set.get_performance(time_lev=self.time_intervals)
        print("Performance set ready")

        alphas = [0.1]  # 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        for alpha in alphas:
            inner_dict = {}
            for func in self.performance_set.metric_func:
                # print(self.performance_set.performance[func.__name__])
                inner_dict[func.__name__] = ModelConfidenceSet(data=self.performance_set.performance[func.__name__],
                                                               alpha=alpha,
                                                               B=3,
                                                               w=1000).run()
            self.results[alpha] = inner_dict
        return None


def main() -> None:
    start_time = time.time()
    aggregation = {
        "Geographical": ["municipality", "street"],
        "Temporal": ["hours", "day"]
    }

    save = False

    output = {}
    for value1 in aggregation["Geographical"]:
        output[value1] = {}

        for value2 in aggregation["Temporal"]:
            print(f'{value1} - {value2}')
            output[value1][value2] = ThesisProcess(aggregate=value1,
                                                   intervals=value2,
                                                   save_data=save)
            output[value1][value2].run()

            for key in output[value1][value2].results:
                for func in output[value1][value2].performance_set.metric_func:
                    print(f'For metric {func} at alpha = {key:.1f}, '
                          f'the MCS includes {output[value1][value2].results[key][func.__name__].included}')

    end_time = time.time()
    print("Time taken: ", end_time - start_time, "seconds")
    return None


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
