from SpiPy.ModelConfidenceSet import *
from SpiPy.ModelProd import *
import time
import warnings


class ThesisProcess:
    def __init__(self, aggregate: str, intervals: str, restricted: bool = False):
        self.results = None
        self.aggregation_level = aggregate
        self.time_intervals = intervals
        self.restricted = restricted
        self.trained_set = None
        self.performance_set = None
        self.train_data = None
        self.test_data = None

    def run(self) -> dict:
        self.train_data, self.test_data = part1(geo_lev=self.aggregation_level,
                                                time_lev=self.time_intervals)
        self.trained_set = ModelSet(train_data=self.train_data,
                                    test_data=self.test_data,
                                    geo_lev=self.aggregation_level,
                                    time_lev=self.time_intervals,
                                    restricted=self.restricted)
        self.trained_set.run()
        print("Train set ready")
        self.performance_set = self.trained_set.get_performance()
        print("Performance set ready")

        output = {}

        alphas = [0.1]  # 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        for alpha in alphas:
            inner_dict = {}
            for func in self.performance_set.metric_func:
                print(self.performance_set.performance[func.__name__])
                inner_dict[func.__name__] = ModelConfidenceSet(data=self.performance_set.performance[func.__name__],
                                                               alpha=alpha,
                                                               B=3,
                                                               w=1000).run()
            output[alpha] = inner_dict
        return output


def main() -> None:
    start_time = time.time()
    aggregation = {
        "Geographical": ["municipality", "street"],
        "Temporal": ["hours", "day"]
    }
    restricted = False

    output = {}
    for value1 in aggregation["Geographical"]:
        for value2 in aggregation["Temporal"]:
            print(f'{value1} - {value2}')
            output[value1][value2] = ThesisProcess(aggregate=value1, intervals=value2, restricted=restricted).run()
            for key in output[value1][value2]:
                for func in output[value1][value2][key].performance_set.metric_func:
                    print(f'For metric {func} at alpha = {key:.1f}, '
                          f'the MCS includes {output[value1][value2][key][func.__name__]}.included')

    end_time = time.time()
    print("Time taken: ", end_time - start_time, "seconds")
    return None


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
