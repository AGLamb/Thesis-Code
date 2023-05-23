from SpiPy.Models.ModelConfidenceSet import *
from SpiPy.RunFlow.ModelProd import *
from SpiPy.RunFlow.Backbone import *


class ThesisProcess:
    def __init__(
            self, aggregate: str,
            intervals: str,
            save_data: bool = False,
            bWorkLaptop: bool = False
    ) -> None:

        self.aggregation_level = aggregate
        self.time_intervals = intervals
        self.save_data = save_data
        self.bWorkLaptop = bWorkLaptop
        self.performance_set = None
        self.trained_set = None
        self.database = None
        self.results = {}

    def run(self) -> None:
        self.database = RunFlow(save_data=self.save_data, bWorkLaptop=self.bWorkLaptop)
        self.database.run(geo_lev=self.aggregation_level,
                          time_lev=self.time_intervals)

        self.trained_set = ModelSet(database=self.database)
        self.trained_set.run()
        print("Train set ready")

        self.performance_set = self.trained_set.get_performance(time_lev=self.time_intervals)
        print("Performance set ready")

        alphas = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        for alpha in alphas:
            inner_dict = {}
            for func in self.performance_set.metric_func:
                inner_dict[func.__name__] = ModelConfidenceSet(data=self.performance_set.performance[func.__name__],
                                                               alpha=alpha,
                                                               B=3,
                                                               w=1000).run()
            self.results[alpha] = inner_dict
        return None
