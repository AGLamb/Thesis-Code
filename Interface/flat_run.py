from Interface.Core import *
from time import time
import warnings


bWorkLaptop: bool = True
save_to_disk: bool = True


class Run(object):
    def __init__(self, save_res: bool = False) -> None:
        self.save = save_res
        self.aggregation = {
            "Geographical": ["municipality", "street"],
            "Temporal": ["hours", "day"]
        }
        self.output = {}
        self.execute()
        return

    def execute(self) -> None:
        for value1 in self.aggregation["Geographical"]:
            self.output[value1] = {}

            for value2 in self.aggregation["Temporal"]:
                print(f'{value1} - {value2}')
                self.output[value1][value2] = ThesisProcess(aggregate=value1,
                                                            intervals=value2,
                                                            save_data=self.save,
                                                            bWorkLaptop=bWorkLaptop)
                self.output[value1][value2].run()

                for key in self.output[value1][value2].results:
                    for func in self.output[value1][value2].performance_set.metric_func:
                        print(f'For metric {func} at alpha = {key:.1f}, '
                              f'the MCS includes {self.output[value1][value2].results[key][func.__name__].included}')
        return None


if __name__ == "__main__":
    with warnings.catch_warnings():
        start_time = time()
        warnings.simplefilter("ignore")
        Run(save_res=save_to_disk)
        end_time = time()
        print("Time taken: ", end_time - start_time, "seconds")
