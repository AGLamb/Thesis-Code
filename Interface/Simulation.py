from SpiPy.RunFlow.Simulation import *


class FlatSimulation(object):
    def __init__(self, length: int = 8760):
        self.periods = length
        self.simulate()

    def simulate(self) -> None:
        Simulation(num_periods=self.periods).generate_simulation()
        return None


