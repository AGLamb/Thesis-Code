from numpy import random


class Simulation:
    def __init__(self, num_periods):
        self.num_periods = num_periods
        self.pollution_levels = None
        self.wind_directions = None
        self.wind_speeds = None

    def generate_pollution_levels(self):
        # Simulate a stationary process
        self.pollution_levels = random.normal(0, 1, size=(self.num_periods, 4))

    def generate_wind_directions(self):
        # Simulate wind direction (0-360 degrees)
        self.wind_directions = random.uniform(0, 360, size=(self.num_periods, 4))

    def generate_wind_speeds(self):
        # Simulate wind speed (arbitrary units)
        self.wind_speeds = random.uniform(0, 10, size=(self.num_periods, 4))

    def generate_simulation(self):
        self.generate_pollution_levels()
        self.generate_wind_directions()
        self.generate_wind_speeds()
