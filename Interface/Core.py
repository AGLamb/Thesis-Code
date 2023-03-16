import pandas as pd

from SpiPy.ModelProd import create_set
from SpiPy.ModelConfidenceSet import *
from SpiPy.Forecast import *
import time


def main():
    start_time = time.time()
    mun_level = "municipality"
    # street_level = "street"
    hour_interval = "hours"
    day_interval = "day"

    train_set_m_h = create_set(geo_lev=mun_level, time_lev=hour_interval)
    print(f'First train set is ready')
    train_set_m_d = create_set(geo_lev=mun_level, time_lev=day_interval)
    print(f'Second train set is ready')

    performance_data_mh = get_performance(input_set=train_set_m_h, geo_lev=mun_level, time_lev=hour_interval)
    print(f'First performance set is ready')
    performance_data_md = get_performance(input_set=train_set_m_d, geo_lev=mun_level, time_lev=day_interval)
    print(f'Second performance set is ready')

    # train_set_s_h = create_set(geo_lev=street_level, time_lev=hour_interval)
    # print(f'Third train set is ready')
    # train_set_s_d = create_set(geo_lev=street_level, time_lev=day_interval)
    # print(f'Fourth train set is ready')
    #
    # performance_data_sh = get_performance(input_set=train_set_s_h, geo_lev=street_level, time_lev=hour_interval)
    # print(f'Third performance set is ready')
    # performance_data_sd = get_performance(input_set=train_set_s_d, geo_lev=street_level, time_lev=day_interval)
    # print(f'Fourth performance set is ready')

    performance_set = pd.concat([performance_data_mh, performance_data_md], axis=1)
                                 # performance_data_sh, performance_data_sd], axis=1)

    alphas = [0.5, 0.2, 0.1, 0.05]
    for i in range(len(alphas)):
        confidence_set = ModelConfidenceSet(data=performance_set, alpha=alphas[i], B=5, w=10000).run()
        print(confidence_set.included)

    end_time = time.time()
    print("Time taken: ", end_time - start_time, "seconds")
    return


if __name__ == "__main__":
    main()
