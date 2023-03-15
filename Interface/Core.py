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

    confidenceSet = ModelConfidenceSet(performance_data_md, 0.1,3, 1000).run()

    end_time = time.time()
    print("Time taken: ", end_time - start_time, "seconds")
    return


if __name__ == "__main__":
    main()
