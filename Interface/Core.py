from Interface.ModelProd import create_set
from Interface.Forecast import *
import time


def main():

    start_time = time.time()
    mun_level = "municipality"
    street_level = "street"

    hour_interval = "hours"
    day_interval = "day"

    train_set_m_h = create_set(mun_level, hour_interval)
    train_set_m_d = create_set(mun_level, day_interval)
    train_set_s_h = create_set(street_level, hour_interval)
    train_set_s_d = create_set(street_level, day_interval)

    performance_data_mh = get_performance(input_set=train_set_m_h, geo_lev=mun_level, time_lev=hour_interval)
    performance_data_md = get_performance(input_set=train_set_m_d, geo_lev=mun_level, time_lev=day_interval)
    performance_data_sh = get_performance(input_set=train_set_s_h, geo_lev=street_level, time_lev=hour_interval)
    performance_data_sd = get_performance(input_set=train_set_s_d, geo_lev=street_level, time_lev=day_interval)

    print(performance_data_md)
    print(performance_data_mh)
    print(performance_data_sh)
    print(performance_data_sd)

    end_time = time.time()
    print("Time taken: ", end_time - start_time, "seconds")
    return


if __name__ == "__main__":
    main()
