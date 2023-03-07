from Interface.ModelProd import create_set
import time


def main():

    start_time = time.time()
    mun_level = "municipality"
    street_level = "street"

    hour_interval = "hours"
    day_interval = "day"

    set_m_h = create_set(mun_level, hour_interval)
    set_m_d = create_set(mun_level, day_interval)
    set_s_h = create_set(street_level, hour_interval)
    set_s_d = create_set(street_level, day_interval)

    end_time = time.time()
    print("Time taken: ", end_time - start_time, "seconds")
    return


if __name__ == "__main__":
    main()
