import matplotlib.pyplot as plt
import pandas as pd

# loc_leg = list(df_pol.columns)
# fitted_values = SWVAR.fittedvalues
# for key in LocDict:
#     print(key)
#     fig, ax = plt.subplots(figsize=(50, 20))
#     ax.plot(filtered_pol[key][800:1200])
#     ax.plot(fitted_values[key][800:1200])
#     ax.set_title("Images/" + key + "Comparisson")
#     plt.show()


# fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(nrows=3, ncols=3,
#     sharey=True, figsize=(50,30))
#
# ax1.plot(df_pol[df_pol.columns[0]])
# ax1.axvline(pd.to_datetime('2020-03-01 01:00:00+00:00'), color='red', linestyle='-')
# ax1.axvline(pd.to_datetime('2022-02-14 01:00:00+00:00'), color='red', linestyle='-')
# ax1.set_xticks(df_pol.index[::2160])
# ax1.set_xticklabels(df_pol.index[::2160], rotation=25)
# ax1.set_title(df_pol.columns[0])
#
# ax2.plot(df_pol[df_pol.columns[1]])
# ax2.axvline(pd.to_datetime('2020-03-01 01:00:00+00:00'), color='red', linestyle='-')
# ax2.axvline(pd.to_datetime('2022-02-14 01:00:00+00:00'), color='red', linestyle='-')
# ax2.set_xticks(df_pol.index[::2160])
# ax2.set_xticklabels(df_pol.index[::2160], rotation=25)
# ax2.set_title(df_pol.columns[1])
#
# ax3.plot(df_pol[df_pol.columns[2]])
# ax3.axvline(pd.to_datetime('2020-03-01 01:00:00+00:00'), color='red', linestyle='-')
# ax3.axvline(pd.to_datetime('2022-02-14 01:00:00+00:00'), color='red', linestyle='-')
# ax3.set_xticks(df_pol.index[::2160])
# ax3.set_xticklabels(df_pol.index[::2160], rotation=25)
# ax3.set_title(df_pol.columns[2])
#
# ax4.plot(df_pol[df_pol.columns[3]])
# ax4.axvline(pd.to_datetime('2020-03-01 01:00:00+00:00'), color='red', linestyle='-')
# ax4.axvline(pd.to_datetime('2022-02-14 01:00:00+00:00'), color='red', linestyle='-')
# ax4.set_xticks(df_pol.index[::2160])
# ax4.set_xticklabels(df_pol.index[::2160], rotation=25)
# ax4.set_title(df_pol.columns[3])
#
# ax5.plot(df_pol[df_pol.columns[4]])
# ax5.axvline(pd.to_datetime('2020-03-01 01:00:00+00:00'), color='red', linestyle='-')
# ax5.axvline(pd.to_datetime('2022-02-14 01:00:00+00:00'), color='red', linestyle='-')
# ax5.set_xticks(df_pol.index[::2160])
# ax5.set_xticklabels(df_pol.index[::2160], rotation=25)
# ax5.set_title(df_pol.columns[4])
#
# ax6.plot(df_pol[df_pol.columns[5]])
# ax6.axvline(pd.to_datetime('2020-03-01 01:00:00+00:00'), color='red', linestyle='-')
# ax6.axvline(pd.to_datetime('2022-02-14 01:00:00+00:00'), color='red', linestyle='-')
# ax6.set_xticks(df_pol.index[::2160])
# ax6.set_xticklabels(df_pol.index[::2160], rotation=25)
# ax6.set_title(df_pol.columns[5])
#
# ax7.plot(df_pol[df_pol.columns[6]])
# ax7.axvline(pd.to_datetime('2020-03-01 01:00:00+00:00'), color='red', linestyle='-')
# ax7.axvline(pd.to_datetime('2022-02-14 01:00:00+00:00'), color='red', linestyle='-')
# ax7.set_xticks(df_pol.index[::2160])
# ax7.set_xticklabels(df_pol.index[::2160], rotation=25)
# ax7.set_title(df_pol.columns[6])
#
# ax8.plot(df_pol, label=df_pol.columns)
# ax8.axvline(pd.to_datetime('2020-03-01 01:00:00+00:00'), color='red', linestyle='-')
# ax8.axvline(pd.to_datetime('2022-02-14 01:00:00+00:00'), color='red', linestyle='-')
# ax8.legend(loc_leg)
# ax8.set_xticks(df_pol.index[::2160])
# ax8.set_xticklabels(df_pol.index[::2160], rotation=25)
# ax8.set_title('Pollution Levels')
#
#
# # plt.savefig('Pol.png')
# plt.show()

# import geopandas as gpd
#
# df = gpd.read_file(r"C:\Users\VY72PC\Downloads\BestuurlijkeGebieden_2023_voorlopig.gpkg")
#
# df.plot(figsize=(50, 30))
