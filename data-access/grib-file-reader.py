import pandas as pd
import xarray as xr
import os

# path to file
path = ''
file = 'europe_evaporation_jan2018.grib'
#'era-5_2018.grib'
filepath = os.path.join(path,file)

bbox_json = {
        "type": "Polygon",
        "coordinates": [
          [
            [
              0.0212860107421875,
              52.12084186372001
            ],
            [
              0.2574920654296875,
              52.12084186372001
            ],
            [
              0.2574920654296875,
              52.25891191821227
            ],
            [
              0.0212860107421875,
              52.25891191821227
            ],
            [
              0.0212860107421875,
              52.12084186372001
            ]
          ]
        ]
}

# Open grib file
ds_grib = xr.open_dataset(filepath, engine='cfgrib')
df = ds_grib.to_dataframe().reset_index()

print(ds_grib)

df.loc[:,'lat_long'] = df[['latitude', 'longitude']].apply(tuple, axis=1)


#print(df[['time','ro']])
#print(pd.unique(df['lat_long']))
#print(sorted(df.columns.values))
