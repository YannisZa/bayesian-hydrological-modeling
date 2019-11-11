import pandas as pd
import xarray as xr
import os

# path to file
path = '/Users/Yannis/Desktop/Cambridge/MRes/Mini-project/data/FF-HadRM3-PPE/'
file = 'FF-HadRM3-afgcx-PE-1980-2009.nc'
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

# Open nc file
ds_grib = xr.open_dataset(filepath)
df = ds_grib.to_dataframe().reset_index()

print(df[df.pe.notnull()])
