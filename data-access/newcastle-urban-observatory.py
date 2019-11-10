from shapely.geometry import shape, Polygon
from requests.auth import HTTPDigestAuth
from pandas.io.json import json_normalize
import pandas as pd
import requests
import fiona
import json
import copy
import os


def http_get(url,is_json=True):

    # Retrieve data
    response = requests.get(url)

    # If data is successfully retrieved
    if(response.ok):

        if is_json:
            # Loading the response data into a dict variable
            data = response.json()
        else:
            print(response.content)
            data = pd.read_csv(io.StringIO(response.content.decode('utf-8')))

        return data

    else:
        # If response code is not ok (200), print the resulting http error code with description
        response.raise_for_status()
        return False

def import_shapefile(path):

    # Load shapefiles of interest
    shapefile = fiona.open(path)
    shapes = [x for x in shapefile]

    aois = copy.deepcopy(shapes)
    for aoi in aois:
        aoi['geometry'] = shape(aoi['geometry']).bounds

    return aois


# Newcastle University Urban Observatory API
# Define static variables

# API key
api_key = '70bt3lgir8zn9sqvtx09t9g9xreaxtae5jjgbkcqrdthho9r5c5p2grk071jba58t4gu6s30ciohd924r6oc5ylm4x'
# Current working directory
cwd = '/Users/Yannis/Desktop/Cambridge/MRes/Mini-project/data'
# Data theme
theme = 'Weather'
# Start and end times for data search
start_time = '20190201'
end_time = '20190228'

# Bounding box coordinates
p1_x = p2_x = p1_y = p2_y = 0

# Sensor name
sensor_name = 'PER_WUNDERGROUND_INEWBURN3'

# Variables in string
all_variables = "Average%20Speed%2CCongestion%2CJourney%20Time%2CTraffic%20Flow%2CDaily%20Accumulation%20Rainfall%2CRain%2CRain%20Acc%2CRain%20Duration%2CRainfall%2CRainInt%2CTemperature"
climatic_variables = "2CDaily%20Accumulation%20Rainfall%2CRain%2CRain%20Acc%2CRain%20Duration%2CRainfall%2CRainInt%2CTemperature"
traffic_variables = "Average%20Speed%2CCongestion%2CJourney%20Time%2CTraffic%20Flow"

# URLs
variable_url = f"http://uoweb3.ncl.ac.uk/api/v1.1/variables/json/?theme={theme}&api_key={api_key}"
sensor_type_url = f"http://uoweb3.ncl.ac.uk/api/v1.1/sensors/types/json/?theme={theme}&api_key={api_key}"
sensor_url = f"http://uoweb3.ncl.ac.uk/api/v1.1/sensors/json/?bbox_p1_x={p1_x}&bbox_p1_y={p1_y}&bbox_p2_x={p2_x}&bbox_p2_y={p2_y}&api_key={api_key}"
sensor_info_url = f"http://uoweb3.ncl.ac.uk/api/v1.1/sensors/json/?variable={climatic_variables}&api_key={api_key}"
sensor_data_feed_url = f"http://uoweb3.ncl.ac.uk/api/v1.1/sensors/data/json/?starttime={start_time}&endtime={end_time}&variables={climatic_variables}&api_key={api_key}"
# sensors of interest
all_sensors = pd.read_csv(os.path.join(cwd,'0-sensors','weather_and_traffic_sensors.csv'))
sensor_ids = []
#[73148,73151,79456,79302,79357,79387,79384,79501,79454,79489,79463]

#print(all_sensors)
#print(all_sensors[all_sensors['Raw ID'].isin(sensor_ids)])


# Define HTTP get url
url = f"http://uoweb3.ncl.ac.uk:80/metadata/api/measurements?variable=Daily%20Accumulation%20Rainfall%2CRain%2CRain%20Acc%2CRain%20Duration%2CRainfall%2CRainInt&api_key={api_key}"
#f"http://uoweb3.ncl.ac.uk:80/metadata/api/measurements?variable=Average%20Speed%2CCongestion%2CJourney%20Time%2CTraffic%20Flow%2CDaily%20Accumulation%20Rainfall%2CRain%2CRain%20Acc%2CRain%20Duration%2CRainfall%2CRainInt%2CTemperature&api_key={api_key}"

data = http_get(url,True)
#print(data)
print(json.dumps(data, indent=4, sort_keys=True))
