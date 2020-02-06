from requests.auth import HTTPDigestAuth
import pandas as pd
import requests
import json

# Stations url
stations = 'val/wxfcs/all/json/sitelist'


#hourly_data_by_location = 'val/wxobs/all/json/locationId'.format(locationId)

# Define resource
resource = stations

# Define HTTP get url
url = 'http://datapoint.metoffice.gov.uk/public/data/{resource}?key={APIkey}'.format(resource=resource,APIkey=APIkey)
print(url)
# Retrieve data
response = requests.get(url)

# If data is successfully retrieved
if(response.ok):

    # Loading the response data into a dict variable
    data = response.json()

    print("The response contains {0} properties".format(len(data)))
    print("\n")
    #print(json.dumps(data, indent=4, sort_keys=True))

    # Convert data to dataframe
    df = pd.DataFrame.from_dict(data['Locations']['Location'])

    # Save dataframe to file
    df.to_csv('met-office-station-locations.csv',index=False)

    # with open('met-office-station-locations.json', 'w') as f:
    #     json.dump(data, f)

else:
    # If response code is not ok (200), print the resulting http error code with description
    response.raise_for_status()
