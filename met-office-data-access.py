from requests.auth import HTTPDigestAuth
import pandas as pd
import requests
import json

APIkey = 'ed84efb9-66bb-4d18-bd73-86a132112a60'
stations = 'val/wxfcs/all/json/sitelist'

resource = stations

url = 'http://datapoint.metoffice.gov.uk/public/data/{resource}?key={APIkey}'.format(resource=resource,APIkey=APIkey)
print(url)
response = requests.get(url)

if(response.ok):

    # Loading the response data into a dict variable
    data = response.json()

    print("The response contains {0} properties".format(len(data)))
    print("\n")
    #print(json.dumps(data, indent=4, sort_keys=True))

    #print(type(data))
    df = pd.DataFrame.from_dict(data['Locations']['Location'])

    df.to_csv('met-office-station-locations.csv',index=False)


    # with open('met-office-station-locations.json', 'w') as f:
    #     json.dump(data, f)

else:
  # If response code is not ok (200), print the resulting http error code with description
    response.raise_for_status()


#print(response.json())

#hourly_data_by_location = 'val/wxobs/all/datatype/locationId'.format(locationId)
