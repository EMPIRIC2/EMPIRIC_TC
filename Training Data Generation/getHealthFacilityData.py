
import pandas as pd

def getHealthFacilityData(file_paths = None):
    """

    :param file_paths:
    :return:
    """
    if file_paths is None:
        file_paths = ['./SPC_health_data_hub_Kiribati.csv', './SPC_health_data_hub_Solomon_Islands.csv', './SPC_health_data_hub_Tonga.csv', './SPC_health_data_hub_Vanuatu.csv']

    locations = []

    for file_path in file_paths:
        df = pd.read_csv(file_path)

        latitudes = df.loc[:, "LATITUDE: Latitude"]
        longitudes = df.loc[:, "LONGITUDE: Longitude"]

        locations += zip(latitudes, longitudes)

    return locations

getHealthFacilityData()