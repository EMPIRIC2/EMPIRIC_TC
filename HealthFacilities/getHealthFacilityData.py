
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
from geopy import distance
import os

class Sites:
    def __init__(self, n_clusters=10):
        self.sites = None
        self.cluster_bounding_boxes = None
        self.clusters_to_sites = None
        self.site_to_index = None

        self.getHealthFacilityData()
        self.createSiteClusters(n_clusters)
        self.set_site_to_index()

    
    def siteTouchedByStormRMax(self, site, lat, lon, rmax, rmax_multiple):
        d = distance.distance(site, (lat, lon)).km
      
        return d <= rmax_multiple*rmax

    @staticmethod
    def boxes_intersect(box1, box2):
        
        '''
        this is just a useful reference for where the corners of the box are. Important to remember that the y axis is negative, but not flipped (more down is more negative)
        this method is based on this 
        top_right = (box1[1], box1[3])
        bottom_left_other = (box2[0], box2[2])
        bottom_left_self = (box1[0], box1[2])
        top_right_other = (box2[1], box2[3])
        '''
        
        return not (box1[3] < box2[2]
         or box1[2] > box2[3]
         or box1[1] < box2[0]
         or box1[0] > box2[1]
    )

    def update_sites_touched_by_storm(self, sites_touched_by_storm, lat, lon, rmax, rmax_multiple, storm_bounding_box):
        
        # also keep track of whether the box hits a site cluster (I use this for debugging purposes)
        box_touches = False
        
        # check each cluster of sites to see if the cluster bounding box hits the storm bounding box
        for i,cluster_bounding_box in self.cluster_bounding_boxes.items():
            if Sites.boxes_intersect(cluster_bounding_box, storm_bounding_box):
                box_touches = True
                for site in self.clusters_to_sites[i]:
                    
                    if site[0] < storm_bounding_box[0] or site[0] > storm_bounding_box[1] or site[1] < storm_bounding_box[2] or site[1] > storm_bounding_box[3]: continue
                    if self.siteTouchedByStormRMax(site, lat, lon, rmax, rmax_multiple):
                        sites_touched_by_storm.add(site)

        
        return sites_touched_by_storm, box_touches


    def create_site_landfall_vector(self):

        if self.sites is None: return None

        site_data = np.zeros((
                len(self.sites),
                6, # storm producing months
                5 # storm categories
        ))

        return site_data

    def set_site_to_index(self):
        if self.sites is None: return
        self.site_to_index = {site: i for i, site in enumerate(self.sites)}

    def createSiteClusters(self, n_clusters=10):
        if self.sites is None:
            return

        est = KMeans(n_clusters=n_clusters)

        est.fit(self.sites)

        cluster_bounding_boxes = {i: [None, None, None, None] for i in range(n_clusters)}
        clusters_to_sites = {i: [] for i in range(n_clusters)}

        for i, site in enumerate(self.sites):
            label = est.labels_[i]

            clusters_to_sites[label].append(site)

            center = est.cluster_centers_[label]

            cluster_bounding_boxes[label][0] = min(site[0], cluster_bounding_boxes[label][0]) if \
            cluster_bounding_boxes[label][0] is not None else site[0]
            cluster_bounding_boxes[label][1] = max(site[0], cluster_bounding_boxes[label][1]) if \
            cluster_bounding_boxes[label][1] is not None else site[0]

            cluster_bounding_boxes[label][2] = min(site[1], cluster_bounding_boxes[label][2]) if \
            cluster_bounding_boxes[label][2] is not None else site[1]
            cluster_bounding_boxes[label][3] = max(site[1], cluster_bounding_boxes[label][3]) if \
            cluster_bounding_boxes[label][3] is not None else site[1]
        
        # expand cluster boxes slightly... not sure this is necessary or not
        for label in cluster_bounding_boxes.keys():
            cluster_bounding_boxes[label][0] -= 1
            cluster_bounding_boxes[label][1] += 1
            cluster_bounding_boxes[label][2] -= 1
            cluster_bounding_boxes[label][3] += 1
            
        self.cluster_bounding_boxes = cluster_bounding_boxes
        self.clusters_to_sites = clusters_to_sites

    def getHealthFacilityData(self):
        """

        :param file_paths:
        :return:
        """
        __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

        files = ['SPC_health_data_hub_Kiribati.csv', 'SPC_health_data_hub_Solomon_Islands.csv', 'SPC_health_data_hub_Tonga.csv', 'SPC_health_data_hub_Vanuatu.csv']
        file_paths = [os.path.join(__location__, file) for file in files]
        locations = []

        for file_path in file_paths:
            df = pd.read_csv(file_path)

            latitudes = df.loc[:, "LATITUDE: Latitude"]
            longitudes = df.loc[:, "LONGITUDE: Longitude"]


            ## adjust because our longitudes go from 0 to 360 not -180 to 180
            for i in range(len(longitudes)):
                if latitudes.loc[i] > -5 or latitudes.loc[i] < -60: continue# not in basin

                if longitudes.loc[i] < 0:
                    df.loc[i, "LONGITUDE: Longitude"] += 360

                locations.append((latitudes.loc[i], longitudes.loc[i]))

        self.sites = locations
