import cv2
import numpy as np
from sklearn.cluster import KMeans

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}

    def get_clustering_model(self, img):
        # Reshape the image to 2D array
        img_2d = img.reshape(-1, 3)

        # Perform K-means with 2 clusters
        clustering = KMeans(n_clusters=2, init="k-means++", n_init=1)
        clustering.fit(img_2d)

        return clustering

    def get_player_color(self, frame, bbox):
        crop = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

        # Skip if crop is too small
        if crop.shape[0] < 10 or crop.shape[1] < 10:
            return np.array([0, 0, 0])

        top_half = crop[0:int(crop.shape[0]/2), :]

        # Skip if top half is empty
        if top_half.size == 0:
            return np.array([0, 0, 0])

        # Get Clustering model
        clustering = self.get_clustering_model(top_half)

        # Get the cluster labels for each pixel
        pixel_labels = clustering.labels_

        # Reshape the labels to the image shape
        label_map = pixel_labels.reshape(top_half.shape[0], top_half.shape[1])

        # Get the player cluster
        corners = [label_map[0,0], label_map[0,-1], label_map[-1,0], label_map[-1,-1]]
        bg_cluster = max(set(corners), key=corners.count)
        jersey_cluster = 1 - bg_cluster

        jersey_color = clustering.cluster_centers_[jersey_cluster]

        return jersey_color

    def assign_team_color(self, frame, player_detections):

        colors = []
        for _, detection in player_detections.items():
            box = detection["bbox"]
            color = self.get_player_color(frame, box)
            colors.append(color)

        clustering = KMeans(n_clusters=2, init="k-means++", n_init=10)
        clustering.fit(colors)

        self.kmeans = clustering

        self.team_colors[1] = clustering.cluster_centers_[0]
        self.team_colors[2] = clustering.cluster_centers_[1]

    def get_player_team(self, frame, player_bbox, player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        color = self.get_player_color(frame, player_bbox)

        team_id = self.kmeans.predict(color.reshape(1, -1))[0]
        team_id += 1

        self.player_team_dict[player_id] = team_id

        return team_id