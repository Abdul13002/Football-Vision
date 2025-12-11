import cv2
import numpy as np
from sklearn.cluster import KMeans

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}
        self.kmeans = None  # Store the trained kmeans model
        self.cluster_to_team = {}

    def get_clustering_model(self, img):
        # Reshape the image to 2D array
        img_2d = img.reshape(-1, 3)

        # Perform K-means with 2 clusters
        # n_init=1 is appropriate here as it's just finding two dominant colors in the small patch
        clustering = KMeans(n_clusters=2, init="k-means++", n_init=1, random_state=42)
        clustering.fit(img_2d)

        return clustering

    def get_player_color(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        crop = frame[y1:y2, x1:x2]

        # Skip if crop is too small
        if crop.shape[0] < 10 or crop.shape[1] < 10:
            return np.array([0, 0, 0])

        # --- IMPROVEMENT 1: Focus on the Torso Region ---
        # The torso provides the most reliable jersey color.
        # Use 15-40% of height and 20-80% of width
        h, w = crop.shape[:2]
        torso_area = crop[
            int(h * 0.15) : int(h * 0.40),
            int(w * 0.20) : int(w * 0.80)
        ]

        # Skip if torso area is empty
        if torso_area.size == 0:
            return np.array([0, 0, 0])

        # Get Clustering model
        clustering = self.get_clustering_model(torso_area)

        # Get the cluster labels and centers
        pixel_labels = clustering.labels_
        centers = clustering.cluster_centers_

        # --- IMPROVEMENT 2: Use HSV to Distinguish Jersey/Background ---
        # The jersey color is almost always darker or much brighter than the grass/background
        # Convert BGR centers to HSV to better filter green colors

        bgr_0 = centers[0].astype(np.uint8)
        hsv_0 = cv2.cvtColor(np.array([[bgr_0]]), cv2.COLOR_BGR2HSV)[0][0]

        bgr_1 = centers[1].astype(np.uint8)
        hsv_1 = cv2.cvtColor(np.array([[bgr_1]]), cv2.COLOR_BGR2HSV)[0][0]

        # Simple Metric: Check if a color is 'greenish' (Hue around 60)
        is_greenish = lambda hsv: 40 < hsv[0] < 80 and hsv[1] > 50

        is_0_greenish = is_greenish(hsv_0)
        is_1_greenish = is_greenish(hsv_1)

        if is_0_greenish and not is_1_greenish:
            # Cluster 0 is grass, Cluster 1 is jersey
            jersey_cluster = 1
        elif is_1_greenish and not is_0_greenish:
            # Cluster 1 is grass, Cluster 0 is jersey
            jersey_cluster = 0
        else:
            # Fallback: Pick the cluster with the lower green channel
            if bgr_0[1] < bgr_1[1]:
                jersey_cluster = 0
            else:
                jersey_cluster = 1

        jersey_color = centers[jersey_cluster]

        return jersey_color

    def assign_team_color(self, frame, player_detections):
        # --- IMPROVEMENT 3: Only Assign Team Colors ONCE ---
        # If team colors are already set, skip this function to ensure stability
        if self.team_colors:
            return

        colors = []
        for _, detection in player_detections.items():
            box = detection["bbox"]
            color = self.get_player_color(frame, box)
            # Filter out near-black colors (failed detection) and overexposed/white colors
            color_sum = np.sum(color)
            if 10 < color_sum < 700:  # Reject black (< 10) and overexposed white (> 700)
                colors.append(color)

        if len(colors) < 2:
            return

        # Use n_init=10 for good results with reasonable speed
        clustering = KMeans(n_clusters=2, init="k-means++", n_init=10, random_state=42)
        clustering.fit(colors)

        self.kmeans = clustering

        # Use composite metric for consistent team assignment
        # Robust for common opposing colors like red/blue or white/black
        color_0 = clustering.cluster_centers_[0]
        color_1 = clustering.cluster_centers_[1]

        # Calculate composite scores (B, G, R format in OpenCV)
        # Using weighted brightness + red channel emphasis
        score_0 = color_0[0] * 0.114 + color_0[1] * 0.587 + color_0[2] * 0.299
        score_1 = color_1[0] * 0.114 + color_1[1] * 0.587 + color_1[2] * 0.299

        score_0 += color_0[2] * 0.1  # Red channel bonus
        score_1 += color_1[2] * 0.1

        # The lower score team gets assigned as Team 1 (arbitrary but consistent)
        if score_0 < score_1:
            self.team_colors[1] = clustering.cluster_centers_[0]
            self.team_colors[2] = clustering.cluster_centers_[1]
            self.cluster_to_team = {0: 1, 1: 2}
        else:
            self.team_colors[1] = clustering.cluster_centers_[1]
            self.team_colors[2] = clustering.cluster_centers_[0]
            self.cluster_to_team = {0: 2, 1: 1}


    def get_player_team(self, frame, player_bbox, player_id):
        # --- IMPROVEMENT 4: Player Team Assignment Persistence ---
        # Once a player is assigned a team, the assignment is sticky
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        if not self.kmeans:
            # Should not happen if assign_team_color was called first, but good safeguard
            return None

        color = self.get_player_color(frame, player_bbox)

        # Only assign team if the player color is not black/error color or overexposed
        color_sum = np.sum(color)
        if color_sum < 10 or color_sum > 700:
            return None

        # Predict the cluster
        cluster_id = self.kmeans.predict(color.reshape(1, -1))[0]
        team_id = self.cluster_to_team[cluster_id]

        # --- IMPROVEMENT 5: Confidence Check for New Assignments ---
        # Calculate the distance to the predicted cluster center
        team_center = self.kmeans.cluster_centers_[cluster_id]
        distance = np.linalg.norm(color - team_center)

        # Distance threshold to prevent misclassification from wrong color extraction
        CONFIDENCE_THRESHOLD = 80

        if distance < CONFIDENCE_THRESHOLD:
            # Assign team ID and make it sticky for this track_id
            self.player_team_dict[player_id] = team_id
            return team_id

        return None
