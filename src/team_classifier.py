import cv2
import numpy as np
from sklearn.cluster import KMeans


def get_jersey_crop(frame, bbox):
    # Extracting the bbox coordinates
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

    #setting the height of box
    height = y2-y1

    #calculating upper body height 40% of total
    upper_body_height = int(height * 0.4)
    y_crop_end = y1 + upper_body_height

    #now putting together the actual crop
    jersey_crop = frame[y1:y_crop_end, x1:x2]

    return jersey_crop

def get_dominant_color(image, k=2):
#reshape the image to be a list of pixels/2d
    pixels = image.reshape(-1,3)
    pixels = np.float32(pixels)

    #kmeans clustering
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(pixels)

    #find biggest clusters 
    labels = kmeans.labels_ # the cluster each label belongs to 
    counts = np.bincount(labels) #counts pixel
    dominant_cluster_index = np.argmax(counts)


    #Get center color of biggest cluster
    dominant_color = kmeans.cluster_centers_[dominant_cluster_index]
    dominant_color = tuple(dominant_color.astype(int))

    return dominant_color



    




