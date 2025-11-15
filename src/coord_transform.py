from ultralytics import YOLO  # Ensure you are importing YOLO from the correct library
import numpy

# Example usage
model = YOLO('/Users/abduladdan/Documents/football-cv-offline/Models/best.pt')  # Load the YOLO model

results = model.predict('/Users/abduladdan/Documents/football-cv-offline/videos/test (14).mp4', save=True)
print(results[0])
print('--------------------------')
for box in results[0].boxes:
    print(box)
