from ultralytics import YOLO 
import numpy


model = YOLO('/Users/abduladdan/Documents/football-cv-offline/Models/best.pt')  

results = model.predict('/Users/abduladdan/Documents/football-cv-offline/videos/test (14).mp4', save=True)
print(results[0])
print('--------------------------')
for box in results[0].boxes:
    print(box)
