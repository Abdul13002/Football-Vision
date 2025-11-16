import cv2

def video_reader(video_path):
    #creating object to read videos, uses video_path as file path
    cap = cv2.VideoCapture(video_path)
    #initializing a list to store all frames
    frames = []
    #creating infinite loop unless explicity broken
    while True:
        # ret is a boolean indicates if the frame was succesfully with cap.read 
        ret, frame = cap.read()
        # if ret returnes fales/ no fram succesfully read then break
        if not ret:
            break
        frames.append(frame)
return frames 


