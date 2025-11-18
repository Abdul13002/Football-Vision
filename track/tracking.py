from ultralytics import YOLO
import supervision as sv
class tracking:
    def __init__(self,model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def Frame_detection(self, frames):
        batch_size = 20
        Detected = []
        for i in range (0, len(frames), batch_size):
            Total_Detected = self.model.predict(frames[i:i+batch_size], conf=0.1)
            Detected += Total_Detected
            break
        return Detected

    def object_tracking(self, frames):
        Detected  = self.Frame_detection(frames)

        for frame_num, Detected in enumerate(Detected):
            cls_names = Detected.names
            cls_names_inv = {v:k for k,v in cls_names.items()}

            Detected_sv = sv.Detections.from_ultralytics(Detected)

            print(Detected_sv)
    


