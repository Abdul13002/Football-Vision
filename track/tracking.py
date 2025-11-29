from ultralytics import YOLO
import supervision as sv
import pickle
import os
import sys
import cv2
sys.path.append('../')
from Views import get_center_bbox, get_width
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
           
        return Detected

    def object_tracking(self, frames, read_from_stub=False, stub_path=None):



        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        
        Detected  = self.Frame_detection(frames)

        tracks={
            "players" : [],
            "referees":[],
            "ball" :[]

        }

        for frame_num, Detected in enumerate(Detected):
            cls_names = Detected.names
            cls_names_inv = {v:k for k,v in cls_names.items()}

            detected_supervision = sv.Detections.from_ultralytics(Detected)

            for object_ind, class_id in enumerate(detected_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detected_supervision.class_id[object_ind] = cls_names_inv['player']

            Track_Det = self.tracker.update_with_detections(detected_supervision)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in Track_Det:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}

                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}

            for frame_detection in detected_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox": bbox}
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks

    def draw_ellpse(self, frame, bbox, track_id, color=(0, 255, 0)):
        y2 = int(bbox[3])
        x_center, _ = get_center_bbox(bbox)
        width = max(25, min(get_width(bbox), 40))

        overlay = frame.copy()

        cv2.ellipse(
            overlay,
            center=(x_center, y2),
            axes=(int(width), int(0.35*width)),
            angle=0,
            startAngle=-95,
            endAngle=235,
            color=(200, 200, 255),  
            thickness=-1,  # Fill the ellipse
            lineType=cv2.LINE_4
        )

        alpha = 0.3  
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35*width)),
            angle=0,
            startAngle=-95,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_AA
        )

        return frame



    def annotations(self, video_frames, tracks):
        output_video_frames = []

        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            referee_dict = tracks["referees"][frame_num]
            ball_dict = tracks["ball"][frame_num]

            # players: green disc
            for track_id, player in player_dict.items():
                frame = self.draw_ellpse(frame, player["bbox"], track_id, color=(255, 20, 0))

            # referees: yellow disc
            for track_id, ref in referee_dict.items():
                frame = self.draw_ellpse(frame, ref["bbox"], track_id, color=(0, 255, 255))

            # ball: simple white circle at bbox center
            for _, ball in ball_dict.items():
                x1, y1, x2, y2 = map(int, ball["bbox"])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                cv2.circle(frame, (cx, cy), 8, (255, 255, 255), thickness=-1)

            output_video_frames.append(frame)

        return output_video_frames