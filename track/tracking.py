from ultralytics import YOLO
import supervision as sv
import pickle
import os
import sys
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
        """
        Draw a 'platform disc' under the player using the bbox.

        bbox: [x1, y1, x2, y2]
        """
        x1, y1, x2, y2 = map(int, bbox)

        # bottom-center of bbox = approximate feet / ground contact point
        cx = (x1 + x2) // 2
        cy = y2

        overlay = frame.copy()

        # radius scaled to player width (min radius 8)
        radius = max(8, (x2 - x1) // 3)

        # filled base disc (semi-transparent)
        cv2.circle(overlay, (cx, cy), radius, color, thickness=-1)

        # outline ring around disc
        cv2.circle(overlay, (cx, cy), radius + 2, color, thickness=2)

        # small vertical connector line (from feet up into body)
        line_height = max(10, (y2 - y1) // 4)
        cv2.line(overlay, (cx, cy), (cx, cy - line_height), color, thickness=2)

        # alpha blending for glow effect
        alpha = 0.4
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # optional: draw track ID slightly above head
        label = str(track_id)
        label_y = max(0, y1 - 10)
        cv2.putText(
            frame,
            label,
            (cx - 10, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA
        )

        return frame   



    def annotations(self, video_frames, tracks):
        output_video_frames = []

        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            # get dicts for this frame
            player_dict = tracks["players"][frame_num]
            referee_dict = tracks["referees"][frame_num]
            ball_dict = tracks["ball"][frame_num]

            # players: green disc
            for track_id, player in player_dict.items():
                frame = self.draw_ellpse(frame, player["bbox"], track_id, color=(0, 255, 0))

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