from ultralytics import YOLO
import supervision as sv
import pickle
import os
import sys
import pandas as pd
import cv2
import numpy as np
sys.path.append('../')
from Views import get_center_bbox, get_width
from src.team_assignment import TeamAssigner
class tracking:
    def __init__(self,model_path):
        self.model = YOLO(model_path)
        
        self.tracker = sv.ByteTrack()

    def ball_interpolation(self, ball_cords):
        ball_cords = [x.get(1,{}).get('bbox',[]) for x in ball_cords]
        df_ball_cords = pd.DataFrame(ball_cords,columns=['x1','y1','x2','y2'])

        #missing values handle
        df_ball_cords = df_ball_cords.interpolate()
        df_ball_cords = df_ball_cords.bfill()

        ball_cords = [{1: {'bbox': x}} for x in df_ball_cords.to_numpy().tolist()]

        return ball_cords

    def Frame_detection(self, frames):
        batch_size = 5
        Detected = []
        for i in range (0, len(frames), batch_size):
            
            Total_Detected = self.model.predict(
                frames[i:i+batch_size],
                conf=0.1,
                half=True,
                device='mps',
                iou=0.5,
                agnostic_nms=True
            )
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

    def assign_teams(self, frames, tracks):
        team_assigner = TeamAssigner()

        # Find first frame with players and assign team colors
        for frame_num, player_track in enumerate(tracks['players']):
            if len(player_track) > 0:
                team_assigner.assign_team_color(frames[frame_num], player_track)
                break

        # Assign teams to all players
        for frame_num, player_track in enumerate(tracks['players']):
            for player_id, track in player_track.items():
                team = team_assigner.get_player_team(frames[frame_num], track['bbox'], player_id)
                tracks['players'][frame_num][player_id]['team'] = team
                # Only assign team_color if team_colors were successfully set
                if team in team_assigner.team_colors:
                    tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]
        
        for frame_num, referee_track in enumerate(tracks['referees']):
            for referee_id, track in referee_track.items():
                tracks['referees'][frame_num][referee_id]['team'] = 0  # 0 or None for no team
                tracks['referees'][frame_num][referee_id]['color'] = (0, 255, 255) # Yellow BGR

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

    def draw_possession_stats(self, frame, possession_stats):
        if possession_stats is None:
            return frame

        BOX_WIDTH = 280
        BOX_HEIGHT = 100
        BOX_X_START = 10
        BOX_Y_START = 10
        BOX_X_END = BOX_X_START + BOX_WIDTH
        BOX_Y_END = BOX_Y_START + BOX_HEIGHT
        
        TEAM_1_COLOR = (255, 0, 0) # Blue
        TEAM_2_COLOR = (0, 0, 255) # Red
        TEXT_COLOR = (255, 255, 255) # White

        overlay = frame.copy()
        cv2.rectangle(
            overlay, 
            (BOX_X_START, BOX_Y_START), 
            (BOX_X_END, BOX_Y_END), 
            (0, 0, 0), # Black color
            thickness=-1
        )
        alpha = 0.5 
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        cv2.rectangle(
            frame, 
            (BOX_X_START, BOX_Y_START), 
            (BOX_X_END, BOX_Y_END), 
            (255, 255, 255), # White color
            thickness=2
        )


        cv2.putText(
            frame, 
            "POSSESSION", 
            (BOX_X_START + 10, BOX_Y_START + 25),
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            TEXT_COLOR, 
            2
        )

        team1_percent = possession_stats.get('team_1_percentage', 0.0)
        team2_percent = possession_stats.get('team_2_percentage', 0.0)
        
        team1_text = f"Team 1: {team1_percent:.1f}%"
        cv2.putText(
            frame, 
            team1_text, 
            (BOX_X_START + 10, BOX_Y_START + 55),
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            TEAM_1_COLOR, 
            2
        )

        team2_text = f"Team 2: {team2_percent:.1f}%"
        cv2.putText(
            frame, 
            team2_text, 
            (BOX_X_END - 100, BOX_Y_START + 55),
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            TEAM_2_COLOR, 
            2
        )
        
        BAR_Y = BOX_Y_START + 75
        BAR_HEIGHT = 15
        BAR_X_MARGIN = 10
        BAR_LENGTH = BOX_WIDTH - (2 * BAR_X_MARGIN)
        
        team1_bar_width = int(BAR_LENGTH * (team1_percent / 100.0))
        
        bar_x1 = BOX_X_START + BAR_X_MARGIN
        bar_x2 = bar_x1 + team1_bar_width
        bar_x3 = BOX_X_END - BAR_X_MARGIN
        
        cv2.rectangle(
            frame,
            (bar_x1, BAR_Y),
            (bar_x2, BAR_Y + BAR_HEIGHT),
            TEAM_1_COLOR,
            thickness=-1
        )

        cv2.rectangle(
            frame,
            (bar_x2, BAR_Y),
            (bar_x3, BAR_Y + BAR_HEIGHT),
            TEAM_2_COLOR,
            thickness=-1
        )
        
        cv2.rectangle(
            frame,
            (bar_x1, BAR_Y),
            (bar_x3, BAR_Y + BAR_HEIGHT),
            (255, 255, 255),
            thickness=1
        )

        return frame

    def annotations(self, video_frames, tracks, possession_stats=None):


        output_video_frames = []

        # Define team colors (BGR format)
        team_colors = {
            1: (255, 0, 0),    # Team 1: Blue
            2: (0, 0, 255)     # Team 2: Red
        }

        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            referee_dict = tracks["referees"][frame_num]
            ball_dict = tracks["ball"][frame_num]

            # players: color based on team or possession
            for track_id, player in player_dict.items():
                # Check if player has possession
                if player.get("has_possesion", False):
                    color = (255, 255, 255)  # White for ball possession
                else:
                    # Get team color, default to green if no team assigned
                    team = player.get("team", None)
                    if team is not None and team in team_colors:
                        color = team_colors[team]
                    else:
                        color = (0, 255, 0)  

                frame = self.draw_ellpse(frame, player["bbox"], track_id, color=color)

            # referees: yellow disc
            for track_id, ref in referee_dict.items():
                frame = self.draw_ellpse(frame, ref["bbox"], track_id, color=(0, 255, 255))

            # ball: simple white circle at bbox center
            for _, ball in ball_dict.items():
                x1, y1, x2, y2 = map(int, ball["bbox"])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                cv2.circle(frame, (cx, cy), 8, (255, 255, 255), thickness=-1)

            # Draw possession stats overlay
            frame = self.draw_possession_stats(frame, possession_stats)

            output_video_frames.append(frame)

        return output_video_frames

    