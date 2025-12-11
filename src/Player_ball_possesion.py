import sys
sys.path.append('../')
from Views import get_center_bbox, get_width, foot_distance_measure

class Player_ball_possesion():
    def __init__(self):
        self.max_distance_from_ball = 25

    def assign_ball(self, players, ball_bbox):
        ball_center = get_center_bbox(ball_bbox)
        min_distance = 10000
        assigneed_player_id = -1

        for player_id, player_data in players.items():
            box = player_data['bbox']
            left_x = box[0]
            right_x = box[2]
            bottom_y = box[-1]
            left_foot = foot_distance_measure((left_x, bottom_y), ball_center)
            Right_foot = foot_distance_measure((right_x, bottom_y), ball_center)
            foot_distance = min(left_foot, Right_foot)
            if foot_distance < self.max_distance_from_ball:
                if foot_distance < min_distance:
                    min_distance = foot_distance
                    assigneed_player_id = player_id

        return assigneed_player_id
                




