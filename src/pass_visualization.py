import cv2
import numpy as np
import sys
sys.path.append('../')
from Views import get_center_bbox
from src.pass_analyzer import PassAnalyzer

def get_player_base_position(bbox):
    # Returns player foot position (x_center, y_bottom) for stable pass map nodes
    x1, y1, x2, y2 = map(int, bbox)
    x_center = (x1 + x2) // 2
    y_base = y2
    return (x_center, y_base)


def calculate_player_average_positions(tracks):
    # Calculates average position for each player across all frames
    player_positions = {}

    for player_dict in tracks['players']:
        for player_id, player_data in player_dict.items():
            bbox = player_data.get('bbox')
            if bbox is not None:
                x_center, y_base = get_player_base_position(bbox)

                if player_id not in player_positions:
                    player_positions[player_id] = []

                player_positions[player_id].append(np.array([x_center, y_base]))

    avg_positions = {}
    for player_id, positions in player_positions.items():
        avg_positions[player_id] = tuple(np.mean(positions, axis=0))

    return avg_positions


def analyze_passes_from_tracks(tracks):
    # Analyzes passes from tracks with possession data
    pass_analyzer = PassAnalyzer(min_possession_frames=3)

    for frame_num in range(len(tracks['players'])):
        player_dict = tracks['players'][frame_num]

        current_players = {}
        player_bboxes = {}

        for player_id, player_data in player_dict.items():
            has_ball = player_data.get('has_possesion', False)
            team = player_data.get('team')
            bbox = player_data.get('bbox')

            if team is not None and bbox is not None:
                current_players[player_id] = {
                    'team': team,
                    'has_possesion': has_ball
                }
                player_bboxes[player_id] = {'bbox': bbox}

        if current_players:
            pass_analyzer.analyze_pass(frame_num, current_players, player_bboxes)

    return pass_analyzer


def draw_pass_map_on_frame(frame, pass_map_data, player_avg_positions, team_colors=None):
    # Draws pass arrows on frame with thickness representing pass frequency
    if team_colors is None:
        team_colors = {
            1: (255, 0, 0),    # Team 1: Blue
            2: (0, 0, 255)     # Team 2: Red
        }

    overlay = frame.copy()

    for pass_data in pass_map_data:
        passer_id = pass_data['passer_id']
        receiver_id = pass_data['receiver_id']

        # Use stable average positions instead of pass-event positions
        start_pos_arr = player_avg_positions.get(passer_id)
        end_pos_arr = player_avg_positions.get(receiver_id)

        if start_pos_arr is None or end_pos_arr is None:
            continue

        start_pos = tuple(map(int, start_pos_arr))
        end_pos = tuple(map(int, end_pos_arr))

        team = pass_data['team']
        count = pass_data['count']
        color = team_colors.get(team, (255, 255, 255))
        thickness = min(2 + count // 2, 8)

        # Draw arrow
        cv2.arrowedLine(overlay, start_pos, end_pos, color, thickness, tipLength=0.3)

        # Draw player nodes
        cv2.circle(overlay, start_pos, 10, color, -1)
        cv2.circle(overlay, end_pos, 10, color, -1)

        # Draw pass count
        mid_x = (start_pos[0] + end_pos[0]) // 2
        mid_y = (start_pos[1] + end_pos[1]) // 2
        cv2.circle(overlay, (mid_x, mid_y), 15, (0, 0, 0), -1)
        cv2.putText(overlay, str(count), (mid_x - 7, mid_y + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    alpha = 0.7
    result = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    return result


def create_static_pass_map(frame_shape, pass_map_data, player_avg_positions, team_colors=None):
    # Uses stable player positions to create clean pass map visualization
    if team_colors is None:
        team_colors = {
            1: (255, 0, 0),
            2: (0, 0, 255)
        }

    canvas = np.ones((frame_shape[0], frame_shape[1], 3), dtype=np.uint8) * 255

    for pass_data in pass_map_data:
        passer_id = pass_data['passer_id']
        receiver_id = pass_data['receiver_id']

        # Use stable average positions instead of pass-event positions
        start_pos_arr = player_avg_positions.get(passer_id)
        end_pos_arr = player_avg_positions.get(receiver_id)

        if start_pos_arr is None or end_pos_arr is None:
            continue

        start_pos = tuple(map(int, start_pos_arr))
        end_pos = tuple(map(int, end_pos_arr))

        team = pass_data['team']
        count = pass_data['count']
        color = team_colors.get(team, (0, 0, 0))
        thickness = min(2 + count // 2, 10)

        # Draw arrow and nodes
        cv2.arrowedLine(canvas, start_pos, end_pos, color, thickness, tipLength=0.3)
        cv2.circle(canvas, start_pos, 8, color, -1)
        cv2.circle(canvas, end_pos, 8, color, -1)

        # Draw pass count
        mid_x = (start_pos[0] + end_pos[0]) // 2
        mid_y = (start_pos[1] + end_pos[1]) // 2
        cv2.circle(canvas, (mid_x, mid_y), 18, (50, 50, 50), -1)
        cv2.putText(canvas, str(count), (mid_x - 8, mid_y + 6),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return canvas


def create_sequential_pass_map(frame_shape, all_passes, player_avg_positions, team_colors=None):
    # Shows passes in chronological order numbered 1, 2, 3, etc.
    if team_colors is None:
        team_colors = {
            1: (255, 0, 0),
            2: (0, 0, 255)
        }

    canvas = np.ones((frame_shape[0], frame_shape[1], 3), dtype=np.uint8) * 255

    # Filter only successful passes and sort by start frame
    successful_passes = [p for p in all_passes if p['successful']]
    successful_passes.sort(key=lambda x: x['start_frame'])

    for idx, pass_data in enumerate(successful_passes, start=1):
        passer_id = pass_data['passer_id']
        receiver_id = pass_data['receiver_id']

        # Use stable average positions
        start_pos_arr = player_avg_positions.get(passer_id)
        end_pos_arr = player_avg_positions.get(receiver_id)

        if start_pos_arr is None or end_pos_arr is None:
            continue

        start_pos = tuple(map(int, start_pos_arr))
        end_pos = tuple(map(int, end_pos_arr))

        team = pass_data['team']
        color = team_colors.get(team, (0, 0, 0))

        # Draw arrow and nodes
        cv2.arrowedLine(canvas, start_pos, end_pos, color, 3, tipLength=0.3)
        cv2.circle(canvas, start_pos, 8, color, -1)
        cv2.circle(canvas, end_pos, 8, color, -1)

        # Draw sequence number
        mid_x = (start_pos[0] + end_pos[0]) // 2
        mid_y = (start_pos[1] + end_pos[1]) // 2
        cv2.circle(canvas, (mid_x, mid_y), 18, (50, 50, 50), -1)
        cv2.putText(canvas, str(idx), (mid_x - 8, mid_y + 6),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return canvas


def add_pass_stats_to_frame(frame, pass_stats):
    # Adds pass statistics overlay to frame showing team accuracy
    y_offset = 150

    for team, stats in pass_stats.items():
        team_name = f"Team {team}"
        accuracy_text = f"{team_name}: {stats['pass_accuracy']:.1f}% accuracy"
        passes_text = f"  ({stats['successful_passes']}/{stats['total_passes']} passes)"

        color = (255, 0, 0) if team == 1 else (0, 0, 255)

        cv2.putText(frame, accuracy_text, (35, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame, passes_text, (35, y_offset + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        y_offset += 60

    return frame