import sys
import os
import datetime
import pickle
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Views import video_reader, save_video
from track import tracking
from src.Player_ball_possesion import Player_ball_possesion
from src.pass_visualization import (
    analyze_passes_from_tracks,
    create_static_pass_map,
    create_sequential_pass_map,
    add_pass_stats_to_frame,
    calculate_player_average_positions
)


def load_video(video_path):
    import cv2
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    frames = video_reader(video_path)
    return frames, fps


def process_tracking(frames, model_path, stub_path, use_stub=False):
    tracker = tracking(model_path)
    tracks = tracker.object_tracking(frames, read_from_stub=use_stub, stub_path=stub_path)
    return tracker, tracks


def apply_team_assignment(tracker, frames, tracks):
    tracks = tracker.assign_teams(frames, tracks)
    return tracks


def interpolate_ball_positions(tracker, tracks):
    tracks["ball"] = tracker.ball_interpolation(tracks["ball"])
    return tracks


def assign_ball_possession(tracks):
    possession_detector = Player_ball_possesion()
    team_ball_control = []

    for frame_num, player_dict in enumerate(tracks["players"]):
        ball_data = tracks["ball"][frame_num].get(1, {})
        ball_box = ball_data.get("bbox")

        if ball_box:
            player_with_ball = possession_detector.assign_ball(player_dict, ball_box)
            if player_with_ball != -1:
                tracks["players"][frame_num][player_with_ball]["has_possesion"] = True
                team_ball_control.append(tracks['players'][frame_num][player_with_ball]['team'])
            else:
                team_ball_control.append(team_ball_control[-1] if team_ball_control else 0)
        else:
            team_ball_control.append(team_ball_control[-1] if team_ball_control else 0)

    tracks["team_ball_control"] = np.array(team_ball_control)
    return tracks


def calculate_team_possession(tracks):
    team_ball_control = tracks.get("team_ball_control", np.array([]))

    team_1_frames = int(np.sum(team_ball_control == 1))
    team_2_frames = int(np.sum(team_ball_control == 2))

    # Only count frames where possession was assigned (not 0)
    possession_frames = team_1_frames + team_2_frames

    if possession_frames == 0:
        return {
            "team_1_frames": 0,
            "team_2_frames": 0,
            "team_1_percentage": 0.0,
            "team_2_percentage": 0.0,
            "total_frames": len(team_ball_control)
        }

    team_1_pct = round((team_1_frames / possession_frames) * 100, 1)
    team_2_pct = round((team_2_frames / possession_frames) * 100, 1)

    return {
        "team_1_frames": team_1_frames,
        "team_2_frames": team_2_frames,
        "team_1_percentage": team_1_pct,
        "team_2_percentage": team_2_pct,
        "total_frames": possession_frames
    }


def save_tracks_to_stub(tracks, stub_path):
    with open(stub_path, 'wb') as f:
        pickle.dump(tracks, f)


def render_annotations(tracker, frames, tracks, possession_stats=None):
    annotated = tracker.annotations(frames, tracks, possession_stats)
    return annotated


def save_output_video(frames, output_dir, fps, timestamp):
    output_path = os.path.join(output_dir, f'output_{timestamp}.mp4')
    save_video(frames, output_path, fps)
    print(f"Video saved to: {output_path}")
    return output_path


def main():
    video_path = '/Users/abduladdan/Documents/football-cv-offline/videos/test (14).mp4'
    model_path = '/Users/abduladdan/Documents/football-cv-offline/Models/bestv2.pt'
    stub_path = '/Users/abduladdan/Documents/football-cv-offline/stubs/stubs.pkl'
    output_dir = '/Users/abduladdan/Documents/football-cv-offline/output_runs'

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    frames, fps = load_video(video_path)
    tracker, tracks = process_tracking(frames, model_path, stub_path, use_stub=False)
    tracks = apply_team_assignment(tracker, frames, tracks)
    tracks = interpolate_ball_positions(tracker, tracks)
    tracks = assign_ball_possession(tracks)
    possession_stats = calculate_team_possession(tracks)
     # Analyze passes and generate pass counts
    pass_analyzer = analyze_passes_from_tracks(tracks)
    pass_map_data = pass_analyzer.get_pass_map_data()
    pass_stats = pass_analyzer.get_team_pass_stats()

    # Calculate stable player positions across all frames
    player_avg_positions = calculate_player_average_positions(tracks)

    # Get all passes for sequential map
    all_passes = pass_analyzer.get_all_passes()

    # Generate pass maps with stable positions
    if pass_map_data:
        import cv2

        # Aggregated pass map (shows count per player pair)
        static_map = create_static_pass_map(frames[0].shape, pass_map_data, player_avg_positions)
        map_path = os.path.join(output_dir, f'pass_map_aggregated_{timestamp}.png')
        cv2.imwrite(map_path, static_map)
        print(f"pass map saved to: {map_path}")

        # Sequential pass map (shows passes numbered 1, 2, 3, etc.)
        sequential_map = create_sequential_pass_map(frames[0].shape, all_passes, player_avg_positions)
        seq_map_path = os.path.join(output_dir, f'pass_map_sequential_{timestamp}.png')
        cv2.imwrite(seq_map_path, sequential_map)
        print(f"Sequential pass map saved to: {seq_map_path}")

    save_tracks_to_stub(tracks, stub_path)
    annotated_frames = render_annotations(tracker, frames, tracks, possession_stats)
    if pass_stats:
        annotated_frames = [add_pass_stats_to_frame(frame, pass_stats) for frame in annotated_frames]

    save_output_video(annotated_frames, output_dir, fps, timestamp)


if __name__ == '__main__':
    main()