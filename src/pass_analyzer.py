import pandas as pd
import numpy as np
import sys
sys.path.append('../')
from Views import get_center_bbox

class PassAnalyzer:
    def __init__(self, min_possession_frames=1):
        # Initializes pass analyzer with minimum possession frame threshold
        self.active_pass = None
        self.passes_data = []
        self.min_possession_frames = min_possession_frames
        self.possession_tracker = {}

    def analyze_pass(self, frame_num, current_players, player_bboxes):
        # Detects pass start/end events by tracking possession changes and records pass data
        player_with_ball = next(
            (id for id, data in current_players.items() if data.get('has_possesion')),
            None
        )

        if player_with_ball is not None:
            if player_with_ball not in self.possession_tracker:
                self.possession_tracker[player_with_ball] = 1
            else:
                self.possession_tracker[player_with_ball] += 1
        else:
            self.possession_tracker = {}

        if self.active_pass:
            passer_id = self.active_pass['passer_id']

            if (player_with_ball is not None and
                player_with_ball != passer_id and
                self.possession_tracker.get(player_with_ball, 0) >= self.min_possession_frames):

                receiver_id = player_with_ball
                passer_team = self.active_pass['passer_team']
                receiver_team = current_players[receiver_id]['team']

                receiver_bbox = player_bboxes.get(receiver_id, {}).get('bbox')
                if receiver_bbox is None:
                    self.active_pass = None
                    return

                receiver_position = get_center_bbox(receiver_bbox)
                is_successful = (passer_team == receiver_team)

                pass_record = {
                    "passer_id": passer_id,
                    "receiver_id": receiver_id,
                    "passer_position": self.active_pass['passer_position'],
                    "receiver_position": receiver_position,
                    "team": passer_team,
                    "start_frame": self.active_pass['start_frame'],
                    "end_frame": frame_num,
                    "duration_frames": frame_num - self.active_pass['start_frame'],
                    "successful": is_successful
                }
                self.passes_data.append(pass_record)
                self.active_pass = None

        if (player_with_ball is not None and
            self.active_pass is None and
            self.possession_tracker.get(player_with_ball, 0) >= self.min_possession_frames):

            passer_id = player_with_ball
            passer_team = current_players[passer_id]['team']

            passer_bbox = player_bboxes.get(passer_id, {}).get('bbox')
            if passer_bbox is None:
                return

            passer_position = get_center_bbox(passer_bbox)

            self.active_pass = {
                "passer_id": passer_id,
                "passer_team": passer_team,
                "passer_position": passer_position,
                "start_frame": frame_num
            }

    def get_pass_map_data(self):
        # Aggregates successful passes between player pairs for visualization
        if not self.passes_data:
            return []

        df = pd.DataFrame(self.passes_data)
        successful = df[df['successful'] == True]

        if successful.empty:
            return []

        aggregated = []
        for (passer, receiver, team), group in successful.groupby(['passer_id', 'receiver_id', 'team']):
            avg_passer_pos = np.mean([p for p in group['passer_position']], axis=0)
            avg_receiver_pos = np.mean([p for p in group['receiver_position']], axis=0)

            aggregated.append({
                'passer_id': passer,
                'receiver_id': receiver,
                'team': team,
                'count': len(group),
                'avg_passer_position': tuple(avg_passer_pos),
                'avg_receiver_position': tuple(avg_receiver_pos),
                'avg_duration': group['duration_frames'].mean()
            })

        return aggregated

    def get_all_passes(self):
        # Returns raw pass data including successful and unsuccessful passes
        return self.passes_data

    def get_team_pass_stats(self):
        # Calculates pass accuracy and statistics for each team
        if not self.passes_data:
            return {}

        df = pd.DataFrame(self.passes_data)
        stats = {}

        for team in df['team'].unique():
            team_passes = df[df['team'] == team]
            successful = team_passes[team_passes['successful'] == True]

            stats[team] = {
                'total_passes': len(team_passes),
                'successful_passes': len(successful),
                'pass_accuracy': (len(successful) / len(team_passes) * 100) if len(team_passes) > 0 else 0,
                'avg_pass_duration': team_passes['duration_frames'].mean()
            }

        return stats