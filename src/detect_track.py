import sys
import os
# Add parent directory to path so we can import Views
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Views import video_reader, save_video
from track import tracking
def main():
    # Video reader
    video_frame = video_reader('/Users/abduladdan/Documents/football-cv-offline/videos/test (14).mp4')

    # Create tracker instance with your model
    tracker = tracking('/Users/abduladdan/Documents/football-cv-offline/Models/best.pt')

    # Run tracking
    tracks = tracker.object_tracking(video_frame, read_from_stub=True, stub_path='/Users/abduladdan/Documents/football-cv-offline/stubs/stubs.pkl')

    # Draw annotations (platform discs) on frames
    annotated_frames = tracker.annotations(video_frame, tracks)

    # save annotated video
    save_video(annotated_frames, '/Users/abduladdan/Documents/football-cv-offline/output_runs/output3.mp4' )




if __name__ == '__main__':
    main()