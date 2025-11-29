import sys
import os
import datetime
# Add parent directory to path so we can import Views
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Views import video_reader, save_video
from track import tracking
def main():
    # Video reader
    video_frame = video_reader('/Users/abduladdan/Documents/football-cv-offline/videos/test (34).mp4')

    # Create tracker instance with  model
    tracker = tracking('/Users/abduladdan/Documents/football-cv-offline/Models/best.pt')

    # Run tracking
    tracks = tracker.object_tracking(video_frame, read_from_stub=False, stub_path='/Users/abduladdan/Documents/football-cv-offline/stubs/stubs.pkl')

    # Draw annotations 
    annotated_frames = tracker.annotations(video_frame, tracks)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f'/Users/abduladdan/Documents/football-cv-offline/output_runs/output_{timestamp}.mp4'

    # save 
    save_video(annotated_frames, output_path)
    print(f"Video saved to: {output_path}")




if __name__ == '__main__':
    main()