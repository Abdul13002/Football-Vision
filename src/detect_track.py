import sys
import os
# Add parent directory to path so we can import Views
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Views import video_reader, save_video
def main():
    # Video reader 
    video_frame = video_reader('/Users/abduladdan/Documents/football-cv-offline/videos/test (14).mp4')

    # save function
    save_video(video_frame, '/Users/abduladdan/Documents/football-cv-offline/output_runs/output2.mp4' )




if __name__ == '__main__':
    main()