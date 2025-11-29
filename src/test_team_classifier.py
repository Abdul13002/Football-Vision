import sys
import os
# Add parent directory to path to import Views
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.team_classifier import get_jersey_crop, get_dominant_color
from Views import video_reader
import cv2

def test_team_classifier():
    """
    Test the jersey crop and dominant color functions
    """
    print("🧪 Testing Team Classifier...")
    
    # Step 1: Load a video frame
    print("\n1️⃣ Loading video...")
    video_frames = video_reader('/Users/abduladdan/Documents/football-cv-offline/videos/test (34).mp4')
    test_frame = video_frames[50]  # Use frame 50 for testing
    print(f"   ✅ Loaded frame shape: {test_frame.shape}")
    
    # Step 2: Create a fake bbox (you'll use real ones later)
    # Let's pretend we have a player at these coordinates
    test_bbox = [300, 200, 400, 350]  # [x1, y1, x2, y2]
    print(f"\n2️⃣ Testing with bbox: {test_bbox}")
    
    # Step 3: Test get_jersey_crop
    print("\n3️⃣ Testing get_jersey_crop()...")
    try:
        jersey_crop = get_jersey_crop(test_frame, test_bbox)
        print(f"   ✅ Jersey crop shape: {jersey_crop.shape}")
        
        # Save the cropped image to see what it looks like
        cv2.imwrite('test_jersey_crop.jpg', jersey_crop)
        print(f"   ✅ Saved jersey crop to: test_jersey_crop.jpg")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return
    
    # Step 4: Test get_dominant_color
    print("\n4️⃣ Testing get_dominant_color()...")
    try:
        dominant_color = get_dominant_color(jersey_crop, k=2)
        print(f"   ✅ Dominant color (BGR): {dominant_color}")
        
        # Convert BGR to RGB for display
        r, g, b = dominant_color[2], dominant_color[1], dominant_color[0]
        print(f"   ✅ Dominant color (RGB): ({r}, {g}, {b})")
        
        # Create a color swatch to visualize
        import numpy as np
        swatch = np.zeros((100, 100, 3), dtype=np.uint8)
        swatch[:] = dominant_color
        cv2.imwrite('test_dominant_color.jpg', swatch)
        print(f"   ✅ Saved color swatch to: test_dominant_color.jpg")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return
    
    print("\n✅ All tests passed! 🎉")

if __name__ == '__main__':
    test_team_classifier()