import json
import numpy as np
import os
import pickle
import cv2
from glob import glob

def extract_video_frames(video_path, output_folder, frame_interval=1):
    """Extract frames from video"""
    os.makedirs(output_folder, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_interval == 0:
            frame_path = os.path.join(output_folder, f"frame_{saved_count:05d}.jpg")
            cv2.imwrite(frame_path, frame)
            saved_count += 1
            
        frame_count += 1
    
    cap.release()
    print(f"Extracted {saved_count} frames from video")
    return saved_count

def create_fixed_camera_transforms(camera_pkl_path, images_folder, output_json_path, total_frames):
    """
    Creates transforms.json for a FIXED camera watching a rotating subject
    """
    # Load camera parameters
    with open(camera_pkl_path, 'rb') as f:
        camera_data = pickle.load(f, encoding='latin1')
    
    # Get all image files
    image_files = sorted(glob(os.path.join(images_folder, "*.jpg")) + 
                        glob(os.path.join(images_folder, "*.png")))
    
    print(f"Found {len(image_files)} images")
    
    # Create transforms data
    transforms_data = {
        "w": camera_data['width'],
        "h": camera_data['height'],
        "fl_x": camera_data['camera_f'][0],
        "fl_y": camera_data['camera_f'][1],
        "cx": camera_data['camera_c'][0],
        "cy": camera_data['camera_c'][1],
        "k1": camera_data['camera_k'][0],
        "k2": camera_data['camera_k'][1],
        "p1": camera_data['camera_k'][2],
        "p2": camera_data['camera_k'][3],
        "k3": camera_data['camera_k'][4],
        "frames": []
    }
    
    # For a fixed camera, we need to ESTIMATE a reasonable camera position
    # Since camera.pkl doesn't have extrinsics, we'll create a plausible setup
    
    # Typical setup: camera looking at the center where person rotates
    # We'll position the camera at a reasonable distance facing the center
    camera_distance = 2.5  # meters - typical distance for human capture
    camera_height = 1.5    # meters - eye level height
    
    # Camera is fixed at this position, looking at the origin (where person rotates)
    camera_position = [0, camera_height, camera_distance]
    
    # Create a fixed camera-to-world transformation
    # This represents a camera looking at the center (0,0,0) from camera_position
    look_at_center = np.array([0, 0, 0])
    camera_pos = np.array(camera_position)
    
    # Calculate camera orientation vectors
    forward = look_at_center - camera_pos
    forward = forward / np.linalg.norm(forward)
    
    # Create a temporary up vector and calculate right vector
    temp_up = np.array([0, 1, 0])
    right = np.cross(temp_up, forward)
    right = right / np.linalg.norm(right)
    
    # Recalculate proper up vector
    up = np.cross(forward, right)
    
    # Build the camera-to-world transformation matrix
    transform_matrix = [
        [right[0], right[1], right[2], camera_pos[0]],
        [up[0], up[1], up[2], camera_pos[1]],
        [forward[0], forward[1], forward[2], camera_pos[2]],
        [0.0, 0.0, 0.0, 1.0]
    ]
    
    for i, img_path in enumerate(image_files):
        if i >= total_frames:
            break
            
        # Use filename without extension as image identifier
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        
        # Normalize time between 0.0 and 1.0
        time = i / max(1, total_frames - 1)
        
        frame = {
            "file_path": f"images/{img_name}",
            "transform_matrix": transform_matrix,  # SAME for all frames - camera is fixed!
            "time": float(time)  # Time changes - person is rotating
        }
        transforms_data["frames"].append(frame)
    
    # Write to file
    with open(output_json_path, 'w') as f:
        json.dump(transforms_data, f, indent=4)
    
    print(f"Created {output_json_path} with {len(transforms_data['frames'])} frames")
    print(f"Camera position: {camera_position}")
    print(f"Looking at: [0, 0, 0]")

def create_orbit_camera_transforms(camera_pkl_path, images_folder, output_json_path, total_frames):
    import json, pickle, os
    import numpy as np
    from glob import glob

    # Load camera parameters
    with open(camera_pkl_path, 'rb') as f:
        camera = pickle.load(f, encoding='latin1')

    # Intrinsics
    fl_x = float(camera['camera_f'][0])
    fl_y = float(camera['camera_f'][1])
    cx   = float(camera['camera_c'][0])
    cy   = float(camera['camera_c'][1])
    w    = int(camera['width'])
    h    = int(camera['height'])

    # Image files
    image_files = sorted(glob(os.path.join(images_folder, "*.jpg")) + 
                         glob(os.path.join(images_folder, "*.png")))
    print(f"Found {len(image_files)} images")

    # Camera orbit setup
    radius = 3.0
    center = np.array([0, 1.5, 0])   # look at person ~1.5m tall

    def pose(theta):
        cam_pos = np.array([radius * np.sin(theta), 1.5, radius * np.cos(theta)])
        forward = center - cam_pos
        forward = forward / np.linalg.norm(forward)
        right = np.cross([0,1,0], forward)
        right = right / np.linalg.norm(right)
        up = np.cross(forward, right)
        m = np.eye(4)
        m[:3,0] = right
        m[:3,1] = up
        m[:3,2] = forward
        m[:3,3] = cam_pos
        return m.tolist()

    # Build JSON
    transforms = {
        "w": w,
        "h": h,
        "fl_x": fl_x,
        "fl_y": fl_y,
        "cx": cx,
        "cy": cy,
        "frames": []
    }

    for i, img_path in enumerate(image_files):
        if i >= total_frames:
            break
        theta = 2.0 * np.pi * (i / total_frames)
        frame = {
            "file_path": f"images/{os.path.splitext(os.path.basename(img_path))[0]}",
            "transform_matrix": pose(theta),
            "time": float(i / max(1, total_frames - 1))
        }
        transforms["frames"].append(frame)

    with open(output_json_path, 'w') as f:
        json.dump(transforms, f, indent=4)

    print(f"Created {output_json_path} with {len(transforms['frames'])} frames")


# Step 1: Extract frames from video
video_path = "female-3-sport.mp4"
images_folder = "images"
total_frames = extract_video_frames(video_path, images_folder, frame_interval=1)

# Step 2: Create transform files with FIXED camera
#create_fixed_camera_transforms('camera.pkl', images_folder, 'transforms_train.json', total_frames) #was used first

create_orbit_camera_transforms('camera.pkl', images_folder, 'transforms_train.json', total_frames) #switched to this

# Create test transforms (sample every 10th frame)
def create_test_transforms(train_json_path, output_test_path, test_interval=10):
    """Create test transforms by sampling from train transforms"""
    with open(train_json_path, 'r') as f:
        train_data = json.load(f)
    
    test_data = train_data.copy()
    test_data['frames'] = []
    
    # Sample every nth frame for testing
    for i, frame in enumerate(train_data['frames']):
        if i % test_interval == 0:
            test_data['frames'].append(frame)
    
    with open(output_test_path, 'w') as f:
        json.dump(test_data, f, indent=4)
    
    print(f"Created {output_test_path} with {len(test_data['frames'])} test frames")


create_test_transforms('transforms_train.json', 'transforms_test.json', test_interval=10)