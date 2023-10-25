"""
Slices a given video into single frames and saves them in a given directory.
"""

import cv2
import os
from tqdm import tqdm

WORKING_DIR = os.getcwd()
OUTPUT_DIR = os.path.join(WORKING_DIR, "frames")
MAX_FRAMES_PER_DIR = 1000


def main(video_file: str, frame_rate: int):
    """
    Extracts frames from a given video file and
    saves them in a given directory.
    
    Before extraction the user is shown the estimated
    number of frames that will be extracted and their
    total size.
    """

    # Create output directory if it does not exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Open video file
    cap = cv2.VideoCapture(video_file)

    # Get video file name
    video_name = os.path.basename(video_file).split('.')[0]

    # Get total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate number of frames to be extracted
    # based on the given frame rate
    num_frames = int(total_frames / frame_rate)

    # Calculate total size of frames to be extracted
    # t = num_frames * x * y * y / z / z
    # t = ( num_frames * x * y² * z ) / z
    total_size = round(num_frames * 3 * 224 * 224 / 1024 / 1024, 2)

    # Show user the estimated number of frames and their total size
    print(f"Estimated number of frames: {num_frames}")
    print(f"Estimated total size: {total_size} MB")

    # Create a directory for the frames
    # and a counter for the frames
    dir_counter = 0
    frame_counter = 0

    # Loop over all frames
    for i in tqdm(range(total_frames), desc="Extracting frames", total=total_frames, position=0, leave=True):
        # Read frame
        ret, frame = cap.read()

        # Skip frame if it could not be read
        if not ret:
            continue

        # Skip frame if it is not the next frame to be extracted
        if i % frame_rate != 0:
            continue

        # Create a new directory if the current one is full
        if frame_counter % MAX_FRAMES_PER_DIR == 0:
            dir_counter += 1
            os.makedirs(os.path.join(OUTPUT_DIR, f"{video_name}_{dir_counter}"))

        # Save frame
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{video_name}_{dir_counter}", f"{video_name}_{frame_counter}.jpg"), frame)

        # Increment frame counter
        frame_counter += 1

    # Release video file
    cap.release()


if __name__ == '__main__':
    video_file = input("Enter video file path: ")

    # Open video file
    cap = cv2.VideoCapture(video_file)

    # Get frame rate
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

    # Release video file
    cap.release()

    main(video_file, frame_rate)

