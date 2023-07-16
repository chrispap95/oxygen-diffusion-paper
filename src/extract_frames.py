import json
import os
import subprocess

# Your top level directory
folder = "/Users/chrispap/Downloads/O2_videos"

# The timestamps of the original videos
timestamps = json.load(open("timestamps_original_videos.json"))

# Create output directory
output_dir = "/Users/chrispap/Documents/Sept2020_alpha/radiationDamage/AlphaSource/O2_videos_frames"


def get_timestamp(video_path):
    rod_name = video_path.split("/")[-2]
    face_name = video_path.split("/")[-1][:-4]
    return timestamps[rod_name][face_name]


for subdir, _dirs, files in os.walk(folder):
    for file in files:
        if file.endswith(".MOV"):
            if "skip" in file:
                continue

            video_path = os.path.join(subdir, file)

            faces = get_timestamp(video_path)

            if "FaceAC" not in video_path and "FaceBD" not in video_path:
                faces = {file[:-4]: faces}

            for face in faces:
                output_path = subdir.split("/")[-1]
                output_full = os.path.join(output_dir, output_path, face)

                os.makedirs(output_full, exist_ok=True)

                timestamp = faces[face]

                for i, ts in enumerate(timestamp):
                    subprocess.call(
                        [
                            "ffmpeg",
                            "-ss",
                            ts,
                            "-i",
                            video_path,
                            "-frames:v",
                            "1",
                            os.path.join(output_full, f"frame{i}.png"),
                        ],
                    )
