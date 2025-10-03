"""
Data replay script for OmniGibson HDF5 demonstrations with scene graph generation.

This script replays HDF5 demonstration files, generates videos from multiple camera angles,
and optionally computes scene graphs and QA metrics.
"""

try:
    import omnigibson as og
except ImportError:
    raise ImportError("omnigibson not found, please install it first!")

from omnigibson.envs import SceneGraphDataPlaybackWrapper, DataPlaybackWrapper
from omnigibson.utils.scene_graph_utils import SceneGraphWriter
from omnigibson.macros import gm
from gello.robots.sim_robot.og_teleop_utils import optimize_sim_settings
from gello.utils.qa_utils import *

import torch as th
import os
import argparse
import sys
import inspect


# Configuration flags (will be set via command-line arguments)
RUN_SCENE_GRAPH = True

# Viewer settings for minimal overhead
gm.RENDER_VIEWER_CAMERA = False
gm.DEFAULT_VIEWER_WIDTH = 128
gm.DEFAULT_VIEWER_HEIGHT = 128


def extract_arg_names(func):
    """Extract parameter names from a function signature."""
    return list(inspect.signature(func).parameters.keys())




def replay_hdf5_file(hdf_input_path, output_dir):
    """
    Replay a single HDF5 demonstration file and generate videos and scene graphs.
    
    Args:
        hdf_input_path: Path to the HDF5 file to replay
        output_dir: Root directory where task folder will be created
    """
    # Create output folder with same name as HDF5 file (without extension)
    base_name = os.path.basename(hdf_input_path)
    folder_name = os.path.splitext(base_name)[0]
    folder_path = os.path.join(output_dir, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    
    # Define output paths
    hdf_output_path = os.path.join(folder_path, f"{folder_name}_replay.hdf5")
    video_dir = folder_path
    
    # Camera resolution settings
    RESOLUTION_DEFAULT = 512
    RESOLUTION_WRIST = 240
    
    # Disable transition rules for data playback
    gm.ENABLE_TRANSITION_RULES = False
    
    # External camera positions and orientations [position, orientation(quaternion)]
    external_camera_poses = [
        [[-0.4, 0, 2.0], [0.2706, -0.2706, -0.6533, 0.6533]],
    ]
    
    # Robot wrist camera configuration
    robot_sensor_config = {
        "VisionSensor": {
            "modalities": ["rgb", "seg_instance"],
            "sensor_kwargs": {
                "image_height": RESOLUTION_WRIST,
                "image_width": RESOLUTION_WRIST,
            },
        },
    }
    
    # Build external sensors configuration
    external_sensors_config = []
    for i, (position, orientation) in enumerate(external_camera_poses):
        external_sensors_config.append({
            "sensor_type": "VisionSensor",
            "name": f"external_sensor{i}",
            "relative_prim_path": f"/controllable__r1pro__robot_r1/base_link/external_sensor{i}",
            "modalities": ["rgb", "seg_instance"],
            "sensor_kwargs": {
                "image_height": RESOLUTION_DEFAULT,
                "image_width": RESOLUTION_DEFAULT,
                "horizontal_aperture": 40.0,
            },
            "position": th.tensor(position, dtype=th.float32),
            "orientation": th.tensor(orientation, dtype=th.float32),
            "pose_frame": "parent",
        })

    # Add robot head camera (ZED camera)
    idx = 1
    external_sensors_config.append({
        "sensor_type": "VisionSensor",
        "name": f"external_sensor{idx}",
        "relative_prim_path": f"/controllable__r1pro__robot_r1/zed_link/external_sensor{idx}",
        "modalities": ["rgb", "seg_instance"],
        "sensor_kwargs": {
            "image_height": RESOLUTION_DEFAULT,
            "image_width": RESOLUTION_DEFAULT,
            "horizontal_aperture": 40.0,
        },
        "position": th.tensor([0.06, 0.01, 0.15], dtype=th.float32),
        "orientation": th.tensor([-1.0, 0.0, 0.0, 0.0], dtype=th.float32),
        "pose_frame": "parent",
    })

    # Extract task name from HDF5 filename
    task_name = hdf_input_path.split("/")[-1].split("_")[:-1]
    task_name = "_".join(task_name)


    # Locate full scene file for the task
    task_scene_file_folder = os.path.join(
        os.path.dirname(os.path.dirname(og.__path__[0])), "joylo", "sampled_task", task_name
    )

    full_scene_file = None
    for file in os.listdir(task_scene_file_folder):
        if file.endswith(".json") and "partial_rooms" not in file:
            full_scene_file = os.path.join(task_scene_file_folder, file)
    assert full_scene_file is not None, f"No full scene file found in {task_scene_file_folder}"

    # Create environment with appropriate wrapper
    common_params = {
        "input_path": hdf_input_path,
        "output_path": hdf_output_path,
        "robot_obs_modalities": ["rgb", "seg_instance"],
        "robot_sensor_config": robot_sensor_config,
        "external_sensors_config": external_sensors_config,
        "exclude_sensor_names": ["zed"],
        "only_successes": False,
        "include_task": True,
        "include_task_obs": False,
        "include_robot_control": False,
        "include_contacts": True,
        "full_scene_file": full_scene_file,
    }
    
    if RUN_SCENE_GRAPH:
        env = SceneGraphDataPlaybackWrapper.create_from_hdf5(
            **common_params,
            n_render_iterations=1,
        )
    else:
        env = DataPlaybackWrapper.create_from_hdf5(
            **common_params,
            n_render_iterations=3,
        )

    # Optimize rendering settings
    og.sim.add_callback_on_play("optimize_rendering", optimize_sim_settings)
    # Setup video and frame writers
    video_writers = []
    video_rgb_keys = []
    frame_writers = []
    frame_rgb_keys = []
    
    # Create video/frame writers for external cameras (skip camera 0)
    for i in range(len(external_sensors_config)):
        if i == 0:
            continue
        camera_name = f"external_sensor{i}"
        video_writers.append(env.create_video_writer(fpath=f"{video_dir}/{camera_name}.mp4"))
        video_rgb_keys.append(f"external::{camera_name}::rgb")
        frame_writers.append(env.create_frame_writer(output_dir=f"{video_dir}/{camera_name}/"))
        frame_rgb_keys.append(f"external::{camera_name}::rgb")

    # Validate single episode format
    assert len(env.input_hdf5["data"].keys()) == 1, \
        f"Only one episode is supported for now, got {len(env.input_hdf5['data'].keys())} from {hdf_input_path}"

    # Replay configuration
    replay_config = {
        "record_visibility": True,
        "record_rgb_keys": ["external::external_sensor1::rgb"],
        "sensors": ["external_sensor1"],
    }

    start_frame = None
    end_frame = None

    # Playback all episodes
    for episode_id in range(env.input_hdf5["data"].attrs["n_episodes"]):
        if RUN_SCENE_GRAPH:
            scene_graph_writer = SceneGraphWriter(
                output_path=os.path.join(folder_path, f"scene_graph_{episode_id}.json"),
                interval=200,
                buffer_size=200,
                write_full_graph_only=True
            )
            env.playback_episode(
                episode_id=episode_id,
                record_data=False,
                video_writers=video_writers,
                video_rgb_keys=video_rgb_keys,
                frame_writers=None,
                frame_rgb_keys=None,
                start_frame=start_frame,
                end_frame=end_frame,
                scene_graph_writer=scene_graph_writer,
                replay_config=replay_config,
            )
        else:
            env.playback_episode(
                episode_id=episode_id,
                record_data=False,
                video_writers=video_writers,
                video_rgb_keys=video_rgb_keys
            )
    
    # Close all video writers
    for writer in video_writers:
        writer.close()

    # Clean up environment
    og.clear()
    
    print(f"Successfully processed {hdf_input_path}")


def main():
    """Main function to parse arguments and replay HDF5 demonstration files."""
    parser = argparse.ArgumentParser(
        description="Replay OmniGibson HDF5 demonstration files and generate videos/scene graphs"
    )
    parser.add_argument("--file", help="HDF5 file to process")
    parser.add_argument("--output_dir", default="data/replayed_activities", 
                        help="Root directory for output (default: data/replayed_activities)")
    parser.add_argument("--scene_graph", action="store_true", default=True,
                        help="Enable scene graph generation (default: True)")
    parser.add_argument("--no_scene_graph", action="store_false", dest="scene_graph",
                        help="Disable scene graph generation")
    parser.add_argument("--headless", action="store_true", default=True,
                        help="Run in headless mode (default: True)")
    parser.add_argument("--no_headless", action="store_false", dest="headless",
                        help="Disable headless mode")

    
    args = parser.parse_args()
    
    # Set global configuration flags based on arguments
    global RUN_SCENE_GRAPH
    RUN_SCENE_GRAPH = args.scene_graph
    gm.HEADLESS = args.headless
    
    if not args.file:
        parser.print_help()
        print("\nError: --file must be specified", file=sys.stderr)
        return
    
    # Check if file exists
    if not os.path.exists(args.file):
        print(f"Error: File {args.file} does not exist", file=sys.stderr)
        return
    
    # Process the file
    replay_hdf5_file(args.file, args.output_dir)

    og.shutdown()


if __name__ == "__main__":
    main()
