"""
Frame extraction script for replayed activity videos.

This script extracts PNG frames from MP4 video files in replayed activity folders.
It automatically detects whether you're processing a single task or multiple tasks.

Usage Examples:
    
    # Single task mode - extract frames from one specific task
    python scripts/helpers/frame_extraction.py --task_folder data/replayed_activities/bringing_water_1750844141719178
    
    # Batch mode - extract frames from ALL tasks in the root folder
    python scripts/helpers/frame_extraction.py --task_folder data/replayed_activities
    
    # With skip_existing flag to avoid reprocessing
    python scripts/helpers/frame_extraction.py --task_folder data/replayed_activities --skip_existing
    
    # Custom video name and chunk size
    python scripts/helpers/frame_extraction.py --task_folder data/replayed_activities --video_name external_sensor1 --chunk_size 2100

Output:
    - For each task folder with an external_sensor1.mp4 file
    - Creates a subfolder named "external_sensor1" in the same task folder
    - Extracts all video frames as PNG files (00001.png, 00002.png, etc.)
"""

import cv2
import multiprocessing as mp
import os
import argparse
import sys

def save_frame(args):
    """
    Worker function for multiprocessing to save a single frame.
    
    Args:
        args: Tuple of (frame, frame_id, output_dir)
    """
    frame, frame_id, output_dir = args
    filename = os.path.join(output_dir, f"{frame_id:05d}.png")
    cv2.imwrite(filename, frame)


def decompose_video_parallel(video_path, output_folder, base_frame_id=0, chunk_size=2100):
    """
    Decompose a video into PNG frames using parallel processing with chunking.
    
    Args:
        video_path: Path to the input video file
        output_folder: Directory to save extracted frames
        base_frame_id: Starting frame ID for numbering (default: 0)
        chunk_size: Number of frames to process in each chunk to avoid memory issues (default: 2100)
    """
    if base_frame_id is None:
        base_frame_id = 0
    base_frame_id = base_frame_id + 1
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")

    # Get video properties
    vid_capture = cv2.VideoCapture(video_path)
    if not vid_capture.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return
    
    total_frames = int(vid_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames to process: {total_frames}")
    
    num_cores = mp.cpu_count() - 2
    print(f"Using {num_cores} cores for video decomposition with chunk size: {chunk_size}")

    frame_id = base_frame_id
    processed_frames = 0
    
    while True:
        # Process frames in chunks to avoid memory explosion
        chunk_tasks = []
        
        # Read a chunk of frames
        for _ in range(chunk_size):
            success, frame = vid_capture.read()
            if not success:
                break
            chunk_tasks.append((frame.copy(), frame_id, output_folder))  # .copy() to avoid reference issues
            frame_id += 1
        
        if not chunk_tasks:
            break
            
        # Process this chunk in parallel
        with mp.Pool(processes=num_cores) as pool:
            pool.map(save_frame, chunk_tasks)
        
        processed_frames += len(chunk_tasks)
        print(f"Processed {processed_frames}/{total_frames} frames ({processed_frames/total_frames*100:.1f}%)")
        
        # Clear the chunk from memory
        del chunk_tasks
    
    # close video capture
    vid_capture.release()
    
    print(f"Successfully decomposed {processed_frames} frames into {output_folder}")


def extract_frames_from_task(task_folder, video_name="external_sensor1", chunk_size=2100, skip_existing=False):
    """
    Extract frames from a video file in a task folder.
    
    Args:
        task_folder: Path to the task folder (e.g., data/replayed_activities/task_name)
        video_name: Name of the video file (without extension, default: "external_sensor1")
        chunk_size: Number of frames to process in each chunk (default: 2100)
        skip_existing: If True, skip folders where frames already exist (default: False)
    
    Returns:
        bool: True if successful, False otherwise
    """
    # Validate task folder exists
    if not os.path.exists(task_folder):
        print(f"Error: Task folder does not exist: {task_folder}")
        return False
    
    if not os.path.isdir(task_folder):
        print(f"Error: Path is not a directory: {task_folder}")
        return False
    
    # Build video path
    video_path = os.path.join(task_folder, f"{video_name}.mp4")
    
    # Check if video file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return False
    
    # Create output folder for frames
    output_folder = os.path.join(task_folder, video_name)
    
    # Check if output folder already exists
    if os.path.exists(output_folder):
        if skip_existing:
            print(f"Skipping (frames already exist): {os.path.basename(task_folder)}")
            return True
        else:
            print(f"Warning: Output folder already exists: {output_folder}")
            response = input("Do you want to continue and overwrite existing frames? (y/n): ")
            if response.lower() != 'y':
                print("Extraction cancelled.")
                return False
    
    print(f"\nExtracting frames from: {video_path}")
    print(f"Output folder: {output_folder}")
    print("-" * 80)
    
    # Extract frames
    decompose_video_parallel(
        video_path=video_path,
        output_folder=output_folder,
        base_frame_id=0,
        chunk_size=chunk_size
    )
    
    print("-" * 80)
    print(f"Frame extraction complete!")
    
    return True


def process_batch(root_folder, video_name="external_sensor1", chunk_size=2100, skip_existing=True):
    """
    Process all task folders in a root directory.
    
    Args:
        root_folder: Path to the root folder containing task folders
        video_name: Name of the video file (without extension, default: "external_sensor1")
        chunk_size: Number of frames to process in each chunk (default: 2100)
        skip_existing: If True, skip folders where frames already exist (default: True)
    
    Returns:
        dict: Statistics about the processing
    """
    # Find all subdirectories
    task_folders = []
    for item in os.listdir(root_folder):
        item_path = os.path.join(root_folder, item)
        if os.path.isdir(item_path):
            # Check if this folder has the video file
            video_path = os.path.join(item_path, f"{video_name}.mp4")
            if os.path.exists(video_path):
                task_folders.append(item_path)
    
    if not task_folders:
        print(f"No task folders with {video_name}.mp4 found in {root_folder}")
        return {"total": 0, "success": 0, "failed": 0, "skipped": 0}
    
    print(f"Found {len(task_folders)} task folders to process")
    print("=" * 80)
    
    stats = {"total": len(task_folders), "success": 0, "failed": 0, "skipped": 0}
    
    for i, task_folder in enumerate(task_folders, 1):
        task_name = os.path.basename(task_folder)
        print(f"\n[{i}/{len(task_folders)}] Processing: {task_name}")
        print("-" * 80)
        
        # Check if already processed
        output_folder = os.path.join(task_folder, video_name)
        if skip_existing and os.path.exists(output_folder):
            print(f"Skipping (frames already exist): {task_name}")
            stats["skipped"] += 1
            continue
        
        # Extract frames
        success = extract_frames_from_task(
            task_folder=task_folder,
            video_name=video_name,
            chunk_size=chunk_size,
            skip_existing=skip_existing
        )
        
        if success:
            stats["success"] += 1
        else:
            stats["failed"] += 1
            print(f"Failed to process: {task_name}")
    
    print("\n" + "=" * 80)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 80)
    print(f"Total folders: {stats['total']}")
    print(f"  - Successfully processed: {stats['success']}")
    print(f"  - Failed: {stats['failed']}")
    print(f"  - Skipped (already exists): {stats['skipped']}")
    print("=" * 80)
    
    return stats


def main():
    """Main function to parse arguments and extract frames from task videos."""
    parser = argparse.ArgumentParser(
        description="Extract PNG frames from MP4 videos in replayed activity folders.\n"
                    "Can process either a single task folder OR a root folder containing multiple tasks.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--task_folder",
        default="data/replayed_activities",
        help="Path to either:\n"
             "  - A single task folder (e.g., data/replayed_activities/bringing_water_1750844141719178)\n"
             "  - Root folder containing multiple task folders (e.g., data/replayed_activities)\n"
             "The script will automatically detect which mode to use."
    )
    parser.add_argument(
        "--video_name",
        default="external_sensor1",
        help="Name of the video file without extension (default: external_sensor1)"
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=2100,
        help="Number of frames to process in each chunk (default: 2100)"
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        default=False,
        help="Skip folders where frames already exist (useful for batch processing)"
    )
    
    args = parser.parse_args()
    
    # Check if folder exists
    if not os.path.exists(args.task_folder):
        print(f"Error: Folder does not exist: {args.task_folder}")
        sys.exit(1)
    
    # Detect mode: check if this folder contains the video file directly (single task)
    # or if it contains subdirectories with video files (batch mode)
    video_path = os.path.join(args.task_folder, f"{args.video_name}.mp4")
    
    if os.path.exists(video_path):
        # Single task mode
        print("=" * 80)
        print("MODE: Single task extraction")
        print("=" * 80)
        success = extract_frames_from_task(
            task_folder=args.task_folder,
            video_name=args.video_name,
            chunk_size=args.chunk_size,
            skip_existing=args.skip_existing
        )
        if not success:
            sys.exit(1)
    else:
        # Batch mode - check if there are subdirectories with video files
        has_task_folders = False
        for item in os.listdir(args.task_folder):
            item_path = os.path.join(args.task_folder, item)
            if os.path.isdir(item_path):
                video_in_subdir = os.path.join(item_path, f"{args.video_name}.mp4")
                if os.path.exists(video_in_subdir):
                    has_task_folders = True
                    break
        
        if has_task_folders:
            # Batch processing mode
            print("=" * 80)
            print("MODE: Batch processing (multiple tasks)")
            print("=" * 80)
            stats = process_batch(
                root_folder=args.task_folder,
                video_name=args.video_name,
                chunk_size=args.chunk_size,
                skip_existing=args.skip_existing
            )
            if stats["failed"] > 0 and stats["success"] == 0:
                sys.exit(1)
        else:
            print(f"Error: No video file found at {video_path}")
            print(f"Error: No task folders with {args.video_name}.mp4 found in {args.task_folder}")
            print("\nMake sure you provide either:")
            print("  1. A single task folder containing an .mp4 file")
            print("  2. A root folder containing task subfolders with .mp4 files")
            sys.exit(1)


if __name__ == "__main__":
    main()