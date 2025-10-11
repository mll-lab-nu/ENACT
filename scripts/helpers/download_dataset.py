#!/usr/bin/env python3
"""
Dataset Download Helper Script

Downloads ENACT datasets from Hugging Face and Google Drive.
Supports downloading QA data, HDF5 files, replayed activities, and segmented activities.
"""

import os
import sys
import argparse
import shutil
import zipfile
from pathlib import Path


def check_dependencies():
    """Check if required dependencies are installed."""
    missing = []
    
    try:
        import gdown
    except ImportError:
        missing.append("gdown")
    
    try:
        import huggingface_hub
    except ImportError:
        missing.append("huggingface-hub")
    
    if missing:
        print(f"Error: Missing required dependencies: {', '.join(missing)}")
        print(f"Please install them with: pip install {' '.join(missing)}")
        sys.exit(1)


def download_enact_qa(output_dir):
    """Download ENACT QA dataset from Hugging Face."""
    from huggingface_hub import snapshot_download
    
    print("\n" + "="*60)
    print("Downloading ENACT QA dataset from Hugging Face...")
    print("="*60)
    
    repo_id = "Inevitablevalor/ENACT"
    temp_dir = output_dir / "temp_enact"
    
    try:
        # Download the entire repo to temp directory
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=temp_dir,
            local_dir_use_symlinks=False
        )
        
        # Find and extract QA.zip file
        qa_zip = temp_dir / "QA.zip"
        qa_target = output_dir / "QA"
        
        if qa_zip.exists():
            # Remove existing QA directory if it exists
            if qa_target.exists():
                print(f"Removing existing QA directory at {qa_target}")
                shutil.rmtree(qa_target)
            
            print(f"Extracting QA.zip to {qa_target}")
            with zipfile.ZipFile(qa_zip, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
            
            print(f"✓ ENACT QA dataset downloaded successfully to {qa_target}")
        else:
            print(f"Warning: QA.zip not found in downloaded repo")
        
        # Clean up temp directory
        if temp_dir.exists():
            print(f"Cleaning up temporary files...")
            shutil.rmtree(temp_dir)
            
    except Exception as e:
        print(f"Error downloading ENACT QA dataset: {e}")
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise


def clear_gdown_cache():
    """Clear gdown cache to resolve cookie issues."""
    cache_dir = Path.home() / ".cache" / "gdown"
    if cache_dir.exists():
        cookies_file = cache_dir / "cookies.txt"
        if cookies_file.exists():
            try:
                cookies_file.unlink()
                print("Cleared gdown cookies cache")
            except Exception:
                pass


def download_from_gdrive(file_id, output_path, extract_to=None):
    """Download file from Google Drive and optionally extract it."""
    import gdown
    
    try:
        # Clear cache before download to avoid cookie issues
        clear_gdown_cache()
        
        # Download the file using fuzzy=True for large files
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"Downloading from Google Drive...")
        print(f"Note: Large files may take a while and might require confirmation...")
        
        # Use fuzzy=True to handle large files and permission issues
        gdown.download(url, str(output_path), quiet=False, fuzzy=True)
        
        # Check if file was actually downloaded
        if not output_path.exists() or output_path.stat().st_size == 0:
            raise Exception(f"Download failed - file not found or empty")
        
        # Extract if it's a zip file
        if extract_to and output_path.suffix == '.zip':
            print(f"Extracting {output_path.name}...")
            with zipfile.ZipFile(output_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
            
            # Remove the zip file after extraction
            print(f"Removing zip file...")
            output_path.unlink()
            print(f"✓ Successfully downloaded and extracted")
        else:
            print(f"✓ Successfully downloaded to {output_path}")
            
    except Exception as e:
        print(f"Error downloading from Google Drive: {e}")
        # Clean up partial downloads
        if output_path.exists():
            output_path.unlink()
        raise


def download_hdf5(output_dir):
    """Download HDF5 dataset from Google Drive."""
    print("\n" + "="*60)
    print("Downloading HDF5 dataset...")
    print("="*60)
    
    file_id = "1B3YTxlV5V7T8UqkY1V4ReF5jkuSu2qrs"
    output_path = output_dir / "raw_hdf5.zip"
    
    download_from_gdrive(file_id, output_path, extract_to=output_dir)


def download_replayed(output_dir):
    """Download Replayed Activities dataset from Google Drive."""
    print("\n" + "="*60)
    print("Downloading Replayed Activities dataset...")
    print("="*60)
    
    file_id = "19rkSTPZmm2eWfuro8juv3acimELgD-xb"
    output_path = output_dir / "replayed_activities.zip"
    
    download_from_gdrive(file_id, output_path, extract_to=output_dir)


def download_segmented(output_dir):
    """Download Segmented Activities dataset from Google Drive."""
    print("\n" + "="*60)
    print("Downloading Segmented Activities dataset...")
    print("="*60)
    
    file_id = "1sPS7Lxw-FBPcWJbh7hOaD-22OIet_QrR"
    output_path = output_dir / "segmented_activities.zip"
    
    download_from_gdrive(file_id, output_path, extract_to=output_dir)


def main():
    """Main function for downloading datasets."""
    parser = argparse.ArgumentParser(
        description="Download ENACT datasets from Hugging Face and Google Drive",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./data',
        help='Output directory for downloaded datasets'
    )
    
    parser.add_argument(
        '--enact',
        action='store_true',
        default=True,
        help='Download ENACT QA dataset from Hugging Face (default: True)'
    )
    
    parser.add_argument(
        '--no-enact',
        action='store_false',
        dest='enact',
        help='Skip downloading ENACT QA dataset'
    )
    
    parser.add_argument(
        '--hdf5',
        action='store_true',
        default=False,
        help='Download HDF5 dataset from Google Drive'
    )
    
    parser.add_argument(
        '--replayed',
        action='store_true',
        default=False,
        help='Download Replayed Activities dataset from Google Drive'
    )
    
    parser.add_argument(
        '--segmented',
        action='store_true',
        default=False,
        help='Download Segmented Activities dataset from Google Drive'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        default=False,
        help='Download all datasets'
    )
    
    args = parser.parse_args()
    
    # Check dependencies first
    check_dependencies()
    
    # Create output directory
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nOutput directory: {output_dir}")
    print(f"Workspace: {Path.cwd()}")
    
    # Determine what to download
    download_enact_flag = args.enact or args.all
    download_hdf5_flag = args.hdf5 or args.all
    download_replayed_flag = args.replayed or args.all
    download_segmented_flag = args.segmented or args.all
    
    # Summary of what will be downloaded
    print("\n" + "="*60)
    print("Download Summary:")
    print("="*60)
    print(f"ENACT QA:            {'Yes' if download_enact_flag else 'No'}")
    print(f"HDF5:                {'Yes' if download_hdf5_flag else 'No'}")
    print(f"Replayed Activities: {'Yes' if download_replayed_flag else 'No'}")
    print(f"Segmented Activities:{'Yes' if download_segmented_flag else 'No'}")
    print("="*60)
    
    # Track success/failure
    results = []
    
    # Download datasets
    try:
        if download_enact_flag:
            try:
                download_enact_qa(output_dir)
                results.append(("ENACT QA", True, None))
            except Exception as e:
                results.append(("ENACT QA", False, str(e)))
        
        if download_hdf5_flag:
            try:
                download_hdf5(output_dir)
                results.append(("HDF5", True, None))
            except Exception as e:
                results.append(("HDF5", False, str(e)))
        
        if download_replayed_flag:
            try:
                download_replayed(output_dir)
                results.append(("Replayed Activities", True, None))
            except Exception as e:
                results.append(("Replayed Activities", False, str(e)))
        
        if download_segmented_flag:
            try:
                download_segmented(output_dir)
                results.append(("Segmented Activities", True, None))
            except Exception as e:
                results.append(("Segmented Activities", False, str(e)))
        
    except KeyboardInterrupt:
        print("\n\nDownload interrupted by user")
        sys.exit(1)
    
    # Print summary
    print("\n" + "="*60)
    print("Download Complete - Summary:")
    print("="*60)
    
    success_count = 0
    for name, success, error in results:
        if success:
            print(f"✓ {name}: Success")
            success_count += 1
        else:
            print(f"✗ {name}: Failed - {error}")
    
    print("="*60)
    print(f"Successfully downloaded {success_count}/{len(results)} datasets")
    print(f"All data saved to: {output_dir}")
    print("="*60)
    
    if success_count < len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()

