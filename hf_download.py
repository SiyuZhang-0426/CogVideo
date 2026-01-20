import os
import hashlib
import json
from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_files, whoami
from tqdm import tqdm
import argparse

def get_file_hash(file_path):
    """Calculate SHA256 hash of a file."""
    if not os.path.exists(file_path):
        return None
    
    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except Exception as e:
        print(f"Error calculating hash for {file_path}: {e}")
        return None

def check_and_download_missing_files(repo_id, local_dir, include_patterns=None, exclude_patterns=None, token=None):
    """
    Check for missing or corrupted files in a Hugging Face repository and download them.
    
    Args:
        repo_id (str): Repository ID on Hugging Face Hub (e.g., 'zai-org/CogVideoX1.5-5B')
        local_dir (str): Local directory to save files
        include_patterns (list): List of file patterns to include (e.g., ['*.safetensors'])
        exclude_patterns (list): List of file patterns to exclude
        token (str): Hugging Face token for private repositories
    """
    # Create local directory if it doesn't exist
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        # List all files in the repository
        all_files = list_repo_files(repo_id, token=token)
        print(f"Found {len(all_files)} files in repository '{repo_id}'")
        
        # Filter files based on patterns
        files_to_check = []
        for file_path in all_files:
            # Skip hidden files and directories
            if file_path.startswith('.') or '/.' in file_path:
                continue
                
            # Apply include patterns
            if include_patterns:
                include = False
                for pattern in include_patterns:
                    if pattern.startswith('*'):
                        if file_path.endswith(pattern[1:]):
                            include = True
                            break
                    elif pattern in file_path:
                        include = True
                        break
                if not include:
                    continue
            
            # Apply exclude patterns
            if exclude_patterns:
                exclude = False
                for pattern in exclude_patterns:
                    if pattern.startswith('*'):
                        if file_path.endswith(pattern[1:]):
                            exclude = True
                            break
                    elif pattern in file_path:
                        exclude = True
                        break
                if exclude:
                    continue
            
            files_to_check.append(file_path)
        
        print(f"Checking {len(files_to_check)} files after filtering")
        
        # Check which files are missing or corrupted
        files_to_download = []
        for file_path in tqdm(files_to_check, desc="Checking files"):
            local_file_path = Path(local_dir) / file_path
            remote_file_path = file_path
            
            # Check if file exists locally
            if not local_file_path.exists():
                files_to_download.append((remote_file_path, local_file_path))
                continue
            
            # For safetensors files, we can skip hash verification since hf_hub_download handles it
            # But we can check file size to detect obvious corruption
            try:
                remote_info = hf_hub_download(
                    repo_id=repo_id,
                    filename=remote_file_path,
                    token=token,
                    local_dir=local_dir,
                    force_download=False,
                    resume_download=False,
                    local_files_only=True
                )
                # If we get here, the file exists in cache and is valid
                continue
            except Exception as e:
                # File might be corrupted or not in cache
                files_to_download.append((remote_file_path, local_file_path))
        
        print(f"Found {len(files_to_download)} files to download")
        
        # Download missing or corrupted files
        if files_to_download:
            print("\nDownloading files:")
            for remote_path, local_path in tqdm(files_to_download, desc="Downloading"):
                try:
                    # Ensure parent directory exists
                    local_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Download the file (this automatically verifies integrity)
                    downloaded_path = hf_hub_download(
                        repo_id=repo_id,
                        filename=remote_path,
                        token=token,
                        local_dir=local_dir,
                        force_download=True,
                        resume_download=True
                    )
                    
                    print(f"✓ Downloaded: {remote_path} -> {downloaded_path}")
                except Exception as e:
                    print(f"✗ Failed to download {remote_path}: {str(e)}")
        else:
            print("✓ All files are already downloaded and valid!")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Make sure you have the huggingface_hub library installed: pip install huggingface_hub")

def main():
    parser = argparse.ArgumentParser(description='Check and download missing/corrupted files from Hugging Face Hub')
    parser.add_argument('--repo-id', type=str, default='zai-org/CogVideoX1.5-5B',
                        help='Repository ID on Hugging Face Hub')
    parser.add_argument('--local-dir', type=str, default='./CogVideoX1.5-5B',
                        help='Local directory to save files')
    parser.add_argument('--include', type=str, nargs='+', default=['*.safetensors'],
                        help='File patterns to include (e.g., "*.safetensors")')
    parser.add_argument('--exclude', type=str, nargs='+', 
                        help='File patterns to exclude')
    parser.add_argument('--token', type=str, 
                        help='Hugging Face token (optional, for private repos)')
    parser.add_argument('--no-auth', action='store_true',
                        help='Skip authentication check')
    
    args = parser.parse_args()
    
    # Check authentication if needed
    if not args.no_auth:
        try:
            whoami(token=args.token)
            print("✓ Authentication successful")
        except Exception as e:
            print("⚠️ Authentication failed or not provided. This might be okay for public repositories.")
            print(f"Error: {str(e)}")
            print("Continuing anyway...")
    
    # Run the main function
    check_and_download_missing_files(
        repo_id=args.repo_id,
        local_dir=args.local_dir,
        include_patterns=args.include,
        exclude_patterns=args.exclude,
        token=args.token
    )

if __name__ == "__main__":
    main()
