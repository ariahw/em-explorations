import os
import fire

from tqdm import tqdm

from pathlib import Path

from src import utils

'''
NEURONPEDIA UTILS

Downloads all neuronpedia explanations from S3 bucket. Significantly faster than using API. 

'''

NEURONPEDIA_DIR = 'neuronpedia_data' # Output directory in local
NEURONPEDIA_BUCKET = "https://neuronpedia-datasets.s3.us-east-1.amazonaws.com"

def sae_dirpath(model_id: str, sae_name: str):
    return f'{NEURONPEDIA_DIR}/{model_id}/{sae_name}'

def data_fpath(model_id: str, sae_name: str, extension: str):
    return f'{sae_dirpath(model_id, sae_name)}/{extension}.jsonl'


def download_s3_bucket_contents(
        bucket_url: str,
        prefix: str,
        local_dir: str,
        max_files: int = None
    ):
    """
    Download all files from the specified S3 bucket prefix
    
    Args:
        bucket_url: Base S3 bucket URL
        prefix: S3 prefix path to download from
        local_dir: Local directory to save files
        max_files: Maximum number of files to download (None for all)
    """
    # Lazy import - only works with macosx group
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    
    # Create local directory if it doesn't exist
    Path(local_dir).mkdir(parents = True, exist_ok = True)
    
    # Parse bucket name from URL
    bucket_name = bucket_url.replace("https://", "").replace("http://", "").split('.')[0]
    
    # Create S3 client for public access (no credentials needed)
    s3_client = boto3.client(
        's3',
        config = Config(signature_version = UNSIGNED),
        region_name = 'us-east-1'
    )
    
    try:
        # List all objects with the given prefix
        print(f"Listing objects in s3://{bucket_name}/{prefix}")
        
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket = bucket_name, Prefix = prefix)
        
        files_downloaded = 0
        
        for page in pages:
            if 'Contents' in page:
                for obj in page['Contents']:
                    key = obj['Key']
                    
                    # Skip if max_files limit reached
                    if max_files and files_downloaded >= max_files:
                        print(f"Reached max_files limit of {max_files}")
                        return files_downloaded
                    
                    # Skip directories (keys ending with '/')
                    if key.endswith('/'):
                        continue
                    
                    # Create local file path
                    local_file_path = Path(local_dir) / key.replace(prefix, "").lstrip('/')
                    local_file_path.parent.mkdir(parents = True, exist_ok = True)
                    
                    # Skip if file already exists
                    if local_file_path.exists():
                        print(f"Skipping existing file: {local_file_path}")
                        continue
                    
                    # Download file
                    print(f"Downloading: {key} -> {local_file_path}")
                    try:
                        s3_client.download_file(bucket_name, key, str(local_file_path))
                        files_downloaded += 1
                    except Exception as e:
                        print(f"Error downloading {key}: {e}")
                        continue
        
        print(f"Downloaded {files_downloaded} files to {local_dir}")
        return files_downloaded
        
    except Exception as e:
        print(f"Error accessing S3 bucket: {e}")
        return 0


def get_all_files_by_extension(local_dir: str, file_extensions: list[str]):
    '''Get all files that have one of a set of extensions'''
    file_names = []
    for ext in file_extensions:
        file_names.extend(list(Path(local_dir).rglob(f"*{ext}")))
    return file_names


def decompress_downloaded_files(
        local_dir: str,
        file_extensions: list[str] = ['.gz', '.zip', '.tar', '.tar.gz', '.bz2'],
        remove_compressed: bool = False
    ):
    """
    Decompress all compressed files in the downloaded directory
    
    Args:
        local_dir: Directory containing downloaded files
        file_extensions: List of file extensions to decompress
        remove_compressed: Whether to remove compressed files after decompression
    """
    import gzip
    import zipfile
    import tarfile
    import bz2
    
    
    local_path = Path(local_dir)
    if not local_path.exists():
        print(f"Directory {local_dir} does not exist")
        return 0
    
    # Find all compressed files
    compressed_files = get_all_files_by_extension(local_path, file_extensions = file_extensions)
    
    if not compressed_files:
        print("No compressed files found")
        return 0
    
    print(f"Found {len(compressed_files)} compressed files")
    decompressed_count = 0
    
    for file_path in tqdm(compressed_files, desc = "Decompressing files"):
        try:
            output_path = file_path.with_suffix('')  # Remove the compression extension
            
            # Skip if decompressed file already exists
            if output_path.exists():
                print(f"Skipping {file_path.name} - already decompressed")
                continue
            
            # Handle different compression formats
            if file_path.suffix == '.gz':
                if file_path.suffixes[-2:] == ['.tar', '.gz']:
                    # Handle .tar.gz files
                    with tarfile.open(file_path, 'r:gz') as tar:
                        tar.extractall(path = file_path.parent)
                else:
                    # Handle .gz files
                    with gzip.open(file_path, 'rb') as f_in:
                        with open(output_path, 'wb') as f_out:
                            f_out.write(f_in.read())
                            
            elif file_path.suffix == '.zip':
                with zipfile.ZipFile(file_path, 'r') as zip_file:
                    zip_file.extractall(path = file_path.parent)
                    
            elif file_path.suffix == '.tar':
                with tarfile.open(file_path, 'r') as tar:
                    tar.extractall(path = file_path.parent)
                    
            elif file_path.suffix == '.bz2':
                with bz2.open(file_path, 'rb') as f_in:
                    with open(output_path, 'wb') as f_out:
                        f_out.write(f_in.read())
            
            decompressed_count += 1
            print(f"Decompressed: {file_path.name}")
            
            # Remove compressed file if requested
            if remove_compressed:
                file_path.unlink()
                print(f"Removed: {file_path.name}")
                
        except Exception as e:
            print(f"Error decompressing {file_path}: {e}")
            continue
    
    print(f"Successfully decompressed {decompressed_count} files")
    return decompressed_count

    
    

def download_all(
        model_id: str,
        sae_id: str,
        max_files: int = None,
        extension = "explanations"
    ):
    """
    Convenience function to download Gemma-2-9B explanations from Neuronpedia
    
    Args:
        local_dir: Local directory to save files
        max_files: Maximum number of files to download (None for all)
    """

    sae_dir = sae_dirpath(model_id, sae_id)

    # Downloads bucket contents
    download_s3_bucket_contents(
        bucket_url = NEURONPEDIA_BUCKET,
        prefix = f"v1/{model_id}/{sae_id}/{extension}/",
        local_dir = NEURONPEDIA_DIR,
        max_files = max_files
    )

    # Decompresses all files
    decompress_downloaded_files(
        local_dir = NEURONPEDIA_DIR,
        remove_compressed = True
    )

    # Load all files into one
    output = []
    all_files = get_all_files_by_extension(sae_dir, file_extensions = ['jsonl'])
    print('Combining', len(all_files), 'files')

    # Combine and delete once combined
    for fpath in all_files:
        output += utils.read_jsonl_all(str(fpath))
        fpath.unlink()
    
    # Write to output path
    utils.save_dataset_jsonl(output, filename = data_fpath(model_id, sae_id, extension))
    print('Created unified dataset at', sae_dir + f'/{extension}.jsonl')



if __name__ == "__main__":
    fire.Fire(download_all)