from pathlib import Path
from typing import *
import requests

from tqdm import tqdm


__all__ = ["download_file", "download_bytes"]

  
def download_file(url: str, filepath: Union[str, Path], headers: dict = None, resume: bool = True) -> None:   
    headers = headers or {}  

    file_path = Path(filepath)  
    downloaded_bytes = 0  

    if resume and file_path.exists():  
        downloaded_bytes = file_path.stat().st_size  
        headers['Range'] = f"bytes={downloaded_bytes}-"  

    with requests.get(url, stream=True, headers=headers) as response:  
        response.raise_for_status()

        total_size = downloaded_bytes + int(response.headers.get('content-length', 0))  

        with (
            tqdm(desc=f"Downloading {file_path.name}", total=total_size, unit='B', unit_scale=True, leave=False) as pbar,
            open(file_path, 'ab') as file, 
        ):  
            pbar.update(downloaded_bytes)  

            for chunk in response.iter_content(chunk_size=4096):  
                file.write(chunk)  
                pbar.update(len(chunk))  
  

def download_bytes(url: str, headers: dict = None) -> bytes:  
    headers = headers or {}  

    with requests.get(url, stream=True, headers=headers) as response:  
        response.raise_for_status()

        return response.content  
  
