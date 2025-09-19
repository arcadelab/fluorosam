import requests
from pathlib import Path
import logging
from typing import Optional
import os
import subprocess
import urllib

from torchvision.datasets.utils import download_url, extract_archive

logger = logging.getLogger(__name__)


def download(url: str, destination_path: str | Path) -> None:
    # Convert the destination_path to a Path object if it's not already
    destination = Path(destination_path)

    # Check if the file already exists
    if not destination.exists():
        logger.info(f"File not found, downloading from {url}...")
        try:
            # Send a GET request to download the file
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Raise an exception for HTTP errors

            # Write the content to a file
            with destination.open("wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            logger.info(f"File downloaded and saved to {destination}.")
        except requests.exceptions.RequestException as e:
            logger.error(f"An error occurred: {e}")
    else:
        logger.info(f"File already exists at {destination}.")


def get_repo_root() -> str:
    """Returns the root directory of the current Git repository."""
    try:
        repo_root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], stderr=subprocess.DEVNULL)
        return repo_root.decode("utf-8").strip()
    except subprocess.CalledProcessError:
        raise RuntimeError("Not inside a Git repository.")


def download(
    url: str,
    filename: Optional[str] = None,
    root: Optional[str] = None,
    md5: Optional[str] = None,
    extract_name: Optional[str] = None,
) -> Path:
    """Download a data file and place it in root.

    Args:
        url (str): The download link.
        filename (str, optional): The name the save the file under. If None, uses the name from the URL. Defaults to None.
        root (str, optional): The directory to place downloaded data in. Can be overriden by setting the environment variable DEEPDRR_DATA_DIR. Defaults to "~/datasets/DeepDRR_Data".
        md5 (str, optional): MD5 checksum of the download. Defaults to None.
        extract_name: If not None, extract the downloaded file to `root / extract_name`.

    Returns:
        Path: The path of the downloaded file, or the extracted directory.
    """
    if root is None:
        root = get_repo_root()
    else:
        root = Path(root)

    if filename is None:
        filename = os.path.basename(url)

    try:
        download_url(url, root, filename=filename, md5=md5)
    except urllib.error.HTTPError:
        logger.warning(f"Pretty download failed. Attempting with wget...")
        subprocess.call(["wget", "-O", str(root / filename), url])
    except FileNotFoundError as e:
        raise RuntimeError(
            f"Download failed. Try installing wget. This is probably because you are on windows."
        )
    except Exception as e:
        raise RuntimeError(f"Download failed: {e}")

    path = root / filename
    if extract_name is not None:
        extract_archive(path, root, remove_finished=True)
        path = root / extract_name

    return path

