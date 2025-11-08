import logging
import boto3
from botocore.exceptions import ClientError
import os
import requests
import zipfile
import shutil
from uuid import uuid4


class Downloader:
    def __init__(self, bucket_name: str, workdir="/home/ec2-user/data"):
        self.s3_client = boto3.client('s3')
        self.bucket_name = bucket_name
        self.workdir = workdir
        os.makedirs(workdir, exist_ok=True)
    
    def download(self, url: str) -> str:
        file_id = str(uuid4())[:8]
        file_path = f"{self.workdir}/data_{file_id}.zip"
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(file_path, "wb") as fd:
            for chunk in response.iter_content(chunk_size=8192):
                fd.write(chunk)
        return file_path
        
    def unzip_file(self, file_path: str) -> str:
        file_id = str(uuid4())[:8]
        unzipped_path = f"{self.workdir}/unzipped_{file_id}"
        os.makedirs(unzipped_path, exist_ok=True)
        with zipfile.ZipFile(file_path, 'r') as zf:
            zf.extractall(unzipped_path)
        return unzipped_path

    def upload_to_s3(self, file_dir: str, prefix: str):
        for root, dirs, files in os.walk(file_dir):
            for filename in files:
                try:
                    local_path = os.path.join(root, filename)
                    rel_path = os.path.relpath(local_path, file_dir)
                    s3_path = f"{prefix.rstrip('/')}/{rel_path}"
                    self.s3_client.upload_file(local_path, self.bucket_name, s3_path)
                except ClientError as e:
                    logging.error(f"Error uploading {filename} to S3: {e}")
                    raise e
    
    def download_unzip_upload(self, url: str, prefix: str):
        """Download, unzip, and upload a file to S3. Delete temporary files."""
        file_path = None
        unzipped_dir = None
        try:
            print("Downloading\n")
            file_path = self.download(url)
            print("Downloaded\n\n")
            print("Unzipping\n")
            unzipped_dir = self.unzip_file(file_path)
            print("Unzipped\n\n")
            print("Uploading to s3\n")
            self.upload_to_s3(unzipped_dir, prefix)
            print("Finished")
        except Exception as e:
            logging.error(f"Error downloading, unzipping, and uploading file to S3: {e}")
            raise e
        finally:
            if file_path:
                os.remove(file_path)
            if unzipped_dir:
                shutil.rmtree(unzipped_dir)

if __name__ == "__main__":
    downloader_obj = Downloader("video-test-anshul")
    downloader_obj.download_unzip_upload("https://ml-site.cdn-apple.com/datasets/egodex/test.zip", "egodex-test-v1")