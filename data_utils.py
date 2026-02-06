from PIL import Image

import zipfile
import gdown
import os


def download_and_extract(
    url: str = 'https://drive.google.com/uc?id=1_gIar-Q89tWll-dnJUE077UujzAVMPxQ',  # Points to a zip file in GDrive
    output_zip_path: str = 'data/dataset.zip',
    force_download: bool = False
) -> str:
    os.makedirs(os.path.dirname(output_zip_path), exist_ok=True)
    if not os.path.exists(output_zip_path) or force_download:
        gdown.download(url, output_zip_path, quiet=False)
    data_dir = output_zip_path.replace('.zip', '')
    with zipfile.ZipFile(output_zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    print(f"Extracted to {data_dir}")
    return data_dir


def load_data(data_dir: str) -> list[dict]:
    dataset: list[dict] = []
    filenames = [f for f in os.listdir(data_dir) if f.endswith('.jpg')]
    for f in filenames:
        path = os.path.join(data_dir, f)
        img = Image.open(path)
        label = int(path.split('_')[-1].replace('.jpg', '').replace('label', ''))
        dataset.append({
            'img': img,
            'label': label,
            'path': path,
        })
    return dataset


if __name__ == "__main__":
    data_dir: str = download_and_extract(
        output_zip_path='data/dataset.zip',
        force_download=False
    )
