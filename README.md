# CelebA Inpainting Dataset Setup

Start by creating a virtual environment using the command: python -m venv venv
Then install all the requirements using: pip install -r requirements.txt

## Dataset

Download the **CelebA image dataset** from the following link: https://www.kaggle.com/datasets/kimjiyeop/celeba-128-onlybg
After downloading and extracting the dataset, follow these steps to prepare it for use.

## 1. Resize the Images

Use the provided script **`resize_celeba.py`** to resize all images in the dataset to `128×128` pixels.

Edit the paths in the script if needed:

```python
src = "path/to/downloaded/celeba"

 