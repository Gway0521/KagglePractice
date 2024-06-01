import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class MedianFilter:
    '''
    中值濾波處理。
    '''
    def __init__(self, kernel_size=3):
        self.kernel_size = kernel_size

    def __call__(self, img):
        img_np = np.array(img)
        img_np = cv2.medianBlur(img_np, self.kernel_size)
        img = Image.fromarray(img_np)
        return img

class NPDataset(Dataset):
    '''
    Dataset，輸入 numpy 形式的 x y 和 transform，
    輸出經過 transform 的 non-Filtered image、Filtered image 與 label。

    Parameters
    -----------
    x: :class:`numpy.ndarray`
        多張灰階影像，通常 dimension 為 [60000, 28, 28]。
    y: :class:`numpy.ndarray`
        灰階影像對應的 label (非 one-hot)，通常 dimension 為 [60000,]。
    transform: :class:`torchvision.transforms.transforms.Compose`
        影像要做的處理 transforms.Compose。

    Returns
    -----------
    img: :class:`torch.tensor`
        經過 transform 的單張 non-Filtered image。
    filtered_img: :class:`torch.tensor`
        經過 transform 的單張 Filtered image。
    self.y[index]: :class:`torch.tensor`
        單個 label。
    '''
    def __init__(self, x, y, transform=None):
        self.x = []
        for np_img in x:
            self.x.append(Image.fromarray(np_img))
        self.y = torch.from_numpy(y).long()
        self.transform = transform

    def __getitem__(self, index):
        # non-Filtered image
        img = np.array(self.x[index])
        # Filtered image
        filtered_img = np.array(MedianFilter()(self.x[index]))

        if (self.transform):
            # 空白的影像，為了湊滿三個通道
            blank_img = np.zeros_like(img)
            # 結合三張灰階影像，變成三個通道的單張影像
            combined_image = np.stack((img, filtered_img, blank_img), axis=-1)
            # 經過 transform
            transformed_image = self.transform(Image.fromarray(combined_image))
            # 拆回三張灰階影像
            img, filtered_img, _ = torch.split(transformed_image, 1, dim=0)

        return img, filtered_img, self.y[index]
    
    def __len__(self):
        return len(self.x)
    





'''
class MedianFilter_v2:

    # tensor -> tensor

    def __init__(self, kernel_size=3):
        self.kernel_size = kernel_size

    def __call__(self, img):
        img_np = img.numpy()
        img_np = cv2.medianBlur(img_np, self.kernel_size)
        img = torch.from_numpy(img_np)
        return img

from scipy.ndimage import median_filter
class MedianFilter:
    def __init__(self, kernel_size=2):
        self.kernel_size = kernel_size

    def __call__(self, img):
        img_np = np.array(img)
        output = median_filter(img_np, size=self.kernel_size)
        img = Image.fromarray(output)
        return img
  return len(self.dataset)

class NPDataset(Dataset):
    # 接受 numpy 形式的 x y

    def __init__(self, x, y, transform=None):
        self.x = []
        for np_img in x:
            self.x.append(Image.fromarray(np_img))
        self.y = torch.from_numpy(y).long()
        self.transform = transform

    def __getitem__(self, index):
        img = self.x[index]
        if self.transform:
            img = self.transform(self.x[index])
        return img, self.y[index]
    
    def __len__(self):
        return len(self.x)
'''