import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
from IPython.core.pylabtools import figsize


class Avg2MaxPooling(nn.Module):
    def __init__(self, kernel_size = 3, stride = 2):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        #Average Pooling
        avg_pool = F.avg_pool2d(x, kernel_size=self.kernel_size, stride=self.stride)

        #Max Pooling (thực hiện 2 lần như trong công thức)
        max_pool1 = F.max_pool2d(x, kernel_size=self.kernel_size, stride=self.stride)
        max_pool2 = F.max_pool2d(x, kernel_size=self.kernel_size, stride=self.stride)

        # Áp dụng công thức: Avg - (Max1 + Max2)
        output = avg_pool - (max_pool1 + max_pool2)
        return output


if __name__ == '__main__':
    image = cv2.imread("back_hold.jpg", cv2.IMREAD_GRAYSCALE)

    image = cv2.resize(image, (256, 256))

    image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)   # (1, 1, H, W)

    avg2max_pooling = Avg2MaxPooling(kernel_size=3, stride=2)

    output_tensor = avg2max_pooling(image_tensor)

    output_image = output_tensor.squeeze().detach().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title("Input")
    axes[0].axis("off")

    axes[1].imshow(output_image, cmap='gray')
    axes[1].set_title("Avg-2Max Pooling")
    axes[1].axis("off")

    plt.show()
