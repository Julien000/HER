
import os
import re

# 对输入字符串进行处理
def cleanSST(string):
    string = re.sub(u"[，。 :,.；|-“”——_/nbsp+&;@、《》～（）())#O！：【】]", "", string)
    return string.strip().lower()


import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def save_images(raw_image, out_path):
    plt.close()
    # raw_image: [1, 3, 224, 224]
    img = raw_image.squeeze(0).detach().cpu()  # [3, 224, 224]
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    # 反归一化
    img = img * std + mean

    # 防止数值越界
    # img = img.clamp(0, 1)

    # CHW → HWC
    img = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    Image.fromarray(img).save(out_path)

    print(f" output dir ==> {out_path}")
    # plt.show()