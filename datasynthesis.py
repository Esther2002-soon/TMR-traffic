import os, glob, random
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter

NIGHTCITY_GT = ""
NIGHTCITY_INPUT = ""
RAIN100L = ""

def load_img(path):
    return TF.to_tensor(Image.open(path).convert("RGB"))

def save_img(t, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    TF.to_pil_image(t.clamp(0,1)).save(path)

def estimate_rain_layer(rainy, clean):
    return (rainy - clean).clamp(0,1)

def gamma_darken(x, gamma):
    return x ** gamma

def add_noise(x, sigma):
    return (x + torch.randn_like(x)*sigma).clamp(0,1)

def motion_blur_pil(img, k=7):
    return img.filter(ImageFilter.GaussianBlur(radius=k/6.0))

def process_split(split="train"):
    gt_files = sorted(glob.glob(os.path.join(NIGHTCITY_GT, split, "*.png")))
    rainy_files = sorted(glob.glob(os.path.join(RAIN100L, "input", "*.png")))
    clean_files = sorted(glob.glob(os.path.join(RAIN100L, "target", "*.png")))
    assert len(rainy_files) == len(clean_files)

    for idx, gt_path in enumerate(gt_files):
        I = load_img(gt_path)
        H, W = I.shape[-2:]

        j = random.randrange(len(rainy_files))
        Ir = load_img(rainy_files[j])
        Ic = load_img(clean_files[j])

        Ir = TF.resize(Ir, [H, W], antialias=True)
        Ic = TF.resize(Ic, [H, W], antialias=True)
        R = estimate_rain_layer(Ir, Ic)

        g = random.uniform(1.8, 3.0)  # degradation
        I_ll = gamma_darken(I, g)

        I_blur = TF.to_tensor(motion_blur_pil(
            TF.to_pil_image(I_ll),
            k=random.choice([5,7,9,11])   # blurring
        ))

        I_noise = add_noise(I_blur, sigma=random.uniform(0.01, 0.05))  # add noise

        '''g = random.uniform(1.3, 2.0)  # gamma
        I_ll = gamma_darken(I, g)
        I_blur = TF.to_tensor(motion_blur_pil(TF.to_pil_image(I_ll), k=random.choice([3,5,7,9])))
        I_noise = add_noise(I_blur, sigma=random.uniform(0.005, 0.03))'''
        

        alpha = random.uniform(0.8, 1.2)  # add rain
        #alpha = random.uniform(0.6, 1.2)
        I_deg = (I_noise + alpha * R).clamp(0,1)

        out_path = gt_path.replace(NIGHTCITY_GT, NIGHTCITY_INPUT)
        save_img(I_deg, out_path)

        if idx % 100 == 0:
            print(f"[{split}] Processed {idx}/{len(gt_files)}")

if __name__ == "__main__":
    process_split("train")
    process_split("test")
    print("Done!")