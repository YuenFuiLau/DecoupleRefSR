import __init_paths
import torch
from face_model.gpen_model import FullGenerator
import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from training import lpips

def img2tensor(img):
    img_t = torch.from_numpy(img).to(device)/255.
    img_t = (img_t - 0.5) / 0.5
    img_t = img_t.permute(2, 0, 1).unsqueeze(0).flip(1) # BGR->RGB
    return img_t

def tensor2img(img_t, pmax=255.0, imtype=np.uint8):

    img_t = img_t * 0.5 + 0.5
    img_t = img_t.squeeze(0).permute(1, 2, 0).flip(2) # RGB->BGR
    img_np = np.clip(img_t.float().cpu().numpy(), 0, 1) * pmax

    return img_np.astype(imtype)

def step_load_img(path, resolution=512, scale=4, lq=True):

        img_gt = cv2.imread(path, cv2.IMREAD_COLOR)
        if lq:
            img_gt = cv2.resize(img_gt, (resolution//scale, resolution//scale), interpolation=cv2.INTER_AREA)
        img_gt = cv2.resize(img_gt, (resolution, resolution), interpolation=cv2.INTER_AREA)
        
        img_gt = img_gt.astype(np.float32)/255.
        #img_gt, img_lq = self.degrader.degrade_process(img_gt)

        img_gt =  (torch.from_numpy(img_gt) - 0.5) / 0.5
        #img_lq =  (torch.from_numpy(img_lq) - 0.5) / 0.5
        
        img_gt = img_gt.permute(2, 0, 1).flip(0) # BGR->RGB
        #img_lq = img_lq.permute(2, 0, 1).flip(0) # BGR->RGB

        return img_gt

def load_ref(file_dir):

    paths = os.listdir(file_dir)
    ref_list = []
    for img_path in paths:
        ref_list.append(step_load_img(f"{os.getcwd()}/{file_dir}/{img_path}", lq=False))
    
    ref_list = torch.stack(ref_list, dim=0)

    return ref_list

device = "cuda"
ckpt = torch.load("./weights/GPEN-BFR-512.pth")
model = FullGenerator(512,512,8,device=device)
model.load_state_dict(ckpt)
model = model.to(device)

real_img = step_load_img("./eval/hqdata/face-593.png", lq=False)
img = step_load_img("./eval/hqdata/face-593.png").unsqueeze(0).to(device)
#ref_img = load_ref("./eval/relevant_ref/face-593").to(device)

fake_img, _ = model(img)

fake_img = tensor2img( fake_img.detach().cpu() )
cv2.imwrite("./results/eval.jpg", fake_img)
real_img = tensor2img( real_img )

psnr_val = psnr(real_img, fake_img)
print(f"PSNR: {psnr_val}")

ssim_val = ssim(real_img, fake_img, multichannel=True)
print(f"SSIM: {ssim_val}")

lpips_func = lpips.LPIPS(net='alex',version='0.1').to(device)

real_img = lpips.im2tensor(real_img).to(device)
fake_img = lpips.im2tensor(fake_img).to(device)
lipips_ = lpips_func.forward(real_img, fake_img)
lipips_ = lipips_.cpu().item()

print(f"Lpips: {lipips_}")