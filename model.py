# ------------------------------------------------------------
# (1) Illumination branch (Retinex-inspired) -> L_hat (smooth, low-freq)
# (2) Frequency branch with learnable spectral mask (FFT-based) -> detail refine
# (3) Cross-branch gating to couple branches
# (4) Final composition: I_hat = R_hat ⊙ L_hat (fast convergence, physically-plausible)
# ------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DWConvBlock(nn.Module):
    #Depthwise separable conv block: lightweight & fast.
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()
    def forward(self, x):
        x = self.dw(x); x = self.pw(x); x = self.bn(x); return self.act(x)

class GatedFuse(nn.Module):
    #Feature gating: x * sigmoid(Wg * guide)
    def __init__(self, feat_c, guide_c=3):
        super().__init__()
        self.g = nn.Conv2d(guide_c, feat_c, 1)
    def forward(self, x, guide):
        guide = self.g(guide)
        gate = torch.sigmoid(guide)
        return x * gate

#learnable spectral mask
class SpectralBlock(nn.Module):
    def __init__(self, c, height=32, width=32):
        super().__init__()
        self.mask = nn.Parameter(torch.ones(1, 1, height, width))
        self.proj_in  = nn.Conv2d(c, c, 1, bias=False)
        self.proj_out = nn.Conv2d(c, c, 1, bias=False)
        self.act = nn.GELU()

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj_in(x)

        X = torch.fft.rfft2(x, norm='ortho')
        mag, pha = torch.abs(X), torch.angle(X)

        Hf = H
        Wf = W//2 + 1
        M = F.interpolate(self.mask, size=(Hf, Wf), mode='bilinear', align_corners=False)  # [1,1,Hf,Wf]
        M = torch.clamp(M, 0.0, 2.0) 

        mag = mag * M

        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        X_new = torch.complex(real, imag)

        x_new = torch.fft.irfft2(X_new, s=(H, W), norm='ortho')
        x_new = self.proj_out(self.act(x_new))
        return x_new

# Illumination UNet
class IllumUNet(nn.Module):
    """
    Very lightweight UNet to estimate L_hat in [0,1].
    """
    def __init__(self, in_ch=3, base=32):
        super().__init__()
        C1, C2, C3 = base, base*2, base*4
        self.enc1 = nn.Sequential(DWConvBlock(in_ch, C1), DWConvBlock(C1, C1))
        self.enc2 = nn.Sequential(nn.MaxPool2d(2), DWConvBlock(C1, C2), DWConvBlock(C2, C2))
        self.enc3 = nn.Sequential(nn.MaxPool2d(2), DWConvBlock(C2, C3), DWConvBlock(C3, C3))
        self.dec2 = nn.Sequential(DWConvBlock(C3+C2, C2), DWConvBlock(C2, C2))
        self.dec1 = nn.Sequential(DWConvBlock(C2+C1, C1), DWConvBlock(C1, C1))
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.out = nn.Conv2d(C1, 3, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        d2 = self.up(e3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        L_hat = torch.sigmoid(self.out(d1))  # [0,1]
        return L_hat

# Reflectance refiner
class ReflectanceRefiner(nn.Module):
    def __init__(self, in_ch=3, base=48, use_spectral=True):
        super().__init__()
        C1, C2, C3 = base, base*2, base*4
        self.use_spectral = use_spectral

        self.enc1 = nn.Sequential(DWConvBlock(in_ch, C1), DWConvBlock(C1, C1))
        self.enc2 = nn.Sequential(nn.MaxPool2d(2), DWConvBlock(C1, C2), DWConvBlock(C2, C2))
        self.enc3 = nn.Sequential(nn.MaxPool2d(2), DWConvBlock(C2, C3), DWConvBlock(C3, C3))

        if use_spectral:
            self.spec = SpectralBlock(C3, height=64, width=64)  # finer frequency control

        self.gate2 = GatedFuse(C2, guide_c=3)
        self.gate3 = GatedFuse(C3, guide_c=3)


        self.dec2 = nn.Sequential(DWConvBlock(C3+C2, C2), DWConvBlock(C2, C2))
        self.dec1 = nn.Sequential(DWConvBlock(C2+C1, C1), DWConvBlock(C1, C1))

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.out = nn.Conv2d(C1, 3, 1)

    def forward(self, x, illum_guide):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        if self.use_spectral:
            e3 = e3 + self.spec(e3)

        g2 = F.interpolate(illum_guide, size=e2.shape[-2:], mode='bilinear', align_corners=False)
        g3 = F.interpolate(illum_guide, size=e3.shape[-2:], mode='bilinear', align_corners=False)
        e2 = self.gate2(e2, g2)
        e3 = self.gate3(e3, g3)

        d2 = self.up(e3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        R_hat = torch.sigmoid(self.out(d1))
        return R_hat

# DFIRNet
class DFIRNet(nn.Module):
    def __init__(self,
                 base_illum=32,
                 base_refl=48,
                 spectral=True,
                 eps=1e-3):
        super().__init__()
        self.illum = IllumUNet(in_ch=3, base=base_illum)
        self.refl  = ReflectanceRefiner(in_ch=3, base=base_refl, use_spectral=spectral)
        self.eps = eps

    def forward(self, I_d):
        L_hat = self.illum(I_d)  
        self.eps = 1e-6 
        R0 = I_d / (L_hat + self.eps)
        R0 = torch.clamp(R0, 0.0, 1.0)
        R_hat = self.refl(R0, L_hat)
        I_hat = torch.clamp(I_d + (R_hat * L_hat - I_d), 0.0, 1.0)
        return I_hat, L_hat, R_hat

# Loss functions
class TVLoss(nn.Module):
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight
    def forward(self, x):
        dx = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
        dy = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
        return self.weight * (dx + dy)

def fft_mag(t, eps=1e-8):
    X = torch.fft.rfft2(t.to(torch.float32), norm='ortho')
    return torch.sqrt(torch.clamp(X.real**2 + X.imag**2, min=eps))

class SpectralLoss(nn.Module):
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight
    def forward(self, x, y):
        return self.weight * F.l1_loss(fft_mag(x), fft_mag(y))


if __name__ == "__main__":
    net = DFIRNet(base_illum=32, base_refl=48, spectral=True)
    x = torch.rand(2,3,256,256)
    y, L, R = net(x)
    print(y.shape, L.shape, R.shape)
