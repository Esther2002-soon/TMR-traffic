# Traffic-Aware Multi-Degradation Restoration
隨著智慧交通與智慧城市的發展，交通場域中車載攝影機（vehicle-mounted camera）與道路監視器已成為道路安全與交通管理的重要基礎。然而，夜間下雨環境往往造成影像品質惡化，包括低光照、雨絲、雨滴遮擋與運動模糊等問題，影響駕駛輔助與監控系統的可靠性。傳統方法多僅針對單一退化（如去雨或增亮），缺乏整合能力，難以滿足真實場景需求。本專題基於此動機，提出交通場域多重退化影像智慧復原系統。
		本系統訓練資料包含借用的公開影像資料集 Rain100L（雨絲影像）以及 NightCity[1]（城市夜景影像），並結合自製之 motion blur kernel 合成模擬真實交通情境的影像退化，以確保模型能在複雜環境下表現穩定。

NightCity dataset
[1] https://dmcv.sjtu.edu.cn/people/phd/tanxin/NightCity/index.html

## 1) Data Synthesis (`datasynthesis.py`)

### 1.1 Inputs & Notation

* Clean base image <img width="157" height="33" alt="Screenshot 2025-09-29 at 9 29 37 PM" src="https://github.com/user-attachments/assets/10dfa074-b8ee-44a9-8e9f-019a5a0fa8e4" />from NightCity (RGB float32).
* Rain/clean pairs from Rain100L to **extract a rain layer** (R).
* Optional blur kernel (K) and Gaussian noise (n).

* 乾淨基底影像 (C) 來自 NightCity（RGB、float32、值域 ([0,1])）。
* 從 Rain100L 的「雨天/乾淨」配對中**估計雨層** (R)。
* 可選：加入**運動模糊核** (K) 與**高斯雜訊** (n)。

---

### 1.2 Rain layer estimation / 雨層估計

We estimate a rain-only layer by a simple positive residual between a rainy image (I^{\text{rainy}}) and its clean mate (I^{\text{clean}}) (both in ([0,1])):

[
\boxed{R = \mathrm{clip}\big(I^{\text{rainy}} - I^{\text{clean}},,0,,1\big)} \tag{1}
]
```python
R = (rainy - clean).clamp(0, 1)   # thresholding if needed
```
由雨天影像與乾淨影像相減取正值，得到雨層 (R)，再夾住於 ([0,1])。

---

### 1.3 Low-light rendering by gamma / 以 γ 調暗

Nighttime low illumination is synthesized by gamma compression:

[
\boxed{I_{\gamma} = C^{\gamma}}, \qquad \gamma \sim \mathcal{U}(\gamma_{\min},\gamma_{\max}),\ \gamma>1. \tag{2}
]

Typical range: (\gamma\in[1.8,3.0]).

```python
gamma = random.uniform(args.gamma_min, args.gamma_max)  # >>> EDIT HERE <<<
I_gamma = C.clamp(0,1).pow(gamma)
```
用冪次（(\gamma>1)）將影像壓暗，模擬低照度。

---

### 1.4 Rain composition / 雨層合成

We blend the rain layer additively with a strength (s\in[0,1]):

[
\boxed{I_{\text{rain}} = \mathrm{clip}\big(I_{\gamma} + s,R,,0,,1\big)},\quad s\sim\mathcal{U}(s_{\min},s_{\max}). \tag{3}
]

(Optionally alpha-composite via a luminance mask (A) if your code supports it:
(I_{\text{rain}}=(1-A),I_{\gamma} + A,\mathrm{clip}(I_{\gamma}+R,0,1)).)

```python
s = random.uniform(0.5, 1.0)  # >>> EDIT HERE <<<
I_rain = (I_gamma + s*R).clamp(0,1)
```
以強度 (s) 將雨層加到暗化後影像上；可改為 alpha 合成。

---

### 1.5 Motion blur / 運動模糊

Convolve with a random linear motion kernel (K) of length (L) and angle (\theta):

[
\boxed{I_{\text{blur}} = K * I_{\text{rain}}},\qquad K = \text{PSF}(L,\theta),\ \sum K=1. \tag{4}
]

```python
L = random.randint(args.blur_len_min, args.blur_len_max)  # >>> EDIT HERE <<<
theta = random.uniform(0,180)
K = make_linear_motion_kernel(L, theta)
I_blur = conv2d_same(I_rain, K)  # normalized kernel
```
用線性運動模糊核卷積影像，核長 (L)、角度 (\theta)，核須歸一化。

---

### 1.6 Sensor noise / 感測雜訊

Add small Gaussian noise (n\sim\mathcal{N}(0,\sigma^2)):

[
\boxed{I_{d} = \mathrm{clip}\big(I_{\text{blur}} + n,,0,,1\big)},\qquad n\sim\mathcal{N}(0,\sigma^2). \tag{5}
]

```python
sigma = args.noise_std  # >>> EDIT HERE <<<
noise = torch.randn_like(I_blur) * sigma
Id = (I_blur + noise).clamp(0,1)
```
加入高斯雜訊並夾住範圍，得到退化影像 (I_d)。**配對標註**即為 (GT=C)。

---

### 1.7 Summary of synthesis / 合成流程總結

[
\boxed{
I_d = \big(K * (C^{\gamma} + sR)\big) + n \quad \xrightarrow{\ \text{clip}\ } [0,1].
} \tag{6}
]
先壓暗、加雨、再模糊、加雜訊，最後夾住 ([0,1])。

<img width="309" height="313" alt="image" src="https://github.com/user-attachments/assets/34f46e3f-2a78-4075-bf63-05d7bd51e0d7" />
<img width="309" height="313" alt="image" src="https://github.com/user-attachments/assets/bf1e446e-8783-4c16-93c6-462b32fb77c7" />

---

## 2) Model (`model.py`)

### 2.1 Overall pipeline / 整體流程
<img width="698" height="337" alt="image" src="https://github.com/user-attachments/assets/6b296474-a8c6-43c3-8a2f-fce4abc1a60e" />

1. **Illumination branch (U-Net)** predicts illumination (\hat L).
2. **Retinex division** initializes reflectance (R_0 = \frac{I_d}{\hat L+\varepsilon}).
3. **Reflectance branch** refines reflectance with
   (a) **Spectral Block** (learnable FFT magnitude mask) and
   (b) **GatedFuse** (illumination-guided gating).
4. **Residual compose** forms the restored image.
上支預測光照、Retinex 分解得到初始反射、下支以頻域遮罩與門控細化，最後殘差合成輸出復原圖。

---

### 2.2 Illumination U-Net / 光照分支

#### Depthwise separable block（DWConvBlock）

A depthwise conv (W_d) followed by pointwise (1\times1) conv (W_p) and activation:

[
\boxed{
\mathrm{DW}(x)=\phi!\Big(\mathrm{BN}\big(W_p*(W_d \star x)\big)\Big)
} \tag{7}
]

* Encoder: 3 stages .
* Decoder: upsample + skip concat; two DW blocks per stage.
* Head: (1\times1) conv + Sigmoid:

[
\boxed{\hat L = \sigma!\big(W_{1\times1}*F_{\text{dec}}\big)\in[0,1]^{3\times H\times W}} \tag{8}
]
用 DW 可大幅降參數與 FLOPs；每個 stage 疊兩次 DW 擴大有效感受野與表達力；最後 (1\times1) + Sigmoid 輸出光照圖 (\hat L)。

---

### 2.3 Retinex division / Retinex 分解

[
\boxed{R_0=\mathrm{clamp}!\left(\frac{I_d}{\hat L+\varepsilon},,0,,1\right)} \tag{9}
]

```python
eps = 1e-6
R0 = (Id / (L_hat + eps)).clamp(0,1)
```
依 Retinex 理論 (I=L!\odot!R)，以 (\hat L) 解耦亮度，得到初始反射 (R_0)。

---

### 2.4 Reflectance branch / 反射分支

#### (a) Spectral Block（learnable FFT magnitude mask）

At the bottleneck feature (r_3\in\mathbb{R}^{B\times C\times H_s\times W_s}):

[
\begin{aligned}
&z=W_{1\times1}*r_3,\quad X=\mathcal{F}(z)=\mathrm{rfft2}(z) \
&A=\lvert X\rvert,\ \Phi=\angle X,\ \ M=2,\sigma\big(\mathrm{interp}(\Theta)\big)\in(0,2) \
&\tilde A = M\odot A,\quad \tilde X=\tilde A,e^{j\Phi},\quad
z'=\mathcal{F}^{-1}(\tilde X)=\mathrm{irfft2}(\tilde X) \
&\boxed{r_3'=\phi!\big(W'_{1\times1}*z'\big)}
\end{aligned} \tag{10}
]

* Optional **DC lock**: (M[...,0,0]=1).
* Mask resolution ((h,w)) is upsampled to ((H_s, W_s/2+1)).
在頻域以可學遮罩 (M) 重加權幅值（不改相位），抑制雨條的窄帶頻率，保持幾何結構。

#### (b) Illumination-guided GatedFuse（two scales）

For encoder features (r_k) (e.g., (k=2,3)):

[
\boxed{
\text{gate}_k=\sigma!\big(W_g^{(k)} * \mathrm{Resize}(\hat L)\big),\qquad
\tilde r_k=r_k\odot \text{gate}_k.
} \tag{11}
]

```python
# Two gates at e2/e3 scales
r2 = self.gate2(r2, L_hat)  
r3 = self.gate3(r3, L_hat)
```
把 (\hat L) 下採樣並投影到相同通道，Sigmoid 成 0~1 的門控圖，逐像素抑制暗區/亮區的錯誤增益。

#### Decoder + head

Upsample + skip concat + DW blocks; head is (1\times1) (+ optional Sigmoid):

[
\boxed{\hat R = \sigma!\big(W^{\text{refl}}*{1\times1}*F^{\text{refl}}*{\text{dec}}\big)\in[0,1]} \tag{12}
]

```python
self.refl_head = nn.Sequential(nn.Conv2d(48,3,1), nn.Sigmoid())
```
解碼後輸出反射 (\hat R)；通常配合值域採用 Sigmoid。

---

### 2.5 Residual composition / 殘差合成

Instead of directly (\hat L\odot\hat R), we do residual locking to stabilize color/contrast:

[
\boxed{
\hat I = \mathrm{clip}!\big(I_d + (\hat L\odot\hat R - I_d),,0,,1\big).
} \tag{13}
]

```python
I_hat = (Id + (L_hat * R_hat - Id)).clamp(0,1)
```
以「替代圖 − 原圖」為殘差更新，顏色更穩定、收斂較快。

---

## 3) Objective functions / 損失函數

Let (I_{gt}) be the clean GT.

### 3.1 Pixel & SSIM（像素與結構相似度）

[
\boxed{\mathcal{L}*{\text{L1}} = \lVert \hat I - I*{gt}\rVert_1} \tag{14}
]

SSIM (windowed) with constants (C_1, C_2):

[
\boxed{\mathrm{SSIM}(x,y)=\frac{(2\mu_x\mu_y+C_1)(2\sigma_{xy}+C_2)}{(\mu_x^2+\mu_y^2+C_1)(\sigma_x^2+\sigma_y^2+C_2)}},\quad
\mathcal{L}*{\text{SSIM}} = 1-\mathrm{SSIM}(\hat I, I*{gt}). \tag{15}
]
像素 L1 與 SSIM（以視覺結構一致性為主）。

### 3.2 Total variation on illumination / 光照平滑 TV

[
\boxed{
\mathcal{L}*{\text{TV}}(\hat L)=
\lambda*{\text{TV}}!\Bigg(\frac{1}{N}!\sum!\lvert \hat L_{i,j+1}-\hat L_{i,j}\rvert +
\frac{1}{N}!\sum!\lvert \hat L_{i+1,j}-\hat L_{i,j}\rvert\Bigg)
} \tag{16}
]
對 (\hat L) 加總變分以去除鋸齒與雜訊。

### 3.3 Spectral magnitude loss / 頻幅一致性

[
\boxed{
\mathcal{L}*{\text{spec}}(\hat I, I*{gt})=
\lambda_{\text{spec}},
\big|,\lvert \mathcal{F}(\hat I)\rvert - \lvert \mathcal{F}(I_{gt})\rvert,\big|_1
} \tag{17}
]
使輸出與 GT 在頻域幅值分布一致，有助於去除週期性雨條與條紋。

### 3.4 Total loss / 總損失

[
\boxed{
\mathcal{L} =
\lambda_1,\mathcal{L}*{\text{L1}} +
\lambda_2,\mathcal{L}*{\text{SSIM}} +
\lambda_3,\mathcal{L}*{\text{TV}} +
\lambda_4,\mathcal{L}*{\text{spec}}.
} \tag{18}
]

```python
w1,w2,w3,w4 = 1.0, 0.5, 0.1, 0.1
loss = w1*L1 + w2*SSIM + w3*TV(L_hat) + w4*Spec(I_hat, I_gt)
```
四項加權：像素、結構、光照平滑、頻域一致。

---

## 4) Architecture diagram mapping / 架構圖對照

<img width="698" height="337" alt="image" src="https://github.com/user-attachments/assets/6b296474-a8c6-43c3-8a2f-fce4abc1a60e" />

* **Illumination U-Net**

  * Encoder: (3!\to!32!\to!64!\to!128) @ (H!\to!H/2!\to!H/4)
  * Decoder: upsample + skip; head (1\times1) + Sigmoid (\Rightarrow \hat L)
* **Retinex division** (R_0=I_d/(\hat L+\varepsilon))
* **Reflectance encoder** (3!\to!48!\to!96!\to!192)
* **Spectral Block (H/4)** on (C{=}192)
* **GatedFuse** at (H/2) and (H/4)
* **Reflectance decoder** → head (1\times1) (+ Sigmoid) (\Rightarrow \hat R)
* **Residual compose** (\hat I = \mathrm{clip}(I_d + (\hat L\hat R - I_d)))


## 🔹 架構元件解釋
1. **Illumination U-Net**
   * **在做什麼**：這個分支像是「手電筒」，專門去估計每個像素有多少光線（亮度）。
   * **為什麼重要**：有了光照圖 (\hat L)，我們就能知道哪裡是暗區、哪裡是亮區，這是後面 Retinex 分解和門控的基礎。
     
2. **Retinex division**
   * **在做什麼**：把原始影像 (I_d) 除以光照 (\hat L)，得到初步的材質/反射層 (R_0)。
   * **為什麼重要**：這一步把「光」和「材質」分開，讓後面的反射分支不用再管亮暗，只專注處理紋理和雨痕。

3. **Reflectance encoder**
   * **在做什麼**：這是反射分支的編碼器，逐層壓縮特徵，把細節、紋理、雨痕都抽取出來。
   * **為什麼重要**：把複雜的材質特徵表示出來，特別是那些細長的雨條。

4. **Spectral Block (H/4)**
   * **在做什麼**：在頻率世界裡用一個學習到的濾波器，專門壓掉「雨條的頻率峰值」。
   * **為什麼重要**：雨條在頻域裡有固定的窄帶特徵，這個模組就像「耳塞」，把那種噪音過濾掉，但保留畫面原本的形狀。

5. **GatedFuse (H/2, H/4)**
   * **在做什麼**：用光照圖 (\hat L) 生成「閘門」，告訴反射分支在某些區域要加強，某些區域要抑制。
   * **為什麼重要**：暗的地方避免放大雜訊，亮的地方避免把雨痕誤當紋理，相當於「光照導遊」幫忙調整。

6. **Reflectance decoder**
   * **在做什麼**：把壓縮後的反射特徵再放大回原圖大小，重建出乾淨的反射圖 (\hat R)。
   * **為什麼重要**：這一步就是「把抽象特徵翻譯回影像」，並用最後的 1×1 卷積 + Sigmoid 限制在正常的 RGB 範圍。

7. **Residual compose**
   * **在做什麼**：把原圖 (I_d) 和修復後的光照 × 反射結果 (\hat L \hat R) 混在一起，用殘差方式輸出最終復原影像 (\hat I)。
   * **為什麼重要**：這樣能保留顏色與對比，不會修過頭，收斂也更穩定。

👉 白話版：
* 上面那條 U-Net 負責「看清楚光照」。
* Retinex 把「光」跟「材質」拆開。
* 下面那條編碼器負責「抓細節、抓雨痕」。
* 中間的 Spectral Block 用「頻率濾波」去雨。
* GatedFuse 用「光照」來調整反射特徵。
* 解碼器把反射細節組回影像。
* 最後 Residual compose 把乾淨的反射和光照重組，輸出一張穩定的清晰影像。

---

## 5) Code locations to modify / 重要可調處（行為不變的安全修改點）

### `datasynthesis.py`

* **Gamma range**: search `gamma_min`, `gamma_max`.
* **Rain strength**: search `s = random.uniform(`.
* **Blur kernel**: `make_linear_motion_kernel`, change length/angle ranges.
* **Noise std**: search `noise_std`.
* **Clamping**: ensure every stage ends with `.clamp(0,1)`.

### `model.py`

* **Base channels**: `base_illum`, `base_refl`.
* **Spectral Block**: `SpectralBlock(ch=..., height=..., width=..., share_channels=True)`; `fix_dc=True`.
* **GatedFuse scales**: add/remove `gate1/gate2/gate3` and corresponding calls.
* **Heads**: add/remove `nn.Sigmoid()` to control range.
* **Residual compose**: the final `clamp(0,1)` is recommended.

