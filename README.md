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

We estimate a rain-only layer by a simple positive residual between a rainy image <img width="51" height="32" alt="Screenshot 2025-09-29 at 9 42 28 PM" src="https://github.com/user-attachments/assets/e136515d-7ab6-4f4e-baed-dae966ef9805" /> and its clean mate <img width="174" height="28" alt="Screenshot 2025-09-29 at 9 42 45 PM" src="https://github.com/user-attachments/assets/814717b3-4277-4cd6-bece-2450778e927d" />
<img width="283" height="47" alt="Screenshot 2025-09-29 at 9 30 37 PM" src="https://github.com/user-attachments/assets/d7992b56-4837-41b7-9f5a-1e17679f3869" />

由雨天影像與乾淨影像相減取正值，得到雨層 (R)，再夾住於 ([0,1])。

---

### 1.3 Low-light rendering by gamma / 以 γ 調暗

Nighttime low illumination is synthesized by gamma compression:
<img width="557" height="102" alt="Screenshot 2025-09-29 at 9 31 01 PM" src="https://github.com/user-attachments/assets/ff660549-c34a-4b76-8f41-5813830ab906" />

用冪次<img width="75" height="27" alt="Screenshot 2025-09-29 at 9 31 17 PM" src="https://github.com/user-attachments/assets/4f860776-3ea0-458b-b8ea-1d405cb877da" />將影像壓暗，模擬低照度。

---

### 1.4 Rain composition / 雨層合成

We blend the rain layer additively with a strength <img width="93" height="25" alt="Screenshot 2025-09-29 at 9 31 35 PM" src="https://github.com/user-attachments/assets/8416e6cc-1356-4827-906b-c0c5e43bd3a2" />
<img width="449" height="49" alt="Screenshot 2025-09-29 at 9 31 48 PM" src="https://github.com/user-attachments/assets/1cba677e-d8c0-49f0-bfe0-2c8515913d01" />

(Optionally alpha-composite via a luminance mask (A) if your code supports it:
<img width="374" height="36" alt="Screenshot 2025-09-29 at 9 32 00 PM" src="https://github.com/user-attachments/assets/443af6b0-e4d9-43e9-a383-03c6927ae5f2" />

以強度 (s) 將雨層加到暗化後影像上；可改為 alpha 合成。

---

### 1.5 Motion blur / 運動模糊

Convolve with a random linear motion kernel (K) of length (L) and angle (\theta):
<img width="469" height="47" alt="Screenshot 2025-09-29 at 9 32 20 PM" src="https://github.com/user-attachments/assets/7050ba37-0d41-445d-ba90-2b9c427016df" />

用線性運動模糊核卷積影像，核長 (L)、角度 <img width="12" height="22" alt="Screenshot 2025-09-29 at 9 44 30 PM" src="https://github.com/user-attachments/assets/ad9843cd-27be-4e42-840d-b0d9755657fb" />，核須歸一化。

---

### 1.6 Sensor noise / 感測雜訊

<img width="570" height="98" alt="Screenshot 2025-09-29 at 9 32 43 PM" src="https://github.com/user-attachments/assets/e8ae6214-ed2d-4623-8688-16f98e2b0aa3" />

加入高斯雜訊並夾住範圍，得到退化影像 <img width="22" height="27" alt="Screenshot 2025-09-29 at 9 45 17 PM" src="https://github.com/user-attachments/assets/09323340-655e-4398-9e88-f99bc6172cab" />。**配對標註**即為 (GT=C)。

---

### 1.7 Summary of synthesis / 合成流程總結

<img width="400" height="60" alt="Screenshot 2025-09-29 at 9 33 27 PM" src="https://github.com/user-attachments/assets/862342aa-1c51-428b-9c8a-a146540ed23e" />

先壓暗、加雨、再模糊、加雜訊，最後夾住 ([0,1])。

<img width="309" height="313" alt="image" src="https://github.com/user-attachments/assets/34f46e3f-2a78-4075-bf63-05d7bd51e0d7" />
<img width="309" height="313" alt="image" src="https://github.com/user-attachments/assets/bf1e446e-8783-4c16-93c6-462b32fb77c7" />

---

## 2) Model (`model.py`)

### 2.1 Overall pipeline / 整體流程
<img width="698" height="337" alt="image" src="https://github.com/user-attachments/assets/6b296474-a8c6-43c3-8a2f-fce4abc1a60e" />

1. **Illumination branch (U-Net)** predicts illumination <img width="18" height="33" alt="Screenshot 2025-09-29 at 9 33 56 PM" src="https://github.com/user-attachments/assets/6bb3319b-bcb0-4459-bb78-5d46325b47fd" />.
2. **Retinex division** initializes reflectance <img width="89" height="40" alt="Screenshot 2025-09-29 at 9 34 10 PM" src="https://github.com/user-attachments/assets/f27628b5-7ea5-4994-869f-4b2f93ec4889" />.
3. **Reflectance branch** refines reflectance with
   (a) **Spectral Block** (learnable FFT magnitude mask) and
   (b) **GatedFuse** (illumination-guided gating).
4. **Residual compose** forms the restored image.
上支預測光照、Retinex 分解得到初始反射、下支以頻域遮罩與門控細化，最後殘差合成輸出復原圖。

---

### 2.2 Illumination U-Net / 光照分支

#### Depthwise separable block（DWConvBlock）

A depthwise conv (W_d) followed by pointwise (1\times1) conv (W_p) and activation:

<img width="334" height="65" alt="Screenshot 2025-09-29 at 9 34 48 PM" src="https://github.com/user-attachments/assets/54d4c69e-f458-480a-a0a2-8d1a6b38a02b" />

* Encoder: 3 stages .
* Decoder: upsample + skip concat; two DW blocks per stage.
* Head: 1 x 1 conv + Sigmoid:
<img width="336" height="56" alt="Screenshot 2025-09-29 at 9 35 02 PM" src="https://github.com/user-attachments/assets/65633ce2-668f-40a4-93e7-14132be9155f" />

用 DW 可大幅降參數與 FLOPs；每個 stage 疊兩次 DW 擴大有效感受野與表達力；最後 1 x 1 + Sigmoid 輸出光照圖 <img width="18" height="29" alt="Screenshot 2025-09-29 at 9 47 31 PM" src="https://github.com/user-attachments/assets/6c074208-b753-4c92-897f-4ac67b01eeea" />。

---

### 2.3 Retinex division / Retinex 分解

<img width="277" height="75" alt="Screenshot 2025-09-29 at 9 35 15 PM" src="https://github.com/user-attachments/assets/3504be92-d382-4283-b356-4be3f2755109" />

依 Retinex 理論 <img width="155" height="28" alt="Screenshot 2025-09-29 at 9 55 08 PM" src="https://github.com/user-attachments/assets/f102306d-8ae2-4d3a-9098-179f9b8ad98a" /> 解耦亮度，得到初始反射 <img width="28" height="27" alt="Screenshot 2025-09-29 at 9 55 34 PM" src="https://github.com/user-attachments/assets/3781cdfb-36ea-48e4-9de7-e3d45abc2b04" />。

---

### 2.4 Reflectance branch / 反射分支

#### (a) Spectral Block（learnable FFT magnitude mask）

At the bottleneck feature <img width="162" height="29" alt="Screenshot 2025-09-29 at 9 35 47 PM" src="https://github.com/user-attachments/assets/89f4c4a4-4b5b-4fe3-beeb-34ec51206420" />:

<img width="507" height="144" alt="Screenshot 2025-09-29 at 9 35 31 PM" src="https://github.com/user-attachments/assets/7d99f667-940b-47cb-b859-8dc94a2ac9ec" />

* Optional **DC lock**: <img width="139" height="31" alt="Screenshot 2025-09-29 at 9 36 06 PM" src="https://github.com/user-attachments/assets/536e7678-3c3d-4ff8-8afa-2e38d890c02a" />.
* Mask resolution ((h,w)) is upsampled to <img width="143" height="36" alt="Screenshot 2025-09-29 at 9 36 18 PM" src="https://github.com/user-attachments/assets/d6e1d317-2755-40e2-bb5e-2410eee7f5a0" />.
在頻域以可學遮罩 (M) 重加權幅值（不改相位），抑制雨條的窄帶頻率，保持幾何結構。

#### (b) Illumination-guided GatedFuse（two scales）

For encoder features <img width="149" height="35" alt="Screenshot 2025-09-29 at 9 36 34 PM" src="https://github.com/user-attachments/assets/35cbe876-04d3-4539-ac2e-4afe475fbd68" />:
<img width="483" height="55" alt="Screenshot 2025-09-29 at 9 36 52 PM" src="https://github.com/user-attachments/assets/84fe1294-b2bb-4281-af17-e08a87a2429b" />

把 <img width="23" height="27" alt="Screenshot 2025-09-29 at 9 37 17 PM" src="https://github.com/user-attachments/assets/a2da9a4f-ca0e-469e-b28d-acdc852d51fe" /> 下採樣並投影到相同通道，Sigmoid 成 0~1 的門控圖，逐像素抑制暗區/亮區的錯誤增益。

#### Decoder + head

Upsample + skip concat + DW blocks; head is (1\times1) (+ optional Sigmoid):
<img width="273" height="51" alt="Screenshot 2025-09-29 at 9 37 33 PM" src="https://github.com/user-attachments/assets/35fab707-6767-4b3a-b779-0201681b98b2" />

解碼後輸出反射 <img width="17" height="28" alt="Screenshot 2025-09-29 at 9 37 47 PM" src="https://github.com/user-attachments/assets/cb217d1a-fbb2-406b-85e0-767f133d7f40" />；通常配合值域採用 Sigmoid。

---

### 2.5 Residual composition / 殘差合成

Instead of directly <img width="59" height="24" alt="Screenshot 2025-09-29 at 9 48 02 PM" src="https://github.com/user-attachments/assets/083ff895-0471-4313-8bfd-a5520ac79b6d" />, we do residual locking to stabilize color/contrast:

<img width="329" height="50" alt="Screenshot 2025-09-29 at 9 48 16 PM" src="https://github.com/user-attachments/assets/fbaebcce-21df-4044-b80e-c5ab2d898052" />

---

## 3) Objective functions / 損失函數

Let <img width="30" height="32" alt="Screenshot 2025-09-29 at 9 38 04 PM" src="https://github.com/user-attachments/assets/dd833100-4c52-48e2-bebf-a890d36b4833" /> be the clean GT.

### 3.1 Pixel & SSIM（像素與結構相似度）

<img width="181" height="52" alt="Screenshot 2025-09-29 at 9 38 24 PM" src="https://github.com/user-attachments/assets/a0718fd9-ff41-4b71-913d-96ab953403bb" />

SSIM (windowed) with constants (C_1, C_2):

<img width="508" height="59" alt="Screenshot 2025-09-29 at 9 38 57 PM" src="https://github.com/user-attachments/assets/8bcdf0ff-ecc6-4072-8484-3258271e1490" />

像素 L1 與 SSIM（以視覺結構一致性為主）。

### 3.2 Total variation on illumination / 光照平滑 TV

<img width="412" height="65" alt="Screenshot 2025-09-29 at 9 39 22 PM" src="https://github.com/user-attachments/assets/e1668e18-6456-42cb-b466-6a34fec35078" />

對 (\hat L) 加總變分以去除鋸齒與雜訊。

### 3.3 Spectral magnitude loss / 頻幅一致性

<img width="291" height="38" alt="Screenshot 2025-09-29 at 9 39 37 PM" src="https://github.com/user-attachments/assets/bab3b3e1-53d3-45ac-b84f-f9af1f040c8d" />

使輸出與 GT 在頻域幅值分布一致，有助於去除週期性雨條與條紋。

### 3.4 Total loss / 總損失

<img width="307" height="39" alt="Screenshot 2025-09-29 at 9 39 51 PM" src="https://github.com/user-attachments/assets/7d609dc6-0e33-47cd-bde4-3b06f7c4ac51" />

四項加權：像素、結構、光照平滑、頻域一致。

---

## 4) Architecture diagram mapping / 架構圖對照

<img width="698" height="337" alt="image" src="https://github.com/user-attachments/assets/6b296474-a8c6-43c3-8a2f-fce4abc1a60e" />

<img width="384" height="201" alt="Screenshot 2025-09-29 at 9 40 16 PM" src="https://github.com/user-attachments/assets/f0e40929-5d0d-4b91-8eb2-547b18bc0ff8" />

## 🔹 架構元件解釋
1. **Illumination U-Net**
   * **在做什麼**：這個分支像是「手電筒」，專門去估計每個像素有多少光線（亮度）。
   * **為什麼重要**：有了光照圖 <img width="16" height="28" alt="Screenshot 2025-09-29 at 9 51 04 PM" src="https://github.com/user-attachments/assets/a70aa7e0-7eef-4f8f-8ad0-10d6e494d099" />，我們就能知道哪裡是暗區、哪裡是亮區，這是後面 Retinex 分解和門控的基礎。
     
2. **Retinex division**
   * **在做什麼**：把原始影像 <img width="23" height="25" alt="Screenshot 2025-09-29 at 9 51 19 PM" src="https://github.com/user-attachments/assets/5fe68f65-8c28-4d71-90ba-9ebdb329c5ea" /> 除以光照 <img width="16" height="28" alt="Screenshot 2025-09-29 at 9 51 04 PM" src="https://github.com/user-attachments/assets/a70aa7e0-7eef-4f8f-8ad0-10d6e494d099" />，得到初步的材質/反射層 <img width="23" height="27" alt="Screenshot 2025-09-29 at 9 52 47 PM" src="https://github.com/user-attachments/assets/9392f522-0f38-4919-972a-0274ac6f68b4" />。
   * **為什麼重要**：這一步把「光」和「材質」分開，讓後面的反射分支不用再管亮暗，只專注處理紋理和雨痕。

3. **Reflectance encoder**
   * **在做什麼**：這是反射分支的編碼器，逐層壓縮特徵，把細節、紋理、雨痕都抽取出來。
   * **為什麼重要**：把複雜的材質特徵表示出來，特別是那些細長的雨條。

4. **Spectral Block (H/4)**
   * **在做什麼**：在頻率世界裡用一個學習到的濾波器，專門壓掉「雨條的頻率峰值」。
   * **為什麼重要**：雨條在頻域裡有固定的窄帶特徵，這個模組就像「耳塞」，把那種噪音過濾掉，但保留畫面原本的形狀。

5. **GatedFuse (H/2, H/4)**
   * **在做什麼**：用光照圖 <img width="16" height="28" alt="Screenshot 2025-09-29 at 9 51 04 PM" src="https://github.com/user-attachments/assets/a70aa7e0-7eef-4f8f-8ad0-10d6e494d099" /> 生成「閘門」，告訴反射分支在某些區域要加強，某些區域要抑制。
   * **為什麼重要**：暗的地方避免放大雜訊，亮的地方避免把雨痕誤當紋理，相當於「光照導遊」幫忙調整。

6. **Reflectance decoder**
   * **在做什麼**：把壓縮後的反射特徵再放大回原圖大小，重建出乾淨的反射圖 <img width="20" height="29" alt="Screenshot 2025-09-29 at 9 52 59 PM" src="https://github.com/user-attachments/assets/859561a3-4bde-4783-803b-2dc6f47b4c7d" />。
   * **為什麼重要**：這一步就是「把抽象特徵翻譯回影像」，並用最後的 1×1 卷積 + Sigmoid 限制在正常的 RGB 範圍。

7. **Residual compose**
   * **在做什麼**：把原圖 <img width="26" height="27" alt="Screenshot 2025-09-29 at 9 52 18 PM" src="https://github.com/user-attachments/assets/2e245030-4ce1-45ed-8c18-b323d231d616" /> 和修復後的光照 × 反射結果 <img width="16" height="28" alt="Screenshot 2025-09-29 at 9 51 04 PM" src="https://github.com/user-attachments/assets/a70aa7e0-7eef-4f8f-8ad0-10d6e494d099" /> 混在一起，用殘差方式輸出最終復原影像 <img width="12" height="23" alt="Screenshot 2025-09-29 at 9 52 04 PM" src="https://github.com/user-attachments/assets/d1f6144f-d963-4ebd-be89-d3d833ffbd87" />。
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

<img width="450" height="105" alt="image" src="https://github.com/user-attachments/assets/7c2edc79-d946-4561-9a16-4f856ef54e38" />

<img width="450" height="105" alt="image" src="https://github.com/user-attachments/assets/ac21d842-a79b-4c33-9c40-f3537a025204" />

<img width="446" height="104" alt="image" src="https://github.com/user-attachments/assets/684079cc-1d98-488f-8e27-5eacdde24958" />
