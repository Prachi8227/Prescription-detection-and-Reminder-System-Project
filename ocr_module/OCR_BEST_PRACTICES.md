# OCR Module – Best Practices & Improvements

This document explains the improvements made to the OCR pipeline and how to get the best accuracy.

---

## 1. Image Preprocessing (Why Each Step)

| Step | Purpose |
|------|--------|
| **Grayscale** | Reduces channels to one; Tesseract expects grayscale. Removes color noise and speeds up later steps. |
| **Resize** | Small images (< ~600px) are **upscaled** so Tesseract has enough resolution (target min edge ~1200px). Low-res images gain the most. Very large images are capped to avoid OOM. |
| **Noise removal (fastNlMeansDenoising)** | Removes grain and scanner noise while keeping text edges. Critical for photos and low-quality scans. |
| **Bilateral filter** | Optional; smooths noise but **preserves edges** (text strokes), unlike a plain Gaussian blur. |
| **Gaussian blur** | Use a **small** kernel (e.g. 1 or 3) only if needed to reduce salt-and-pepper; large kernels blur text and hurt accuracy. |
| **Adaptive thresholding** | Handles **uneven lighting** (common in phone photos of prescriptions). Each region gets its own threshold. |
| **Otsu fallback** | When the image is bimodal (e.g. clear black text on white), Otsu can give a cleaner binary than adaptive. |
| **Morphology (open + dilate)** | **Open** (erode then dilate) removes thin noise and small dots. **Dilate** closes small gaps inside characters. |

All of this is implemented in `preprocess_image_advanced()` and used by `preprocess_image()`.

---

## 2. Pytesseract Configuration

- **`--oem 3`**  
  Use both LSTM and legacy engines (default). Best general accuracy.

- **`--psm 6`** (default)  
  “Assume a single uniform block of text.” Good for prescription body text.

- **Fallback PSM modes** (used if the first attempt returns empty or errors):  
  `11` (sparse text), `4` (single column), `3` (fully automatic), `13` (raw line).  
  Implemented in `extract_text_tesseract()` with automatic retries.

- **Language**  
  Pass `language_codes` (e.g. `["eng", "hin"]`) so the config includes `-l eng+hin`.  
  Required for multi-language prescriptions.

---

## 3. Error Handling & Logging

- **Logging**  
  The module uses Python `logging` with logger name `__name__`. Set level to `DEBUG` when troubleshooting:
  ```python
  import logging
  logging.getLogger("ocr_module.enhanced_ocr").setLevel(logging.DEBUG)
  ```

- **Tesseract path**  
  Set env `TESSERACT_CMD` to the full path of `tesseract` if it’s not in the default Windows/Linux paths.

- **Failures**  
  Preprocessing and Tesseract calls are wrapped in try/except; failures are logged and the pipeline continues (e.g. fallback PSM, or EasyOCR/PaddleOCR when enabled).

---

## 4. Handwritten Text

- **Tesseract** is built for **printed** text. Handwriting accuracy is limited.

- **EasyOCR** and **PaddleOCR** in this project are used for **handwriting**:
  - In the app, use OCR mode **“Force handwriting”** (EasyOCR) or **“Force PaddleOCR”** for handwritten content.
  - EasyOCR: `run_handwriting_ocr(image_path, language_codes)`.
  - PaddleOCR: `run_paddleocr(image_path, language_codes)`.

- **Best practice for handwritten**  
  Preprocess lightly (grayscale, mild denoise, maybe adaptive threshold). Avoid heavy morphology that can merge or remove strokes. Rely on EasyOCR/PaddleOCR rather than Tesseract for handwritten lines.

---

## 5. Low-Resolution Images

- **Upscaling**  
  The pipeline **resizes** small images (shortest edge &lt; 600px) up so the shortest edge is at least ~1200px before Tesseract. This improves character recognition on low-res inputs.

- **Sharpening**  
  Optional: after upscaling, apply mild unsharp mask (see `skimage.filters.unsharp_mask`) to enhance edges. Too much can amplify noise.

- **Super-resolution**  
  For very poor resolution, consider a super-resolution model (e.g. OpenCV DNN or ESRGAN) to upscale the image first, then run the same preprocessing + OCR.

---

## 6. Alternative: EasyOCR / Deep Learning

- **EasyOCR**  
  Already integrated. Good for handwriting and mixed scripts. Use `run_handwriting_ocr()`. Slower than Tesseract; enable GPU if available.

- **PaddleOCR**  
  Already integrated. Strong for printed and handwritten text, multiple languages. Use `run_paddleocr()` or select “Force PaddleOCR” in the app.

- **When to use which**  
  - **Printed, clean, high-res** → Tesseract (current default) with the new preprocessing.  
  - **Handwritten or messy** → EasyOCR or PaddleOCR.  
  - **Low-res** → Preprocessing (resize + denoise + threshold) + Tesseract first; if result is poor, try EasyOCR/PaddleOCR on the same preprocessed image.

---

## 7. Production Checklist

- [ ] Tesseract installed and path set (or `TESSERACT_CMD` in env).
- [ ] Required language packs installed (e.g. `eng`, `hin` for Hindi).
- [ ] Logging level set appropriately (INFO for production, DEBUG for debugging).
- [ ] Temp files (e.g. `enhanced_*`, `inverted_temp.jpg`) cleaned periodically if stored under uploads.
- [ ] For handwriting, EasyOCR/PaddleOCR dependencies installed and OCR mode “Force handwriting” or “Force PaddleOCR” available to users.

---

## 8. Quick Reference – Key Functions

| Function | Role |
|----------|------|
| `preprocess_image(image_path)` | Main entry: full preprocessing, returns `original`, `enhanced`, `processed_image`, `enhanced_inverted`. |
| `preprocess_image_advanced(...)` | Tune parameters (denoise, blur, adaptive block, morphology). |
| `extract_text_tesseract(image, config=..., language_codes=...)` | Single Tesseract run with config and PSM fallbacks. |
| `run_multiple_ocr_passes(image_data, language_codes=...)` | Runs Tesseract on enhanced, original, and inverted with multiple configs; merges results. |
| `run_handwriting_ocr(image_path, language_codes)` | EasyOCR for handwriting. |
| `run_paddleocr(image_path, language_codes)` | PaddleOCR for printed/handwritten. |
