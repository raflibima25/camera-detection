# Camera Detection

Python desktop application for real-time detection from a webcam:
- Face recognition + face landmarks (InsightFace + CUDA)
- Person detection + pose skeleton (MediaPipe)
- License plate recognition (fast-alpr)

Output: `cv2.imshow` window directly on the desktop.  
Face enrollment: web browser at `http://localhost:8000`.

---

## Setup

### 1. Create a virtual environment

**Windows (CMD):**
```cmd
python -m venv .venv
.venv\Scripts\activate
```

**Windows (Git Bash):**
```bash
python -m venv .venv
source .venv/Scripts/activate
```

**Linux / WSL:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

**Windows (CMD/Git Bash):**
```cmd
.venv\Scripts\pip.exe install -r requirements.txt
```

**Linux / WSL:**
```bash
pip install -r requirements.txt
```

> **NVIDIA GPU:** Make sure `onnxruntime-gpu` is installed (already in requirements.txt).  
> Without GPU: switch to `onnxruntime==1.21.0` in requirements.txt and set `DEVICE=cpu` in `.env`.

### 3. Configure

**Windows:**
```cmd
copy .env.example .env
```

**Linux / WSL:**
```bash
cp .env.example .env
```

Edit `.env` as needed:

```env
CAMERA_INDEX=0      # webcam index (0 = default)
DEVICE=cuda         # cuda | cpu
ENABLE_FACE=true    # enable/disable each detector
ENABLE_POSE=true
ENABLE_PLATE=true
```

---

## Running the Desktop Application

**Windows (CMD):**
```cmd
.venv\Scripts\python.exe -m app.main
```

**Windows (Git Bash) / Linux:**
```bash
python -m app.main
```

The camera window opens. Press **`q`** to quit.

---

## Enroll Faces via CLI

**Windows (CMD):**
```cmd
.venv\Scripts\python.exe -m app.enroll --photo C:\path\photo.jpg --name "Full Name"
.venv\Scripts\python.exe -m app.enroll --list
.venv\Scripts\python.exe -m app.enroll --delete 3
```

**Windows (Git Bash) / Linux:**
```bash
python -m app.enroll --photo /path/photo.jpg --name "Full Name"
python -m app.enroll --list
python -m app.enroll --delete 3
```

---

## Web Face Enrollment

Run in a separate terminal (the camera app can keep running):

**Windows (CMD) — recommended:**
```cmd
.venv\Scripts\python.exe -m uvicorn enroll_web.server:app --port 8000
```

**Windows (Git Bash) / Linux:**
```bash
python -m uvicorn enroll_web.server:app --port 8000
```

Open `http://localhost:8000` in your browser.

> **Note for Git Bash + `--reload`:** Uvicorn `--reload` on Windows Git Bash sometimes spawns a subprocess using the system Python instead of the venv. Use CMD or run without `--reload` (restart manually when editing code).

---

## NVIDIA GPU — CUDA Setup

Required for InsightFace to run on GPU (~30+ FPS vs ~4 FPS on CPU):

1. Install **CUDA Toolkit 12.x** from [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)
2. Install **cuDNN 9.x** from [developer.nvidia.com/cudnn-downloads](https://developer.nvidia.com/cudnn-downloads)  
   (Select Windows → x86_64 → exe local → install)
3. Set `.env`: `DEVICE=cuda`
4. Make sure `onnxruntime-gpu==1.21.0` is installed

> The code automatically detects the CUDA bin from the `CUDA_PATH` env var or by scanning the filesystem. No manual PATH setup required.

---

## Phase Status

| Phase | Description | Status |
|---|---|---|
| 0 | Scaffold + camera window | ✅ Done |
| 1 | Face recognition + name + landmarks | ✅ Done |
| 2 | Person detection + pose skeleton | ✅ Done |
| 3 | License plate recognition | ✅ Done |
| 4 | Web face enrollment + auto-reload | ✅ Done |

---

## Troubleshooting

### `cv2.imshow` not appearing (WSL2)
- **Windows 11** — WSLg is built-in, works out of the box.
- **Windows 10** — install [VcXsrv](https://sourceforge.net/projects/vcxsrv/) and set `DISPLAY=:0`.
- Alternative: run Python directly from Windows (not WSL Python).

### InsightFace stays on CPU despite `DEVICE=cuda`
- Make sure `onnxruntime-gpu` is installed (not `onnxruntime-directml`):
  ```cmd
  .venv\Scripts\pip.exe show onnxruntime-gpu
  ```
- Make sure CUDA Toolkit 12.x is installed and `cudart64_12.dll` is present.

### `ModuleNotFoundError` when running uvicorn
- Use the venv Python explicitly: `.venv\Scripts\python.exe -m uvicorn ...`
- Do not use `uvicorn` directly as it may resolve to the system Python.

### MediaPipe `mp.solutions` not found
- Install a compatible version: `.venv\Scripts\pip.exe install mediapipe==0.10.14`
