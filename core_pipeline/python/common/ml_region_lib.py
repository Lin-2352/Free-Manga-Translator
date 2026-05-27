"""
ml_region_lib.py  —  Multi-Model manga text detection & replacement pipeline.

Model A:   comic-text-detector (ONNX) — detects Japanese text bounding boxes
           AND produces pixel-level text segmentation masks.
Model A-S: Magi (ragavsachdeva/magi, PyTorch) — SOTA manga understanding model.
           Provides supplementary text detection + dialogue confidence scores.
Model B:   manga109-segmentation-bubble (YOLOv11n-seg, PyTorch) — segments speech bubbles.
Model D:   manga-ocr (kha-white/manga-ocr-base, PyTorch) — Japanese OCR for text
           content analysis. Used to classify SFX vs dialogue via Unicode script analysis.
Model C:   LaMa (Large Mask Inpainting, ONNX) — deep learning inpainting that
           reconstructs background art (screentones, line art) behind removed text.

All inference models are LOCKED to CUDA (RTX 4060). No CPU fallback.

Workflow:
  Step 1: Model A + Magi detect text boxes (union for recall). Model A provides seg mask.
  Step 2: Model B segments speech bubble pixel masks.
  Step 2b: Route each text box:
           - Inside bubble → "bubble_dialogue" → erase
           - Outside bubble → OCR with Model D → classify as dialogue or SFX
  Step 3: Dynamic mask builder (route-specific dilation + bubble clipping).
  Step 4: Selective erasure — route-aware mask fed to LaMa.
  Step 5: English text rendered inside erased regions.
"""

from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import os

                                                                     
                                            
                                                                     
                                                                          
                                                                              
                                                                          
                                               
try:
    import torch
    torch_lib = os.path.join(os.path.dirname(torch.__file__), 'lib')
    if os.path.exists(torch_lib):
        os.environ['PATH'] = torch_lib + os.pathsep + os.environ.get('PATH', '')
        if hasattr(os, 'add_dll_directory'):
            os.add_dll_directory(torch_lib)
except ImportError:
    pass
                                                                     


                                                                             
     
                                                                             

@dataclass(frozen=True)
class Box:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> int:
        return self.width * self.height

    def expanded(self, pad: int, img_w: int, img_h: int) -> Box:
        return Box(
            x1=max(0, self.x1 - pad),
            y1=max(0, self.y1 - pad),
            x2=min(img_w, self.x2 + pad),
            y2=min(img_h, self.y2 + pad),
        )

    def to_dict(self) -> dict:
        return {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2,
                "width": self.width, "height": self.height}


                                                                             
        
                                                                             

@dataclass
class MLConfig:
                                   
    text_model_path: str = "models/comictextdetector.pt.onnx"
    confidence_threshold: float = 0.05
    nms_iou_threshold: float = 0.45
    input_size: int = 1024

                                                                            
    semantic_model_path: str = "ragavsachdeva/magi"
    semantic_confidence: float = 0.20
    semantic_nms_iou: float = 0.45
    semantic_input_size: int = 1600
    semantic_max_det: int = 400

                                                                 
    semantic_dialogue_confidence: float = 0.40
    semantic_onomatopoeia_confidence: float = 0.55
    semantic_post_nms_iou: float = 0.35
    semantic_min_box_area: int = 160
    semantic_max_box_area_ratio: float = 0.18
    semantic_min_dim: int = 10
    semantic_min_ink_ratio: float = 0.015
    semantic_max_ink_ratio: float = 0.70
    semantic_min_edge_ratio: float = 0.004

                                                     
    bubble_model_path: str = "models/manga109_bubble/best.pt"
    bubble_confidence: float = 0.50
    bubble_overlap_threshold: float = 0.50                                              

                                    
    lama_model_path: str = "models/lama/lama_fp32.onnx"

              
    seg_threshold: float = 0.50                                   
    seg_dilate_kernel: int = 3                                                    
    seg_dilate_iterations: int = 1

                                          
    mask_padding: int = 8
    adaptive_block_size: int = 15
    adaptive_c: int = 4
    dilate_kernel_size: int = 3
    dilate_iterations: int = 1
    inpaint_radius: int = 3

                  
    font_path: Optional[str] = None
    font_size_max: int = 28
    font_size_min: int = 8
    font_color: Tuple[int, int, int] = (0, 0, 0)
    line_spacing: float = 1.3

                         
    green_color: Tuple[int, int, int] = (0, 255, 0)                  
    red_color: Tuple[int, int, int] = (0, 0, 255)                      
    yellow_color: Tuple[int, int, int] = (0, 255, 255)                  
    bubble_outline_color: Tuple[int, int, int] = (255, 180, 0)                         
    box_thickness: int = 2


                                                                     
                                        
                                                                     

def load_text_model(model_path: str, allow_cpu: bool = False):
    """Load ONNX text detector. STRICT CUDA-only."""
    import onnxruntime as ort

    try:
        session = ort.InferenceSession(
            model_path, providers=['CUDAExecutionProvider'],
        )
    except Exception as e:
        raise RuntimeError(
            f"CUDA GPU REQUIRED for Model A (text detector) but unavailable.\n"
            f"Error: {e}\n"
            f"Fix: Add CUDA 12.x bin/ to PATH (cublasLt64_12.dll, cudnn*.dll)\n"
        ) from e

    print("  [Model A] Text detector: CUDA LOCKED")
    return session


def load_semantic_model(model_path: str, allow_cpu: bool = False):
    import transformers
    from transformers import AutoModel, AutoConfig
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    
                     
    if getattr(transformers.PreTrainedModel, "all_tied_weights_keys", None) is None or isinstance(transformers.PreTrainedModel.all_tied_weights_keys, property):
        def get_tied(self): 
            v = getattr(self, '_tied_weights_keys', {})
            return v if (v is not None and getattr(v, "keys", None)) else {}
        def set_tied(self, value): self._tied_weights_keys = value
        transformers.PreTrainedModel.all_tied_weights_keys = property(get_tied, set_tied)

    repo_id = model_path or "ragavsachdeva/magi"
    if repo_id == "magi":
        repo_id = "ragavsachdeva/magi"
    config = AutoConfig.from_pretrained(repo_id, trust_remote_code=True)
    if getattr(config, "detection_model_config", None) and getattr(config.detection_model_config, "backbone_config", None) is None:
        from transformers import TimmBackboneConfig
        config.detection_model_config.backbone_config = TimmBackboneConfig(
            backbone="resnet50", num_channels=3, features_only=True, use_pretrained_backbone=False, out_indices=[4]
        )
        config.detection_model_config.use_timm_backbone = True

    model = AutoModel.from_config(config, trust_remote_code=True)
    sf_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors")
    
    state_dict = load_file(sf_path)
    new_state_dict = {}
    for k, v in state_dict.items():
        new_k = k
        if "conv_encoder.model" in new_k: new_k = new_k.replace("conv_encoder.model", "model")
                                                                                             
        if new_k.startswith("detection_transformer."):
            new_k = new_k.replace(".fc1.", ".mlp.fc1.").replace(".fc2.", ".mlp.fc2.")
            for sa in ["v", "qpos", "kpos", "qcontent", "kcontent"]:
                new_k = new_k.replace(f".sa_{sa}_proj.", f".self_attn.{sa}_proj.")
            new_k = new_k.replace(".self_attn.out_proj.", ".self_attn.o_proj.")
            for ca in ["v", "qpos", "kpos", "qcontent", "kcontent", "qpos_sine"]:
                new_k = new_k.replace(f".ca_{ca}_proj.", f".encoder_attn.{ca}_proj.")
            new_k = new_k.replace(".encoder_attn.out_proj.", ".encoder_attn.o_proj.")
                                                                                              
            for prefix in ["self_attn.", "encoder_attn."]:
                for old, new in [("kcontent_proj", "k_content_proj"), ("kpos_proj", "k_pos_proj"),
                                 ("qcontent_proj", "q_content_proj"), ("qpos_proj", "q_pos_proj"),
                                 ("qpos_sine_proj", "q_pos_sine_proj")]:
                    new_k = new_k.replace(f".{prefix}{old}", f".{prefix}{new}")
        new_state_dict[new_k] = v
        
    model.load_state_dict(new_state_dict, strict=False)
    model.to('cuda:0')
    print("  [Model A-S] Semantic detector (Magi): CUDA:0 LOCKED")
    return model


def load_ocr_model(force_cpu: bool = False):
    """Load manga-ocr (kha-white/manga-ocr-base) for Japanese text recognition."""
    from manga_ocr import MangaOcr
    mocr = MangaOcr(force_cpu=force_cpu)
    print("  [Model D] manga-ocr: LOADED")
    return mocr


def classify_text_by_content(text: str) -> str:
    """
    Classify OCR'd CJK text as 'dialogue', 'sfx', 'english', or 'noise'.
    Aggressively filters out all non-dialogue elements.
    Returns 'dialogue' for definite multi-character Japanese, Korean, or Chinese
    text while still treating short emphatic glyph runs as SFX/noise.
    """
    if not text or not text.strip():
        return 'noise'

    text_clean = text.strip()

                                                                             
    letters = [c for c in text_clean if c.isalpha()]
    if letters:
        ascii_letters = [c for c in letters if c.isascii()]
        if len(ascii_letters) / len(letters) > 0.20:
            return 'english'

                                                       
    script_text = "".join(c for c in text_clean if c.isalnum())
    if not script_text:
        return 'noise'

                                                                                               
    if script_text.isdigit():
        return 'noise'

                               
    katakana = sum(1 for c in script_text if '\u30A0' <= c <= '\u30FF' or c == 'ー')
    hiragana = sum(1 for c in script_text if '\u3040' <= c <= '\u309F')
    kanji    = sum(1 for c in script_text if '\u4E00' <= c <= '\u9FFF')
    hangul   = sum(
        1 for c in script_text
        if '\uAC00' <= c <= '\uD7AF'
        or '\u1100' <= c <= '\u11FF'
        or '\u3130' <= c <= '\u318F'
    )
    cjk_total = katakana + hiragana + kanji + hangul

    if cjk_total == 0:
                                    
        return 'noise'

    katakana_ratio = katakana / cjk_total

                                           
                                                                               
                                                                             
    if hangul >= 3:
        return 'dialogue'
    if hangul >= 2 and len(script_text) >= 3:
        return 'dialogue'
    if hangul > 0 and len(script_text) <= 2:
        return 'sfx'

                                                                            
                                                                             
    if kanji >= 2 and hiragana == 0 and katakana == 0:
        return 'dialogue'
    if kanji == 1 and hiragana == 0 and katakana == 0:
        return 'sfx'

                                                 
                                                    
    if katakana_ratio >= 0.65 and len(script_text) <= 10:
        return 'sfx'
                                                                          
    if len(script_text) <= 4 and kanji == 0:
        return 'sfx'
                                                                             
    if hiragana == cjk_total and len(script_text) <= 5:
        return 'sfx'

                                                                                         
    if kanji >= 1:
        return 'dialogue'
    if hiragana >= 3:
        return 'dialogue'
    if hiragana >= 2 and katakana >= 1:
        return 'dialogue'

                                                                            
    return 'noise'



def ocr_classify_region(
    ocr_model,
    image: np.ndarray,
    box: "Box",
    pad: int = 4,
) -> Tuple[str, str]:
    """
    Crop the image region defined by `box`, run manga-ocr, classify the text.
    
    Returns: (ocr_text, classification) where classification is 'dialogue' or 'sfx'
    """
    from PIL import Image
    h, w = image.shape[:2]
    y1 = max(0, box.y1 - pad)
    y2 = min(h, box.y2 + pad)
    x1 = max(0, box.x1 - pad)
    x2 = min(w, box.x2 + pad)
    
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return ('', 'dialogue')
    
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    pil_crop = Image.fromarray(crop_rgb)
    
    try:
        ocr_text = ocr_model(pil_crop)
    except Exception:
        ocr_text = ''
    
    classification = classify_text_by_content(ocr_text)
    return (ocr_text, classification)


def _preprocess(image: np.ndarray, input_size: int) -> np.ndarray:
    resized = cv2.resize(image, (input_size, input_size))
    blob = resized.astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))
    blob = np.expand_dims(blob, axis=0)
    return blob


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> List[int]:
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep: List[int] = []
    while len(order) > 0:
        i = order[0]
        keep.append(int(i))
        if len(order) == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[np.where(iou <= iou_thresh)[0] + 1]
    return keep


@dataclass
class TextDetectionResult:
    boxes: List[Box]
    seg_mask: np.ndarray                                                     


@dataclass
class SemanticModelHandle:
    model: object
    device: str
    backend: str                            
    input_name: Optional[str] = None
    size_input_name: Optional[str] = None
    label_map: Dict[int, str] = field(default_factory=dict)


@dataclass
class SemanticTextRegion:
    box: Box
    class_id: int
    raw_class_name: str
    semantic_class: str                                
    action: str                                     
    confidence: float


@dataclass
class SemanticDetectionResult:
    regions: List[SemanticTextRegion]


def detect_text(session, image: np.ndarray, cfg: MLConfig) -> TextDetectionResult:
    """
    Run comic-text-detector. Returns BOTH bounding boxes AND pixel-level
    text segmentation mask at original image resolution.
    """
    h_orig, w_orig = image.shape[:2]
    sx, sy = w_orig / cfg.input_size, h_orig / cfg.input_size
    blob = _preprocess(image, cfg.input_size)

                                                                       
    outputs = session.run(None, {"images": blob})
    blk = outputs[0][0]                        
    seg_raw = outputs[1][0][0]                               

                                    
    obj = blk[:, 4]
    mask = obj > cfg.confidence_threshold
    preds, obj_filtered = blk[mask], obj[mask]
    boxes = []
    if len(preds) > 0:
        cx, cy, w, h = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
        corners = np.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], axis=1)
        keep = _nms(corners, obj_filtered, cfg.nms_iou_threshold)
        corners = corners[keep]
        for d in corners:
            bx1, by1 = int(max(0, d[0]*sx)), int(max(0, d[1]*sy))
            bx2, by2 = int(min(w_orig, d[2]*sx)), int(min(h_orig, d[3]*sy))
            if bx2 > bx1 and by2 > by1:
                boxes.append(Box(x1=bx1, y1=by1, x2=bx2, y2=by2))

                                                              
    seg_binary = (seg_raw > cfg.seg_threshold).astype(np.uint8)
    seg_resized = cv2.resize(seg_binary, (w_orig, h_orig),
                             interpolation=cv2.INTER_NEAREST)
                                                      
    if cfg.seg_dilate_kernel > 0:
        kernel = np.ones((cfg.seg_dilate_kernel, cfg.seg_dilate_kernel), np.uint8)
        seg_resized = cv2.dilate(seg_resized, kernel,
                                 iterations=cfg.seg_dilate_iterations)
    seg_resized = seg_resized * 255            

    return TextDetectionResult(boxes=boxes, seg_mask=seg_resized)


def detect_semantic_text_regions(
    semantic_model,
    image: np.ndarray,
    cfg: MLConfig,
) -> SemanticDetectionResult:
    """
    Run Magi to extract text bounding boxes AND dialogue confidence scores.
    
    Magi's `is_this_text_a_dialogue` MLP head outputs a per-text-box confidence
    score indicating whether the text is dialogue (high) or non-dialogue/SFX (low).
    We now extract this alongside the bounding boxes for more informed routing.
    """
    import torch
    from PIL import Image

    h, w = image.shape[:2]
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    with torch.no_grad():
        magi_res = semantic_model.predict_detections_and_associations([np.array(image_pil)])[0]
        
    magi_texts = magi_res.get('texts', [])
    magi_dialog_conf = magi_res.get('dialog_confidences', [])
    
    regions = []
    for idx, tx in enumerate(magi_texts):
        bx = Box(
            x1=int(max(0, min(w, tx[0]))),
            y1=int(max(0, min(h, tx[1]))),
            x2=int(max(0, min(w, tx[2]))),
            y2=int(max(0, min(h, tx[3])))
        )
                                                              
        dialog_conf = float(magi_dialog_conf[idx]) if idx < len(magi_dialog_conf) else 0.5
        
        regions.append(
            SemanticTextRegion(
                box=bx,
                class_id=1,
                raw_class_name='dialogue_detected',
                semantic_class='dialogue',
                action='erase',
                confidence=dialog_conf
            )
        )
        
    return SemanticDetectionResult(regions=regions)


                                                                     
                                                                 
                                                                     

def load_bubble_model(model_path: str, allow_cpu: bool = False):
    """Load YOLOv11n bubble segmentor. STRICT CUDA-only."""
    import torch
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU REQUIRED for Model B (bubble segmentor) but unavailable.\n"
            "Fix: Install PyTorch with CUDA support."
        )

    model = YOLO(model_path)
    model.to('cuda:0')
    device = 'cuda:0'

    print(f"  [Model B] Bubble segmentor: CUDA LOCKED")
    return model, device


def detect_bubbles(
    model,
    device: str,
    image: np.ndarray,
    cfg: MLConfig,
) -> List[np.ndarray]:
    """
    Run bubble segmentation. Returns list of binary masks [H, W] at original
    image resolution. Each mask: 255 = inside bubble, 0 = outside.
    
    Post-processing: Validates that each detected region is actually a speech
    bubble by checking that the interior is predominantly white (>55% of 
    pixels above brightness threshold 200). This eliminates false positives
    on clothes, character bodies, and background art.
    """
    h, w = image.shape[:2]
    results = model.predict(
        source=image, device=device, conf=cfg.bubble_confidence,
        imgsz=1600, verbose=False, retina_masks=True,
    )
    r = results[0]
    masks = []
    if r.masks is not None:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        for m in r.masks.data:
            mask_np = m.cpu().numpy()
            mask_resized = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_LINEAR)
            binary = (mask_resized > 0.5).astype(np.uint8) * 255

            interior_pixels = gray[binary > 127]
            if len(interior_pixels) < 200:
                continue                                 

                                                                     
                                                        
            white_ratio = np.mean(interior_pixels > 200)
            if white_ratio < 0.70:
                continue                                                           

                                                                                  
            ys_b, xs_b = np.nonzero(binary)
            if len(ys_b) == 0:
                continue
            bw = int(xs_b.max()) - int(xs_b.min()) + 1
            bh = int(ys_b.max()) - int(ys_b.min()) + 1
                                                                                               
                                                                                    
            if bh > bw * 2.5 and len(interior_pixels) < 5000:
                continue

            masks.append(binary)
    return masks


                                                                     
                                         
                                                                     

def load_lama_model(model_path: str, allow_cpu: bool = False):
    """Load LaMa inpainting ONNX model. STRICT CUDA-only."""
    import onnxruntime as ort

    try:
        session = ort.InferenceSession(
            model_path, providers=['CUDAExecutionProvider'],
        )
    except Exception as e:
        raise RuntimeError(
            f"CUDA GPU REQUIRED for Model C (LaMa inpainter) but unavailable.\n"
            f"Error: {e}\n"
            f"Fix: Add CUDA 12.x bin/ to PATH"
        ) from e

    print("  [Model C] LaMa inpainter: CUDA LOCKED")
    return session


def lama_inpaint(
    lama_session,
    image: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """
    Run LaMa inpainting on an image region.
    image: BGR uint8 [H, W, 3]
    mask:  uint8 [H, W] — 255 = inpaint, 0 = keep
    Returns: inpainted BGR uint8 [H, W, 3]
    """
    h_orig, w_orig = image.shape[:2]

                                                                                    
                                                                                           
    pad_h = (32 - (h_orig % 32)) % 32
    pad_w = (32 - (w_orig % 32)) % 32

    if pad_h > 0 or pad_w > 0:
        img_padded = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
        mask_padded = cv2.copyMakeBorder(mask, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
    else:
        img_padded = image
        mask_padded = mask

                                                 
    img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
    img_tensor = img_rgb.astype(np.float32) / 255.0
    img_tensor = np.transpose(img_tensor, (2, 0, 1))[np.newaxis]

                                               
    mask_tensor = (mask_padded > 127).astype(np.float32)
    mask_tensor = mask_tensor[np.newaxis, np.newaxis]

    try:
                                       
        output = lama_session.run(None, {
            "image": img_tensor,
            "mask": mask_tensor,
        })[0]
    except Exception as e:
                                                                 
        print(f"LaMa native resolution failed ({e}), falling back to 512x512 resize...")
        img_resized = cv2.resize(image, (512, 512))
        mask_resized = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
        
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_tensor = (img_rgb.astype(np.float32) / 255.0).transpose((2, 0, 1))[np.newaxis]
        mask_tensor = (mask_resized > 127).astype(np.float32)[np.newaxis, np.newaxis]
        
        output = lama_session.run(None, {"image": img_tensor, "mask": mask_tensor})[0]
        
        result = output[0]
        result = np.transpose(result, (1, 2, 0))
        result = np.clip(result * 255, 0, 255).astype(np.uint8)
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        result = cv2.resize(result, (w_orig, h_orig))
        return result

                                                      
    result = output[0]
    result = np.transpose(result, (1, 2, 0))
    result = np.clip(result * 255, 0, 255).astype(np.uint8)
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    
                           
    result = result[:h_orig, :w_orig]

    return result


                                                                     
                                                          
                                                                     

@dataclass
class ClassifiedText:
    box: Box
    expanded_box: Box
    text_type: str                                                                  
    bubble_idx: int                                      
    overlap: float                                       
    semantic_type: str = "unknown"                                            
    route_state: str = "unknown"                                                               
    action: str = "unknown"                                                      
    mask_mode: str = "stroke"                                      
    raw_class_name: str = ""
    confidence: float = 0.0
    overlap_collision: bool = False                                   
    green_polygon: list = field(default_factory=list)


def _find_maximal_rectangle(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """O(M*N) maximal rectangle in binary image using histogram stack approach."""
    rows, cols = mask.shape
    if rows == 0 or cols == 0:
        return None
    max_area = 0
    best_rect = None
    heights = np.zeros(cols, dtype=np.int32)
    for i in range(rows):
        for j in range(cols):
            if mask[i, j] > 0:
                heights[j] += 1
            else:
                heights[j] = 0
        stack = []
        for j in range(cols + 1):
            h = heights[j] if j < cols else 0
            while stack and h < heights[stack[-1]]:
                top = stack.pop()
                width = j if not stack else j - stack[-1] - 1
                area = heights[top] * width
                if area > max_area:
                    max_area = area
                    x_start = j - width
                    y_start = i - heights[top] + 1
                    best_rect = (x_start, y_start, j, i + 1)
            stack.append(j)
    return best_rect


def _get_bubble_mir_internal(bubble_mask: np.ndarray, cx: int, cy: int):
    """Find the MIR of the bubble component containing (cx, cy)."""
    h, w = bubble_mask.shape
    cx = max(0, min(w - 1, cx))
    cy = max(0, min(h - 1, cy))
    
                               
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((bubble_mask > 127).astype(np.uint8), connectivity=8)
    
                                                                       
    label_id = labels[cy, cx]
    if label_id == 0:
        best_d = 10000
        for r in range(1, 51, 5):
            for dy in [-r, r]:
                for dx in range(-r, r + 1, 5):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= nx < w and 0 <= ny < h and labels[ny, nx] > 0:
                        d = dx*dx + dy*dy
                        if d < best_d:
                            best_d = d
                            label_id = labels[ny, nx]
            if label_id > 0: break
            
    if label_id == 0: return None

    comp_mask = (labels == label_id).astype(np.uint8) * 255
    x, y, bw, bh = stats[label_id, cv2.CC_STAT_LEFT], stats[label_id, cv2.CC_STAT_TOP], \
                   stats[label_id, cv2.CC_STAT_WIDTH], stats[label_id, cv2.CC_STAT_HEIGHT]
    
    roi_mask = comp_mask[y:y+bh, x:x+bw]
    mir = _find_maximal_rectangle(roi_mask)
    if mir:
        return (x + mir[0], y + mir[1], x + mir[2], y + mir[3])
    return None

def _clamp_to_mask_pixel_perfect(x1, y1, x2, y2, mask, safety=4):
    """
    Strictly shrinks a rectangle until all pixels inside it are contained 
    within the mask (mask > 127). Handles coordinate mismatches robustly.
    """
    h, w = mask.shape
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    if x2 <= x1 or y2 <= y1: return x1, y1, x2, y2

                                                               
    roi = mask[y1:y2, x1:x2]
    if roi.size > 0 and np.all(roi > 127):
        return int(x1 + safety), int(y1 + safety), int(x2 - safety), int(y2 - safety)

                                                         
    for _ in range(300):
        roi = mask[y1:y2, x1:x2]
        if roi.size == 0: break
        if np.all(roi > 127):
            break
            
                                                                                        
                                                 
        bad_top = np.count_nonzero(mask[y1, x1:x2] <= 127)
        bad_bot = np.count_nonzero(mask[y2-1, x1:x2] <= 127)
        bad_lft = np.count_nonzero(mask[y1:y2, x1] <= 127)
        bad_rgt = np.count_nonzero(mask[y1:y2, x2-1] <= 127)
        
        mx = max(bad_top, bad_bot, bad_lft, bad_rgt)
        if mx == 0:                
            x1 += 1
            x2 -= 1
            y1 += 1
            y2 -= 1
        elif mx == bad_top: y1 += 1
        elif mx == bad_bot: y2 -= 1
        elif mx == bad_lft: x1 += 1
        elif mx == bad_rgt: x2 -= 1
            
        if x2 <= x1 or y2 <= y1: break
        
                         
    x1, y1, x2, y2 = x1 + safety, y1 + safety, x2 - safety, y2 - safety
    
                                                                                      
    if x2 <= x1 or y2 <= y1:
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        return int(cx), int(cy), int(cx + 1), int(cy + 1)
    
    return int(x1), int(y1), int(x2), int(y2)

def _clamp_to_mask_centered(cx, cy, x1, y1, x2, y2, mask, safety=4):
    """
    Symmetrically shrinks the box [x1, y1, x2, y2] around (cx, cy) 
    until it is entirely contained within the mask. 
    MANDATE: Must share the exact center of the Red Box.
    """
    h, w = mask.shape
    cx, cy = float(cx), float(cy)
    
                                   
    dx = max(0.0, abs(x1 - cx), abs(x2 - cx))
    dy = max(0.0, abs(y1 - cy), abs(y2 - cy))
    
    for _ in range(300):
                                     
        nx1, ny1 = int(cx - dx), int(cy - dy)
        nx2, ny2 = int(cx + dx), int(cy + dy)
        
                                  
        cx1, cy1 = max(0, nx1), max(0, ny1)
        cx2, cy2 = min(w, nx2), min(h, ny2)
        
        if cx2 <= cx1 or cy2 <= cy1: break
        
        roi = mask[cy1:cy2, cx1:cx2]
        if roi.size > 0 and np.all(roi > 127):
                                               
            return int(cx1 + safety), int(cy1 + safety), int(cx2 - safety), int(cy2 - safety)
            
                                                                      
                                            
        bad_h = np.count_nonzero(mask[cy1, cx1:cx2] <= 127) + np.count_nonzero(mask[cy2-1, cx1:cx2] <= 127)
        bad_v = np.count_nonzero(mask[cy1:cy2, cx1] <= 127) + np.count_nonzero(mask[cy1:cy2, cx2-1] <= 127)
        
        if bad_h >= bad_v and bad_h > 0:
            dy -= 1.0
        elif bad_v > 0:
            dx -= 1.0
        else:                                  
            dx -= 1.0
            dy -= 1.0
            
        if dx < 0 or dy < 0: break
        
    return int(cx), int(cy), int(cx + 1), int(cy + 1)

def consolidate_by_bubble(routed: List[ClassifiedText], seg_mask: np.ndarray, bubble_masks: List[np.ndarray], cfg: MLConfig, gray_image: np.ndarray = None) -> List[ClassifiedText]:
    """
    Groups and merges ClassifiedText regions using Hierarchical Masking,
    Edge-Based Voronoi Partitioning, and Dynamic Shrink.
    """
    if not routed: return []
    h, w = seg_mask.shape

                                             
    valid_bubbles = []
    for bmask in bubble_masks:
        if gray_image is not None:
            ys, xs = np.nonzero(bmask)
            if len(ys) > 0:
                mean_val = np.mean(gray_image[ys, xs])
                if mean_val >= 180:
                    valid_bubbles.append(bmask)
            else:
                valid_bubbles.append(bmask)
        else:
            valid_bubbles.append(bmask)

    bubble_masks.clear()
    bubble_masks.extend(valid_bubbles)

    consolidated = []
    handled_mask = np.zeros((h, w), dtype=np.uint8)

                                                        
    for b_idx, bmask in enumerate(bubble_masks):
        M_bubble_text = cv2.bitwise_and(seg_mask, bmask)
        ys, xs = np.nonzero(M_bubble_text > 127)
        if len(ys) == 0:
            continue

                                                               
        canvas = np.zeros((h, w), dtype=np.uint8)
        canvas[ys, xs] = 255

                                                                         
        run_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 30))
        closed_runs = cv2.morphologyEx(canvas, cv2.MORPH_CLOSE, run_kernel)

        run_contours, _ = cv2.findContours(closed_runs, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        individual_red_boxes = []
        for c in run_contours:
            c_x, c_y, c_w, c_h = cv2.boundingRect(c)
            if c_w * c_h < 30:
                continue
                                                                         
            c_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(c_mask, [c], -1, 255, -1)
            raw_in_run = cv2.bitwise_and(canvas, c_mask)
            ys_r, xs_r = np.nonzero(raw_in_run)
            if len(ys_r) == 0:
                continue
                                                      
            rx1 = int(xs_r.min())
            ry1 = int(ys_r.min())
            rx2 = int(xs_r.max()) + 1
            ry2 = int(ys_r.max()) + 1
            if (rx2 - rx1) * (ry2 - ry1) >= 30:
                individual_red_boxes.append({
                    'coords': (rx1, ry1, rx2, ry2)
                })

                                                         
                                                                                                           
        merged = True
        while merged:
            merged = False
            for i in range(len(individual_red_boxes)):
                for j in range(i + 1, len(individual_red_boxes)):
                    b1 = individual_red_boxes[i]['coords']
                    b2 = individual_red_boxes[j]['coords']
                    
                    h_gap = max(0, max(b1[0], b2[0]) - min(b1[2], b2[2]))
                    v_overlap = max(0, min(b1[3], b2[3]) - max(b1[1], b2[1]))
                    min_h = min(b1[3] - b1[1], b2[3] - b2[1])
                    
                    if h_gap < 25 and v_overlap > 0.3 * min_h:
                        new_rx1 = min(b1[0], b2[0])
                        new_ry1 = min(b1[1], b2[1])
                        new_rx2 = max(b1[2], b2[2])
                        new_ry2 = max(b1[3], b2[3])
                        
                        individual_red_boxes[i]['coords'] = (new_rx1, new_ry1, new_rx2, new_ry2)
                        individual_red_boxes.pop(j)
                        merged = True
                        break
                if merged:
                    break
        
        bubble_red_boxes = individual_red_boxes
        
        if not bubble_red_boxes:
            continue
            
                        
        ys_b, xs_b = np.nonzero(bmask)
        bx1, by1 = int(xs_b.min()), int(ys_b.min())
        bx2, by2 = int(xs_b.max()), int(ys_b.max())
        bw = bx2 - bx1
        bh = by2 - by1
        kernel_size = max(7, int(min(bw, bh) * 0.06)) 
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        safe_bubble = cv2.erode(bmask, kernel, iterations=1)
        
        if np.count_nonzero(safe_bubble) == 0:
            safe_bubble = bmask.copy()
            
        ys_safe, xs_safe = np.nonzero(safe_bubble)
        territory_map = np.full((h, w), -1, dtype=np.int32)
        
        if len(ys_safe) > 0:
            dist_maps = []
                                                               
            for box in bubble_red_boxes:
                rx1, ry1, rx2, ry2 = box['coords']
                                                                 
                mask = np.ones_like(bmask, dtype=np.uint8) * 255
                cv2.rectangle(mask, (int(rx1), int(ry1)), (int(rx2), int(ry2)), 0, -1)
                
                                                                  
                dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
                dist_maps.append(dist)

                                                                     
            if dist_maps:
                dist_stack = np.stack(dist_maps)
                territory_ids_arr = np.argmin(dist_stack, axis=0)
                territory_map[ys_safe, xs_safe] = territory_ids_arr[ys_safe, xs_safe]
            
        for i, box in enumerate(bubble_red_boxes):
            grp_rx1, grp_ry1, grp_rx2, grp_ry2 = box['coords']
            individual_boxes = box.get('individual_boxes', [{'coords': (grp_rx1, grp_ry1, grp_rx2, grp_ry2)}])

            territory_mask = ((territory_map == i) & (safe_bubble > 0)).astype(np.uint8) * 255
            t_contours, _ = cv2.findContours(territory_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            poly_points = []
            if t_contours:
                largest_c = max(t_contours, key=cv2.contourArea)
                poly_points = [point[0].tolist() for point in largest_c]

            if not poly_points or len(poly_points) < 3:
                poly_points = [[grp_rx1, grp_ry1], [grp_rx2, grp_ry1],
                               [grp_rx2, grp_ry2], [grp_rx1, grp_ry2]]

            gx_arr = [p[0] for p in poly_points]
            gy_arr = [p[1] for p in poly_points]
            gx1, gy1 = min(gx_arr), min(gy_arr)
            gx2, gy2 = max(gx_arr), max(gy_arr)

                                                                  
            for ind_box in individual_boxes:
                rx1, ry1, rx2, ry2 = ind_box['coords']

                intersecting_routes = []
                max_conf = 0.0
                for r in routed:
                    ix1 = max(rx1, r.box.x1)
                    iy1 = max(ry1, r.box.y1)
                    ix2 = min(rx2, r.box.x2)
                    iy2 = min(ry2, r.box.y2)
                    if ix1 < ix2 and iy1 < iy2:
                        intersecting_routes.append(r.route_state)
                        max_conf = max(max_conf, r.confidence)

                primary_route = max(set(intersecting_routes), key=intersecting_routes.count) if intersecting_routes else "bubble_dialogue"

                consolidated.append(ClassifiedText(
                    box=Box(rx1, ry1, rx2, ry2),
                    expanded_box=Box(gx1, gy1, gx2, gy2),
                    text_type=primary_route,
                    bubble_idx=b_idx,
                    overlap=1.0,
                    semantic_type="dialogue" if primary_route != "onomatopoeia" else "onomatopoeia",
                    route_state=primary_route,
                    action="erase",
                    mask_mode="bubble_interior",
                    confidence=max_conf if max_conf > 0 else 0.5,
                    raw_class_name=f"cluster_{b_idx}",
                    overlap_collision=False,
                    green_polygon=poly_points
                ))
            
        handled_mask = np.maximum(handled_mask, bmask)

                                     
    floating_seg = cv2.bitwise_and(seg_mask, cv2.bitwise_not(handled_mask))
    floating_canvas = floating_seg.copy()

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 12))
    closed_floating_canvas = cv2.morphologyEx(floating_canvas, cv2.MORPH_CLOSE, kernel)

    final_floating_contours, _ = cv2.findContours(closed_floating_canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in final_floating_contours:
        rx, ry, rw, rh = cv2.boundingRect(c)
                                                 
        if rw * rh < 300: continue
                                                                          
        if rw < 10 or rh < 10: continue

        rx1, ry1 = rx, ry
        rx2, ry2 = rx + rw, ry + rh
        cx, cy = rx + rw / 2.0, ry + rh / 2.0

        intersecting_routes = []
        max_conf = 0.0
        for r in routed:
            ix1 = max(rx1, r.box.x1)
            iy1 = max(ry1, r.box.y1)
            ix2 = min(rx2, r.box.x2)
            iy2 = min(ry2, r.box.y2)
            if ix1 < ix2 and iy1 < iy2:
                intersecting_routes.append(r.route_state)
                max_conf = max(max_conf, r.confidence)

        primary_route = max(set(intersecting_routes), key=intersecting_routes.count) if intersecting_routes else "floating_dialogue"
        if primary_route == "bubble_dialogue":
            primary_route = "floating_dialogue"
            
        scale = 1.118
        new_w = rw * scale
        new_h = rh * scale

        gx1 = max(0, int(cx - (new_w / 2.0)))
        gy1 = max(0, int(cy - (new_h / 2.0)))
        gx2 = min(w, int(cx + (new_w / 2.0)))
        gy2 = min(h, int(cy + (new_h / 2.0)))
        
        poly_points = [[gx1, gy1], [gx2, gy1], [gx2, gy2], [gx1, gy2]]

        consolidated.append(ClassifiedText(
            box=Box(rx1, ry1, rx2, ry2),
            expanded_box=Box(gx1, gy1, gx2, gy2),
            text_type=primary_route,
            bubble_idx=-1,
            overlap=0.0,
            semantic_type="dialogue" if primary_route != "onomatopoeia" else "onomatopoeia",
            route_state=primary_route,
            action="careful_erase" if primary_route == "floating_dialogue" else "skip_protect",
            mask_mode="stroke",
            confidence=max_conf if max_conf > 0 else 0.5,
            raw_class_name=f"floating_{primary_route}",
            overlap_collision=False,
            green_polygon=poly_points
        ))

                                               
                                                                     
                                                                              
    for ct in consolidated:
        rb = ct.box
        gb = ct.expanded_box
        poly = ct.green_polygon

                                           
        min_gx1 = int(rb.x1 - max(1, rb.width * 0.005))
        min_gy1 = int(rb.y1 - max(1, rb.height * 0.005))
        min_gx2 = int(rb.x2 + max(1, rb.width * 0.005))
        min_gy2 = int(rb.y2 + max(1, rb.height * 0.005))

        new_gx1 = min(gb.x1, min_gx1)
        new_gy1 = min(gb.y1, min_gy1)
        new_gx2 = max(gb.x2, min_gx2)
        new_gy2 = max(gb.y2, min_gy2)

                                                         
        if ct.bubble_idx != -1 and ct.bubble_idx < len(bubble_masks):
            bm = bubble_masks[ct.bubble_idx]
            ys_b, xs_b = np.nonzero(bm)
            if len(ys_b) > 0:
                b_x1, b_y1 = int(xs_b.min()), int(ys_b.min())
                b_x2, b_y2 = int(xs_b.max()) + 1, int(ys_b.max()) + 1
                bw = b_x2 - b_x1
                bh = b_y2 - b_y1
                margin_x = max(1, int(bw * 0.05))
                margin_y = max(1, int(bh * 0.05))
                new_gx1 = max(new_gx1, b_x1 + margin_x)
                new_gy1 = max(new_gy1, b_y1 + margin_y)
                new_gx2 = min(new_gx2, b_x2 - margin_x)
                new_gy2 = min(new_gy2, b_y2 - margin_y)

        ct.expanded_box = Box(new_gx1, new_gy1, new_gx2, new_gy2)
        if poly and len(poly) >= 3:
                                                
            ct.green_polygon = [
                [max(new_gx1, min(new_gx2, p[0])), max(new_gy1, min(new_gy2, p[1]))]
                for p in poly
            ]

    return consolidated


def build_step2_routing_state(
    text_result: TextDetectionResult,
    semantic_result: SemanticDetectionResult,
    bubble_masks: List[np.ndarray],
    cfg: MLConfig,
    img_w: int,
    img_h: int,
    ocr_model=None,
    image: np.ndarray = None,
) -> List[ClassifiedText]:
    """
    Step 2 routing-state builder (OCR-powered).

    Produces one of exactly three route states:
      1) bubble_dialogue   (inside bubble)                          -> erase
      2) floating_dialogue  (outside bubble, OCR says dialogue)     -> careful_erase
      3) onomatopoeia       (outside bubble, OCR says SFX)          -> skip_protect
      
    Classification strategy:
      - Text inside bubbles → always bubble_dialogue (no OCR needed, white bg is safe)
      - Text outside bubbles → run manga-ocr on the cropped region → classify by
        Unicode script composition (katakana = SFX, hiragana/kanji = dialogue)
      - Magi dialogue confidence used as supporting signal (not sole classifier)
      
    Detection recall boost:
      - After processing all Model A boxes, Magi text boxes that don't overlap
        with any already-routed region are added as supplementary detections.
    """
    routed: List[ClassifiedText] = []

    def _is_point_in_box(px, py, b: Box):
        return b.x1 <= px <= b.x2 and b.y1 <= py <= b.y2
    
    def _boxes_overlap_significantly(b1: Box, b2: Box, threshold: float = 0.3) -> bool:
        """Check if two boxes overlap by at least `threshold` of the smaller box's area."""
        ix1 = max(b1.x1, b2.x1)
        iy1 = max(b1.y1, b2.y1)
        ix2 = min(b1.x2, b2.x2)
        iy2 = min(b1.y2, b2.y2)
        if ix2 <= ix1 or iy2 <= iy1:
            return False
        inter = (ix2 - ix1) * (iy2 - iy1)
        min_area = min(max(1, b1.area), max(1, b2.area))
        return inter / min_area > threshold
        
    magi_dialogues = [reg.box for reg in semantic_result.regions if reg.semantic_class == "dialogue"]
    magi_dialog_confs = {id(reg.box): reg.confidence for reg in semantic_result.regions}

    for original_box in text_result.boxes:
        roi_text = text_result.seg_mask[original_box.y1:original_box.y2, original_box.x1:original_box.x2].copy()
        
                                                             
        for i, bmask in enumerate(bubble_masks):
            roi_bmask = bmask[original_box.y1:original_box.y2, original_box.x1:original_box.x2]
            text_in_this_bubble = cv2.bitwise_and(roi_text, roi_bmask)
            
            if np.count_nonzero(text_in_this_bubble) > 20:
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 5))
                dilated = cv2.dilate(text_in_this_bubble, kernel, iterations=1)
                contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                                                                         
                ys_b, xs_b = np.nonzero(bmask)
                if len(ys_b) == 0: continue
                b_x1, b_y1, b_x2, b_y2 = xs_b.min(), ys_b.min(), xs_b.max() + 1, ys_b.max() + 1
                
                for c in contours:
                    cx, cy, cw, ch = cv2.boundingRect(c)
                    if cw * ch > 100:                                 
                        c_roi = text_in_this_bubble[cy:cy+ch, cx:cx+cw]
                        ys, xs = np.nonzero(c_roi)
                        if len(xs) > 0 and len(ys) > 0:
                            strict_box = Box(
                                int(original_box.x1 + cx + xs.min()), 
                                int(original_box.y1 + cy + ys.min()),
                                int(original_box.x1 + cx + xs.max() + 1), 
                                int(original_box.y1 + cy + ys.max() + 1)
                            )

                            if strict_box.area < 120:                          
                                continue

                                                                                           
                            ocr_text = ''
                            ocr_class = 'dialogue'
                            if ocr_model is not None and image is not None:
                                ocr_text, ocr_class = ocr_classify_region(ocr_model, image, strict_box)
                            
                                                                         
                            if ocr_class in ['english', 'sfx', 'noise']:
                                continue
                                
                            expanded = Box(
                                x1=max(b_x1, strict_box.x1 - cfg.mask_padding),
                                y1=max(b_y1, strict_box.y1 - cfg.mask_padding),
                                x2=min(b_x2, strict_box.x2 + cfg.mask_padding),
                                y2=min(b_y2, strict_box.y2 + cfg.mask_padding)
                            )
                            
                            routed.append(ClassifiedText(
                                box=strict_box,
                                expanded_box=expanded,
                                text_type="bubble_dialogue", 
                                bubble_idx=i,
                                overlap=1.0,
                                semantic_type="dialogue",
                                route_state="bubble_dialogue",
                                action="erase",
                                raw_class_name=f"bubble_dialogue|{ocr_text}",
                                confidence=1.0,
                            ))
                
                                                                               
                roi_text = cv2.bitwise_and(roi_text, cv2.bitwise_not(roi_bmask))

                                                                                      
        if np.count_nonzero(roi_text) > 30:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
            dilated = cv2.dilate(roi_text, kernel, iterations=1)
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for c in contours:
                cx, cy, cw, ch = cv2.boundingRect(c)
                if cw * ch > 100:                      
                    c_roi = roi_text[cy:cy+ch, cx:cx+cw]
                    ys, xs = np.nonzero(c_roi)
                    if len(xs) > 0 and len(ys) > 0:
                        strict_box = Box(
                            int(original_box.x1 + cx + xs.min()), 
                            int(original_box.y1 + cy + ys.min()),
                            int(original_box.x1 + cx + xs.max() + 1), 
                            int(original_box.y1 + cy + ys.max() + 1)
                        )

                        if strict_box.area < 120:
                            continue
                        
                                                            
                        ocr_text = ''
                        ocr_class = 'dialogue'                
                        
                        if ocr_model is not None and image is not None:
                            ocr_text, ocr_class = ocr_classify_region(ocr_model, image, strict_box)
                        
                                                                      
                        if ocr_class in ['english', 'sfx', 'noise']:
                            continue

                        route_state = "floating_dialogue"
                        action = "careful_erase"
                        semantic_type = "dialogue"
                            
                        expanded = strict_box.expanded(cfg.mask_padding, img_w, img_h)
                        routed.append(ClassifiedText(
                            box=strict_box,
                            expanded_box=expanded,
                            text_type=route_state, 
                            bubble_idx=-1,
                            overlap=0.0,
                            semantic_type=semantic_type,
                            route_state=route_state,
                            action=action,
                            raw_class_name=f"{route_state}|{ocr_text}",
                            confidence=1.0,
                        ))

                                                                                               
    routed_boxes = [r.box for r in routed]
    for magi_reg in semantic_result.regions:
        m_box = magi_reg.box
        if m_box.area < 100:
            continue
        
        already_covered = False
        for rb in routed_boxes:
            if _boxes_overlap_significantly(m_box, rb, 0.3):
                already_covered = True
                break
        
        if already_covered:
            continue
            
                                                            
                                       
        inside_bubble = False
        best_bubble_idx = -1
        for i, bmask in enumerate(bubble_masks):
            roi_bmask = bmask[m_box.y1:m_box.y2, m_box.x1:m_box.x2]
            if roi_bmask.size > 0 and np.count_nonzero(roi_bmask) / max(1, roi_bmask.size) > 0.4:
                inside_bubble = True
                best_bubble_idx = i
                break
        
        if inside_bubble:
            ys_b, xs_b = np.nonzero(bubble_masks[best_bubble_idx])
            if len(ys_b) == 0: continue
            b_x1, b_y1, b_x2, b_y2 = xs_b.min(), ys_b.min(), xs_b.max() + 1, ys_b.max() + 1
            expanded = Box(
                x1=max(b_x1, m_box.x1 - cfg.mask_padding),
                y1=max(b_y1, m_box.y1 - cfg.mask_padding),
                x2=min(b_x2, m_box.x2 + cfg.mask_padding),
                y2=min(b_y2, m_box.y2 + cfg.mask_padding)
            )
            routed.append(ClassifiedText(
                box=m_box, expanded_box=expanded,
                text_type="bubble_dialogue", bubble_idx=best_bubble_idx,
                overlap=1.0, semantic_type="dialogue",
                route_state="bubble_dialogue", action="erase",
                raw_class_name="magi_supplementary_bubble",
                confidence=magi_reg.confidence,
            ))
        else:
                                           
            ocr_text = ''
            ocr_class = 'dialogue'
            if ocr_model is not None and image is not None:
                ocr_text, ocr_class = ocr_classify_region(ocr_model, image, m_box)
            
                                          
            if ocr_class in ['english', 'sfx', 'noise']:
                continue
            
            route_state = "floating_dialogue"
            action = "careful_erase"
            semantic_type = "dialogue"
            
            expanded = m_box.expanded(cfg.mask_padding, img_w, img_h)
            routed.append(ClassifiedText(
                box=m_box, expanded_box=expanded,
                text_type=route_state, bubble_idx=-1,
                overlap=0.0, semantic_type=semantic_type,
                route_state=route_state, action=action,
                raw_class_name=f"magi_supp_{route_state}|{ocr_text}",
                confidence=magi_reg.confidence,
            ))

                            
                                                                                                                    
    routed_bubble_indices = set(r.bubble_idx for r in routed if r.bubble_idx != -1)
    
    for i, bmask in enumerate(bubble_masks):
        if i not in routed_bubble_indices:
            ys_b, xs_b = np.nonzero(bmask)
            if len(ys_b) == 0: continue
            
            cy, cx = int(ys_b.mean()), int(xs_b.mean())
            mir = _get_bubble_mir_internal(bmask, cx, cy)
            
            if mir:
                mx1, my1, mx2, my2 = mir
                if (mx2 - mx1) * (my2 - my1) > 100:
                    rescue_box = Box(mx1, my1, mx2, my2)
                    
                    ocr_text = ''
                    ocr_class = 'dialogue'
                    if ocr_model is not None and image is not None:
                        ocr_text, ocr_class = ocr_classify_region(ocr_model, image, rescue_box)
                    
                    if ocr_class == 'english' or not ocr_text.strip():
                        continue 
                        
                    b_x1, b_y1, b_x2, b_y2 = xs_b.min(), ys_b.min(), xs_b.max() + 1, ys_b.max() + 1
                    expanded = Box(
                        x1=max(b_x1, rescue_box.x1 - cfg.mask_padding),
                        y1=max(b_y1, rescue_box.y1 - cfg.mask_padding),
                        x2=min(b_x2, rescue_box.x2 + cfg.mask_padding),
                        y2=min(b_y2, rescue_box.y2 + cfg.mask_padding)
                    )
                    
                    routed.append(ClassifiedText(
                        box=rescue_box, expanded_box=expanded,
                        text_type="bubble_dialogue", bubble_idx=i,
                        overlap=1.0, semantic_type="dialogue",
                        route_state="bubble_dialogue", action="erase",
                        raw_class_name="empty_bubble_rescue",
                        confidence=0.5,
                    ))

    return routed


def draw_step2_routing_debug(
    image: np.ndarray,
    routed: List[ClassifiedText],
) -> np.ndarray:
    """
    Step 2 debug drawing:
      - bubble_dialogue in Green
      - floating_dialogue in Cyan (Yellow in BGR)
      - onomatopoeia in Red
    """
    debug = image.copy()
    for ct in routed:
        if ct.route_state == "bubble_dialogue":
            color = (0, 255, 0)
        elif ct.route_state == "floating_dialogue":
            color = (255, 255, 0)                                                                                                                                                   
        elif ct.route_state == "onomatopoeia":
            color = (0, 0, 255)
        else:
            continue

        b = ct.box
        cv2.rectangle(debug, (b.x1, b.y1), (b.x2, b.y2), color, 2)
        label = f"{ct.route_state} ({ct.raw_class_name}) {ct.confidence:.2f}"
        cv2.putText(
            debug, label, (b.x1, max(12, b.y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA,
        )
    return debug


                                                                     
                                                                
                                                                     

def _extract_text_strokes(image: np.ndarray, coords: Tuple[int, int, int, int], is_bubble: bool) -> np.ndarray:
    x1, y1, x2, y2 = coords
    bh = y2 - y1
    bw = x2 - x1
    if bh <= 0 or bw <= 0:
        return np.zeros((max(1, bh), max(1, bw)), dtype=np.uint8)
    
                    
    img_roi = image[y1:y2, x1:x2]
    
                  
    gray = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
    
    if is_bubble:
                                                          
                                                                     
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
                                                                          
        _, strokes = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        
                                           
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(strokes, connectivity=8)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < 3:
                strokes[labels == i] = 0
                
                                           
        strokes = cv2.morphologyEx(strokes, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
        
    else:
                                                                    
                                                                            
        median_val = np.median(gray)
                                                                                            
                                                                
        thresh_type = cv2.THRESH_BINARY if median_val < 160 else cv2.THRESH_BINARY_INV
        
                                                                          
        gray = cv2.medianBlur(gray, 3)
        
                           
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
                                                                   
        strokes = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresh_type,
            17,                          
            12,                             
        )
        
                                                            
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(strokes, connectivity=8)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < 10:                                                 
                strokes[labels == i] = 0
                
                                        
        strokes = cv2.morphologyEx(strokes, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
        
    return strokes


def build_step3_dynamic_mask(
    image: np.ndarray,
    routed: "List[ClassifiedText]",
    seg_mask: np.ndarray,
    cfg: "MLConfig",
    bubble_masks: "List[np.ndarray]" = None,
) -> np.ndarray:
    """
    Step 3: Build a full-page 1-channel binary inpainting mask from actual
    text stroke pixels extracted via adaptive thresholding.
    
    Route-specific dilation (strict limits applied after smoothing):
      - bubble_dialogue:   (3,3) kernel, 2 iterations — fat enough to prevent
                           LaMa smudges on AA edges.
      - floating_dialogue: (2,2) kernel, 1 iteration — tight to protect art.
      - onomatopoeia:      SKIP — artwork fully protected, zero mask.
    
    CRITICAL: For bubble_dialogue, the final dilated mask is AND-ed against
    the actual pixel-level bubble segmentation mask so that NO stroke pixel
    ever leaks outside the organic contour of the speech bubble.
    
    Returns: 1-channel uint8 mask, shape (H, W), 0=keep / 255=erase.
             Strictly binary. No debug text, no bounding boxes, no RGB.
    """
    h, w = image.shape[:2]
    bubble_mask_out = np.zeros((h, w), dtype=np.uint8)
    floating_mask_out = np.zeros((h, w), dtype=np.uint8)
    
                                                                           
    bubble_union = None
    if bubble_masks:
        bubble_union = np.zeros((h, w), dtype=np.uint8)
        for bm in bubble_masks:
            bubble_union = np.maximum(bubble_union, bm)
    
    for ct in routed:
        box = ct.box
        
                                                                            
        pad = 4
        y1, y2 = max(0, box.y1 - pad), min(h, box.y2 + pad)
        x1, x2 = max(0, box.x1 - pad), min(w, box.x2 + pad)
        
                                                                                        
                                                                                            
                                                                                  
                                                        
        if ct.route_state == "floating_dialogue":
            y1 = max(0, y1 - 50)
            
        padded_coords = (x1, y1, x2, y2)
        
        if ct.route_state == "bubble_dialogue":
                                       
            box_area = (x2 - x1) * (y2 - y1)
            roi_seg_check = seg_mask[y1:y2, x1:x2]
            seg_density = np.count_nonzero(roi_seg_check) / max(1, box_area)
            if box_area > 8000 and seg_density < 0.05:
                continue
            if box_area > 3000 and seg_density < 0.03:
                continue
                
            roi_seg = seg_mask[y1:y2, x1:x2].copy()
            _, roi_seg_bin = cv2.threshold(roi_seg, 127, 255, cv2.THRESH_BINARY)
            
            if np.count_nonzero(roi_seg_bin) < 10:
                strokes = _extract_text_strokes(image, (x1, y1, x2, y2), is_bubble=True)
            else:
                strokes = roi_seg_bin
                
                                                  
            kernel = np.ones((3, 3), np.uint8)
            strokes = cv2.dilate(strokes, kernel, iterations=2)
            
                                                                        
            if bubble_union is not None:
                bubble_roi = bubble_union[y1:y2, x1:x2]
                strokes = cv2.bitwise_and(strokes, bubble_roi)
                
            bubble_mask_out[y1:y2, x1:x2] = np.maximum(bubble_mask_out[y1:y2, x1:x2], strokes)
        
        elif ct.route_state == "floating_dialogue":
                                                                              
                                                                           
                                                                                  
                                                                 
            box_area = (x2 - x1) * (y2 - y1)
            roi_seg_check = seg_mask[y1:y2, x1:x2]
            seg_density = np.count_nonzero(roi_seg_check) / max(1, box_area)
            if box_area > 5000 and seg_density < 0.05:
                continue                                                               
            
                                                                                      
            roi_seg = seg_mask[y1:y2, x1:x2].copy()
            _, roi_seg_bin = cv2.threshold(roi_seg, 127, 255, cv2.THRESH_BINARY)
            
                                                                                                                                                   
            if np.count_nonzero(roi_seg_bin) >= 10:
                roi_mask = roi_seg_bin
            else:
                roi_mask = _extract_text_strokes(image, (x1, y1, x2, y2), is_bubble=False)
                
                                                                     
            dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            dilated_strokes = cv2.dilate(roi_mask, dilate_kernel, iterations=1)
            
            floating_mask_out[y1:y2, x1:x2] = np.maximum(
                floating_mask_out[y1:y2, x1:x2], dilated_strokes
            )

                                     
    
                                          
    final_full_mask = np.maximum(bubble_mask_out, floating_mask_out)
    return final_full_mask


                                                                     
                                         
                                                                     

def expand_boxes(boxes: List[Box], pad: int, img_w: int, img_h: int) -> List[Box]:
    return [b.expanded(pad, img_w, img_h) for b in boxes]


def erase_regions_selective(
    image: np.ndarray,
    classified: List[ClassifiedText],
    seg_mask: np.ndarray,
    bubble_masks: List[np.ndarray],
    lama_session,
    cfg: MLConfig,
) -> np.ndarray:
    """
    Only inpaint text regions classified as "bubble_text" using LaMa.
    Uses the pixel-level seg mask from comic-text-detector (not adaptive threshold).
    Floating text is left untouched (background art protected).
    """
    h, w = image.shape[:2]

                                      
                                                                            
    final_mask = np.zeros((h, w), dtype=np.uint8)

    for ct in classified:
        if ct.text_type != "bubble_text":
            continue                      

                                     
        b = ct.expanded_box
                                                   
        roi_seg = seg_mask[b.y1:b.y2, b.x1:b.x2]
                               
        final_mask[b.y1:b.y2, b.x1:b.x2] = np.maximum(
            final_mask[b.y1:b.y2, b.x1:b.x2], roi_seg
        )

                                            
    if np.count_nonzero(final_mask) == 0:
        return image.copy()

                                                       
    inpainted = lama_inpaint(lama_session, image, final_mask)

                                                                            
    result = image.copy()
    mask_bool = final_mask > 127
    result[mask_bool] = inpainted[mask_bool]

    return result


                                                                     
                                 
                                                                     

def insert_text(image, expanded_boxes, translations, cfg):
    result = image.copy()
    for box, text in zip(expanded_boxes, translations):
        if not text.strip():
            continue
        box_w, box_h = box.x2 - box.x1, box.y2 - box.y1
        if cfg.font_path:
            _insert_text_pillow(result, box, text, cfg)
        else:
            _insert_text_opencv(result, box, text, box_w, box_h, cfg)
    return result


def _insert_text_opencv(image, box, text, box_w, box_h, cfg):
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    scale, min_scale = cfg.font_size_max / 20.0, cfg.font_size_min / 20.0
    while scale >= min_scale:
        thickness = max(1, int(scale))
        (cw, ch), _ = cv2.getTextSize("W", font_face, scale, thickness)
        if cw == 0:
            break
        lines = textwrap.wrap(text, width=max(1, box_w // cw))
        lh = int(ch * cfg.line_spacing)
        if lh * len(lines) <= box_h and all(
            cv2.getTextSize(ln, font_face, scale, thickness)[0][0] <= box_w for ln in lines
        ):
            break
        scale -= 0.05
    if scale < min_scale:
        scale = min_scale
        thickness = max(1, int(scale))
        (cw, ch), _ = cv2.getTextSize("W", font_face, scale, thickness)
        lines = textwrap.wrap(text, width=max(1, box_w // cw) if cw > 0 else 10)
        lh = int(ch * cfg.line_spacing)
    y = box.y1 + max(0, (box_h - lh * len(lines)) // 2) + ch
    for ln in lines:
        tw = cv2.getTextSize(ln, font_face, scale, thickness)[0][0]
        cv2.putText(image, ln, (box.x1 + max(0, (box_w - tw) // 2), y),
                    font_face, scale, cfg.font_color, thickness, cv2.LINE_AA)
        y += lh


def _insert_text_pillow(image, box, text, cfg):
    from PIL import Image, ImageDraw, ImageFont
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    bw, bh = box.x2 - box.x1, box.y2 - box.y1
    fs = cfg.font_size_max
    while fs >= cfg.font_size_min:
        font = ImageFont.truetype(cfg.font_path, fs)
        wrapped = _wrap_text_pillow(draw, text, font, bw)
        lh = int(fs * cfg.line_spacing)
        if lh * len(wrapped) <= bh and all(draw.textlength(l, font=font) <= bw for l in wrapped):
            break
        fs -= 1
    if fs < cfg.font_size_min:
        fs = cfg.font_size_min
        font = ImageFont.truetype(cfg.font_path, fs)
        wrapped = _wrap_text_pillow(draw, text, font, bw)
        lh = int(fs * cfg.line_spacing)
    ys = box.y1 + max(0, (bh - lh * len(wrapped)) // 2)
    rgb = (cfg.font_color[2], cfg.font_color[1], cfg.font_color[0])
    for i, l in enumerate(wrapped):
        tw = draw.textlength(l, font=font)
        draw.text((box.x1 + max(0, (bw - int(tw)) // 2), ys + i * lh), l, fill=rgb, font=font)
    np.copyto(image, cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR))


def _wrap_text_pillow(draw, text, font, max_w):
    words = text.split()
    lines, cur = [], ""
    for w in words:
        test = f"{cur} {w}".strip()
        if draw.textlength(test, font=font) <= max_w:
            cur = test
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines or [text]


                                                                     
                     
                                                                     

def draw_debug_boxes(
    image: np.ndarray,
    classified: List[ClassifiedText],
    bubble_masks: List[np.ndarray],
    seg_mask: np.ndarray,
    cfg: MLConfig,
) -> np.ndarray:
    """
    Clean 3-color debug visualization:
      Blue  (255,0,0)   — Speech bubble boundaries (absolute wall)
      Red   (0,0,255)   — Source text erasure zone (tight ink bounding box)
      Green (0,255,0)   — Typesetting layout zone (polygon or expanded box)
    No yellow, no orange, no collision indicators.
    """
    debug = image.copy()

                                                                        
    for bmask in bubble_masks:
        contours, _ = cv2.findContours(bmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(debug, contours, -1, (255, 0, 0), 2)             

                                                                        
    for ct in classified:
        rb = ct.box
                                 
        cv2.rectangle(debug, (rb.x1, rb.y1), (rb.x2, rb.y2), (0, 0, 255), 2)

                                                                    
        poly = getattr(ct, 'green_polygon', [])
        if poly and len(poly) >= 3:
            pts = np.array(poly, np.int32).reshape((-1, 1, 2))
            cv2.polylines(debug, [pts], isClosed=True, color=(0, 255, 0), thickness=1)
        else:
            gb = ct.expanded_box
            cv2.rectangle(debug, (gb.x1, gb.y1), (gb.x2, gb.y2), (0, 255, 0), 1)

    return debug


def draw_semantic_class_debug(
    image: np.ndarray,
    semantic_result: SemanticDetectionResult,
) -> np.ndarray:
    """
    Draw semantic detector boxes + labels for visual Step-1 verification.
    """
    debug = image.copy()
    for region in semantic_result.regions:
        b = region.box
        color = (0, 255, 0) if region.semantic_class == "dialogue" else (0, 0, 255)
        cv2.rectangle(debug, (b.x1, b.y1), (b.x2, b.y2), color, 2)
        label = f"{region.semantic_class} ({region.raw_class_name}) {region.confidence:.2f}"
        cv2.putText(
            debug, label, (b.x1, max(12, b.y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
        )
    return debug


                                                                     
               
                                                                     

def run_pipeline(
    image_path: str | Path,
    cfg: MLConfig,
    text_session=None,
    bubble_model=None,
    bubble_device: str = "cuda:0",
    lama_session=None,
    semantic_handle=None,
    ocr_model=None,
) -> dict:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")
    h, w = image.shape[:2]

                                                    
    text_result = detect_text(text_session, image, cfg)
    seg_mask = text_result.seg_mask
    print(f"  [Step 1] Model A: {len(text_result.boxes)} text regions")

                                           
    if semantic_handle is None:
        semantic_handle = load_semantic_model(cfg.semantic_model_path)
    semantic_result = detect_semantic_text_regions(semantic_handle, image, cfg)
    print(f"  [Step 1b] Magi: {len(semantic_result.regions)} semantic regions")

                                       
    bubble_masks = detect_bubbles(bubble_model, bubble_device, image, cfg)
    print(f"  [Step 2] Model B: {len(bubble_masks)} speech bubbles")

                                     
    if ocr_model is None:
        ocr_model = load_ocr_model()
    
    routed = build_step2_routing_state(
        text_result, semantic_result, bubble_masks, cfg, w, h, ocr_model, image
    )
    
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    consolidated = consolidate_by_bubble(routed, seg_mask, bubble_masks, cfg, gray_img)
    
    n_bubble = sum(1 for c in consolidated if c.route_state == "bubble_dialogue")
    n_float = sum(1 for c in consolidated if c.route_state == "floating_dialogue")
    print(f"  [Step 3] Consolidation: {n_bubble} bubble dialogue, {n_float} floating dialogue")

                                                              
    erased_image = erase_regions_selective(
        image, consolidated, seg_mask, bubble_masks, lama_session, cfg,
    )
    print(f"  [Step 4] LaMa inpainted {n_bubble} bubble regions")

                 
    debug_image = draw_debug_boxes(image, consolidated, bubble_masks, seg_mask, cfg)

    return {
        "classified": consolidated,
        "bubble_masks": bubble_masks,
        "seg_mask": seg_mask,
        "erased_image": erased_image,
        "debug_image": debug_image,
        "bubble_text": [c for c in consolidated if c.route_state == "bubble_dialogue"],
        "floating_text": [c for c in consolidated if c.route_state == "floating_dialogue"],
    }


                                                                     
              
                                                                     

SAMPLE_MAP = {
    "sample1": "sample.jpg",
    "sample2": "sample 2.jpeg",
    "sample3": "sample 3.jpg",
    "sample4": "sample 4.jpg",
    "sample5": "sample 5.jpg",
    "sample6": "sample 6.jpg",
}



def run_all_samples(
    cfg: MLConfig,
    samples_dir: Path,
    run_name: str,
    text_session,
    bubble_model,
    bubble_device: str,
    lama_session,
):
                                                        
    print("Pre-loading models for batch run...")
    semantic_handle = load_semantic_model(cfg.semantic_model_path)
    ocr_model = load_ocr_model()

    for sample_name, img_file in SAMPLE_MAP.items():
        img_path = samples_dir / sample_name / img_file
        if not img_path.exists():
            print(f"[SKIP] {img_path} not found")
            continue

        sample_dir = samples_dir / sample_name
                                 
        out_dir = sample_dir / "step_final_output"
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Processing {sample_name}: {img_file}")
        print(f"{'='*60}")

        result = run_pipeline(
            img_path, cfg,
            text_session=text_session,
            bubble_model=bubble_model,
            bubble_device=bubble_device,
            lama_session=lama_session,
            semantic_handle=semantic_handle,
            ocr_model=ocr_model,
        )

                                                          
        cv2.imwrite(str(out_dir / "seg_mask.png"), result["seg_mask"])
        cv2.imwrite(str(out_dir / "erased_inpainted.png"), result["erased_image"])
        cv2.imwrite(str(out_dir / "debug_layout_boxes.jpg"), result["debug_image"])

        report = {
            "sample": sample_name,
            "image": img_file,
            "text_conf": cfg.confidence_threshold,
            "bubble_conf": cfg.bubble_confidence,
            "overlap_threshold": cfg.bubble_overlap_threshold,
            "padding": cfg.mask_padding,
            "inpainter": "LaMa",
            "num_text": len(result["classified"]),
            "num_bubble_text": len(result["bubble_text"]),
            "num_floating_text": len(result["floating_text"]),
            "num_bubbles": len(result["bubble_masks"]),
            "bubble_text": [
                {"box": c.box.to_dict(), "overlap": round(c.overlap, 3), "bubble_idx": c.bubble_idx}
                for c in result["bubble_text"]
            ],
            "floating_text": [
                {"box": c.box.to_dict(), "overlap": round(c.overlap, 3)}
                for c in result["floating_text"]
            ],
        }
        (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"  Processed {sample_name}. Output saved to {out_dir}/")


def run_semantic_test_all_samples(
    cfg: MLConfig,
    samples_dir: Path,
    semantic_handle: SemanticModelHandle,
):
    """
    Step-1 verification utility.
    Runs semantic model on all known samples and saves:
      samples/<sampleN>/semantic_test/debug_classes.png
      samples/<sampleN>/semantic_test/report.json
    """
    for sample_name, img_file in SAMPLE_MAP.items():
        img_path = samples_dir / sample_name / img_file
        if not img_path.exists():
            print(f"[SKIP] {img_path} not found")
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            print(f"[SKIP] cannot load {img_path}")
            continue

        semantic_result = detect_semantic_text_regions(semantic_handle, image, cfg)
        debug_img = draw_semantic_class_debug(image, semantic_result)

        out_dir = samples_dir / sample_name / "semantic_test"
        out_dir.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(out_dir / "debug_classes.png"), debug_img)
        report = {
            "sample": sample_name,
            "image": img_file,
            "semantic_model": cfg.semantic_model_path,
            "num_semantic_detections": len(semantic_result.regions),
            "regions": [
                {
                    "box": r.box.to_dict(),
                    "class_id": r.class_id,
                    "raw_class_name": r.raw_class_name,
                    "semantic_class": r.semantic_class,
                    "action": r.action,
                    "confidence": round(r.confidence, 4),
                }
                for r in semantic_result.regions
            ],
        }
        (out_dir / "report.json").write_text(json.dumps(report, indent=2))
        print(f"[Semantic Test] {sample_name}: {len(semantic_result.regions)} detections -> {out_dir}")


def run_step2_routing_test(
    cfg: MLConfig,
    image_path: Path,
    semantic_handle: SemanticModelHandle,
    bubble_model,
    bubble_device: str,
):
    """
    Step 2 visual test:
      1) Run Model A semantic detection with class remap (bubble class dropped)
      2) Use Model B bubble geometry to assign route states
      3) Draw only dialogue (green) and onomatopoeia (red)
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Cannot load step2 test image: {image_path}")

    h, w = image.shape[:2]
    semantic_result = detect_semantic_text_regions(semantic_handle, image, cfg)
    bubble_masks = detect_bubbles(bubble_model, bubble_device, image, cfg)
    routed = build_step2_routing_state(semantic_result, bubble_masks, cfg, w, h)

    debug = draw_step2_routing_debug(image, routed)

    out_dir = image_path.parent / "semantic_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_img = out_dir / "debug_step2_routing.png"
    cv2.imwrite(str(out_img), debug)

    report = {
        "image": str(image_path),
        "semantic_model": cfg.semantic_model_path,
        "num_semantic_regions_after_remap": len(semantic_result.regions),
        "num_bubbles_model_b": len(bubble_masks),
        "route_counts": {
            "bubble_dialogue": sum(1 for r in routed if r.route_state == "bubble_dialogue"),
            "floating_dialogue": sum(1 for r in routed if r.route_state == "floating_dialogue"),
            "onomatopoeia": sum(1 for r in routed if r.route_state == "onomatopoeia"),
        },
        "regions": [
            {
                "box": r.box.to_dict(),
                "raw_class_name": r.raw_class_name,
                "semantic_type": r.semantic_type,
                "route_state": r.route_state,
                "action": r.action,
                "confidence": round(r.confidence, 4),
                "bubble_idx": r.bubble_idx,
                "overlap": round(r.overlap, 4),
            }
            for r in routed
        ],
    }
    (out_dir / "report_step2.json").write_text(json.dumps(report, indent=2))

    print(f"[Step2 Test] semantic_regions(after remap): {len(semantic_result.regions)}")
    print(f"[Step2 Test] bubbles: {len(bubble_masks)}")
    print(f"[Step2 Test] saved: {out_img}")


                                                                     
     
                                                                     

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Dual-model manga text detection pipeline")
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--all-samples", action="store_true")
    parser.add_argument("--samples-dir", type=Path, default=Path("samples"))
    parser.add_argument("--run-name", type=str, default="lama_run_1")
    parser.add_argument("--text-model", type=str, default="models/comictextdetector.pt.onnx")
    parser.add_argument("--bubble-model", type=str, default="models/manga109_bubble/best.pt")
    parser.add_argument("--lama-model", type=str, default="models/lama/lama_fp32.onnx")
    parser.add_argument("--semantic-model", type=str,
                        default="ragavsachdeva/magi")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--bubble-conf", type=float, default=0.50)
    parser.add_argument("--overlap", type=float, default=0.50)
    parser.add_argument("--padding", type=int, default=8)
    parser.add_argument("--semantic-test", action="store_true",
                        help="Run Step-1 semantic detector test on one image and save labeled output.")
    parser.add_argument("--semantic-test-all", action="store_true",
                        help="Run Step-1 semantic detector test on all known samples.")
    parser.add_argument("--step2-routing-test", action="store_true",
                        help="Run Step-2 class-remap + routing-state visual test.")
    parser.add_argument("--semantic-test-image", type=Path,
                        default=Path("samples/sample4/sample4.jpg"))
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()

    cfg = MLConfig(
        text_model_path=args.text_model,
        semantic_model_path=args.semantic_model,
        bubble_model_path=args.bubble_model,
        lama_model_path=args.lama_model,
        confidence_threshold=args.conf,
        bubble_confidence=args.bubble_conf,
        bubble_overlap_threshold=args.overlap,
        mask_padding=args.padding,
    )

                                                                     
    if args.semantic_test:
        test_image_path = args.semantic_test_image

                                                                   
        if not test_image_path.exists() and str(args.semantic_test_image).replace("\\", "/") == "samples/sample4/sample4.jpg":
            candidates = [
                Path("samples/sample4/sample4.jpg"),
                Path("samples/sample4/sample 4 i want.jpg"),
                Path("samples/sample4/sample 4.png"),
            ]
            for c in candidates:
                if c.exists():
                    test_image_path = c
                    break

        image = cv2.imread(str(test_image_path))
        if image is None:
            raise FileNotFoundError(
                f"Cannot load semantic test image: {test_image_path}\n"
                f"Try: samples/sample4/sample4.jpg or samples/sample4/sample 4.png"
            )

        semantic_handle = load_semantic_model(cfg.semantic_model_path, allow_cpu=args.allow_cpu)
        semantic_result = detect_semantic_text_regions(semantic_handle, image, cfg)
        debug_img = draw_semantic_class_debug(image, semantic_result)

        out_dir = test_image_path.parent / "semantic_test"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "debug_classes.png"

        cv2.imwrite(str(out_path), debug_img)
        print(f"[Semantic Test] detections: {len(semantic_result.regions)}")
        print(f"[Semantic Test] image: {test_image_path}")
        print(f"[Semantic Test] saved: {out_path}")
        return

                                                                    
    if args.semantic_test_all:
        semantic_handle = load_semantic_model(cfg.semantic_model_path, allow_cpu=args.allow_cpu)
        run_semantic_test_all_samples(cfg, args.samples_dir, semantic_handle)
        return

                                                                 
    if args.step2_routing_test:
        test_image_path = args.semantic_test_image
        if not test_image_path.exists() and str(args.semantic_test_image).replace("\\", "/") == "samples/sample4/sample4.jpg":
            candidates = [
                Path("samples/sample4/sample4.jpg"),
                Path("samples/sample4/sample 4 i want.jpg"),
                Path("samples/sample4/sample 4.png"),
            ]
            for c in candidates:
                if c.exists():
                    test_image_path = c
                    break

        semantic_handle = load_semantic_model(cfg.semantic_model_path, allow_cpu=args.allow_cpu)
        bubble_model, bubble_device = load_bubble_model(cfg.bubble_model_path, allow_cpu=args.allow_cpu)
        run_step2_routing_test(cfg, test_image_path, semantic_handle, bubble_model, bubble_device)
        return

    print("Loading models...")
    text_session = load_text_model(cfg.text_model_path, allow_cpu=args.allow_cpu)
    bubble_model, bubble_device = load_bubble_model(cfg.bubble_model_path, allow_cpu=args.allow_cpu)
    lama_session = load_lama_model(cfg.lama_model_path, allow_cpu=args.allow_cpu)

    if args.all_samples:
        run_all_samples(cfg, args.samples_dir, args.run_name,
                        text_session, bubble_model, bubble_device, lama_session)
    elif args.image:
        result = run_pipeline(args.image, cfg, text_session, bubble_model,
                              bubble_device, lama_session)
        stem = args.image.stem
        cv2.imwrite(f"{stem}_debug_dual.png", result["debug_image"])
        cv2.imwrite(f"{stem}_erased.png", result["erased_image"])
        n_b = len(result["bubble_text"])
        n_f = len(result["floating_text"])
        print(f"Saved. Bubble text: {n_b}, Floating text: {n_f}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
