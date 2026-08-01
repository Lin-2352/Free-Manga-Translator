from __future__ import annotations

import sys
from pathlib import Path

_BOOTSTRAP_FILE = Path(__file__).resolve()
for _candidate in _BOOTSTRAP_FILE.parents:
    if (_candidate / "samples").exists() and (_candidate / "python").exists():
        _PROJECT_ROOT_FOR_IMPORTS = _candidate
        break
else:
    _PROJECT_ROOT_FOR_IMPORTS = _BOOTSTRAP_FILE.parents[2]
for _rel in ("python/common", "python/steps"):
    _path = str(_PROJECT_ROOT_FOR_IMPORTS / _rel)
    if _path not in sys.path:
        sys.path.insert(0, _path)
del _BOOTSTRAP_FILE, _candidate, _PROJECT_ROOT_FOR_IMPORTS, _rel, _path

import os


def test_torch_is_importable_before_paddle_in_paddleocr_reader() -> None:
    """Regression guard for the Windows cuDNN basename collision (WinError 127):
    both torch and paddle vendor their own cuDNN 9.x sub-DLLs, and whichever loads
    second in-process can bind the other's cuDNN siblings at the wrong minor version.
    Reproduced directly: importing paddle before torch in a fresh process raises
    OSError on torch\\lib\\shm.dll. Every real caller of _paddleocr_reader() already
    has torch loaded first (detection/magi/manga-ocr load before this lazy import),
    so this test asserts the source itself enforces that ordering rather than relying
    on caller discipline.
    """
    source = (Path(__file__).resolve().parents[1] / "steps" / "run_step5_ocr.py").read_text(encoding="utf-8")
    import_torch_idx = source.index("import torch  # noqa: F401")
    import_paddle_idx = source.index("import paddle", import_torch_idx)
    assert import_torch_idx < import_paddle_idx, (
        "run_step5_ocr.py must import torch before paddle inside _paddleocr_reader() -- "
        "reversing this order reproduces WinError 127 on torch\\lib\\shm.dll"
    )


def test_paddle_device_respects_manga_paddle_device_override(monkeypatch) -> None:
    """MANGA_PADDLE_DEVICE=cpu must force CPU even when paddle reports CUDA-compiled,
    giving an operator a kill switch without touching code."""
    import run_step5_ocr

    monkeypatch.setenv("MANGA_PADDLE_DEVICE", "cpu")
    monkeypatch.setattr(run_step5_ocr, "_PADDLEOCR_READERS", {})
    monkeypatch.setattr(run_step5_ocr, "_PADDLEOCR_UNAVAILABLE", set())

    calls = []

    class _FakePaddleOCR:
        def __init__(self, lang, device, **kwargs):
            calls.append(device)

    class _FakePaddleDevice:
        @staticmethod
        def is_compiled_with_cuda():
            return True

    class _FakePaddleModule:
        device = _FakePaddleDevice()

    import sys as _sys
    monkeypatch.setitem(_sys.modules, "paddle", _FakePaddleModule())
    fake_paddleocr_module = type(_sys)("paddleocr")
    fake_paddleocr_module.PaddleOCR = _FakePaddleOCR
    monkeypatch.setitem(_sys.modules, "paddleocr", fake_paddleocr_module)

    run_step5_ocr._paddleocr_reader("ch")
    assert calls == ["cpu"], f"MANGA_PADDLE_DEVICE=cpu override was not honored, got device={calls}"


def test_gpu_init_failure_falls_back_to_cpu_not_crash(monkeypatch) -> None:
    """A CUDA-OOM or other GPU init failure at PaddleOCR construction must degrade to
    CPU for the process rather than failing the OCR request outright."""
    import run_step5_ocr

    monkeypatch.delenv("MANGA_PADDLE_DEVICE", raising=False)
    monkeypatch.setattr(run_step5_ocr, "_PADDLEOCR_READERS", {})
    monkeypatch.setattr(run_step5_ocr, "_PADDLEOCR_UNAVAILABLE", set())

    calls = []

    class _FakePaddleOCR:
        def __init__(self, lang, device, **kwargs):
            calls.append(device)
            if device == "gpu:0":
                raise RuntimeError("simulated CUDA OOM at init")

    class _FakePaddleDevice:
        @staticmethod
        def is_compiled_with_cuda():
            return True

    class _FakePaddleModule:
        device = _FakePaddleDevice()

    import sys as _sys
    monkeypatch.setitem(_sys.modules, "paddle", _FakePaddleModule())
    fake_paddleocr_module = type(_sys)("paddleocr")
    fake_paddleocr_module.PaddleOCR = _FakePaddleOCR
    monkeypatch.setitem(_sys.modules, "paddleocr", fake_paddleocr_module)

    reader = run_step5_ocr._paddleocr_reader("ch")
    assert calls == ["gpu:0", "cpu"], f"expected a GPU attempt then a CPU fallback, got {calls}"
    assert reader is not None, "OCR reader must still be constructed after GPU failure, not None"


def main() -> int:
    test_torch_is_importable_before_paddle_in_paddleocr_reader()

    class _Monkeypatch:
        def __init__(self):
            self._undo = []

        def setenv(self, name, value):
            old = os.environ.get(name)
            self._undo.append(lambda: (os.environ.pop(name, None) if old is None else os.environ.__setitem__(name, old)))
            os.environ[name] = value

        def delenv(self, name, raising=False):
            old = os.environ.get(name)
            if old is not None:
                self._undo.append(lambda: os.environ.__setitem__(name, old))
                os.environ.pop(name, None)

        def setattr(self, obj, name, value):
            old = getattr(obj, name)
            self._undo.append(lambda: setattr(obj, name, old))
            setattr(obj, name, value)

        def setitem(self, mapping, key, value):
            had = key in mapping
            old = mapping.get(key)
            self._undo.append(lambda: (mapping.__setitem__(key, old) if had else mapping.pop(key, None)))
            mapping[key] = value

        def undo_all(self):
            for fn in reversed(self._undo):
                fn()
            self._undo.clear()

    mp = _Monkeypatch()
    try:
        test_paddle_device_respects_manga_paddle_device_override(mp)
    finally:
        mp.undo_all()

    mp2 = _Monkeypatch()
    try:
        test_gpu_init_failure_falls_back_to_cpu_not_crash(mp2)
    finally:
        mp2.undo_all()

    print("paddleocr_gpu_device_selection=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
