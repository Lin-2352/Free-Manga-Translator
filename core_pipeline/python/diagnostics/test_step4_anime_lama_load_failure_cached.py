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

import run_step4_inpaint


class _FakeTorch:
    class jit:
        @staticmethod
        def load(*_args, **_kwargs):
            raise RuntimeError("fake: corrupt checkpoint")

    class cuda:
        @staticmethod
        def is_available():
            return True

    @staticmethod
    def device(name):
        return name


def test_corrupt_checkpoint_load_is_guarded_and_remembered() -> None:
    """Regression test for Tier B #7: the anime-LaMa loader's torch.jit.load()
    call used to be unwrapped, AND _ANIME_LAMA_LOAD_ATTEMPTED was set to True
    only AFTER the (successful) load -- so a genuinely bad/corrupt checkpoint
    file (a) crashed instead of falling back, and (b) if somehow caught upstream,
    would still be retried on every single subsequent request forever (no
    negative caching). The fix mirrors the sibling manga-cleaner loader: guard
    the load with try/except, and set the attempted-flag BEFORE attempting."""
    original_torch = run_step4_inpaint.torch
    original_model = run_step4_inpaint._ANIME_LAMA_MODEL
    original_device = run_step4_inpaint._ANIME_LAMA_DEVICE
    original_attempted = run_step4_inpaint._ANIME_LAMA_LOAD_ATTEMPTED
    original_path = run_step4_inpaint.ANIME_LAMA_PATH

    load_calls = {"n": 0}
    real_load = _FakeTorch.jit.load

    def counting_load(*args, **kwargs):
        load_calls["n"] += 1
        return real_load(*args, **kwargs)

    _FakeTorch.jit.load = staticmethod(counting_load)

    try:
        run_step4_inpaint.torch = _FakeTorch
        run_step4_inpaint._ANIME_LAMA_MODEL = None
        run_step4_inpaint._ANIME_LAMA_DEVICE = None
        run_step4_inpaint._ANIME_LAMA_LOAD_ATTEMPTED = False

        # Point at a path that exists (any real file) so the "missing model
        # file" early-return doesn't short-circuit before reaching torch.jit.load.
        fake_model_path = Path(__file__).resolve()
        run_step4_inpaint.ANIME_LAMA_PATH = fake_model_path

        model1, device1 = run_step4_inpaint._get_anime_lama_model()
        assert model1 is None and device1 is None, (
            f"a failed load must fall back to (None, None), got ({model1!r}, {device1!r})"
        )
        assert run_step4_inpaint._ANIME_LAMA_LOAD_ATTEMPTED is True, (
            "the attempted-flag must be set even though the load failed, so it isn't retried forever"
        )
        assert load_calls["n"] == 1, f"expected exactly one load attempt so far, got {load_calls['n']}"

        # A second call must NOT retry the load (negative caching).
        model2, device2 = run_step4_inpaint._get_anime_lama_model()
        assert model2 is None and device2 is None
        assert load_calls["n"] == 1, (
            f"a corrupt checkpoint must be remembered and NOT retried on every request, "
            f"but load was attempted {load_calls['n']} times across 2 calls"
        )
    finally:
        run_step4_inpaint.torch = original_torch
        run_step4_inpaint._ANIME_LAMA_MODEL = original_model
        run_step4_inpaint._ANIME_LAMA_DEVICE = original_device
        run_step4_inpaint._ANIME_LAMA_LOAD_ATTEMPTED = original_attempted
        run_step4_inpaint.ANIME_LAMA_PATH = original_path


def main() -> int:
    test_corrupt_checkpoint_load_is_guarded_and_remembered()
    print("step4_anime_lama_load_failure_cached=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
