# Architecture

The runtime path for each page is:

1. Extension captures panel image.
2. Extension sends payload to `POST /v1/translate-image`.
3. Backend invokes `run_extension_pipeline_server.py`.
4. Pipeline executes OCR, layout, translation, inpainting, and typesetting.
5. Backend returns final image and metadata.
6. Extension overlays translated output on page.
