FROM python:3.12-slim

WORKDIR /app

# ffmpeg is used for real, local work the dialogue pipeline does itself:
# stitching every real clip into one film (dialogue_pipeline.py's
# _stitch), building the shared caption group photo (_build_caption_group_image),
# and rendering on-screen caption clips (_render_caption_clip).
#
# fonts-noto-core: a real, confirmed-installable Debian/Ubuntu package.
# Added 2026-10-20 for on-screen caption lines (dialogue_pipeline.py's
# spoken=False path, _render_caption_clip) - ffmpeg's own drawtext filter
# needs an actual font FILE with Arabic glyph coverage, and this base
# image's apt-get ffmpeg install pulls in no fonts at all. Without this,
# Arabic captions would either find no font or render wrong/garbled -
# verified live (a real rendered frame, properly shaped Arabic, checked
# before this was added) that fonts-noto-core's NotoNaskhArabic-Regular.ttf
# actually fixes this, not guessed.
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg fonts-noto-core && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
# Real bug found live, 2026-10-19: the image build never copied this
# directory, so every real Character.reference_image_path
# (character_photos.py) pointed at a file that did not exist inside the
# running container - DialoguePipeline raised DialoguePipelineError on
# the very first line of the very first real production run, and that
# exception was never surfaced to the owner (see runner.py's own fix,
# same commit) - they waited over two hours for a reply that could
# never have come.
COPY assets ./assets

ENV OUTPUT_DIR=/data/output
VOLUME ["/data"]

CMD ["python", "-m", "src.main"]
