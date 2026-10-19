FROM python:3.12-slim

WORKDIR /app

# ffmpeg is used for real, local voiceover merging (episode_pipeline.py
# _attach_narration/_stitch_local) - without it every episode is produced
# silent again even with AZURE_SPEECH_KEY set.
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

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
