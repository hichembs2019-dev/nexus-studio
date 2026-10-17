FROM python:3.12-slim

WORKDIR /app

# ffmpeg is used for real, local voiceover merging (episode_pipeline.py
# _attach_narration/_stitch_local) - without it every episode is produced
# silent again even with AZURE_SPEECH_KEY set.
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src

ENV OUTPUT_DIR=/data/output
VOLUME ["/data"]

CMD ["python", "-m", "src.main"]
