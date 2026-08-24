FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends nginx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN rm -f /etc/nginx/sites-enabled/default
COPY nginx.conf /etc/nginx/conf.d/default.conf

# ingestion/output/ is gitignored (regenerated, not committed), so a fresh
# clone needs `ingestion.run` before `retrieval.build_index` can find chunks.jsonl.
RUN python -m ingestion.run

# Bakes the retrieval index (embeddings + BM25) into the image at build time so
# /health's index_loaded is true immediately at container start, and so the
# corpus doesn't get re-embedded with bge-m3 on every container start. This
# downloads bge-m3's ~2.27GB weights during the build — expect a slow first build.
RUN python -m retrieval.build_index

RUN chmod +x start.sh

EXPOSE 7860

CMD ["bash", "start.sh"]
