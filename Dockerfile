FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends nginx \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Docker Spaces run the container as a non-root user (uid 1000)
# by default. Create that user now so ownership can be handed over to it
# below, before anything (Nginx paths, the HF model cache, the Chroma store)
# gets written as root during the build.
RUN useradd -m -u 1000 user

WORKDIR /app

COPY requirements.txt .

# sentence-transformers pulls in torch as a transitive dependency. Installing
# it here first, from PyTorch's CPU-only wheel index, satisfies that
# dependency before the requirements.txt install below runs — otherwise pip
# resolves torch's default GPU/CUDA build (torch itself plus ~15 nvidia-*
# CUDA packages, several GB) even though this image only ever runs on CPU
# (HF Spaces' free tier has no GPU).
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN rm -f /etc/nginx/sites-enabled/default
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Redirect the HF Hub cache (bge-m3's weights land here) to a path under /app
# that the runtime `user` will own, instead of root's default ~/.cache.
ENV HF_HOME=/app/.cache/huggingface

# ingestion/output/ is gitignored (regenerated, not committed), so a fresh
# clone needs `ingestion.run` before `retrieval.build_index` can find chunks.jsonl.
RUN python -m ingestion.run

# Bakes the retrieval index (embeddings + BM25) into the image at build time so
# /health's index_loaded is true immediately at container start, and so the
# corpus doesn't get re-embedded with bge-m3 on every container start. This
# downloads bge-m3's ~2.27GB weights during the build — expect a slow first build.
RUN python -m retrieval.build_index

RUN chmod +x start.sh

# Redirect nginx's pid file to /tmp (world-writable by default): the stock
# /run/nginx.pid path isn't reachable by the non-root runtime user, since
# /var/run (aliased to /run on Debian) is a symlink that `chown -R .../var/run`
# doesn't actually traverse into. Editing nginx.conf directly here rather than
# passing `-g "pid ...;"` at runtime in start.sh — nginx treats a directive
# passed via -g as a duplicate (fatal error) when the same directive already
# exists in the config file, which the stock nginx.conf's `pid /run/nginx.pid;`
# does.
RUN sed -i 's#^pid /run/nginx.pid;#pid /tmp/nginx.pid;#' /etc/nginx/nginx.conf

# Everything above this point (apt-get nginx install, pip install, ingestion,
# embedding/index build) needed root. Hand ownership of everything the
# runtime process touches — the app dir (incl. the baked index and HF cache)
# and Nginx's log/body paths — over to the non-root runtime user, then switch
# to it for the actual running container.
RUN chown -R user:user /app /var/log/nginx /var/lib/nginx

USER user

EXPOSE 7860

CMD ["bash", "start.sh"]
