FROM python:3.11-slim-bookworm@sha256:0bee7276f83efd4a1ee05bbbf4281d95ed28e079220a9457f25a93e3f1e3c31b

# apt packages are deliberately not version-pinned. An exact Debian pin
# (e.g. nginx=1.22.1-9+deb12u9) breaks every future rebuild the moment the
# security pocket rotates and that version leaves the mirror, and it also
# freezes known-old nginx/openssl. Reproducibility that matters is already
# covered by the digest-pinned base image above and the hash-pinned pip set
# below; `apt-get update` on that fixed base just pulls current security
# patches for three low-risk packages (nginx, tini, ca-certificates).
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        nginx \
        tini \
    && rm -rf /var/lib/apt/lists/*

# Run the application as a non-root user (uid 1000) for least privilege and
# because Hugging Face Docker Spaces run containers as uid 1000.
# Create that user before the Nginx paths, model cache, and Chroma store are
# written during the build.
RUN useradd -m -u 1000 user

WORKDIR /app

# Non-recursive and cheap (a single directory inode, not the large layers
# created later) — lets `user` create brand-new nested paths that don't exist
# in the copied source tree at all (ingestion/output/, retrieval/output/,
# .cache/huggingface are all gitignored, never present in `COPY --chown` below).
RUN chown user:user /app

COPY requirements-lock.txt .

# Install the exact tested dependency set in one resolver transaction. The
# additive PyTorch index supplies the pinned CPU-only wheel.
RUN pip install --no-cache-dir --require-hashes -r requirements-lock.txt \
        --extra-index-url https://download.pytorch.org/whl/cpu \
    && pip check

COPY --chown=1000:1000 . .

RUN rm -f /etc/nginx/sites-enabled/default
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Redirect nginx's pid file to /tmp (world-writable by default): the stock
# /run/nginx.pid path isn't reachable by the non-root runtime user, since
# /var/run (aliased to /run on Debian) is a symlink that a chown of /var/run
# doesn't actually traverse into. Editing nginx.conf directly here rather than
# passing `-g "pid ...;"` at runtime in start.sh — nginx treats a directive
# passed via -g as a duplicate (fatal error) when the same directive already
# exists in the config file, which the stock nginx.conf's `pid /run/nginx.pid;`
# does.
RUN sed -i \
        -e 's#^pid /run/nginx.pid;#pid /tmp/nginx.pid;#' \
        -e 's#^error_log /var/log/nginx/error.log;#error_log /dev/stderr;#' \
        -e 's#access_log /var/log/nginx/access.log;#access_log /dev/stdout;#' \
        /etc/nginx/nginx.conf \
    && chmod 0755 start.sh \
    && nginx -t \
    && rm -f /tmp/nginx.pid \
    && rm -rf /tmp/nginx-client /tmp/nginx-fastcgi /tmp/nginx-proxy /tmp/nginx-scgi /tmp/nginx-uwsgi
# nginx -t (run as root, above) leaves a root-owned /tmp/nginx.pid and
# www-data-owned, mode-0700 /tmp/nginx-* temp-path directories baked into the
# image — both unwritable by the uid-1000 runtime user, so nginx would fail
# with "Permission denied" at container start. Deleting them here lets the
# real nginx process recreate them as uid 1000 at startup, owning them from
# creation — same non-recursive-chown philosophy as the /app ownership above.

USER user

# Redirect the HF Hub cache (bge-m3's weights land here) to the runtime user's
# home. It is created as uid 1000 and never needs a later ownership rewrite.
ENV HF_HOME=/home/user/.cache/huggingface

# ingestion/output/ is gitignored (regenerated, not committed), so a fresh
# clone needs the ingestion CLI before the retrieval CLI can find chunks.jsonl.
# Runs as `user` (see USER above), so the output file is owned by `user` from
# creation.
RUN python -m src.features.ingestion.cli

# Bakes the retrieval index (embeddings + BM25) into the image at build time so
# /health's index_loaded is true immediately at container start, and so the
# corpus doesn't get re-embedded with bge-m3 on every container start. This
# downloads bge-m3's ~2.27GB weights during the build — expect a slow first
# build, and note every image rebuild repeats this from scratch (nothing here
# is cached across builds beyond Docker's own layer cache). Runs as `user`,
# so the baked index and cached weights are owned by `user` from creation —
# never needing a later chown -R over this multi-GB content.
RUN python -m src.features.retrieval.cli

ENV HF_HUB_OFFLINE=1

EXPOSE 7860

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/app/start.sh"]
