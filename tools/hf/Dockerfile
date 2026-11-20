# Docker Space for the mandala-gnn cockpit.
#
# The same server that runs locally, on port 7860: real forward passes, per-head
# attention, no recorded answers. The checkpoint is 1.7 MB, so CPU basic is
# plenty; one pass is ~50 ms on the Space hardware.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# torch's CPU wheel wants libgomp at runtime; slim does not ship it.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# CPU-only torch, explicitly: the default index would pull the CUDA stack.
RUN pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install numpy pillow

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app
COPY --chown=user . $HOME/app

EXPOSE 7860

# Lazy by design: /api/state answers from demo.json before torch is imported, so
# the cockpit paints immediately and the first click pays for the load. The
# server binds 0.0.0.0 itself; Spaces proxies port 7860.
CMD ["python", "-m", "mandalanet.serve", "--model", "runs/main-hex/model.pt", "--port", "7860", "--host", "0.0.0.0"]
