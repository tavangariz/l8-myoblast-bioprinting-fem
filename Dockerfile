# ────────────────────────────────────────────────────────────────
# L8 Myoblast Bioprinting — FEM + ML Surrogate
# Base: official DOLFINx stable (includes FEniCS, gmsh, MPI)
# ────────────────────────────────────────────────────────────────
FROM ghcr.io/fenics/dolfinx/dolfinx:v0.10.0

LABEL maintainer="tavangariz"
LABEL description="L8 myoblast FEM+ML bioprinting pipeline"
LABEL version="1.0"

# Run as an unprivileged user. UID/GID default to 1000, which matches the first
# user on most Linux hosts, so a bind-mounted working directory stays writable.
# Override at build time if your host UID differs: --build-arg UID=$(id -u)
ARG USER=fenics
ARG UID=1000
ARG GID=1000
RUN (groupadd -g ${GID} ${USER} || true) \
 && (useradd -m -u ${UID} -g ${GID} -s /bin/bash ${USER} || true)

WORKDIR /workspace

# Install Python ML/data science packages not in base image
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . /workspace

# Create output directories and hand the workspace to the unprivileged user
RUN mkdir -p results/figures results/optimal_params \
 && chown -R ${UID}:${GID} /workspace

USER ${UID}:${GID}

# Expose Jupyter port
EXPOSE 8888

# Default: Jupyter Lab.
#
# --ip=0.0.0.0 binds inside the container so a published port can reach it.
# Isolation comes from publishing on the host loopback only:
#     docker run -p 127.0.0.1:8888:8888 ...
# Authentication is left at the Jupyter default, so the server generates a
# one-time token and prints it to the container log. Do not disable it.
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", \
     "--no-browser", \
     "--notebook-dir=/workspace"]
