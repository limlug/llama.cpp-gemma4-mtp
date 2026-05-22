ARG CUDA_VERSION=12.8.1
ARG UBUNTU_VERSION=24.04
ARG BASE_CUDA_DEV_CONTAINER=nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}
ARG BASE_CUDA_RUN_CONTAINER=nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

# ─── build stage ───────────────────────────────────────────────────────────
FROM ${BASE_CUDA_DEV_CONTAINER} AS build

ARG CUDA_DOCKER_ARCH=default

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      gcc-14 g++-14 build-essential cmake python3 python3-pip git \
      libssl-dev libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV CC=gcc-14 CXX=g++-14 CUDAHOSTCXX=g++-14

WORKDIR /src
COPY . /src

RUN if [ "${CUDA_DOCKER_ARCH}" != "default" ]; then \
        CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=${CUDA_DOCKER_ARCH}"; \
    fi && \
    cmake -B /build \
        -DGGML_NATIVE=OFF \
        -DGGML_CUDA=ON \
        -DGGML_BACKEND_DL=ON \
        -DGGML_CPU_ALL_VARIANTS=ON \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_CURL=OFF \
        -DCMAKE_BUILD_TYPE=Release \
        ${CMAKE_ARGS} \
        /src && \
    cmake --build /build --config Release -j"$(nproc)" --target llama-server

# Pre-stage the runtime files into /out so the runtime stage can COPY
# from a single tidy tree.
RUN mkdir -p /out/bin /out/lib && \
    cp /build/bin/llama-server /out/bin/ && \
    find /build -name '*.so*' -exec cp -P {} /out/lib/ \;

# ─── runtime stage ─────────────────────────────────────────────────────────
FROM ${BASE_CUDA_RUN_CONTAINER} AS runtime

RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

COPY --from=build /out/lib/ /app/
COPY --from=build /out/bin/llama-server /app/

ENV LD_LIBRARY_PATH=/app
ENV LLAMA_ARG_HOST=0.0.0.0
ENV LLAMA_ARG_PORT=8080

WORKDIR /app
EXPOSE 8080
HEALTHCHECK CMD curl -fsS http://localhost:8080/health || exit 1
ENTRYPOINT ["/app/llama-server"]
