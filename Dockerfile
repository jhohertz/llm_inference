FROM ubuntu:26.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
  curl wget ca-certificates && \
  rm -rf /var/lib/apt/lists/*
WORKDIR /home/ubuntu
USER 1000
RUN curl -fsSL jsoftware.com/download/j9.7/jinstall.sh | sh -s -- --qt none
CMD ["/bin/bash"]
