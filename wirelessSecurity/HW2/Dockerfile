FROM ubuntu:22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install required packages
RUN apt-get update && apt-get install -y \
    gcc \
    make \
    build-essential \
    zlib1g-dev \
    check \
    clang-format \
    zip \
    unzip \
    git \
    gdb \
    valgrind \
    vim \
    nano \
    && rm -rf /var/lib/apt/lists/*

# Create a working directory
WORKDIR /homework

# Set environment variables for Check framework debugging
ENV CK_FORK=no

# Create a script to handle user permissions dynamically
RUN echo '#!/bin/bash\n\
# Create or modify student user to match host UID/GID\n\
if [ "$MY_HOST_UID" != "0" ]; then\n\
    groupmod -g $MY_HOST_GID student 2>/dev/null || groupadd -g $MY_HOST_GID student\n\
    usermod -u $MY_HOST_UID -g $MY_HOST_GID student 2>/dev/null || useradd -u $MY_HOST_UID -g $MY_HOST_GID -m -s /bin/bash student\n\
    chown -R student:student /homework\n\
    exec gosu student "$@"\n\
else\n\
    exec "$@"\n\
fi' > /entrypoint.sh && chmod +x /entrypoint.sh

# Install gosu for user switching
RUN apt-get update && apt-get install -y gosu && rm -rf /var/lib/apt/lists/*

# Add a non-root user (will be modified at runtime)
RUN useradd -m -s /bin/bash student

ENTRYPOINT ["/entrypoint.sh"]

CMD ["/bin/bash"]