# hw2 Development Environment Setup Guide

This document provides instructions for setting up the development environment for the hw2 assignment. You can choose between Docker environment or local environment setup.

## Docker Environment Setup (Recommended)

Using Docker ensures a consistent development environment.

### Prerequisites - Docker

- Docker installation and execution
- Docker Compose (optional)

For Docker installation, refer to: [https://docs.docker.com/compose/install/]

### Usage

**Check**: Docker must be running before setup

#### Method 1: Using Docker Compose (Recommended)

```bash
# Go to root directory of this HW

# Set current user UID/GID for Docker execution
export MY_UID=$(id -u) MY_GID=$(id -g)

# Build Docker image and run container
docker-compose up -d

# Access container
docker-compose exec hw2-environment bash

# Exit container
exit

# Stop container after work completion
docker-compose down
```

#### Method 2: Direct Docker Usage

```bash
# Build Docker image
docker build -t hw2-env .

# Run container
docker run -it \
  -e MY_UID=$(id -u) \
  -e MY_GID=$(id -g) \
  -v $(pwd):/homework \
  hw2-env
```

**If docker-compose is not available**: Use direct Docker commands.

```bash
# Go to root directory of this HW

# Run development environment in background
docker run -d --name hw2-container \
  -e MY_UID=$(id -u) \
  -e MY_GID=$(id -g) \
  -v $(pwd):/homework \
  hw2-env tail -f /dev/null

# Access container
docker exec -it hw2-container bash

# Cleanup after work completion
docker stop hw2-container && docker rm hw2-container
```

#### Method 3: Simple Root Execution (Ignoring Permission Issues)

If the above methods are too complex:

```bash
# Go to root directory of this HW

# Build Docker image
docker build -t hw2-env .

# Run with root privileges (simple but not recommended)
docker run -it --user root -v $(pwd):/homework hw2-env
```

### Docker Troubleshooting

#### When Permission Issues Occur

**Check**: First Check Docker is running

**Method 1**: Explicitly set MY_UID/MY_GID

```bash
# Check your UID, GID
id

# Run container using those values
docker run -it -e MY_UID={YOUR_UID} -e MY_GID={YOUR_GID} -v $(pwd):/homework hw2-env
```

**Method 2**: Modify file permissions (on host)

```bash
# Change permissions on host
sudo chown -R $USER:$USER src/ reports/
```

**Method 3**: Run as root then fix permissions

```bash
# Run container as root
docker run -it --user root -v $(pwd):/homework hw2-env

# Inside container
chown -R 1000:1000 /homework  # Change 1000 to your UID
```

#### When Container Rebuild is Needed

```bash
# When using Docker Compose
docker-compose down
docker-compose build --no-cache
export UID=$(id -u) GID=$(id -g)
docker-compose up -d

# When using Docker directly
docker build --no-cache -t hw2-env .
```

### Docker Quick Start

```bash
# 1. Start Docker environment in current directory (resolve UID conflicts)
export MY_UID=$(id -u) MY_GID=$(id -g) && docker-compose up -d

# 2. Access container
docker-compose exec hw2-environment bash

# 3. Build project and run tests
cd src && make all && ./tests

# 4. Cleanup after completion
exit && docker-compose down
```

---

## Local Environment Setup

Method for developing directly in local environment without Docker.

### Prerequisites - Local

#### Ubuntu/Debian Systems

```bash
sudo apt-get update
sudo apt-get install -y \
    gcc \
    make \
    build-essential \
    zlib1g-dev \
    check \
    clang-format \
    gdb \
    valgrind
```

#### macOS (Using Homebrew)

```bash
# Install Homebrew (if not available)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install required packages
brew install gcc make check clang-format gdb
```

#### CentOS/RHEL/Fedora Systems

```bash
# CentOS/RHEL 8+
sudo dnf install -y gcc make zlib-devel check-devel clang-tools-extra gdb valgrind

# Or for Fedora
sudo dnf install -y gcc make zlib-devel check-devel clang-tools-extra gdb valgrind
```

### Environment Verification

After installation, verify that essential tools are properly installed:

```bash
# Check compiler
gcc --version

# Check Make
make --version

# Check library verification (important!)
pkg-config --libs check || echo "Check library not found - install 'check' package"

# Check clang-format
clang-format --version
```

---

## Common Workflow

Once the environment setup is complete, you can proceed with the project by referring to the build and test methods in the INSTRUCTIONS.md file.

## Installed Tools (Docker Environment)

- **Development Tools**: gcc, make, build-essential
- **Libraries**: zlib1g-dev, check (unit testing)
- **Formatting**: clang-format
- **Debugging**: gdb, valgrind
- **Others**: zip, unzip, git, vim, nano

## Important Notes

1. All changes are saved in the `src/` and `reports/` folders
2. Use `CK_FORK=no` environment variable to easily debug Check tests with gdb
3. When using Docker, your work files are preserved even if the container is restarted
4. **If permission issues occur, refer to the troubleshooting methods above**

## Additional Docker Tips

### When docker-compose command is not available

```bash
# Use Docker CLI instead of docker-compose
export MY_UID=$(id -u) MY_GID=$(id -g)
docker build -t hw2-env .
docker run --rm -e MY_UID=$MY_UID -e MY_GID=$MY_GID -v $(pwd):/homework hw2-env bash -c "cd src && make all && ./tests"
```

### Maintaining Development Environment in Container

```bash
# Start development container (background)
docker run -d --name hw2-dev -e MY_UID=$(id -u) -e MY_GID=$(id -g) -v $(pwd):/homework hw2-env tail -f /dev/null

# Development work
docker exec -it hw2-dev bash

# Cleanup after completion
docker stop hw2-dev && docker rm hw2-dev
```
