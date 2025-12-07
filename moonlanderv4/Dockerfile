FROM python:3.9-slim

# install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    swig \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# copy project files
COPY *.py ./

# set environment variables for headless mode
ENV SDL_VIDEODRIVER=dummy \
    DISPLAY=:0 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

# create output directories
RUN mkdir -p logs plots

CMD ["sh", "-c", "python experiment.py && python generate_plots.py"]
