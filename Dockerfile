
FROM nvcr.io/nvidia/pytorch:24.12-py3

WORKDIR /app

# Disable bytecode generation
ENV PYTHONDONTWRITEBYTECODE=1

# Send the python output directly to the terminal without first buffering it
ENV PYTHONUNBUFFERED=1

# -y --> Assume "yes" to all prompts and run non-interactively
RUN apt-get update -y
# libsm6 is used for graphics and image processing
# RUN apt-get install -y libsm6 libxext6 libxrender1

COPY requirements.txt .

RUN python3 -m pip install --upgrade --no-cache-dir -r requirements.txt

# RUN apt-get update
# RUN apt-get install libgl1 -y
