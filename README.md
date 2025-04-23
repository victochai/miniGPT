# miniGPT

This repository contains a simple implementation of a mini GPT model using PyTorch. The model is designed to be lightweight and easy to understand, making it suitable for educational purposes and small-scale applications.

## Installation

It is recommended to use Docker for installation to ensure a consistent environment. The Dockerfile provided in this repository sets up the necessary dependencies and configurations.

To build the image, run the following command in the terminal:

```
docker build  -f Dockerfile --rm -t minigpt:latest .
```

To run the container, use the following command:

```
docker run -it --rm \
            --gpus all \
           -v /full/path/to/project:/app \
           minigpt:latest \
           bash
```
Note: The `--gpus all` flag is used to enable GPU support. If you don't have a GPU, you can remove this flag.
