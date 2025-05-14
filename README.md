# Orpheus Neural Text-to-Speech Engine using RkLLM

## Overview
This project implements a  [Orpheus](https://github.com/canopyai/Orpheus-TTS) text-to-speech (TTS) system that converts text input into natural-sounding speech. It uses a combination of neural networks and ONNX runtime for efficient inference.

## Features
- Text to speech conversion using neural networks
- ONNX model integration for efficient inference
- Support for 24kHz audio output
- Code generation and audio synthesis pipeline
- C++ and Python implementation

## Prerequisites
- Python 3.11+
- C++ compiler with C++11 support
- Required Python packages:
  - numpy
  - soundfile
  - onnxruntime
  - opencv-python
  - pillow
  - protobuf
  - scipy
  - sympy

## Installation

1. Clone the repository

`git clone https://github.com/N-E-W-T-O-N/Orpheus-RKLLM.git` 

2. Install Python dependencies:
Make sure uv already install in you device.

`uv sync`

3. Build the C++ component 
The project need `onnxruntime` to run onnx model & `sndfile` library to convert waveform(a list of float) into audio file 
 
 ```bash
 sudo apt-get install  libsndfile1-dev libasound-dev autoconf autogen automake build-essential libasound2-dev   libflac-dev libogg-dev libtool libvorbis-dev libopus-dev libmp3lame-dev   libmpg123-dev pkg-config  libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0
 ```
 
 Now simply run the following script
 
``` bash
bash build.sh -b
```

or 

``` bash
g++ Inference.cpp -lrkllmrt input.cpp output.cpp -I onnxruntime/include -L onnxruntime/lib -L onnxruntime/lib/libonnxruntime* -lpthread -ldl -lm -lsndfile -o llm

```


## Usage

### Python Interface

``` bash
uv run cli.py 1500 2000 "Hey there my name is EDISON, <giggles> and I'm a speech generation model that can sound like a person.I Am a badass person"
```

### Cpp Interface 

- NOTE : Since Huggingface dont have tokenizer in CPP I am using `Input.py` to create Input_Ids which is used by the rkllm model 

``` bash
export LD_LIBRARY_PATH=$(pwd)/onnxruntime/lib:$LD_LIBRARY_PATH  # Model required this Enviroment Variable to run the onnx model 

./llm orpheus_3b_0.1_ft_w8a8_RK3588_GGUF_F16.rkllm 1000 2000 "Features of Good Design Before we proceed to the actual patterns, let’s discuss the process of designing software architecture: things to aim for and things you’d better avoid.Code reuse Cost and time are two of the most valuable metrics when developing any software product. Less time in development means entering the market earlier than competitors. Lower development costs mean more money is left for marketing and a broader reach to potential customers." 

```

### Monitoring

To monitor the inference performance of RKLLM on the board like the above figure, you can use
the command:

``` bash 
export RKLLM_LOG_LEVEL=1
```

``` bash
Process Finish..
I rkllm: --------------------------------------------------------------------------------------
I rkllm:  Stage         Total Time (ms)  Tokens    Time per Token (ms)      Tokens per Second      
I rkllm: --------------------------------------------------------------------------------------
I rkllm:  Init          189073.94        /         /                        /                      
I rkllm:  Prefill       1523.53          47        32.42                    30.85                  
I rkllm:  Generate      308636.08        999       308.95                   3.24                   
I rkllm: --------------------------------------------------------------------------------------
I rkllm:  Memory Usage (GB)
I rkllm:  6.66        
I rkllm: --------------------------------------------------------------------------------------
```


This will display the number of tokens processed and the inference time for both the Prefill and Generate
stages after each inference, as shown in the figure below. This information will help you evaluate the
performance by providing detailed logging of how long each stage of the inference process takes.
If you need to view more detailed logs, such as the tokens after encoding the prompt, you can use the following
command:

``` bash 
export RKLLM_LOG_LEVEL=2
```



