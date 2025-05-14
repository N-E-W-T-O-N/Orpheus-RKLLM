import argparse
import os
import resource
import sys

from Input import InputTokenizer
from Output import create_codes, create_audio, run_onnx
from RKLLM import RKLLM

Input: list[int] = []

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--rkllm_model_path', type=str, required=True,
                        help='Absolute path of the converted RKLLM model on the Linux board;')
    parser.add_argument('--target_platform', type=str, required=True,
                        help='Target platform: e.g., rk3588/rk3576;',
                        default="rk3588")
    parser.add_argument('-i', '--input', type=str, required=True, help='Prompt that need to be convert into audio')
    parser.add_argument("-v", "--voice", type=str, help="voice character")
    parser.add_argument('--file', type=str, help='Name of output file', default="output.wav")

    args = parser.parse_args()

    if not os.path.exists(args.rkllm_model_path):
        print("Error: Please provide the correct rkllm model path, and ensure it "
              "is the absolute path on the board.")
        sys.stdout.flush()
        exit()

    if not (args.target_platform in ["rk3588", "rk3576"]):
        print("Error: Please specify the correct target platform: rk3588/rk3576.")
        sys.stdout.flush()
        exit()

    if args.input is None:
        print("Error: Please Input Prompt that you want to transform into Audio.")
        sys.stdout.flush()
        exit()

    file_name = ""
    if args.file is None:
        file_name = "output.wav"
    # Set resource limit
    resource.setrlimit(resource.RLIMIT_NOFILE, (102400, 102400))

    prompt = args.input
    input_string = InputTokenizer(prompt, args.voice)

    print(f"Input string: {input_string}")

    input_ids: list[int] = list(map(int, input_string.split(',')))

    # Initialize RKLLM model
    print("=========init....===========")
    sys.stdout.flush()
    model_path = args.rkllm_model_path

    rkllm_model = RKLLM(model_path)
    print("RKLLM Model has been initialized successfully！")
    print("==============================")
    sys.stdout.flush()

    output_Ids = rkllm_model.run_token(input_ids)

    sys.stdout.flush()
    print("====================")
    print("RKLLM model inference completed, releasing "
          "RKLLM model resources...")
    rkllm_model.release()
    print("====================")
    print("\n")
    print("\n====================")
    codes = create_codes(output_Ids)
    sys.stdout.flush()
    print("Start Running the Decoder model...")
    wave_form = run_onnx(codes)
    sys.stdout.flush()
    print("Decoder model run Successfully...")
    sys.stdout.flush()
    print("Create Audio File....")
    sys.stdout.flush()
    create_audio(wave_form[0][0], file_name)
    print(f"Audio file {file_name} created successfully...")
    print("\n====================")
    sys.stdout.flush()
