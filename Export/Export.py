import torch 

try:
    from rkllm.api import RKLLM
except ImportError:
    print("RKLLM not installed. Please install from wheel 'https://github.com/airockchip/rknn-llm'.") 


llm = RKLLM()

from getpass import getpass
from huggingface_hub import snapshot_download, hf_hub_download

def DownloadHF(token)  :


    print("Downloading main model from Hugging Face Hub...")
    repo_id = "Prince-1/orpheus_3b_0.1_ft_16bit"
    local_dir = "OrpheusMain" #"/content/OrpheusMain"  # Choose a local directory
    snapshot_download(repo_id=repo_id, local_dir=local_dir, token= token)#userdata.get("HF_TOKEN"))
    print("Main model downloaded successfully.")


    return "OrpheusMain"

def DownloadGGUF(token) :
    print("Downloading GGUF model from Hugging Face Hub...")
    path = hf_hub_download(repo_id="Prince-1/orpheus_3b_0.1_GGUF", filename="unsloth.F16.gguf",token= token,local_dir="GGUF")  
    print("GGUF model downloaded successfully.")
    return path 


def UsingHf(llm,modelpath,modelLora) :

    print("Loading model...")
    print(modelpath)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ret = llm.load_huggingface(model=modelpath, model_lora = modelLora,device=device)
    
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    return llm     

def UsingGGUF(llm,modelpath) :
    print("Loading model...")
    ret = llm.load_gguf(model=modelpath)
    
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    return llm    


password = getpass("Please Enter your Hugging Face Token: ")
if password == "" :
    print("No token provided.")
    exit(1)



while True :
    print("Do you want to download Lora model or GGUF model ?")
    print("1. Lora")
    print("2. GGUF")
    i = input()
    if i == "1" :
        main = DownloadHF(password)
        UsingHf(llm,main)

        break
    elif i == "2" :
        gguf = DownloadGGUF(password)
        UsingGGUF(llm,gguf)
        break
    else :
        print("Invalid input. Please enter 1 or 2.")
        continue



# Build model
dataset = None
qparams = None
target_platform = "RK3588"
optimization_level = 1
quantized_dtype = "w8a8" #"w4a16_g32" #w4a16_g64 or w4a16_g128
quantized_algorithm = "normal"
num_npu_core = 3

print("Building model...")
ret = llm.build(
    do_quantization=False,optimization_level=optimization_level,
    quantized_dtype=quantized_dtype,quantized_algorithm=quantized_algorithm,
    target_platform=target_platform, num_npu_core=num_npu_core,
    extra_qparams=qparams, dataset=dataset)
if ret != 0:
  print('Build model failed!')
  exit(ret)

print("Model Build successfully.")

# Export rkllm model
ret =llm.export_rkllm(f"orpheus_3b_0.1_ft_{quantized_dtype}_{target_platform[2:]}.rkllm")
if ret != 0:
  print('Export model failed!')
  exit(ret)
  
print("Model Export successfully.")
