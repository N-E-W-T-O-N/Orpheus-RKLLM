try :
    from huggingface_hub import hf_hub_download
except ImportError:
    print("Please install huggingface_hub first.")
    exit(1)

print("Download the model `orpheus_3b_0.1_ft_w8a8_RK3588_16bit.rkllm`...")

hf_hub_download(repo_id="Prince-1/orpheus_3b_0.1_rkllm", filename="orpheus_3b_0.1_ft_w8a8_RK3588_16bit.rkllm",local_dir=".")  

print("Download the model Successfully...")