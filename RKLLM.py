import ctypes
import platform
import sys
import threading

Response_Ids: list[int] = list()

# Check if OS is `arm64` &CPU is `RockChip`
cpu_arch = platform.machine()

if cpu_arch not in ["aarch64", "arm"]:
    print(f"Following application is not meant for following OS:{cpu_arch}")
    exit(1)

# Set the dynamic library path
try:
    rkllm_lib = ctypes.CDLL('/lib/librkllmrt.so')
except OSError:
    print("Library `librkllmrt.so` not found.Make sure Library exist in path `/lib`")

# A handle used to manage and interact with the large language model.
RKLLM_Handle_t = ctypes.c_void_p
userdata = ctypes.c_void_p(None)

# Describes the possible states of an LLM call.
LLMCallState = ctypes.c_int
LLMCallState.RKLLM_RUN_NORMAL = 0  # The LLM call is in a normal running
LLMCallState.RKLLM_RUN_WAITING = 1  # The LLM call is waiting for complete UTF - 8 encoded
LLMCallState.RKLLM_RUN_FINISH = 2  # The LLM call has finished execution.
LLMCallState.RKLLM_RUN_ERROR = 3  # An error occurred during the LLM call.

# Defines the types of inputs that can be fed into the LLM.
RKLLMInputMode = ctypes.c_int
RKLLMInputMode.RKLLM_INPUT_PROMPT = 0  # Input is a text prompt.
RKLLMInputMode.RKLLM_INPUT_TOKEN = 1  # Input is a sequence of tokens.
RKLLMInputMode.RKLLM_INPUT_EMBED = 2  # Input is an embedding vector.
RKLLMInputMode.RKLLM_INPUT_MULTIMODAL = 3  # Input is multimodal(e.g., text and image).

# Specifies the inference modes of the LLM.
RKLLMInferMode = ctypes.c_int
RKLLMInferMode.RKLLM_INFER_GENERATE = 0  # The LLM generates text based on input.
RKLLMInferMode.RKLLM_INFER_GET_LAST_HIDDEN_LAYER = 1  # The LLM retrieves the last hidden layer for further processing.
RKLLMInferMode.RKLLM_INFER_GET_LOGITS = 2  # The LLM retrieves logits for further processing.


class RKLLMExtendParam(ctypes.Structure):
    """
    The extent parameters for configuring an LLM instance.
    """

    _fields_ = [
        ("base_domain_id", ctypes.c_int32),
        ("embed_flash", ctypes.c_int8),
        # Indicates whether to query word embedding vectors from flash memory(1) or not (0)
        ("enabled_cpus_num", ctypes.c_int8),  # Number of CPUs enabled for inference.
        ("enabled_cpus_mask", ctypes.c_uint32),  # Bitmask indicating which CPUs to enable for inference.
        ("reserved", ctypes.c_uint8 * 106)
    ]


class RKLLMParam(ctypes.Structure):
    """
    Defines the parameters for configuring an LLM instance.
    """
    _fields_ = [
        ("model_path", ctypes.c_char_p),  # Path to the model file.
        ("max_context_len", ctypes.c_int32),  # Maximum number of tokens in the context window.
        ("max_new_tokens", ctypes.c_int32),  # Maximum number of new tokens to generate.
        ("top_k", ctypes.c_int32),  # Top - K sampling parameter for token generation.
        ("n_keep", ctypes.c_int32),  # Number of kv cache to keep at the beginning when shifting the context window
        ("top_p", ctypes.c_float),  # (nucleus) sampling parameter.
        ("temperature", ctypes.c_float),  # Sampling temperature, affecting the randomness of token selection.
        ("repeat_penalty", ctypes.c_float),  # Penalty for repeating tokens in a generation.
        ("frequency_penalty", ctypes.c_float),  # Penalizes frequent tokens during generation.
        ("presence_penalty", ctypes.c_float),  # Penalizes tokens based on their presence in the input.
        ("mirostat", ctypes.c_int32),  # Mirostat sampling strategy flag(0 to disable)
        ("mirostat_tau", ctypes.c_float),  # Tau parameter for Mirostat sampling.
        ("mirostat_eta", ctypes.c_float),  # Eta parameter for Mirostat sampling.
        ("skip_special_token", ctypes.c_bool),  # Whether to skip special tokens during generation.
        ("is_async", ctypes.c_bool),  # Whether to run inference asynchronously.
        ("img_start", ctypes.c_char_p),  # Starting position of an image in multimodal input.
        ("img_end", ctypes.c_char_p),  # Ending position of an image in multimodal input.
        ("img_content", ctypes.c_char_p),  # Pointer to the image content.
        ("extend_param", RKLLMExtendParam),  # Extend parameters.
    ]


class RKLLMLoraAdapter(ctypes.Structure):
    """
    Defines parameters for a Lora adapter used in model fine - tuning.
    """
    _fields_ = [
        ("lora_adapter_path", ctypes.c_char_p),  # Path to the Lora adapter file.
        ("lora_adapter_name", ctypes.c_char_p),  # Name of the Lora adapter.
        ("scale", ctypes.c_float)  # Scaling factor for applying the Lora adapter.
    ]


class RKLLMEmbedInput(ctypes.Structure):
    """
    Represents an embedding input to the LLM.
    """
    _fields_ = [
        ("embed", ctypes.POINTER(ctypes.c_float)),  # Pointer to the embedding vector (of size n_tokens * n_embed).
        ("n_tokens", ctypes.c_size_t)  # Number of tokens represented in the embedding.
    ]


class RKLLMTokenInput(ctypes.Structure):
    """
    Represents token input to the LLM.
    """
    _fields_ = [
        ("input_ids", ctypes.POINTER(ctypes.c_int32)),
        ("n_tokens", ctypes.c_size_t)
    ]


class RKLLMMultiModelInput(ctypes.Structure):
    _fields_ = [
        ("prompt", ctypes.c_char_p),
        ("image_embed", ctypes.POINTER(ctypes.c_float)),
        ("n_image_tokens", ctypes.c_size_t),
        ("n_image", ctypes.c_size_t),
        ("image_width", ctypes.c_size_t),
        ("image_height", ctypes.c_size_t)
    ]


class RKLLMInputUnion(ctypes.Union):
    _fields_ = [
        ("prompt_input", ctypes.c_char_p),
        ("embed_input", RKLLMEmbedInput),
        ("token_input", RKLLMTokenInput),
        ("multimodal_input", RKLLMMultiModelInput)
    ]


class RKLLMInput(ctypes.Structure):
    _fields_ = [
        ("input_mode", ctypes.c_int),
        ("input_data", RKLLMInputUnion)
    ]


class RKLLMLoraParam(ctypes.Structure):
    _fields_ = [
        ("lora_adapter_name", ctypes.c_char_p)
    ]


class RKLLMPromptCacheParam(ctypes.Structure):
    _fields_ = [
        ("save_prompt_cache", ctypes.c_int),
        ("prompt_cache_path", ctypes.c_char_p)
    ]


class RKLLMInferParam(ctypes.Structure):
    _fields_ = [
        ("mode", RKLLMInferMode),
        ("lora_params", ctypes.POINTER(RKLLMLoraParam)),
        ("prompt_cache_params", ctypes.POINTER(RKLLMPromptCacheParam)),
        ("keep_history", ctypes.c_int)
    ]


class RKLLMResultLastHiddenLayer(ctypes.Structure):
    _fields_ = [
        ("hidden_states", ctypes.POINTER(ctypes.c_float)),
        ("embd_size", ctypes.c_int),
        ("num_tokens", ctypes.c_int)
    ]


class RKLLMResultLogits(ctypes.Structure):
    _fields_ = [
        ("logits", ctypes.POINTER(ctypes.c_float)),
        ("vocab_size", ctypes.c_int),
        ("num_tokens", ctypes.c_int)
    ]


class RKLLMResult(ctypes.Structure):
    _fields_ = [
        ("text", ctypes.c_char_p),
        ("token_id", ctypes.c_int),
        ("last_hidden_layer", RKLLMResultLastHiddenLayer),
        ("logits", RKLLMResultLogits)
    ]


# Create a lock to control multi-user access to the server.
lock = threading.Lock()

# Create a global variable to indicate whether the server is currently in a blocked state.
is_blocking = False

# Define global variables to store the callback function output for displaying in the Gradio interface
global_text = ''
global_state = -1
split_byte_data = bytes(b"")  # Used to store the segmented byte data


# Define the callback function


# Define the callback function
def callback_impl(result, userdata, state):
    global global_text, global_state, split_byte_data
    if state == LLMCallState.RKLLM_RUN_FINISH:
        global_state = state
        print("\n")
        sys.stdout.flush()
    elif state == LLMCallState.RKLLM_RUN_ERROR:
        global_state = state
        print("run error")
        sys.stdout.flush()
    elif state == LLMCallState.RKLLM_RUN_NORMAL:
        global_state = state
        print(f"text : {result.text} token_id : {result.token_id}")
        Response_Ids.append(result.token_id)


# Connect the callback function between the Python side and the C++ side
# Return Type: None
# Input 1: result Pointer to the LLM result.
# Input 2: userdata Pointer to user data for the callback.
# Input 3: state of the LLM call(e.g., finished, error).
callback_type = ctypes.CFUNCTYPE(None, ctypes.POINTER(RKLLMResult), ctypes.c_void_p, ctypes.c_int)
callback = callback_type(callback_impl)


# Define the RKLLM class, which includes initialization, inference, and release operations for the RKLLM model in the dynamic library
class RKLLM(object):
    """
    RKLLM Class
    """

    def __init__(self, model_path: str, lora_model_path: str = None, prompt_cache_path: str = None,
                 max_context_len: int = 4096, max_new_tokens: int = 1000, skip_special_token: bool = True,
                 n_keep: int = - 1, top_k: float = 1, top_p: float = 0.9):

        rkllm_param = RKLLMParam()
        rkllm_param.model_path = bytes(model_path, 'utf-8')

        rkllm_param.max_context_len = max_context_len
        rkllm_param.max_new_tokens = max_new_tokens
        rkllm_param.skip_special_token = skip_special_token  # Default True
        # rkllm_param.n_keep = -1
        rkllm_param.top_k = top_k  # Deafult 1 rkllm_param.top_p = top_p #Default 0.9 rkllm_param.temperature = 0.8 rkllm_param.repeat_penalty = 1.1 rkllm_param.frequency_penalty = 0.0 rkllm_param.presence_penalty = 0.0

        rkllm_param.mirostat = 0
        rkllm_param.mirostat_tau = 5.0
        rkllm_param.mirostat_eta = 0.1

        rkllm_param.is_async = False

        rkllm_param.img_start = "".encode('utf-8')
        rkllm_param.img_end = "".encode('utf-8')
        rkllm_param.img_content = "".encode('utf-8')

        rkllm_param.extend_param.base_domain_id = 0
        rkllm_param.extend_param.enabled_cpus_num = 4
        rkllm_param.extend_param.enabled_cpus_mask = (1 << 4) | (1 << 5) | (1 << 6) | (1 << 7)

        self.handle = RKLLM_Handle_t()

        self.rkllm_init = rkllm_lib.rkllm_init
        self.rkllm_init.argtypes = [ctypes.POINTER(RKLLM_Handle_t), ctypes.POINTER(RKLLMParam), callback_type]
        self.rkllm_init.restype = ctypes.c_int
        self.rkllm_init(ctypes.byref(self.handle), ctypes.byref(rkllm_param), callback)

        self.rkllm_run = rkllm_lib.rkllm_run
        self.rkllm_run.argtypes = [RKLLM_Handle_t, ctypes.POINTER(RKLLMInput), ctypes.POINTER(RKLLMInferParam),
                                   ctypes.c_void_p]
        self.rkllm_run.restype = ctypes.c_int

        self.set_chat_template = rkllm_lib.rkllm_set_chat_template
        self.set_chat_template.argtypes = [RKLLM_Handle_t, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
        self.set_chat_template.restype = ctypes.c_int

        system_prompt = "<|im_start|>system You are a helpful assistant. <|im_end|>"
        prompt_prefix = "<|im_start|>user"
        prompt_postfix = "<|im_end|><|im_start|>assistant"
        # self.set_chat_template(self.handle, ctypes.c_char_p(system_prompt.encode('utf-8')), ctypes.c_char_p(prompt_prefix.encode('utf-8')), ctypes.c_char_p(prompt_postfix.encode('utf-8')))

        self.rkllm_destroy = rkllm_lib.rkllm_destroy
        self.rkllm_destroy.argtypes = [RKLLM_Handle_t]
        self.rkllm_destroy.restype = ctypes.c_int

        rkllm_lora_params = None
        if lora_model_path:
            lora_adapter_name = "test"
            lora_adapter = RKLLMLoraAdapter()
            ctypes.memset(ctypes.byref(lora_adapter), 0, ctypes.sizeof(RKLLMLoraAdapter))
            lora_adapter.lora_adapter_path = ctypes.c_char_p((lora_model_path).encode('utf-8'))
            lora_adapter.lora_adapter_name = ctypes.c_char_p((lora_adapter_name).encode('utf-8'))
            lora_adapter.scale = 1.0

            rkllm_load_lora = rkllm_lib.rkllm_load_lora
            rkllm_load_lora.argtypes = [RKLLM_Handle_t, ctypes.POINTER(RKLLMLoraAdapter)]
            rkllm_load_lora.restype = ctypes.c_int
            rkllm_load_lora(self.handle, ctypes.byref(lora_adapter))
            rkllm_lora_params = RKLLMLoraParam()
            rkllm_lora_params.lora_adapter_name = ctypes.c_char_p((lora_adapter_name).encode('utf-8'))

        self.rkllm_infer_params = RKLLMInferParam()
        ctypes.memset(ctypes.byref(self.rkllm_infer_params), 0, ctypes.sizeof(RKLLMInferParam))
        self.rkllm_infer_params.mode = RKLLMInferMode.RKLLM_INFER_GENERATE

        # self.rkllm_infer_params.lora_params = ctypes.pointer(rkllm_lora_params) if rkllm_lora_params else None

        # self.rkllm_infer_params.keep_history = 0

        self.prompt_cache_path = None
        if prompt_cache_path:
            self.prompt_cache_path = prompt_cache_path

            rkllm_load_prompt_cache = rkllm_lib.rkllm_load_prompt_cache
            rkllm_load_prompt_cache.argtypes = [RKLLM_Handle_t, ctypes.c_char_p]
            rkllm_load_prompt_cache.restype = ctypes.c_int
            rkllm_load_prompt_cache(self.handle, ctypes.c_char_p((prompt_cache_path).encode('utf-8')))

    # String
    def run_prompt(self, prompt: str):
        rkllm_input = RKLLMInput()
        rkllm_input.input_mode = RKLLMInputMode.RKLLM_INPUT_PROMPT
        rkllm_input.input_data.prompt_input = ctypes.c_char_p(prompt.encode('utf-8'))
        self.rkllm_run(self.handle, ctypes.byref(rkllm_input), ctypes.byref(self.rkllm_infer_params), None)
        return

    # Embedding
    def run_embedding(self, embed_input: list[float]):
        rkllm_input = RKLLMEmbedInput()
        pass

    # Token
    def run_token(self, token_input: list[int]):
        rkllm_input = RKLLMInput()
        rkllm_input.input_mode = RKLLMInputMode.RKLLM_INPUT_TOKEN

        rkllm_token_input = RKLLMTokenInput()
        rkllm_token_input.input_ids = token_input
        rkllm_token_input.n_tokens = len(token_input)

        rkllm_input.input_data.token_input = rkllm_token_input
        self.rkllm_run(self.handle, ctypes.byref(rkllm_input), ctypes.byref(self.rkllm_infer_params), None)

        return Response_Ids

    # Multimodel
    def run_multimodel(self, prompt: str, image_embed: list[float], n_image_tokens: int, n_image: int, image_width: int,
                       image_height: int):
        pass

    def release(self):
        self.rkllm_destroy(self.handle)
