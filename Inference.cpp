#include "input.h"
#include "output.hpp"
#include "rkllm.h"
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string.h>
#include <string>
#include <vector>

std::vector<int64_t> Response_Ids;

// callback function which will be called by LLMs every time they generate a
// token. generated text is recorded in text attribute RKLLMResult struct when
// last callback is called in a conversation LLMCallState parameter is equal to
// RKLLM_RUN_FINISH LLMCallState is originally defined as: typedef enum {
//     RKLLM_RUN_NORMAL  = 0, /**< The LLM call is in a normal running state. */
//     RKLLM_RUN_WAITING = 1, /**< The LLM call is waiting for complete UTF-8
//     encoded character. */ RKLLM_RUN_FINISH  = 2, /**< The LLM call has
//     finished execution. */ RKLLM_RUN_ERROR   = 3, /**< An error occurred
//     during the LLM call. */
// } LLMCallState;
void callback(RKLLMResult *result, void *userdata, LLMCallState state) {

  if (state == RKLLM_RUN_NORMAL) {

    // printf("\n---------------\nRESULT %s %d", result->text,result->token_id);

    Response_Ids.push_back(result->token_id);

  } else if (state == RKLLM_RUN_FINISH) {
    std::cout << "\nProcess Finish..." << std::endl;

  } else if (state == RKLLM_RUN_ERROR) {
    std::cout << "Runtime error" << std::endl;
  }
}

int main(int argc, char **argv) {
  if (argc < 5) {
    std::cerr
        << "Usage: " << argv[0]
        << " model_path max_new_tokens max_context_len input audio_name\n";
    return 1;
  }

  // std::cout<<argv[0] <<" "<<argv[1] <<" "<<argv[2]<<" "<<argv[3]<<" "<<argv[4]  ;

  std::cout << "Convert Input to Ids" << std::endl;
  std::string input_str = argv[4];
  
  // Note : Model takes INput as Ids , and I didn't find any tokenizer in cpp so this is a work-arround
  std::vector<int> input_ids = call_python_input(input_str);
  std::cout << "Ids generated successfully" << std::endl;
  std::cout << "Booting the Model\n";

  // initialize llmHandle to store pointer to loaded llm model
  LLMHandle llmHandle = nullptr;

  // initializing RKLLMParam struct for setting parameters required for LLMs
  // RKLLMParam struct is originally defined as:
  // typedef struct {
  //     const char* model_path;         /**< Path to the model file. */
  //     int32_t max_context_len;        /**< Maximum number of tokens in the
  //     context window. */ int32_t max_new_tokens;         /**< Maximum number
  //     of new tokens to generate. */ int32_t top_k;                  /**<
  //     Top-K sampling parameter for token generation. */ float top_p; /**<
  //     Top-P (nucleus) sampling parameter. */ float temperature; /**< Sampling
  //     temperature, affecting the randomness of token selection. */ float
  //     repeat_penalty;           /**< Penalty for repeating tokens in
  //     generation. */ float frequency_penalty;        /**< Penalizes frequent
  //     tokens during generation. */ float presence_penalty;         /**<
  //     Penalizes tokens based on their presence in the input. */ int32_t
  //     mirostat;               /**< Mirostat sampling strategy flag (0 to
  //     disable). */ float mirostat_tau;             /**< Tau parameter for
  //     Mirostat sampling. */ float mirostat_eta;             /**< Eta
  //     parameter for Mirostat sampling. */ bool skip_special_token; /**<
  //     Whether to skip special tokens during generation. */ bool is_async;
  //     /**< Whether to run inference asynchronously. */ const char* img_start;
  //     /**< Starting position of an image in multimodal input. */ const char*
  //     img_end;            /**< Ending position of an image in multimodal
  //     input. */ const char* img_content;        /**< Pointer to the image
  //     content. */ RKLLMExtendParam extend_param; /**< Extend parameters. */
  // } RKLLMParam;

  RKLLMParam param = rkllm_createDefaultParam();
  param.model_path = argv[1];

  param.top_k = 1;
  param.top_p = 0.95;
  param.temperature = 0.6;
  param.repeat_penalty = 1.1;
  param.frequency_penalty = 0.0;
  param.presence_penalty = 0.0;
  param.skip_special_token = true;
  param.extend_param.base_domain_id = 0;

  param.max_new_tokens = std::atoi(argv[2]);
  param.max_context_len = std::atoi(argv[3]);

  // loading LLM model onto memory for conversations
  int ret = rkllm_init(&llmHandle, &param, callback);
  if (ret == 0) {
    std::cout << "rkllm init success" << std::endl;
  } else {
    std::cerr << "rkllm init failed" << std::endl;
    exit(1);
  }

  // std::string text;

  // struct to pass input to LLM
  RKLLMInput rkllm_input;

  // struct define kind of interaction with LLM
  RKLLMInferParam rkllm_infer_params;
  memset(&rkllm_infer_params, 0, sizeof(RKLLMInferParam));

  rkllm_infer_params.mode = RKLLM_INFER_GENERATE; // RKLLM_INFER_GET_LOGITS; // RKLLM_INFER_GET_LAST_HIDDEN_LAYER 

  std::cout << std::endl;

  RKLLMTokenInput t_input;
  t_input.input_ids = input_ids.data(); // Pointer to the array
  t_input.n_tokens = input_ids.size();

  std::cout << "Set token Input\n";

  rkllm_input.input_type = RKLLM_INPUT_TOKEN;
  rkllm_input.token_input = t_input;
  printf("AI: ");

  // Running LLM model
  rkllm_run(llmHandle, &rkllm_input, &rkllm_infer_params, NULL);

  std::cout << "OUTPUT ARRAY " << std::endl;
  for (auto i : Response_Ids) {
    std::cout << i << ",";
  }
  // releasing llmHandle after work is done.
  rkllm_destroy(llmHandle);

  // Create Codes from Model Output
  std::tuple<std::vector<int64_t>, std::vector<int64_t>, std::vector<int64_t>>
      codes = redistribute_codes(Response_Ids);

  // Create audio waveform mrom SAC decoder model
  std::vector<float> audio =
      run_onnx(std::get<0>(codes), std::get<1>(codes), std::get<2>(codes));

  // std::string audio =(argc > 5 ) ? argc[5] : "output.wav"
  //  Convert waveform to audio
  saveWav(audio, "output.wav", 24000);

  return 0;
}

// g++ Inference.cpp -lrkllmrt input.cpp output.cpp -I onnxruntime/include -L onnxruntime/lib -L onnxruntime/lib/libonnxruntime*  -lpthread -ldl -lm -lsndfile  -o llm

// ./llm orpheus_3b_0.1_ft_w8a8_RK3588_GGUF_F16.rkllm 1000 2000 "TEXT TO SPPECH"
