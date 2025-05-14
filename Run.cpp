#include "rkllm.h"
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string.h>
#include <string>

// callback function which will be called by LLMs every time they generate a
// token. generated text is recorded in text attribute RKLLMResult struct when
// last callback is called in a conversation LLMCallState parameter is equal to
// RKLLM_RUN_FINISH LLMCallState is originally defined as: typedef enum {
//     RKLLM_RUN_NORMAL  = 0, /**< The LLM call is in a normal running state. */
//     RKLLM_RUN_WAITING = 1, /**< The LLM call is waiting for complete UTF-8
//     encoded character. */ RKLLM_RUN_FINISH  = 2, /**< The LLM call has
//     finished execution. */ RKLLM_RUN_ERROR   = 3, /**< An error occurred
//     during the LLM call. */ RKLLM_RUN_GET_LAST_HIDDEN_LAYER = 4 /**< Retrieve
//     the last hidden layer during inference. */
// } LLMCallState;
void callback(RKLLMResult *result, void *userdata, LLMCallState state) {
  if (state == RKLLM_RUN_FINISH) {
    std::cout << "\nBYE" << std::endl;

  } else if (state == RKLLM_RUN_ERROR) {
    std::cout << "Runtime error" << std::endl;
  } else if (state ==
             RKLLM_RUN_NORMAL) { // RKLLM_INFER_GET_LAST_HIDDEN_LAYER) { //
                                 // RKLLM_RUN_GET_LAST_HIDDEN_LAYER

    std::cout << "\n---------------\n";
    if (result->text != NULL)
      printf("\nRESULT %s %d", result->text, result->token_id);

    // std::cout<<"\nvocab_size :" << result->logits.vocab_size << std::endl ;
    // std::cout<<"num_tokens :" << result->logits.num_tokens << std::endl ;

    // for (int i=0; i<result->logits.num_tokens; i++) { printf("\ntoken_id
    // logprob: %f", result->logits.logits[i]);        }

    // if(result->last_hidden_layer != NULL)
    // std::cout<<"\nIS HIDDEN LAYER EXIST\n";            std::cout<<
    // result->last_hidden_layer.embd_size <<"
    // "<<result->last_hidden_layer.num_tokens<<std::endl;

    // Embedding
    // if (result->last_hidden_layer.embd_size != 0 &&
    // result->last_hidden_layer.num_tokens != 0) {
    //    std::cout<< "Using this Method\n";
    //    int data_size = result->last_hidden_layer.embd_size *
    //    result->last_hidden_layer.num_tokens * sizeof(float); std::cout <<
    //    "data_size: " << data_size << std::endl; std::ofstream
    //    outFile("last_hidden_layer.bin", std::ios::binary); if
    //    (outFile.is_open()) {
    //        outFile.write(reinterpret_cast<const
    //        char*>(result->last_hidden_layer.hidden_states), data_size);
    //        outFile.close();
    //        std::cout << "Data saved to output.bin successfully!" <<
    //        std::endl;
    //    } else {
    //        std::cerr << "Failed to open the file for writing!" << std::endl;
    //    }
    //}
  }
}

int main(int argc, char **argv) {
  if (argc < 5) {
    std::cerr << "Usage: " << argv[0]
              << " model_path max_new_tokens max_context_len input\n";

    return 1;
  }
  std::cout << argv[0] << " " << argv[1] << " " << argv[2] << " " << argv[3]
            << " " << argv[4];
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

  param.max_new_tokens = std::atoi(argv[2]);
  param.max_context_len = std::atoi(argv[3]);
  param.skip_special_token = true;
  param.extend_param.base_domain_id = 0;

  std::string text = argv[4];
  std::cout << text << std::endl;

  // loading LLM model onto memory for conversations
  int ret = rkllm_init(&llmHandle, &param, callback);
  if (ret == 0) {
    std::cout << "rkllm init success" << std::endl;
  } else {
    std::cerr << "rkllm init failed" << std::endl;
    exit(1);
  }

  // struct to pass input to LLM
  RKLLMInput rkllm_input;

  // struct define kind of interaction with LLM
  RKLLMInferParam rkllm_infer_params;
  memset(&rkllm_infer_params, 0, sizeof(RKLLMInferParam));

  rkllm_infer_params.mode = RKLLM_INFER_GENERATE; // RKLLM_INFER_GET_LOGITS; //
                                                  // ;

  // while (true)    {
  std::cout << std::endl;
  std::cout << "User: ";
  std::cout << text << std::endl;
  // std::getline(std::cin, text);
  if (text == "exit") {
    return 1;
    // break;
  }

  // input_type attribute is of enum type and offered input types are:
  // RKLLM_INPUT_PROMPT      = 0,  Input is a text prompt.
  // RKLLM_INPUT_TOKEN       = 1,  Input is a sequence of tokens.
  // RKLLM_INPUT_EMBED       = 2,  Input is an embedding vector.
  // RKLLM_INPUT_MULTIMODAL  = 3,  Input is multimodal (e.g., text and image).

  rkllm_input.prompt_input = (char *)text.c_str();
  // {101, 102, 103, 104};  // Example token IDs

  int32_t tokens[] = {128259, 128000, 73,    434,   25,   28653,  1070,  856,
                      836,    374,    16421, 53076, 11,   366,    70,    20831,
                      645,    29,     323,   358,   2846, 264,    8982,  9659,
                      1646,   430,    649,   5222,  1093, 264,    1732,  2506,
                      3383,   264,    87402, 1732,  694,  70,     13671, 29,
                      694,    9810,   3168,  273,   29,   128009, 128260};
  size_t num_tokens = sizeof(tokens) / sizeof(tokens[0]);

  RKLLMTokenInput t_input;
  // t_input.input_ids = tokens;  // Pointer to the array
  // t_input.n_tokens = num_tokens;

  std::cout << "Set token Input\n";

  rkllm_input.input_type =
      RKLLM_INPUT_PROMPT; // RKLLM_INPUT_TOKEN ; //RKLLM_INPUT_PROMPT;
  // rkllm_input.token_input = t_input ;
  printf("AI: ");

  // Running LLM model
  rkllm_run(llmHandle, &rkllm_input, &rkllm_infer_params, NULL);
  //}

  // releasing llmHandle after work is done.
  rkllm_destroy(llmHandle);

  return 0;
}
