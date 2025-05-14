#include "output.hpp"
#include "sndfile.h"
#include <filesystem>
#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <vector>

// #include "onnxruntime/include/onnxruntime_cxx_api.h"

// Remove specific tokens
void remove_value(std::vector<int64_t> &vec, int64_t t) {
  vec.erase(std::remove(vec.begin(), vec.end(), t), vec.end());
}

// Create Tuple Codes
std::tuple<std::vector<int64_t>, std::vector<int64_t>, std::vector<int64_t>>
redistribute_codes(std::vector<int64_t> &output) {

  const int64_t token_to_find = 128257;
  const int64_t token_to_remove = 128258;

  // Find the Last Occurrence of the value
  auto result = std::find(output.rbegin(), output.rend(), token_to_find);

  std::vector<int64_t> cropped;
  if (result != output.rend()) {
    // Saved Everything in the new Vector
    int index = std::distance(result, output.rend()) - 1;
    cropped = std::vector(output.begin() + index + 1, output.end());
  } else
    cropped = std::vector<int64_t>(output);

  // Remove
  remove_value(cropped, token_to_remove);

  size_t new_idx = (cropped.size() / 7) * 7;

  std::vector<int64_t> trimmed_vect =
      std::vector<int64_t>(cropped.begin(), cropped.begin() + new_idx);

  // Created single codex
  for (auto &val : trimmed_vect) {
    val -= 128266;
  }

  std::vector<int64_t> c1, c2, c3;

  // Codex of Size 3

  // trimmed_vect.size() is a multiple of 7, and we want all values
  for (size_t i = 0; i < (trimmed_vect.size()) / 7; ++i) {

    c1.push_back(trimmed_vect[7 * i]);
    c2.push_back(trimmed_vect[7 * i + 1] - 4096);

    c3.push_back(trimmed_vect[7 * i + 2] - (2 * 4096));
    c3.push_back(trimmed_vect[7 * i + 3] - (3 * 4096));

    c2.push_back(trimmed_vect[7 * i + 4] - (4 * 4096));

    c3.push_back(trimmed_vect[7 * i + 5] - (5 * 4096));
    c3.push_back(trimmed_vect[7 * i + 6] - (6 * 4096));
  }

  return std::make_tuple(c1, c2, c3);
}

// Run Onnx model
std::vector<float> run_onnx(const std::vector<int64_t> &input0,
                            const std::vector<int64_t> &input1,
                            const std::vector<int64_t> &input2) {

  // Initialize ONNX runtime
  const Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNXRuntime");
  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(1);

  // Load the ONNX model
  std::filesystem::path source_path = __FILE__;
  std::filesystem::path current_path = source_path.parent_path();
  std::filesystem::path file_path = current_path / "decoder_model.onnx";
  Ort::Session session(env, file_path.c_str(), session_options);

  Ort::AllocatorWithDefaultOptions allocator;

  // Input names (update if your model uses different names)
  const char *input_names[] = {"audio_codes.0", "audio_codes.1",
                               "audio_codes.2"};
  const char *output_names[] = {"audio_values"};

  std::vector<std::vector<int64_t>> input_node_dims;
  std::vector<std::string> input_name = session.GetInputNames();

  auto num_input_nodes = session.GetInputCount();
  for (size_t i = 0; i < num_input_nodes; i++) {

    //  input_node_names[i] = input_name;

    Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    input_node_dims.push_back(tensor_info.GetShape());
  }

  // Prepare input shapes
  std::vector<int64_t> shape0 = {1, static_cast<int64_t>(input0.size())};
  std::vector<int64_t> shape1 = {1, static_cast<int64_t>(input1.size())};
  std::vector<int64_t> shape2 = {1, static_cast<int64_t>(input2.size())};

  Ort::MemoryInfo memory_info =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  Ort::Value input_tensor_0 = Ort::Value::CreateTensor<int64_t>(
      memory_info, const_cast<int64_t *>(input0.data()), input0.size(),
      shape0.data(), shape0.size());

  Ort::Value input_tensor_1 = Ort::Value::CreateTensor<int64_t>(
      memory_info, const_cast<int64_t *>(input1.data()), input1.size(),
      shape1.data(), shape1.size());

  Ort::Value input_tensor_2 = Ort::Value::CreateTensor<int64_t>(
      memory_info, const_cast<int64_t *>(input2.data()), input2.size(),
      shape2.data(), shape2.size());

  std::vector<Ort::Value> input_tensors;
  input_tensors.emplace_back(std::move(input_tensor_0));
  input_tensors.emplace_back(std::move(input_tensor_1));
  input_tensors.emplace_back(std::move(input_tensor_2));

  // Run inference
  auto output_tensors =
      session.Run(Ort::RunOptions{nullptr}, input_names, input_tensors.data(),
                  input_tensors.size(), output_names, 1);

  // Process output
  float *output_data = output_tensors.front().GetTensorMutableData<float>();
  auto output_shape =
      output_tensors.front().GetTensorTypeAndShapeInfo().GetShape();

  std::vector<float> result;
  // result.reserve(output_shape[0]);
  //  for (size_t i = 0; i < output_shape[0]; i++) {
  //      result.emplace_back(output_data + i * output_shape[1],
  //                          output_data + (i + 1) * output_shape[1]);
  //  }

  std::cout << std::endl << "Output shape : " << std::endl;
  for (auto dim : output_shape)
    std::cout << dim << " ";

  for (int i = 0; i < output_shape.back(); ++i) {
    result.push_back(output_data[i]);
  }

  // Log the first few values
  // std::cout << "\nOutput[0:5]: ";
  // for (int i = 0; i < std::min<size_t>(5, output_shape.back()); ++i)
  //     std::cout << output_data[i] << " ";
  // std::cout << std::endl;

  // You can reshape the output into a vector<vector<int64_t>> if needed.
  // Placeholder logic, replace with your actual reshape logic.
  return result;
}

// Save Input To file
void saveWav(const std::vector<float> &output,
             const std::string &name = "output.wav", int sampleRate = 24000) {
  SF_INFO sfinfo;
  sfinfo.samplerate = sampleRate;
  sfinfo.channels = 1; // Mono
  sfinfo.format = SF_FORMAT_WAV | SF_FORMAT_PCM_16;
  sfinfo.frames = output.size();

  std::filesystem::path source_path = __FILE__;
  std::filesystem::path current_path = source_path.parent_path();
  std::filesystem::path file_path = current_path / name;

  std::cout << "Print audio file : " << file_path << std::endl;
  SNDFILE *sndfile = sf_open(file_path.c_str(), SFM_WRITE, &sfinfo);

  if (!sndfile) {
    std::cerr << "Error opening file: " << file_path << sf_strerror(nullptr)
              << std::endl;
    return;
  }

  sf_writef_float(sndfile, output.data(), output.size());
  sf_close(sndfile);
}

// g++ -std=c++17 -I/path/to/onnxruntime/include     -L/path/to/onnxruntime/lib
// -lonnxruntime     main.cpp -o onnx_runner
