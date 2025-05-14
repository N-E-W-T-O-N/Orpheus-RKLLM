std::tuple<std::vector<int64_t>,std::vector<int64_t>,std::vector<int64_t>> redistribute_codes(std::vector<int64_t>& output) ;
std::vector<float>run_onnx (    const std::vector<int64_t>& input0,    const std::vector<int64_t>& input1,    const std::vector<int64_t>& input2) ;
void saveWav( const std::vector<float>& output,const std::string& name,int sampleRate);
