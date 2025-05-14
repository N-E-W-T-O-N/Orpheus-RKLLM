#include "input.h"
#include <cstdio>
#include <iostream>
#include <sstream>
#include <vector>

std::vector<int> call_python_input(const std::string &input) {
  std::string cmd = "python Input.py \"" + input + "\"";
  FILE *pipe = popen(cmd.c_str(), "r");
  if (!pipe)
    return {};

  char buffer[10000];
  std::string output;
  while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
    output += buffer;
  }
  pclose(pipe);

  // std::cout<<output;

  std::stringstream ss(output);
  std::vector<int> result;
  int val;
  char del = ',';
  std::string t;

  while (getline(ss, t, del)) {
    if (!t.empty())
      result.push_back(std::stoi(t));
  }
  // std::cout << "\"" << t << "\"" << " ";

  // while (ss >> val) {        result.push_back(val);    }

  return result;
}
