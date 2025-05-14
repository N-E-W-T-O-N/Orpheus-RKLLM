#include "input.h"
#include <iostream>
#include <string>
#include <vector>

int main() {
  std::string input =
      "Hey there my name is EDISON, <giggles> and I'm a speech generation "
      "model that can sound like a person.I Am a badass person";
  std::cout << input << std::endl;
  std::vector<int> result = call_python_input(input);

  for (auto i : result) {
    std::cout << i << "\n";
  }
}

// g++ Testing_input.cpp input.cpp
