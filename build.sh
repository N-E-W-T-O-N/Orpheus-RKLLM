#!/bin/bash

show_help() {
  echo "Usage: $0 [options]"
  echo "  -h                          Show help"
  echo "  -b                          Build binary"
  echo "  -r <file> <n1> <n2> <text>  Run inference"
  echo "  -t                          Run tests"
}

run_build() {
  echo "🔧 Building binary..."
  g++ Inference.cpp input.cpp output.cpp -lrkllmrt \
    -I onnxruntime/include \
    -L onnxruntime/lib -L onnxruntime/lib/libonnxruntime* \
    -lpthread -ldl -lm -lsndfile \
    -o llm
  echo "✅ Build completed."
}

run_inference() {
  if [[ $# -lt 4 ]]; then
    echo "❌ Error: -r requires 4 arguments: <file> <n1> <n2> <text>"
    show_help
    exit 1
  fi

  local file="$1"
  local n1="$2"
  local n2="$3"
  local text="$4"

  echo "🚀 Running inference with:"
  echo "  File            : $file"
  echo "  max_new_tokens  : $n1"
  echo "  max_context_len : $n2"
  echo "  Text            : $text"

  ./llm "$file" "$n1" "$n2" "$text"
}

run_tests() {
  echo "🧪 Running tests..."

  echo "📦 Testing input.cpp..."
  g++ Testing_input.cpp input.cpp -o testing1 && ./testing1

  echo "📦 Testing output.cpp..."
  export LD_LIBRARY_PATH=$(pwd)/onnxruntime/lib:$LD_LIBRARY_PATH
  g++ Testing_output.cpp \
    -I onnxruntime/include \
    -L onnxruntime/lib \
    -lonnxruntime -lsndfile \
    -o testing2 && ./testing2
  echo ""
  echo "✅ Tests completed."
}

# Main
if [[ "$1" == "-r" ]]; then
  shift
  run_inference "$@"
  exit 0
fi

while getopts "hbt" opt; do
  case "$opt" in
    h) show_help; exit 0 ;;
    b) run_build ;;
    t) run_tests ;;
    *) show_help; exit 1 ;;
  esac
done

