from fastapi import FastAPI

app = FastAPI()

# Create a lock to control multi-user access to the server.
lock = threading.Lock()

# Create a global variable to indicate whether the server is currently in a blocked state.
is_blocking = False

# Define global variables to store the callback function output for displaying in the Gradio interface
global_text = ''
global_state = -1
split_byte_data = bytes(b"") # Used to store the segmented byte data


    # Create a function to receive data sent by the user using a request
    @app.route('/rkllm_chat', methods=['POST'])
    def receive_message():
        # Link global variables to retrieve the output information from the callback function
        global global_text, global_state
        global is_blocking

        # If the server is in a blocking state, return a specific response.
        if is_blocking or global_state==0:
            return jsonify({'status': 'error', 'message': 'RKLLM_Server is busy! Maybe you can try again later.'}), 503
        
        lock.acquire()
        try:
            # Set the server to a blocking state.
            is_blocking = True

            # Get JSON data from the POST request.
            data = request.json
            if data and 'messages' in data:
                # Reset global variables.
                global_text = []
                global_state = -1

                # Define the structure for the returned response.
                rkllm_responses = {
                    "id": "rkllm_chat",
                    "object": "rkllm_chat",
                    "created": None,
                    "choices": [],
                    "usage": {
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "total_tokens": None
                    }
                }

                if not "stream" in data.keys() or data["stream"] == False:
                    # Process the received data here.
                    messages = data['messages']
                    print("Received messages:", messages)
                    for index, message in enumerate(messages):
                        input_prompt = message['content']
                        rkllm_output = ""
                        
                        # Create a thread for model inference.
                        model_thread = threading.Thread(target=rkllm_model.run, args=(input_prompt,))
                        model_thread.start()

                        # Wait for the model to finish running and periodically check the inference thread of the model.
                        model_thread_finished = False
                        while not model_thread_finished:
                            while len(global_text) > 0:
                                rkllm_output += global_text.pop(0)
                                time.sleep(0.005)

                            model_thread.join(timeout=0.005)
                            model_thread_finished = not model_thread.is_alive()
                        
                        rkllm_responses["choices"].append(
                            {"index": index,
                            "message": {
                                "role": "assistant",
                                "content": rkllm_output,
                            },
                            "logprobs": None,
                            "finish_reason": "stop"
                            }
                        )
                    return jsonify(rkllm_responses), 200
                else:
                    messages = data['messages']
                    print("Received messages:", messages)
                    for index, message in enumerate(messages):
                        input_prompt = message['content']
                        rkllm_output = ""
                        
                        def generate():
                            model_thread = threading.Thread(target=rkllm_model.run, args=(input_prompt,))
                            model_thread.start()

                            model_thread_finished = False
                            while not model_thread_finished:
                                while len(global_text) > 0:
                                    rkllm_output = global_text.pop(0)

                                    rkllm_responses["choices"].append(
                                        {"index": index,
                                        "delta": {
                                            "role": "assistant",
                                            "content": rkllm_output,
                                        },
                                        "logprobs": None,
                                        "finish_reason": "stop" if global_state == 1 else None,
                                        }
                                    )
                                    yield f"{json.dumps(rkllm_responses)}\n\n"

                                model_thread.join(timeout=0.005)
                                model_thread_finished = not model_thread.is_alive()

                    return Response(generate(), content_type='text/plain')
            else:
                return jsonify({'status': 'error', 'message': 'Invalid JSON data!'}), 400
        finally:
            lock.release()
            is_blocking = False
        
    # Start the Flask application.
    # app.run(host='0.0.0.0', port=8080)
    app.run(host='0.0.0.0', port=8080, threaded=True, debug=False)

    print("====================")
    print("RKLLM model inference completed, releasing RKLLM model resources...")
    rkllm_model.release()
    print("====================")
