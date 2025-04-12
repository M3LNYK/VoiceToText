import ollama


def test_ollama_model(model_name="mistral"):
    try:
        print(f"Testing connection to {model_name}...")

        # Send a simple test prompt
        response = ollama.chat(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": 'Hello! Please fix this sentence: "Thes is a tst of my new modle."',
                }
            ],
        )

        # Print the response
        print("\nModel response:")
        print(response["message"]["content"])
        print("\nTest completed successfully! Your model is working.")

    except Exception as e:
        print(f"\nError: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure Ollama is running")
        print("2. Check if the model is installed (run 'ollama list' in terminal)")
        print("3. Verify your environment variable is set correctly")


if __name__ == "__main__":
    # Replace "mistral" with your model name if different
    test_ollama_model("mistral")
