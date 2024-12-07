import openai
import textwrap

# Initialize your API key (replace with your actual API key)
openai.api_key = 'sk-proj-RNnrhY8CT2tWPIK7R2iTT3BlbkFJFWgbYOz4bFhFUHtPabTy'

# Define color codes
GREEN = '\033[92m'
YELLOW = '\033[93m'
RESET = '\033[0m'

def chat_with_gpt(messages, model="gpt-4o"):
    try:
        # Call OpenAI's API to generate a response
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            max_tokens=100,  # Increase response length
            n=1,             # We want only one response
            stop=None,       # Stop generation at a full response
            temperature=0.9  # Increase randomness in the output
        )

        # Extract the text response from the API
        reply = response['choices'][0]['message']['content']
        return reply
    except Exception as e:
        return f"Error: {e}"

# Main program loop
if __name__ == "__main__":
    print("Chatbot initialized. Type 'quit' to exit.")

    # Initialize conversation history
    messages = [
        {"role": "system", "content": "You are a therapist for helping patients that have insomnia. Answer empathetically and kindly."}
    ]

    for i in range(100):
        user_input = input(f"{GREEN}You:{RESET} ")

        if user_input.lower() == 'quit':
            print("Exiting chatbot.")
            break

        # Add user message to conversation history
        messages.append({"role": "user", "content": user_input})

        # Get response from the chatbot
        response = chat_with_gpt(messages)

        # Add assistant's response to conversation history
        messages.append({"role": "assistant", "content": response})

        # Print the response in a more readable format
        print(f"{YELLOW}Therapist:{RESET}")
        for paragraph in response.split('\n'):
            print(textwrap.fill(paragraph, width=70))  # Adjust width as needed
