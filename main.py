import openai
import textwrap

# Initialize your API key (replace with your actual API key)
openai.api_key = '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'

def chat_with_gpt(messages, model="gpt-3.5-turbo"):
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
        {"role": "system", "content": "You are a therapist for helping patients that have insomnia Answer empathetically and kindly"}
    ]

    for i in range(100):
        user_input = input("You: ")

        if user_input.lower() == 'quit':
            print("Exiting chatbot.")
            break

        # # Append the generated prompt to the messages
        # messages.append({"role": "system", "content": ""})

        # Add user message to conversation history
        messages.append({"role": "user", "content": user_input})

        # Get response from the chatbot
        response = chat_with_gpt(messages)

        # Add assistant's response to conversation history
        messages.append({"role": "assistant", "content": response})

        # Print the response in a more readable format
        print("Therapist:")
        for paragraph in response.split('\n'):
            print(textwrap.fill(paragraph, width=70))  # Adjust width as needed
