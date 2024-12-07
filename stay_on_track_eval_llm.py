import openai
import concurrent.futures

# Initialize the API key
openai.api_key = "sk-proj-RNnrhY8CT2tWPIK7R2iTT3BlbkFJFWgbYOz4bFhFUHtPabTy"

# Conversation format: conversation = [("message from person 1"), ("message from person 2"), ...]
def evaluate_conversation_stay_on_track(conversation):
    """
    Evaluates how well the sleep therapist model stays on track when the patient says something unrelated.
    :param conversation: List of tuples representing the conversation.
    :return: Evaluation score and detailed feedback.
    """
    total_off_topic = 0
    successful_redirections = 0

    def evaluate_message(patient_message, therapist_message, previous_messages):
        # Use LLM to determine if the patient's message is off-topic
        context = "\n".join(previous_messages)

        patient_response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system",
                 "content": "You are an evaluator that determines if a patient's statement is off-topic in a sleep therapy session. The context of the conversation is provided to help you decide. Note that closing remarks such as 'thank you', 'take care', or 'see you next week' are considered on-topic if they are part of the natural conversation flow."},
                {"role": "user",
                 "content": f"Conversation context:\n{context}\n\nThe patient's message is: '{patient_message}'. Is this message related to sleep therapy? Answer 'yes' or 'no'."}
            ]
        )
        patient_evaluation = patient_response["choices"][0]["message"]["content"].strip().lower()

        if "no" in patient_evaluation:
            # Use LLM to determine if the therapist's response brings the conversation back on track
            redirection_response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[
                    {"role": "system",
                     "content": "You are an evaluator that determines if a therapist's response successfully redirects the conversation back to sleep therapy. The context of the conversation is provided to help you decide. Note that closing remarks such as 'thank you', 'take care', or 'see you next week' are considered acceptable and on-topic as part of the natural conversation flow."},
                    {"role": "user",
                     "content": f"Conversation context:\n{context}\n\nThe patient's off-topic message was: '{patient_message}'. The therapist's response is: '{therapist_message}'. Does this response bring the conversation back to sleep therapy? Answer 'yes' or 'no'."}
                ]
            )
            redirection_evaluation = redirection_response["choices"][0]["message"]["content"].strip().lower()

            return "off-topic", redirection_evaluation
        else:
            return "on-topic", None

    # Use ThreadPoolExecutor to parallelize evaluations
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_evaluation = {}
        previous_messages = []
        for patient_message, therapist_message in conversation:
            future = executor.submit(evaluate_message, patient_message, therapist_message, previous_messages.copy())
            future_to_evaluation[future] = (patient_message, therapist_message)
            previous_messages.append(f"Patient: {patient_message}")
            previous_messages.append(f"Therapist: {therapist_message}")

        for future in concurrent.futures.as_completed(future_to_evaluation):
            patient_message, therapist_message = future_to_evaluation[future]
            try:
                evaluation_result, redirection_evaluation = future.result()
                if evaluation_result == "off-topic":
                    total_off_topic += 1
                    if redirection_evaluation and "yes" in redirection_evaluation:
                        successful_redirections += 1
            except Exception as exc:
                print(f"Generated an exception: {exc}")

    # Calculate score
    if total_off_topic == 0:
        feedback = "\033[1;93mNo off-topic remarks by the patient, hence therapist stayed on track by default.\033[0m"
        return -1, feedback

    else:
        stay_on_track_score = successful_redirections / total_off_topic
        feedback = (f"Out of \033[1;91m{total_off_topic} off-topic remarks\033[0m, "
                    f"\033[1;92m{successful_redirections} were successfully redirected\033[0m.")
        return stay_on_track_score, feedback


conversation = [
    (
        "hi i have trouble falling asleep",
        "Hi there, I'm sorry to hear you're having difficulty falling asleep. "
        "It sounds like it might be really frustrating and exhausting for you. "
        "Sleep is so important for our overall well-being. Can you tell me a little more about what you've been experiencing? "
        "Maybe we can explore some strategies together to help you find a more restful night."
    ),
    (
        "did you watch football today?",
        "I haven't watched the game, but I'm here to help you with your sleep concerns. "
        "If you're finding it difficult to fall asleep after watching a game or if sports excitement is affecting your rest, "
        "we can talk about some relaxation techniques to help ease your mind before bed. "
        "How do you usually feel after watching a game at night?"
    )
]





# Evaluate the example conversation
score, feedback = evaluate_conversation_stay_on_track(conversation)
# print(f"Stay on Track Score: \033[1;92m{score:.2f}\033[0m")
#
# print(f"Feedback: {feedback}")
