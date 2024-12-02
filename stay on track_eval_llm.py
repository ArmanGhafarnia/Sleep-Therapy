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

            if "yes" in redirection_evaluation:
                print(f"Off-topic message: '{patient_message}'")
                print(f"Therapist's successful redirection: '{therapist_message}'\n")
            else:
                print(f"Off-topic message: '{patient_message}'")
                print(f"Therapist's failed redirection: '{therapist_message}'\n")

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
        return 1.0, "No off-topic remarks by the patient, hence therapist stayed on track by default."
    else:
        stay_on_track_score = successful_redirections / total_off_topic
        feedback = f"Out of {total_off_topic} off-topic remarks, {successful_redirections} were successfully redirected."
        return stay_on_track_score, feedback


# Example conversation to evaluate
conversation = [
   ("I've been feeling really tired lately and it's hard to fall asleep.",
    "It sounds like you're struggling with getting enough rest. Can you tell me more about your sleep routine?"),
   ("Well, I usually go to bed around midnight, but sometimes I stay up later watching TV.",
    "Watching TV late at night can make it harder for your body to wind down. Have you tried setting a consistent bedtime?"),
   ("I think I should, but it’s just hard to turn off my mind sometimes.",
    "That makes sense. A lot of people have trouble quieting their minds before bed. Have you tried any relaxation techniques, like deep breathing or meditation?"),
   ("I tried meditation once, but I got frustrated because I couldn't focus.",
    "It's common to feel that way at first. Meditation takes practice, but even a few minutes can help calm your mind. Would you be open to trying it again with some guidance?"),
   ("Maybe. By the way, did you hear about the new movie that just came out? It’s supposed to be amazing.",
    "I haven't heard about it, but let's try to focus on your sleep for now. We can talk about relaxation strategies instead."),
   ("Okay, I also worry about my job a lot. My boss has been really demanding lately.",
    "Work stress can definitely impact sleep. Let’s try to find ways to manage that stress before bedtime. It might help to write down your worries earlier in the evening so you can set them aside."),
   ("That's a good idea. I usually just lay there thinking about everything I need to do.",
    "Exactly. Writing down your thoughts can help your mind relax, knowing that you've addressed those concerns for now. Let's discuss creating a calming bedtime routine that helps you let go of those thoughts."),
   ("Actually, did you see the football game last night? It was crazy!",
    "I didn't catch it, but let's bring our focus back to your sleep. Remember, reducing stimulating activities can help you fall asleep more easily."),
   ("Yeah, I guess. I also worry about my diet. I’ve been eating a lot of junk food lately.",
    "Diet can have an impact on sleep quality as well. Let's discuss some foods that might help you sleep better, and consider cutting back on things like caffeine and sugar in the evening."),
   ("Alright. By the way, I’m thinking of getting a new car. What do you think about electric cars?",
    "Electric cars are interesting, but for now, let's focus on your sleep and what we can do to improve it. We can revisit this topic another time."),
   ("Okay, I’ll give that a try.",
    "Great. Are there other things you do before bed, like using your phone or computer?"),
   ("Yeah, I’m often on my phone until I feel tired.",
    "Using your phone before bed can make it harder to fall asleep because of the blue light. It might help to put your phone away at least 30 minutes before bedtime and do something relaxing instead."),
   ("I guess I could try reading a book instead.",
    "That’s a great idea. Reading can be a good way to relax, especially if it’s something light and enjoyable."),
   ("What about exercise? I’ve heard it can help with sleep.",
    "Absolutely. Regular exercise can improve sleep quality, but try to avoid vigorous exercise too close to bedtime. Activities like yoga or a gentle walk can be especially helpful."),
   ("I’ll keep that in mind. Sometimes I feel too exhausted to exercise, though.",
    "That’s understandable. Even a small amount of movement, like stretching, can help. The key is to find something that feels manageable and consistent."),
   ("Oh, I almost forgot, I wanted to ask if you know a good restaurant in town.",
    "I might have some suggestions, but let’s stay focused on your sleep concerns for now. We can discuss this after we’ve worked on your sleep routine."),
   ("Okay, I’ll see if I can fit in some light exercise.",
    "That sounds like a good plan. Let’s also talk about setting some sleep goals. What would you like your sleep schedule to look like ideally?"),
   ("I think I’d like to be asleep by 11 pm and wake up around 7 am.",
    "That’s a great goal. We can work together to gradually shift your bedtime earlier and establish habits that support that schedule."),
   ("That would be really helpful. Thanks.",
    "You’re welcome. Remember, improving sleep is a process, and it’s okay to take it step by step. Let’s focus on making small, consistent changes."),
   ("Sounds good. I feel a bit better already.",
    "I’m glad to hear that. You’re doing a great job by being open and willing to make changes. Let’s keep working on this together.")
]








# Evaluate the example conversation
score, feedback = evaluate_conversation_stay_on_track(conversation)
print(f"Stay on Track Score: {score:.2f}")
print(f"Feedback: {feedback}")
