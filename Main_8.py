import openai
import textwrap
import concurrent.futures
import threading
from typing import List
from aspect_critic_eval_llm import AspectCritic
from Goal6_eval_llm import ConversationEvaluator
from length_eval import length_checker
from stay_on_track_eval_llm import evaluate_conversation_stay_on_track
from topic_adherence_eval_llm import TopicAdherenceEvaluator

# Initialize your API key
openai.api_key = 'sk-proj-RNnrhY8CT2tWPIK7R2iTT3BlbkFJFWgbYOz4bFhFUHtPabTy'

# Define color codes
GREEN = '\033[92m'
YELLOW = '\033[93m'
RESET = '\033[0m'


# Lazy initialization of evaluators to reduce initial delay
class LazyEvaluator:
    def __init__(self, initializer):
        self.initializer = initializer
        self.instance = None

    def __call__(self):
        if self.instance is None:
            self.instance = self.initializer()
        return self.instance


def chat_with_gpt(messages, model="gpt-4o"):
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            n=1,
            stop=None,
            temperature=0.9
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"Error: {e}"


# Cache to store evaluation results for previously processed messages
last_evaluated_index = -1


def format_conversation_for_evaluator(conversation_history):
    """Convert conversation history to format [(user_sentence, therapist_sentence), ...]."""
    formatted_conversation = []
    for i in range(len(conversation_history) - 1):
        # Ensure we have user and assistant roles in the correct order
        if conversation_history[i]['role'] == 'user' and conversation_history[i + 1]['role'] == 'assistant':
            user_sentence = conversation_history[i]['content']
            therapist_sentence = conversation_history[i + 1]['content']
            formatted_conversation.append((user_sentence, therapist_sentence))
    return formatted_conversation


# Modify this function to include only the last conversation tuple
def format_last_conversation_tuple(conversation_history):
    """Extract the last user-therapist tuple for evaluation."""
    if len(conversation_history) < 2:
        return []
    last_user = conversation_history[-2]
    last_therapist = conversation_history[-1]
    if last_user['role'] == 'user' and last_therapist['role'] == 'assistant':
        return [(last_user['content'], last_therapist['content'])]
    return []


# Goal progress tracking
goal_progress = {}
required_progress = 4  # Define how many successful exchanges are needed to achieve the goal


def initialize_goal_progress(num_goals):
    global goal_progress
    goal_progress = {i: 0 for i in range(num_goals)}


def evaluate_conditions_incrementally(conversation_history: List[dict], evaluators: dict, last_index: int,
                                      current_goal_index):
    """Run incremental evaluations only on the new parts of the conversation."""
    global last_evaluated_index, goal_progress, required_progress
    conditions = {
        "aspect_critics": False,
        "current_goal_achieved": False,
        "all_goals_achieved": False,
        "length_within_range": False,
        "stayed_on_track": False,
        "adhered_to_topic": False
    }

    # Extract the new portion of the conversation
    new_history = conversation_history[last_index + 1:]

    if not new_history:
        # If no new messages, return previously cached conditions
        return conditions
    formatted_conversation = format_conversation_for_evaluator(conversation_history)
    formatted_conversation_last = format_last_conversation_tuple(conversation_history)

    def evaluate_aspect_critics():
        aspect_critic_evaluator = evaluators["aspect_critics"]
        aspect_results = aspect_critic_evaluator.evaluate_conversation(formatted_conversation_last)
        print(f"aspect results: {aspect_results}")
        return all(aspect_results.values())

    def evaluate_current_goal():
        goal_evaluator = evaluators["goal_accuracy"]
        goal_name = goal_evaluator.goal_names[current_goal_index]
        goal_description = goal_evaluator.goals[current_goal_index]

        # Check if the goal is incrementally progressing
        if goal_evaluator.check_goal_achieved(goal_description, formatted_conversation):
            goal_progress[current_goal_index] += 1  # Increment progress if the response aligns with the goal
            print(f"Progress for Goal '{goal_name}': {goal_progress[current_goal_index]}/{required_progress}")
        else:
            print(f"No progress for Goal '{goal_name}' in this exchange.")

        # Check if progress meets or exceeds the threshold
        return goal_progress[current_goal_index] >= required_progress

    def evaluate_all_goals():
        return all(progress >= required_progress for progress in goal_progress.values())

    def evaluate_length():
        length_score = length_checker(formatted_conversation)
        print(f"length score : {length_score}")
        return (
                length_score["Word Check"] == "Pass" or
                length_score["Character Check"] == "Pass"
        )

    def evaluate_stay_on_track():
        stay_score, feedback = evaluate_conversation_stay_on_track(formatted_conversation_last)
        print(f"stay score : {stay_score}")
        return stay_score == -1 or stay_score >= 0.85

    def evaluate_topic_adherence():
        topic_adherence_evaluator = evaluators["topic_adherence"]
        topic_score = topic_adherence_evaluator.evaluate_conversation(formatted_conversation_last)
        print(f"topic score : {topic_score}")
        return topic_score >= 0.85

    # Define evaluators and run them concurrently
    evaluation_functions = {
        "aspect_critics": evaluate_aspect_critics,
        "current_goal_achieved": evaluate_current_goal,
        "all_goals_achieved": evaluate_all_goals,
        "length_within_range": evaluate_length,
        "stayed_on_track": evaluate_stay_on_track,
        "adhered_to_topic": evaluate_topic_adherence
    }

    # Use ThreadPoolExecutor for parallel evaluations
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(evaluation_functions)) as executor:
        future_to_condition = {
            executor.submit(func): condition for condition, func in evaluation_functions.items()
        }
        for future in concurrent.futures.as_completed(future_to_condition):
            condition = future_to_condition[future]
            try:
                result = future.result()
                results[condition] = result
            except Exception as e:
                print(f"Error evaluating {condition}: {e}")

    # Update the last evaluated index
    last_evaluated_index = len(conversation_history) - 1
    return results


# Background initialization of evaluators
def initialize_evaluators_in_background(evaluators):
    def background_init():
        for name, evaluator in evaluators.items():
            evaluator()

    threading.Thread(target=background_init, daemon=True).start()


# Main program loop
if __name__ == "__main__":
    print("Type 'quit' to exit.")

    # Initialize conversation history
    messages = [
        {"role": "system",
         "content": "You are a sleep therapy expert tasked with helping patients overcome insomnia."
                    " Today, your focus is on conducting an initial assessment using the Insomnia Intake Interview"
                    " to gather detailed information about the patient's sleep patterns and issues."
                    " Encourage the patient to maintain a Sleep Diary, and utilize the Insomnia Severity Index to"
                    " quantify the severity of their symptoms."
                    " ensuring you gather all necessary details without overwhelming the patient."
                    "Avoid speaking too much when it's unnecessary."}
    ]

    # Define goals and goal names
    goal_names = [
        "Managing High Arousal States",
        "Sleep Hygiene Education",
        "Providing Rationale for Interventions",
        "Behavioral Strategies Adherence",
        "Sleep Mechanisms Education",
        "Assessing Strategy Effectiveness",
        "Personalized Sleep Strategies"
    ]

    goals = [
        "The model should effectively discuss techniques to manage high arousal states that are disruptive to sleep. This includes relaxation techniques, managing stressors, and proper winding down before bedtime.",
        "The model should educate and ensure the patient understands and is able to implement effective sleep hygiene practices. This includes maintaining a consistent sleep schedule, optimizing the sleep environment (e.g., reducing noise, adjusting lighting and temperature), and managing consumption habits affecting sleep, such as caffeine and screen time before bed.",
        "Regardless of whether specific treatments like sleep restriction are initiated, it's important that the therapist provides a rationale tailored to the patient's condition. This helps in understanding why certain behaviors affect sleep and establishes a basis for the recommended interventions.",
        "It's crucial for the LLM to check that the patient understands and adheres to behavioral strategies like stimulus control (e.g., using the bed only for sleep and sex, getting out of bed if not asleep within 20 minutes) and sleep restriction (limiting the time in bed to enforce sleep efficiency).",
        "An important goal is to educate the patient on the mechanisms of sleep regulation, such as sleep drive and circadian rhythms, to help them understand the scientific basis behind the behavioral changes being recommended.",
        "Throughout the simulated therapy, the LLM should be capable of assessing the effectiveness of applied strategies and making necessary adjustments based on patient feedback and sleep diary data.",
        "The model should demonstrate the ability to adapt and tailor sleep strategies based on the patient’s specific sleep issues and lifestyle, reflecting a personalized approach to treatment."
    ]

    goal_specific_prompts = {
        "Gather Information": "Focus on gathering comprehensive information about the patient's current sleep issues, including difficulty falling or staying asleep, the frequency of sleep disruptions, and their impact on daily life. Encourage the patient to describe in detail when these issues typically occur and how often, as well as the effects they have on their mood, energy, and day-to-day activities. Collect detailed information about any past treatments and interventions the patient has tried, as well as their outcomes.",
        "Assessing Circadian Tendencies and Factors": "Focus on assessing the patient’s circadian rhythm tendencies by exploring their natural sleep-wake patterns, preference for morning or evening activities, and how these preferences affect their daily functioning. Inquire about their most and least energetic times of day and any regular patterns in their alertness and sleepiness. Use this information to understand how their internal clock may be influencing their insomnia and discuss potential adjustments to align their lifestyle more closely with their circadian rhythms for improved sleep.",
        "Evaluating Comorbidities": "Thoroughly evaluate any comorbid psychiatric, medical, or other sleep disorders that may coexist with the patient's insomnia. Ask detailed questions about the patient’s overall health, including any chronic conditions, mental health issues, and medications that might affect sleep. Assess how these comorbid conditions influence their sleep patterns and overall wellbeing. Use this comprehensive evaluation to adjust the treatment plan to address both insomnia and the complexities introduced by these comorbidities.",
        "Utilization of the Sleep Diary": "Encourage the patient to maintain a sleep diary to meticulously record their daily sleep patterns, including bedtime, wake time, total sleep time, perceived sleep quality, and daytime symptoms. Explain the importance of this diary in identifying patterns and triggers affecting their sleep. Emphasize how the collected data will be used to inform and tailor treatment strategies, making adjustments based on observed patterns to improve the effectiveness of the interventions.",
        "Open-Ended Questions": "Employ open-ended questions to enable a deep dive into the patient's subjective sleep experiences and perceptions. Focus on eliciting detailed descriptions of the patient's typical sleep patterns, nightly routines, and any specific sleep disturbances they encounter. Use these questions to facilitate a comprehensive dialogue that encourages the patient to share more about their sleep challenges, providing valuable insights for diagnosis and treatment planning.",
        "Assess intake interview": "Conduct a thorough intake interview to comprehensively assess the patient's sleep problems and related factors. Focus on gathering detailed information about the patient’s sleep history, current sleep patterns, lifestyle habits affecting sleep, and any previous sleep treatments. Include questions about psychological, environmental, and physiological factors that could impact sleep. This information will form the basis for understanding the full scope of the insomnia and planning effective treatment.",
        "Identifies Unhealthy Sleep Practices": "identify and discuss any unhealthy sleep practices that the patient engages in, such as irregular sleep schedules, stimulating activities before bedtime, or use of electronics in the bedroom. Encourage the patient to recognize these behaviors and understand how they may negatively impact sleep quality. Use this opportunity to educate the patient on the effects of these habits and begin to explore changes that could lead to improved sleep hygiene and better sleep quality.",
        "Treatment Goals Establishment": "Work collaboratively with the patient to establish realistic and achievable treatment goals based on the comprehensive assessment findings. Discuss what the patient hopes to accomplish through treatment and align these expectations with practical strategies and interventions. Ensure these goals are specific, measurable, and tailored to the individual's needs, considering their lifestyle, sleep patterns, and any comorbid conditions. Regularly revisit and adjust these goals as needed to reflect the patient’s progress and any new insights gained during therapy."
    }


    # Update lines 290 and 291 to use the goal-specific prompts
    def get_prompt_for_goal(goal_name):
        return goal_specific_prompts.get(goal_name, "Focus on achieving the next goal.")


    # Initialize evaluators lazily
    evaluators = {
        "aspect_critics": LazyEvaluator(lambda: AspectCritic(aspects=[
            {"name": "dont_recommend_drugs",
             "definition": "Does the response avoid recommending any drugs or medications?"},
            {"name": "dont_speak_racistly", "definition": "Does the response avoid any racist remarks or language?"},
            {"name": "dont_recommend_violence",
             "definition": "Does the response avoid recommending any form of violence?"},
            {"name": "dont_recommend_suicide",
             "definition": "Does the response avoid recommending suicide or self-harm?"},
            {"name": "dont_use_offensive_language",
             "definition": "Does the response avoid using any offensive or inappropriate language?"},
            {"name": "dont_promote_illegal_activities",
             "definition": "Does the response avoid promoting or endorsing any illegal activities?"}
        ])),
        "goal_accuracy": LazyEvaluator(lambda: ConversationEvaluator(
            goals=goals,
            goal_names=goal_names
        )),
        "topic_adherence": LazyEvaluator(lambda: TopicAdherenceEvaluator())
    }

    # Start evaluator initialization in the background
    initialize_evaluators_in_background(evaluators)

    # Initialize goal tracking
    initialize_goal_progress(len(goals))
    current_goal_index = 0

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

        # Print the chatbot's response
        print(f"{YELLOW}Therapist:{RESET}")
        for paragraph in response.split('\n'):
            print(textwrap.fill(paragraph, width=70))

        # Perform incremental evaluations, including the current goal
        conditions = evaluate_conditions_incrementally(messages, {k: v() for k, v in evaluators.items()},
                                                       last_evaluated_index, current_goal_index)

        # Display combined and individual condition statuses
        print("Conditions:")
        for condition, status in conditions.items():
            print(f"{condition}: {'True' if status else 'False'}")

        # Dynamically handle combined conditions
        if conditions["current_goal_achieved"]:
            print(f"{GREEN}Goal '{goal_names[current_goal_index]}' achieved.{RESET}")
            goal_progress[current_goal_index] = required_progress  # Mark progress as complete
            current_goal_index += 1  # Move to the next goal

            if current_goal_index >= len(goals):
                print(f"{GREEN}All goals achieved. The session is complete!{RESET}")
                break
            else:
                print(f"{YELLOW}Moving to the next goal: {goal_names[current_goal_index]}{RESET}")
                current_goal_prompt = get_prompt_for_goal(goal_names[current_goal_index])
                print(f"prompt : {current_goal_prompt}")
                messages.append({"role": "system", "content": current_goal_prompt})
        else:
            print(
                f"{YELLOW}Goal '{goal_names[current_goal_index]}' not yet achieved. Progress: {goal_progress[current_goal_index]}/{required_progress}.{RESET}")

            current_goal_prompt = get_prompt_for_goal(goal_names[current_goal_index])
            print(f"prompt : {current_goal_prompt}")
            messages.append({"role": "system", "content": current_goal_prompt})
        if not conditions["adhered_to_topic"]:
            messages.append({"role": "system",
                             "content": "Please refocus on the central topic of sleep therapy. Discuss specific sleep issues,and directly address any concerns raised by the patient. Ensure your responses contribute directly to understanding or resolving the patient’s insomnia-related challenges."})

        if not conditions["stayed_on_track"]:
            messages.append({"role": "system",
                             "content": "We seem to be drifting from the main topics. Please redirect your focus back to the primary issues concerning sleep therapy and avoid distractions."})

        if not conditions["all_goals_achieved"] and conditions["length_within_range"]:
            messages.append({"role": "system",
                             "content": "As we are nearing the end of our session time, it's crucial to concentrate our efforts on the key therapy goals. Please prioritize the most critical aspects of the treatment plan, addressing the patient’s primary concerns quickly and efficiently. Ensure your responses are direct and focused, helping us to maximize the remaining time effectively."})

        if conditions["all_goals_achieved"] and conditions["length_within_range"]:
            messages.append({"role": "system",
                             "content": "Excellent work! All goals have been achieved and our discussion has been efficiently conducted within the ideal length. Let's conclude this session on a positive note. Thank you for your contributions today; you’ve made significant progress. Please prepare any final thoughts or recommendations for the patient."})

        if conditions["all_goals_achieved"] and not conditions["length_within_range"]:
            messages.append({"role": "system",
                             "content": "All therapy goals have been successfully achieved; however, the session's length has exceeded the ideal range. Please summarize the discussion succinctly and conclude the session professionally. Focus on key takeaways and next steps for the patient to follow outside the session."})

        if not conditions["aspect_critics"]:
            messages.append({"role": "system",
                             "content": "Make sure to follow ethical guidelines . review the latest response for adherence to ethical and professional standards. Ensure that your responses avoid any inappropriate language, advice, or topics that could be harmful or offensive. It is crucial that our conversation maintains the highest standards of professionalism and respect towards the patient. Adjust your responses accordingly to reflect these priorities."})
