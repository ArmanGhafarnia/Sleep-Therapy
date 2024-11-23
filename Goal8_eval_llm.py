from typing import List, Dict
import openai


class ConversationEvaluator:
    def __init__(self, goals: List[str], goal_names: List[str]):
        self.goals = goals
        self.goal_names = goal_names
        openai.api_key = "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"

    def evaluate_conversation(self, conversation: List[tuple]) -> Dict[str, Dict[str, object]]:
        """
        Evaluate a conversation between the sleep therapist model and the patient using an LLM.
        :param conversation: The entire conversation as a list of tuples (patient_statement, therapist_response).
        :return: A dictionary containing goal accuracy evaluation for each goal.
        """
        evaluation_results = {}
        conversation_text = "\n".join([f"Patient: {p}\nTherapist: {t}" for p, t in conversation])

        for goal, goal_name in zip(self.goals, self.goal_names):
            evaluation_results[goal_name] = {
                "Description": goal,
                "Achieved": self.check_goal_achieved(goal, conversation_text)
            }

        return evaluation_results

    def check_goal_achieved(self, goal: str, conversation: str) -> bool:
        """
        Use an LLM to determine if a specific goal has been achieved in the conversation.
        :param goal: The goal description.
        :param conversation: The entire conversation.
        :return: True if the goal has been achieved, False otherwise.
        """
        prompt = f"""
        You are an expert evaluator in the field of cognitive behavioral therapy for insomnia. Below is a conversation between a therapist and a patient. Your task is to determine whether the following goal has been achieved in the conversation:

        Goal: "{goal}"

        Conversation:
        {conversation}

        Please respond with "Achieved" if the goal has been achieved, or "Not Achieved" if the goal has not been achieved.
        """
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system",
                 "content": "You are an expert evaluator in the field of cognitive behavioral therapy for insomnia."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=30,
            temperature=0.0
        )
        answer = response['choices'][0]['message']['content'].strip()
        return answer == "Achieved"

# Example usage

goal_names = [
    "Managing High Arousal States",
    "Sleep Hygiene Education",
    "Behavioral Strategies Adherence",
    "Personalized Sleep Strategies",
    "Assessing Strategy Effectiveness",
    "Sleep Mechanisms Education",
    "Providing Rationale for Interventions"
]

goals = [
    "The model should effectively discuss techniques to manage high arousal states that are disruptive to sleep. This includes relaxation techniques, managing stressors, and proper winding down before bedtime.",
    "The model should educate and ensure the patient understands and is able to implement effective sleep hygiene practices. This includes maintaining a consistent sleep schedule, optimizing the sleep environment (e.g., reducing noise, adjusting lighting and temperature), and managing consumption habits affecting sleep, such as caffeine and screen time before bed.",
    "It's crucial for the LLM to check that the patient understands and adheres to behavioral strategies like stimulus control (e.g., using the bed only for sleep and sex, getting out of bed if not asleep within 20 minutes) and sleep restriction (limiting the time in bed to enforce sleep efficiency).",
    "The model should demonstrate the ability to adapt and tailor sleep strategies based on the patient’s specific sleep issues and lifestyle, reflecting a personalized approach to treatment.",
    "Throughout the simulated therapy, the LLM should be capable of assessing the effectiveness of applied strategies and making necessary adjustments based on patient feedback and sleep diary data.",
    "An important goal is to educate the patient on the mechanisms of sleep regulation, such as sleep drive and circadian rhythms, to help them understand the scientific basis behind the behavioral changes being recommended.",
    "Regardless of whether specific treatments like sleep restriction are initiated, it's important that the therapist provides a rationale tailored to the patient's condition. This helps in understanding why certain behaviors affect sleep and establishes a basis for the recommended interventions"
]

conversation = [
    ("Good evening, Dr. I've been trying the things we discussed, but honestly, I feel like nothing's really working. I'm still waking up multiple times at night and it's really frustrating.",
     "Good evening! I'm really sorry to hear that. Let's start by taking a look at your sleep diary to see if there's anything we can spot together that might help us understand what's going on.")
    ,
    ("Yeah, here it is. I mean, you can see that most nights I'm waking up around 3 AM, and then I just can't seem to fall back asleep for hours.",
     "Right, I do see that pattern. It must be very exhausting to have such frequent wake-ups. What do you typically do when you wake up at that time?")
    ,
    ("Honestly, I just lie there and hope I fall back asleep. But after a while, I usually get frustrated and pick up my phone to scroll through social media until I feel tired again.",
     "I understand. It can be very tempting to use your phone to pass the time, but unfortunately, the blue light from screens can make it even harder to get back to sleep.")
    ,
    ("Yeah, I know it's not helping, but I just don't know what else to do. Lying there in the dark just feels so frustrating.",
     "I get it. Maybe we could think of an alternative activity. Something less stimulating like reading a book or listening to calming music might be a good idea."),
    ("I could try reading, but honestly, when I wake up like that, I just feel stuck in bed and don't want to get up.",
     "That's understandable. Sometimes getting out of bed is the last thing you want to do when you're tired. It might help, though, if you prepare something relaxing ahead of time—maybe leave a book on the nightstand to make it easier.")
    ,
    ("I guess I could try that. It's just so hard to think straight in the middle of the night.",
     "Of course. That's why planning ahead could make it easier to do in the moment. Let's also check in on your bedtime routine—how have evenings been for you?")
    ,
    ("Well, I try to keep things calm, but I still end up working on my laptop a lot. I know I shouldn't, but I just have a lot to do.",
     "I completely understand. It can be difficult to avoid work in the evenings, especially if you have a lot on your plate. Maybe cutting back just a little bit or setting a specific end time for work could help."),
    ("I can try, but honestly, it's tough. There's just so much to do, and I feel like I need to catch up.",
     "That makes sense. Even just a small reduction might help over time. Do you do anything specific to help you unwind after work?")
    ,
    ("Not really. I usually just watch TV until I feel sleepy.",
     "TV can be relaxing, but sometimes the content or the light can interfere with winding down. Maybe we could swap in some reading or listening to music as a wind-down activity?")
    ,
    ("I don't know. TV is just what I've always done, and I don't think I have the energy to change that right now.",
     "That’s totally fair. We don't have to change everything at once. Let’s keep TV if it works for you, but perhaps try lowering the brightness or choosing more calming shows.")
    ,
    ("Okay, I could do that. Should I still be trying to get up at the same time every day?",
     "Yes, consistency with wake time is still important. But I understand it can be tough, especially when sleep is so broken. Just do your best, and we'll adjust as needed.")
    ,
    ("Alright. I feel like I keep trying, but I'm not seeing much improvement.",
     "It can be a slow process, and it's natural to feel frustrated. Even small changes can eventually lead to improvement, so let's keep taking it step by step."),
    ("Yeah, I just hope it starts to get better soon. I’m tired of being tired all the time.",
     "I hear you, and I know it’s hard. We’re in this together, and I’m confident that things will improve. Let’s keep tracking what we can in the sleep diary and see what we learn over the next week.")
    ,
    ("Alright. Thanks for listening, Dr. I’ll keep trying.",
     "You’re welcome. I know you’re doing your best, and that matters. Let’s touch base again next week and see if anything shifts.")
]





evaluator = ConversationEvaluator(goals, goal_names)

results = evaluator.evaluate_conversation(conversation)

# Print evaluation results
for goal, result in results.items():
    achieved_text = "\033[92mTrue\033[0m" if result['Achieved'] else "\033[91mFalse\033[0m"
    print(f"{goal}: {achieved_text}")
