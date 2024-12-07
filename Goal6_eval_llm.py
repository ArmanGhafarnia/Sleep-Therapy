from typing import List, Dict
import openai


class ConversationEvaluator:
    def __init__(self, goals: List[str], goal_names: List[str]):
        self.goals = goals
        self.goal_names = goal_names
        openai.api_key = "sk-proj-RNnrhY8CT2tWPIK7R2iTT3BlbkFJFWgbYOz4bFhFUHtPabTy"

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
            model="gpt-4o",
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
    "Gather Information",
    "Identifies Unhealthy Sleep Practices",
    "Assessing Circadian Tendencies and Factors",
    "Evaluating Comorbidities",
    "Treatment Goals Establishment",
    "Utilization of the Sleep Diary",
    "Assess intake interview",
    "Open-Ended Questions"
]

goals = [
    "The model should effectively gather comprehensive information about the patient's current sleep issues, including difficulty falling or staying asleep, the frequency of sleep disruptions, and their impact on daily life.",
    "The model identifies and discusses unhealthy sleep practices, such as poor sleep hygiene, the use of substances that disrupt sleep (like caffeine or alcohol close to bedtime), and other behaviors detrimental to sleep like excessive bedtime worry or screen time before sleep.",
    "The model needs to accurately assess the patient's circadian rhythm influences on sleep problems, such as being a \"night owl\" or \"morning person\" and how these tendencies affect their sleep quality and timing.",
    "It is crucial that the model explores and identifies any psychiatric, medical, or other sleep disorders that coexist with the insomnia.",
    "The model should be able to help the patient set realistic and achievable sleep improvement goals based on the assessment findings.",
    "The model should encourage the patient to maintain a sleep diary as a critical tool for collecting accurate data about their sleep patterns.",
    "Assess the model's proficiency in conducting a thorough intake interview that covers key areas necessary for an accurate understanding and subsequent treatment of insomnia. This includes gathering detailed information on the patient's sleep patterns, lifestyle and environmental influences, psychological and emotional factors, medical history.",
    "The model should ask open-ended questions that encourage the patient to describe their sleep problems in detail."
]


conversation = [
    ("Hi, I hear you're having trouble sleeping. Can you tell me more about that?", "Yeah, I've been struggling to get enough sleep. I feel like I'm awake most of the night."),
    ("That sounds really frustrating. Have you always had trouble sleeping, or is this something new?", "It's been going on for a few months now. I used to sleep fine."),
    ("I see. Do you think there's been any major change in your life recently? Work stress, maybe?", "Yeah, work has been pretty overwhelming. I just can't stop thinking about it when I go to bed."),
    ("Stress can definitely impact sleep. Have you tried relaxing before bed, like taking a bath or listening to music?", "I listen to music sometimes, but it doesn't always help."),
    ("That's a start. What kind of music do you listen to?", "Usually something calming, like instrumental or nature sounds."),
    ("That's great. Maybe stick to that. Have you tried meditating or doing yoga?", "Not really. I'm not sure I'd know how to start."),
    ("It's pretty simple to get into. You could look up some videos online or download an app. A lot of people find that helpful.", "I might try that. But honestly, I feel like my brain just doesn't shut off."),
    ("That can be really tough. Have you thought about journaling? Sometimes writing down your thoughts before bed can help clear your mind.", "I've tried it a couple of times, but it felt like it just made me think more."),
    ("Hmm, that’s tricky. Maybe you just need to focus on positive thoughts instead. Like, think about things you're grateful for.", "I guess I could try that, but it doesn’t seem to help much."),
    ("Okay. How’s your bedroom environment? Is it quiet, dark, and cool?", "It's pretty quiet, but I like to have a little light on. Total darkness feels weird."),
    ("Maybe try using a dim nightlight instead of a regular lamp. Sometimes too much light can make it harder to sleep.", "I suppose I could try that."),
    ("Do you have a consistent bedtime, or does it change depending on how tired you feel?", "It depends. Some nights I'm up until midnight, other nights I crash earlier."),
    ("Consistency is important. Maybe aim to go to bed at the same time every night?", "Yeah, but I don't feel tired at the same time every night."),
    ("That’s understandable. You could try staying up until you feel really sleepy and then go to bed.", "I already do that, but it doesn’t help."),
    ("Well, it might just take some time. Have you tried drinking herbal tea before bed? Chamomile can be really soothing.", "Not really. I’m not much of a tea person."),
    ("You could give it a shot. Or maybe warm milk, like people say.", "I don’t think milk will help."),
    ("Okay, that's fine. Do you think you get enough physical activity during the day?", "Not really. I’m so tired all the time that I don’t have the energy to exercise."),
    ("Exercise can be really important for sleep. Even just a short walk during the day might help.", "Maybe. I’ll try to work that in."),
    ("Good idea. And how about your diet? Do you eat close to bedtime?", "Sometimes I have a snack, but nothing big."),
    ("That’s fine, as long as it’s not too heavy. Maybe avoid sugar late at night.", "Yeah, I don’t really eat sweets at night."),
    ("Great. Well, it sounds like you’re already doing a lot of the right things. Sleep will probably improve on its own over time.", "I hope so. It’s been tough, though."),
    ("I get that. Just try to stay positive and keep experimenting with what works best for you. Let’s touch base in a few weeks to see how things are going.", "Alright. Thanks for your time.")
]





evaluator = ConversationEvaluator(goals, goal_names)

results = evaluator.evaluate_conversation(conversation)

# # Print evaluation results
# for goal, result in results.items():
#     achieved_text = "\033[92mTrue\033[0m" if result['Achieved'] else "\033[91mFalse\033[0m"
#     print(f"{goal}: {achieved_text}")