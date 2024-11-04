from typing import List, Dict
import openai


class ConversationEvaluator:
    def __init__(self, goals: List[str]):
        self.goals = goals
        openai.api_key = "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"

    def evaluate_conversation(self, conversation: List[tuple]) -> Dict[str, Dict[str, object]]:
        """
        Evaluate a conversation between the sleep therapist model and the patient using an LLM.
        :param conversation: The entire conversation as a list of tuples (patient_statement, therapist_response).
        :return: A dictionary containing goal accuracy evaluation for each goal.
        """
        evaluation_results = {}
        conversation_text = "\n".join([f"Patient: {p}\nTherapist: {t}" for p, t in conversation])

        for idx, goal in enumerate(self.goals, start=1):
            evaluation_results[f"Goal {idx}"] = {
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
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 "content": "You are an expert evaluator in the field of cognitive behavioral therapy for insomnia."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=10,
            temperature=0.0
        )
        answer = response['choices'][0]['message']['content'].strip()
        return answer == "Achieved"


# Example usage
goals = [
    "The model should effectively gather comprehensive information about the patient's current sleep issues, including difficulty falling or staying asleep, the frequency of sleep disruptions, and their impact on daily life.",
    "The model identifies and discusses unhealthy sleep practices, such as poor sleep hygiene, the use of substances that disrupt sleep (like caffeine or alcohol close to bedtime), and other behaviors detrimental to sleep like excessive bedtimes worry or screen time before sleep.",
    "The model needs to accurately assess the patient's circadian rhythm influences on sleep problems, such as being a \"night owl\" or \"morning person\" and how these tendencies affect their sleep quality and timing.",
    "It is crucial that the model explores and identifies any psychiatric, medical, or other sleep disorders that coexist with the insomnia, as these can have significant implications for management and treatment.",
    "The model should be able to help the patient set realistic and achievable sleep improvement goals based on the assessment findings.",
    "The model should encourage the patient to maintain a sleep diary as a critical tool for collecting accurate data about their sleep patterns, which is essential for making informed treatment decisions.",
    "Assess the model's proficiency in conducting a thorough intake interview that covers key areas necessary for an accurate understanding and subsequent treatment of insomnia."
]

conversation = [
    ("Hi there! I understand you’re having some trouble with sleep. Could you share a bit about what's been happening?", "Yes, I’ve been struggling with sleep for a while now. I can’t seem to fall asleep at a normal time, and I wake up a lot during the night."),
    ("That sounds really difficult. Let’s start by getting a comprehensive picture. How often do you have trouble falling asleep, and how often do you wake up during the night?", "It happens almost every night. I go to bed around 11 PM, but I don’t fall asleep until 2 or 3 AM. I also wake up a few times, usually around 4 or 5 AM."),
    ("Thank you for sharing that. It sounds like both falling asleep and staying asleep are challenging. How would you say this impacts your daily life?", "It makes me really tired all day. I struggle to concentrate at work, and I get irritable with my family. It’s like I’m always on edge."),
    ("I can imagine how exhausting that must be. It sounds like it’s affecting both your mood and ability to function day-to-day. Let’s talk a bit about your sleep environment and habits. Do you use your bed for activities other than sleep, like watching TV or being on your phone?", "Yes, I do. I watch TV in bed a lot, and I also scroll through my phone before trying to sleep."),
    ("Okay, that’s helpful to know. Using your bed for activities other than sleep can sometimes make it harder for your brain to associate it with rest. One goal we could work on is to reserve your bed for sleep only—no screens or other activities. Would you be willing to try that?", "I guess I could try. It’ll be tough, but I’m willing to give it a shot."),
    ("That’s great. It can definitely be challenging, but it often helps people get better sleep over time. Let’s also talk about your bedtime routine. Are there any particular habits or things you do right before bed?", "Usually, I watch TV until I feel tired, and then I just try to sleep. Sometimes I’ll have a glass of wine to help me relax."),
    ("Thank you for sharing that. Alcohol might seem to help at first, but it can actually disrupt sleep later in the night. How about we try replacing that with something else? Perhaps a relaxing activity like reading or listening to calming music?", "Yeah, I think I could do that."),
    ("Wonderful. Developing a consistent, calming bedtime routine can help your body know when it’s time to wind down. Let’s also explore how your body clock might be affecting your sleep. Would you describe yourself as more of a night owl or a morning person?", "Definitely a night owl. I’ve always found it easier to stay up late, but mornings are a struggle."),
    ("That’s important to recognize. Your natural circadian rhythm might be influencing your sleep schedule. We can try adjusting your routine gradually, maybe moving your bedtime earlier by 15 minutes at a time until you find a pattern that works better. How does that sound?", "That sounds doable. I’ve never really thought about it that way."),
    ("I think it’s a good place to start. Sometimes small adjustments can make a big difference. I’d also like to ask about any medical or psychological factors that might be contributing. Have you experienced anxiety, depression, or any health issues that might affect your sleep?", "I do get anxious a lot, especially about work. It’s hard to turn off my thoughts when I’m trying to sleep."),
    ("That’s very common, and it makes sense that it would affect your sleep. One approach we could try is cognitive restructuring—challenging those anxious thoughts before bed and replacing them with more calming ones. We could also consider keeping a sleep diary to help track what’s happening and identify patterns. Would you be open to that?", "Sure, I’ll try anything at this point."),
    ("That’s a great attitude. The sleep diary will help us get a clearer picture of your sleep patterns, and it will help guide us in making more effective changes. Let’s set some goals together: maintaining a sleep diary, creating a consistent bedtime routine without screens or alcohol, and working on addressing those anxious thoughts. How do those sound to you?", "That sounds good. I feel like I have a plan now."),
    ("I’m really glad to hear that. You’re not alone in this, and we’ll work together to find what helps. Let’s connect again next week to see how these strategies are working and make any necessary adjustments.", "Thank you so much. I feel more hopeful about this now."),
    ("You’re very welcome. Take care, and I’ll see you next week.", "See you next week.")
]



evaluator = ConversationEvaluator(goals)
results = evaluator.evaluate_conversation(conversation)

# Print evaluation results
for goal, result in results.items():
    print(f"{goal}: Achieved: {result['Achieved']}")
