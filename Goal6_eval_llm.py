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
    "The model identifies and discusses unhealthy sleep practices, such as poor sleep hygiene, the use of substances that disrupt sleep (like caffeine or alcohol close to bedtime), and other behaviors detrimental to sleep like excessive bedtime worry or screen time before sleep.",
    "The model needs to accurately assess the patient's circadian rhythm influences on sleep problems, such as being a \"night owl\" or \"morning person\" and how these tendencies affect their sleep quality and timing.",
    "It is crucial that the model explores and identifies any psychiatric, medical, or other sleep disorders that coexist with the insomnia.",
    "The model should be able to help the patient set realistic and achievable sleep improvement goals based on the assessment findings.",
    "The model should encourage the patient to maintain a sleep diary as a critical tool for collecting accurate data about their sleep patterns.",
    "Assess the model's proficiency in conducting a thorough intake interview that covers key areas necessary for an accurate understanding and subsequent treatment of insomnia. This includes gathering detailed information on the patient's sleep patterns, lifestyle and environmental influences, psychological and emotional factors, medical history.",
    "Establishing realistic treatment goals based on the patient's specific needs and circumstances."
]

conversation = [
    ("Hi there, I understand you're having some trouble with sleep. Can you tell me a bit more about what's been going on?", "Sure. I just feel like I can't seem to sleep well at night, no matter what I do. It's been really frustrating."),
    ("I see, that sounds really difficult. Can you describe what happens when you try to sleep? Do you have trouble falling asleep, staying asleep, or both?", "It's kind of both. I have trouble falling asleep most nights, and even when I do fall asleep, I wake up a lot. It's hard to get a full night of rest.",),
    ("That must be exhausting. How often would you say this happens? Is it every night or just sometimes?", "Honestly, it's almost every night. There are a few nights here and there when I sleep better, but it's pretty rare."),
    ("Thank you for sharing that. It sounds like it's having a big impact on you. How would you say this lack of sleep affects your daily life?", "Oh, it's definitely affecting me. I'm tired all the time, which makes it hard to focus at work. I get irritable with people around me, and I just don't feel like myself anymore."),
    ("I'm really sorry to hear that. It sounds like it's taking a toll on both your mood and your ability to function. Let's talk a little about your bedtime routine. Do you use your phone or watch TV in bed before trying to sleep?", "Yeah, I do. I usually watch TV in bed until I feel tired, and sometimes I'll scroll through my phone until I fall asleep."),
    ("Okay, thank you for sharing that. Using screens in bed can sometimes make it harder for your brain to associate the bed with sleep. It might be helpful to try and reserve your bed just for sleep. Maybe instead of watching TV or using your phone in bed, you could read a book or do something else relaxing but not involving a screen. How do you feel about that idea?", "I guess I could try, but it's just a habit I've had for a long time. It might be tough."),
    ("I completely understand. Breaking habits can be really challenging, especially when they're part of your routine. Do you also drink caffeine or alcohol in the evenings?", "Yeah, I usually have a cup of coffee in the afternoon, and sometimes I'll have a glass of wine at night to relax."),
    ("Thank you for sharing that. Both caffeine and alcohol can actually make it harder for you to get good quality sleep. Caffeine can stay in your system for hours, and alcohol, while it might make you feel relaxed initially, can disrupt your sleep later in the night. Would you be open to trying to cut back on those, maybe avoiding caffeine in the late afternoon and skipping the wine before bed?", "I didn't realize that about alcohol. I guess I could try cutting back, but I'm not sure how else to relax."),
    ("That's a really good point. Finding ways to relax without relying on alcohol can be tough, but there are some other things we could try. For example, deep breathing exercises, listening to calming music, or even gentle stretching might help. Do any of those sound like something you could try?", "I could try the breathing exercises. I think that might help me calm down."),
    ("Great. It might also be helpful for us to understand a bit more about your natural sleep preferences. Would you say you're more of a night owl or a morning person?", "I'm definitely a night owl. I've always stayed up late, even when I was a kid. Mornings are just really hard for me."),
    ("Thanks for letting me know. It sounds like your natural rhythm might be making it harder for you to get to sleep earlier. One thing we could try is gradually shifting your bedtime earlier, by maybe 15 minutes at a time, to see if we can help your body adjust. Would you be willing to try that?", "I suppose I could try, but I'm not sure how well it will work."),
    ("That's totally fair. It might take some time, but even small changes can make a big difference. I'd also like to ask if you have any medical conditions or if you've been diagnosed with any mental health issues like anxiety or depression that might be affecting your sleep?", "No, I haven't been diagnosed with anything, but I do get really anxious about work sometimes. I just can't turn my brain off at night."),
    ("I understand. Anxiety can be a major factor when it comes to sleep problems. One thing we could work on is trying to address those anxious thoughts before bed. For example, writing down your worries or making a to-do list for the next day can sometimes help get those thoughts out of your head. Would you be open to trying that?", "Yeah, I think that could help. I always have so many thoughts racing through my head when I'm trying to sleep."),
    ("I'm glad to hear that you're open to trying it. Another tool that can be really helpful is keeping a sleep diary. This would involve writing down when you go to bed, when you wake up, and any factors that might have affected your sleep. It could help us identify patterns and make more effective changes. How do you feel about that?", "I guess I could do that. It sounds like it might help me see what's going on."),
    ("That's great. It doesn't have to be anything complicated—just jotting down a few notes each day can make a big difference. So, let's summarize some of the things we've discussed today: avoiding screens in bed, cutting back on caffeine and alcohol, trying some relaxation techniques like deep breathing, gradually shifting your bedtime, writing down your worries before bed, and keeping a sleep diary. How does that sound to you?", "It sounds like a lot, but I think it's worth trying. I really want to get better sleep."),
    ("I know it sounds like a lot, but remember, you don't have to do everything all at once. Small steps can lead to big changes over time. I'm here to support you, and we can make adjustments as we go. Let's check in again next week to see how you're doing and talk about what worked and what didn't. How does that sound?", "That sounds good. Thank you for your help."),
    ("You're very welcome. Take care, and I'll talk to you next week.", "Thanks, see you next week.")
]




evaluator = ConversationEvaluator(goals)
results = evaluator.evaluate_conversation(conversation)

# Print evaluation results
for goal, result in results.items():
    print(f"{goal}: Achieved: {result['Achieved']}")
