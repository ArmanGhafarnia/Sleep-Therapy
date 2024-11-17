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
   ("Hi, I'm here to help you with your sleep issues. Could you start by telling me a bit about what's been happening?", "Sure. I've been having trouble sleeping for a while. I can't fall asleep easily, and I wake up a lot during the night."),
   ("I see. That sounds very frustrating. Could you tell me how often you experience these issues—both falling asleep and waking up?", "It happens almost every night. I usually lie in bed for hours before I fall asleep, and I wake up at least twice each night."),
   ("Thank you for sharing that. How would you say this is impacting your life during the day?", "I'm tired all the time. It's hard to focus at work, and I'm often in a bad mood. It feels like I can never catch up on rest."),
   ("That sounds really challenging. Let's talk a bit about your evening routine. Do you use your phone or watch TV in bed before trying to sleep?", "Yes, I watch TV until I feel really tired, and then I try to sleep. I also scroll through my phone in bed."),
   ("Thank you for letting me know. Using screens in bed can make it harder for your brain to relax and associate the bed with sleep. Could we try to keep the bed for sleep only? Maybe replace TV and phone time with a relaxing activity like reading a book or listening to soothing music?", "I could try that, but it will be a big change."),
   ("I understand. It can be tough to change habits, but it's often very helpful for improving sleep quality. Do you also consume any caffeine or alcohol before bedtime?", "Yes, I usually have a cup of coffee in the late afternoon, and sometimes I have a glass of wine to help me relax."),
   ("Thanks for sharing that. Both caffeine and alcohol can disrupt your sleep. Caffeine can stay in your system for several hours, and alcohol might make you sleepy initially but can cause disruptions later in the night. Would you be willing to reduce caffeine intake in the afternoon and try avoiding alcohol before bed?", "I didn't realize that about alcohol. I can try cutting back on both."),
   ("That sounds like a great plan. Let's also talk about your natural sleep preferences. Would you say you're more of a night owl or a morning person?", "I'm definitely more of a night owl. I usually stay up late and have a hard time getting up early."),
   ("Thanks for sharing that. Being a night owl can definitely affect your sleep schedule. We could work on gradually shifting your bedtime earlier, by about 15 minutes at a time, to see if that helps you get more rest. Does that sound like something you'd be open to?", "Yeah, I think I could try that."),
   ("Perfect. Small adjustments like that can make a big difference. I'd also like to check if there are any other factors affecting your sleep. Do you experience anxiety, depression, or any health issues that could be impacting your sleep?", "Yes, I get really anxious sometimes, especially about work. It keeps me up at night."),
   ("I appreciate you sharing that. Anxiety is a common factor that can disrupt sleep. One approach we could try is practicing relaxation techniques before bed, such as deep breathing or guided imagery, to help calm your mind. Another helpful tool is keeping a sleep diary, where you record your sleep patterns and what might be affecting them. Would you be willing to try that?", "Yes, I think that could help."),
   ("That's wonderful. The sleep diary will give us a lot of useful information to help us make more informed decisions about your sleep. Let's set some goals together: keeping a sleep diary, creating a consistent bedtime routine without screens, and gradually shifting your bedtime earlier. How do those sound?", "That sounds like a good plan. I feel like I'm getting some direction now."),
   ("I'm really glad to hear that. Let's explore a bit more about your general health. Have you ever been diagnosed with any medical conditions, such as sleep apnea, restless legs syndrome, or any other conditions that might interfere with your sleep?", "No, I haven't been diagnosed with anything like that. But sometimes I feel like my legs are restless at night.",),
   ("That's helpful to know. Restless legs can be uncomfortable and might contribute to difficulty falling asleep. It could be worth discussing this with your healthcare provider to see if there's anything specific they recommend. In the meantime, some people find that stretching or gentle leg massage before bed helps. Would you be open to trying that?", "Yes, I could try that.",),
   ("Great. Another important factor is your sleeping environment. Can you tell me about your bedroom? Is it dark, quiet, and at a comfortable temperature when you try to sleep?", "My bedroom is pretty dark, but sometimes there's noise from outside, and it can get too warm in the summer.",),
   ("Thank you for sharing that. Having a cool, quiet environment can make a big difference in sleep quality. You could try earplugs or a white noise machine to help with outside noise, and perhaps a fan to keep the room cooler. How does that sound?", "I think I could try those suggestions.",),
   ("That sounds like a good step forward. I also want to ask about any routines or thoughts that keep you awake once you're in bed. Do you often find yourself worrying or thinking a lot when you're trying to sleep?", "Yes, I can't stop thinking about work or things I need to do the next day.",),
   ("That's really common. One thing that might help is writing down your thoughts before bed. It could be a to-do list for the next day or just a journal entry to get your thoughts out of your head. Would you be open to trying that as part of your bedtime routine?", "I think that could help. I'll give it a try.",),
   ("Wonderful. Let's add that to your plan: writing down your thoughts before bed, along with keeping the sleep diary, creating a consistent routine, reducing caffeine and alcohol, and gradually shifting your bedtime. These changes might take some time, but they can have a big impact on your sleep quality. How do you feel about everything we've discussed?", "I feel more hopeful now. It seems like I have a lot of things I can try, and it's good to have a plan.",),
   ("I'm really glad to hear that. You're not alone, and I'm here to support you every step of the way. Let's check in again in a week to see how these strategies are working and make any adjustments if needed. Take care, and I'll talk to you next week.", "Thank you so much for your help. Talk to you next week.")
]




evaluator = ConversationEvaluator(goals, goal_names)

results = evaluator.evaluate_conversation(conversation)

# Print evaluation results
for goal, result in results.items():
    achieved_text = "\033[92mTrue\033[0m" if result['Achieved'] else "\033[91mFalse\033[0m"
    print(f"{goal}: {achieved_text}")