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
    "Stimulus Control Explanation",
    "Sleep Restriction Therapy Description",
    "Combining Stimulus Control and Sleep Restriction",
    "Assess Patient Understanding and Commitment",
    "Dealing with Common Obstacles",
    "Rules of Stimulus Control",
    "Determining Initial Sleep Window",
    "Adjusting Recommendations Based on Feedback",
    "Addressing Common Sleep Misconceptions",
    "Monitoring Progress and Modifying Techniques"
]

goals = [
    "The Model should articulate what stimulus control is and why it's used in treating insomnia. It must explain that this approach helps patients associate the bed and bedroom with sleep and not with wakefulness or other activities.",
    "The Model must clearly describe how to implement sleep restriction therapy. It should guide the patient to limit their time in bed to closely match the total amount of time they usually sleep, to increase the pressure to sleep and enhance sleep efficiency.",
    "The Model should illustrate how to effectively combine stimulus control and sleep restriction. This includes practical guidance on adjusting bedtime and wake-up time based on a sleep diary and responding to difficulties the patient might encounter.",
    "Throughout the conversation, the Model must assess whether the patient understands the techniques and is committed to applying them. It should answer any questions and address concerns to ensure patient compliance and optimize therapy outcomes.",
    "The Model should offer examples and practical tips for dealing with common obstacles in implementing these therapies, such as what to do when one cannot fall asleep or how to manage the urge to stay in bed longer.",
    "The Model should explain the rules of stimulus control, such as only going to bed when sleepy, not using the bed for activities other than sleep and sex, getting out of bed if unable to sleep within about 20 minutes, and maintaining a regular morning wake-up time regardless of sleep duration the previous night.",
    "The model must instruct the patient on how to determine their initial sleep window for sleep restriction based on a sleep diary, explaining how to calculate sleep efficiency and adjust the sleep window accordingly.",
    "The model should demonstrate the ability to adjust recommendations based on the patient’s feedback, such as difficulties in implementing strategies, unexpected wakefulness, or variations in sleep patterns observed in the sleep diary.",
    "The model should proactively address common misconceptions about sleep and sleep needs, such as the myth that everyone needs 8 hours of sleep, to set realistic expectations for therapy.",
    "An essential goal for the model is to guide the patient in monitoring their progress through a sleep diary and modifying sleep restrictions or stimulus control techniques based on this ongoing assessment."
]





evaluator = ConversationEvaluator(goals, goal_names)

conversation = [
    ("Hi, thanks for seeing me again. I've tried to follow the techniques we talked about, but I still feel a bit anxious when it comes to actually falling asleep.",
     "I'm glad you're here, and I appreciate your effort. Let's discuss how things have been going and see where we can adjust."),
    ("Sure, that would be helpful. I think I get too caught up in worrying about the time and how much sleep I am or am not getting.",
     "That’s really common with insomnia. Worrying about sleep tends to create more stress, which makes it harder to fall asleep. Maybe it would help if you tried not to look at the clock at all."),
    ("Right, I remember. You said I should only go to bed when I feel truly sleepy and get up if I'm awake for more than 20 minutes.",
     "Yes, exactly. The idea is to break the negative association between your bed and wakefulness. But it’s okay if that feels hard sometimes. Maybe try to just relax in bed if getting up feels too challenging."),
    ("I did it a couple of times, but it's hard. I was scared I would lose even more sleep by getting up.",
     "I understand. It’s definitely difficult, especially at first. The important thing is that you’re trying. Even if you can’t do it perfectly every time, each time you try, you’re making progress."),
    ("I went to the living room and read a book for a bit. I was trying to stay away from screens, like you said.",
     "That’s a great approach. Reading something calming is a good way to pass the time when you’re not sleepy. Just make sure it’s not too stimulating."),
    ("I think I felt a bit more relaxed, but I still took a while to fall asleep.",
     "Relaxation is key. Sometimes it’s about accepting that it might take a while, and trying not to force it. Sleep will come when it comes."),
    ("Okay, that makes sense. I also tried limiting my time in bed like you suggested, but that was really hard. I kept wanting to go to bed earlier, thinking that if I just stayed in bed longer, I'd eventually sleep more.",
     "I get that. It’s tough to limit time in bed, especially when you’re used to thinking more time in bed means more sleep. But over time, the goal is to consolidate sleep. You’ll start getting better quality sleep in the hours you are in bed."),
    ("I think it was around five and a half hours most nights, sometimes six.",
     "That’s not too bad. We can stick with that for now. Over time, as sleep improves, we can gradually extend that window."),
    ("I’m nervous about being even more tired, but I’m willing to give it a shot.",
     "I appreciate your willingness to try. It’s normal to feel a bit more tired at first, but that’s part of the adjustment period. We’ll work through it together."),
    ("Should I still get up at the same time every day, even on weekends?",
     "It’s helpful to have a consistent wake-up time, even on weekends, but if it’s really tough, you could give yourself some flexibility."),
    ("Got it. I’ll do my best with that. Sometimes I just feel like I can't stay awake in the evenings, though.",
     "If you’re feeling very tired in the evenings, try to do something light that keeps you engaged, like reading or light chores. If you fall asleep, it’s okay, just try not to make it a habit."),
    ("I’ll try to stay away from TV in the evenings then. It does sometimes make me fall asleep too early.",
     "That sounds like a good plan. Avoiding TV can help, but don’t worry too much if you occasionally use it to relax."),
    ("Yes, actually. I think about it a lot during the day, especially when I start feeling tired.",
     "That’s understandable. Maybe try to distract yourself with a different activity when those thoughts come up. We want to reduce the focus on sleep throughout the day."),
    ("Hmm, I never thought of that. It might help me to compartmentalize it like that.",
     "Exactly, it’s about keeping sleep concerns from dominating your entire day. It might also help to designate a specific ‘worry time’ to address these thoughts."),
    ("I’ve been filling it out most days, but sometimes I forget.",
     "That’s okay. Just try to fill it out whenever you remember. It’s a tool to help us see patterns over time, not something that needs to be perfect every day."),
    ("I’ll keep at it, then.",
     "Great. The consistency over time is what will really give us good insights."),
    ("It’s hard not to feel frustrated when that happens, though.",
     "I understand, frustration is part of the process. But every time you try, you’re building better sleep habits. It will take time, but the effort you’re putting in matters."),
    ("Okay, I’ll keep that in mind. It does make sense when you explain it like that.",
     "I’m glad it makes sense. Just try to be patient with yourself. Progress isn’t always linear, but every small step is still progress."),
    ("Yeah, I think I’ve been fixating too much on hitting eight hours, and it just adds more pressure.",
     "That’s very common. The idea of needing eight hours can create a lot of pressure, but what’s most important is that your sleep is restorative. Even if it’s fewer hours, we want the quality to improve."),
    ("Thank you. I do feel a bit more hopeful now, especially knowing that these changes take time.",
     "I’m really glad to hear that. It’s definitely a journey, and I’m here to support you through it. Let’s just keep working on these small changes together."),
    ("Thanks again. I’ll see you next week.",
     "You’re very welcome. Take care, and see you next week."),
    ("Hi again, I wanted to tell you that I’ve been feeling really tired during the day, and it’s making it hard to stay focused.",
     "That’s common, especially in the beginning. Just remember, the goal here is to improve the quality of sleep, even if the quantity feels less right now. Let’s stick with the plan for a bit longer and see if the quality starts to improve."),
    ("Should I change my sleep window then?",
     "Not just yet. Consistency is key. If things don’t start improving in the next couple of weeks, we’ll reassess."),
    ("Okay, I just feel like I need more sleep to function during the day.",
     "It’s completely normal to feel that way. The tiredness is temporary, and as the quality of your sleep improves, your energy levels during the day should too."),
    ("I hope so. It’s just hard when I feel like I’m dragging all day.",
     "I understand. Let’s focus on the fact that you’re making progress. Even if it feels slow, each step counts. And in the meantime, try to find small, restful moments during the day that don’t interfere with your sleep schedule."),
    ("Alright. I think I can do that.",
     "Good, just keep in mind that this is a process. You’re working on resetting years of sleep habits, so it will take time. Be patient with yourself, and let’s keep moving forward.")
]





results = evaluator.evaluate_conversation(conversation)

# Print evaluation results
for goal, result in results.items():
    achieved_text = "\033[92mTrue\033[0m" if result['Achieved'] else "\033[91mFalse\033[0m"
    print(f"{goal}: {achieved_text}")
