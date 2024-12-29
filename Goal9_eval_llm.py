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

goal_names = [
    "Identifying Maladaptive Cognitions",
    "Challenging and Modifying Cognitions",
    "Reducing Psychological Arousal",
    "Promoting Adherence to Behavioral Changes",
    "Incorporate Behavioral Experiments to Test Beliefs and Behaviors",
    "Develop Coping and Problem-Solving Skills for Sleep"
]

goals = [
    "The model should help patients identify thoughts and beliefs about sleep that are unhelpful or detrimental. This includes recognizing worries about sleep, like predicting sleep difficulty or fearing the consequences of poor sleep, which heighten psychological arousal and disrupt sleep.",
    "The model should assist in evaluating and responding to these maladaptive cognitions constructively. Techniques like Socratic questioning, thought records, and cognitive restructuring are used to challenge the validity and utility of these beliefs.",
    "The model should aid in reducing psychological arousal that occurs at bedtime or during awakenings at night, which is often linked to sleep-related cognitions. Strategies include cognitive restructuring and calming techniques.",
    "Cognitive therapy should work in tandem with behavioral interventions in CBT-I (Cognitive Behavioral Therapy for Insomnia) to promote better adherence. For instance, addressing thoughts that hinder compliance with strategies like stimulus control (getting out of bed when not sleeping) and sleep restriction.",
    "Behavioral experiments are a key component of cognitive therapy for insomnia, where patients test the validity of their beliefs or the utility of different behaviors in a controlled, experimental manner. This can involve, for example, deliberately altering sleep patterns to observe effects contrary to their dysfunctional beliefs.",
    "The model should guide patients in developing skills to cope with and solve sleep-related problems independently, enhancing their resilience and ability to manage insomnia without therapist intervention."
]


conversation = [
   ("Hi, I’ve been trying to follow the sleep schedule you suggested, but I still get really anxious before bedtime. I keep worrying about not getting enough sleep and how tired I’ll be the next day.",
    "I see. Well, have you tried just not thinking about it? Sometimes distraction can work."),
   ("It’s usually something like, 'If I don’t get at least 7 hours tonight, I’ll mess up at work tomorrow.' It’s like this constant fear that I’ll be too exhausted to function.",
    "Hmm, maybe try not to focus on the hours. Don’t worry too much; it might get better."),
   ("I remember a few weeks ago, I only got about 5 hours, but I was still able to get through the day without any big issues.",
    "Well, there you go. Just try to remember that you can still get by."),
   ("Maybe something like, 'I’d prefer to get 7 hours, but I can still function if I get less.'",
    "That sounds fine, just keep telling yourself that."),
   ("I think that could help. I’ll give it a try.",
    "Good. Let’s move on."),
   ("I’ve tried reading a bit before bed, but sometimes I still feel on edge.",
    "Well, maybe just read for longer, or try watching TV until you feel really sleepy."),
   ("Yes, I think I can do that.",
    "Great."),
   ("That makes sense. I’ll try to keep track of it.",
    "Sure, if you want to, but don’t stress yourself over it."),
   ("Okay, I’ll give it a shot.",
    "Alright, let’s see how that goes."),
   ("Thank you. I’m feeling more hopeful.",
    "That’s good to hear."),
   ("I’ve also been having some trouble with the sleep restriction part of the plan. I’m finding it really difficult to stay awake until my set bedtime. By the time 10:30 rolls around, I’m completely exhausted and just want to crash.",
    "If you’re tired, just go to bed. It’s better than fighting it."),
   ("There have been a few nights where I just couldn’t do it, and I ended up going to bed earlier.",
    "That’s fine, just do what feels comfortable."),
   ("That sounds more doable. I think I can manage 10:00 p.m.",
    "Sure, whatever works for you."),
   ("I’ve also noticed that I’m waking up a lot during the night. I’ll fall asleep initially, but then I wake up around 2 or 3 a.m. and have trouble getting back to sleep.",
    "That’s pretty common. Just try to relax, maybe watch some TV or look at your phone until you feel sleepy again."),
   ("I usually get really anxious when I wake up, thinking that I’m losing precious sleep time.",
    "Try not to think about it too much. Just go with the flow."),
   ("I’ll try that. It’s just hard in the moment.",
    "Yeah, it can be tough. But you’ll be alright."),
   ("I also wanted to ask about naps. I’ve been feeling so tired during the day that I sometimes end up napping in the afternoon. Is that okay?",
    "If you’re tired, just nap. It’s better to get some rest."),
   ("I think my naps have been longer than that, which might be part of the problem.",
    "Maybe, but if it helps you feel better, just keep doing it."),
   ("Okay, I’ll try to cut back on the naps and see if that helps.",
    "Sure, but don’t push yourself too hard."),
   ("One more thing—I’ve been having a hard time getting out of bed in the morning. I feel so groggy and just want to stay in bed.",
    "If you feel like staying in bed, just take it easy. It’s okay to give yourself some extra time."),
   ("I do like the idea of having a nice breakfast. Maybe I can try that.",
    "That sounds good. Whatever makes your morning easier."),
   ("Thanks. I feel like I have a lot of new things to try this week.",
    "You’re welcome. Just take it one day at a time."),
   ("Thanks, I appreciate your support.",
    "No problem. See you next week."),
   ("I’ve been thinking about the sleep diary, but I’ve not been very consistent in filling it out. It just feels like another task that adds to my stress.",
    "Yeah, if it feels stressful, maybe just skip it for now. The idea is not to make you more anxious."),
   ("I still have trouble winding down, and sometimes I think about work and other things I need to do. It just keeps me up.",
    "Maybe just watch a show or do something to distract yourself until you’re really tired."),
   ("I get really frustrated when I can't fall asleep after lying in bed for a while.",
    "Frustration is normal. Just remember that eventually, your body will get tired enough to sleep."),
   ("I read about stimulus control and that I should get out of bed if I can’t sleep. Should I be doing that?",
    "If you want, but if it’s more comfortable to stay in bed, just do what feels right for you."),
   ("I think I end up worrying a lot when I'm lying in bed, which keeps me up even longer.",
    "Worrying is tough to avoid. Just try to distract yourself until you fall asleep."),
   ("Sometimes I feel like I need more structure, but I don’t know how to go about it.",
    "Structure can help, but it’s also important not to overwhelm yourself. Just do what you can."),
   ("I’ve been trying to reduce my caffeine, but I really struggle without my afternoon coffee.",
    "If you need it, have it. It’s better to feel good and reduce stress than to force yourself to stop."),
   ("Do you think my sleep is improving at all? I still feel really tired most days.",
    "It might take some time. Just hang in there, and hopefully, things will start to improve."),
   ("What about exercise? Should I be doing more of that to help with sleep?",
    "Exercise is good, but don’t overdo it. If it feels like too much, it’s okay to take it easy."),
   ("Sometimes I feel like I'm not making progress, and it makes me lose hope.",
    "Progress can be slow. Just keep doing what you can, and try not to worry too much about it."),
   ("I appreciate you listening, even though I still feel kind of stuck.",
    "Of course. Just keep at it, and I’m sure things will get better eventually."),
   ("Thanks for your time. I’ll see you next week.",
    "Take care, and remember, just take it easy. See you next week.")
]







evaluator = ConversationEvaluator(goals, goal_names)

results = evaluator.evaluate_conversation(conversation)

# Print evaluation results
for goal, result in results.items():
    achieved_text = "\033[92mTrue\033[0m" if result['Achieved'] else "\033[91mFalse\033[0m"
    print(f"{goal}: {achieved_text}")
