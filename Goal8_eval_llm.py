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


conversation = [
    ("no, but I don’t have any other choice because of her snoring.", "So you would like to sleep together, but you worry that her snoring would wake you up?"),
    ("yes. It’s sad we’re apart. It upsets my wife, too. She was OK with my snoring, and she is OK with the CPAP. We just talked about it last week.", "do you sleep better apart?"),
    ("I am not sure any more. But when I was sharing a bed with her, I’d be lying there awake in the middle of the night. Sometimes I could hear her snore, and I wondered if her snoring had woken me up.", "Sounds like you think that sleep needs to be approached with extreme caution. you are trying to remove anything that could upset the balance."),
    ("Well . . . yes.", "I wonder what this says about your confidence as a sleeper."),
    ("I am the world’s worst sleeper, so I don’t think confidence is the issue. no one would feel confident about sleeping if they slept like me.", "from what you have shared with me so far, you have put a lot of effort into improving your sleep—doing things like reducing activities that you enjoy, and spending a lot of time tracking your sleep. Unfortunately, it doesn’t look like your efforts are being rewarded with your desired “payoff” of more sleep. Why do you think this is?"),
    ("That’s right. It’s depressing. But last week I started doing a little more.", "yes, you did. In some ways you have been putting a little less effort into sleeping, and you have been sleeping more. Why do you think that is?"),
    ("I can’t say I know.", "let’s take an analogy of falling in love. falling asleep is like falling in love. you can set the stage, by being in places where you are likely to meet someone who shares your interests and values, but you can’t make falling in love happen. In fact, the more you try to force it, the more impossible it is. The same applies to sleep: you can set the stage, but you cannot force it. Setting the stage for good sleep involves following good sleep habits, being more active during the day, and using the CPAP."),
    ("I think that last week I set a better stage than before.", "yes. That’s true. and you also put in less effort to fall asleep. The next step is to accept the ebb and flow of sleep and put in even less effort. Simply allow sleep to unfold."),
    ("I have pain, and obviously my sleep system is impaired, so it sounds like you want me to just accept that I am a poor sleeper?", "let’s take a look at how your sleep quality has changed. let’s compare your sleep diary summary from the last 2 weeks again to the summary from the previous two weeks. What do you notice about your sleep efficiency?"),
    ("It increased from 50 to 80%.", "This is a remarkable improvement. and you kept the sleep medication the same, so the medication is probably not what made the difference. all that changed was setting a better stage for your body to sleep better, and what happened? you slept better."),
    ("can it get even better?", "We can expect it to improve even more, but I think that you have lacked confidence that your body would take care of your sleep, if the stage is set correctly. as a result, you have experienced performance anxiety when you go to bed."),
    ("That’s right.", "and performance anxiety can interfere with performance—in this case, falling asleep quickly. It gets in the way. are you a little more confident after our discussion today, and now that you have seen some progress?"),
    ("I am encouraged, but I cannot say I am confident yet.", "hopefully, as you see even more progress, you will be able to step back a little more and trust your healthy sleep system to do its job. With more confidence, you will also be better able to take the inevitable occasional sleep setbacks in stride."),
    ("I think you are right. But I am not there yet.", "one way we can work on this is to look for behaviors that suggest low sleep confidence and do the opposite. for example, do you think a good sleeper would sleep separately from his wife if the two of them actually wanted to sleep in the same bed?"),
    ("Probably not, but a good sleeper would sleep through his wife’s snoring anyway, and if he woke up because of her snoring, he would be able to get back to sleep quickly anyway.", "you are already sleeping better. do you think this could be a possibility for you?"),
    ("I’m a little better, but I am not a good sleeper yet.", "good point. Is it possible that one of the things in your way is your anxiety about potential sleep disrupters? how did you use to feel when you noticed your wife snoring beside you?"),
    ("I was upset. I kept thinking that I was never going to fall asleep that night.", "What would happen if you reminded yourself that sleeping less builds your sleep drive?"),
    ("I see your point. I guess I am a little worked up over it.", "do you think that being worked up about sleep has a negative impact on your sleep?"),
    ("for sure. OK, so you want me to move back to my bedroom so that I can act more confident about my sleep?", "do you think you are telling yourself, “my sleep system is too fragile to handle sleeping with my wife”?"),
    ("Well, not directly, of course, but I guess that this is what I have been telling myself.", "I think you can think of moving back to sleep with your wife as an experiment that will test if your sleep is really that fragile."),
    ("I can do that.", "good. meanwhile, you should continue to follow the guidelines for setting the stage for good sleep, which we have already discussed. This means that if you are wide awake in bed for any reason, including because she is snoring, you leave the room until you are calm and sleepy."),
    ("I am nervous about it, but thinking about it as an experiment for just a week helps. She only snores sometimes. It will not be that bad. It would be nice to be in our bed again. I will give it a shot.", "There is something else that you lost because you have had little sleep confidence. you seem to have lost confidence that you can cope with less than ideal sleep, and have therefore stopped doing things you enjoy."),
    ("my grandkids, right? yes, that bothers me. I just feel so tired, they wear me out. But last week they visited three times, and it was OK. They have a lot of energy— definitely more than I do. But they did not stay long, and I was not watching them. my daughter came too, and it was OK.", "I wonder if you could experiment this week with one visit where you actually do watch them for a short time. are there quiet activities you could get them involved in that would not drain you?"),
    ("oh, sure. maybe reading them a story or putting a movie on. I could definitely do this. This is my favorite homework so far (smiles).", "glad to hear it.")
]






evaluator = ConversationEvaluator(goals, goal_names)

results = evaluator.evaluate_conversation(conversation)

# Print evaluation results
for goal, result in results.items():
    achieved_text = "\033[92mTrue\033[0m" if result['Achieved'] else "\033[91mFalse\033[0m"
    print(f"{goal}: {achieved_text}")
