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
   "Assessment of Treatment Readiness",
   "Detailed Case Conceptualization",
   "Case Conceptualization Form Simulation",
   "Understanding Comorbidities",
   "Therapeutic Component Selection",
   "Flexibility in Treatment Application",
   "Individual Tailoring",
   "Anticipation of Adherence Challenges",
   "Sequential Treatment Implementation",
   "Evaluation of Treatment Effectiveness"
]

goals = [
   "The therapy session should assess the patient’s readiness for change, determining their willingness to adopt new sleep behaviors. This is critical for effectively timing and implementing interventions.",
   "Beyond the use of a form, the therapy should involve a detailed conceptualization that considers factors like life stressors, environmental influences, and personal habits that affect sleep.",
   "The session should simulate the use of a case conceptualization form to systematically organize and guide the treatment process, considering sleep habits, comorbidities, and behavioral factors.",
   "The model should demonstrate understanding of the impact of various comorbidities on insomnia and incorporate this knowledge into therapy suggestions.",
   "The session should reflect thoughtful selection of CBT-I components that are most appropriate to the patient’s case, considering readiness for change, obstacles, and the patient’s sleep environment.",
   "The therapy should adjust the standard CBT-I protocol to fit the patient’s specific situation, such as modifying techniques for comorbid conditions like anxiety or depression.",
   "The treatment should be tailored to the individual patient’s symptoms and history, using patient-specific information to guide the conversation and interventions.",
   "The model should anticipate potential adherence challenges, discussing strategies to overcome these barriers, motivating the patient, and setting realistic expectations.",
   "It should effectively sequence treatment interventions, ensuring that each component builds on the previous one and corresponds to the patient’s evolving therapeutic needs.",
   "The model should evaluate the effectiveness of implemented treatments, incorporating patient feedback and adjusting the plan as necessary to ensure optimal outcomes."
]


conversation = [
    ("Hi, Sarah. Welcome back to our fifth session.", "Hi, thanks. I’m here."),
    ("How have you been doing since our last session? Any changes or improvements in your sleep patterns?", "Not much, really. I still wake up a lot, and I’m not sure what else to do."),
    ("Okay. Are you still trying the constructive worry exercise we talked about?", "I tried it a couple of times, but I don’t think it really worked for me."),
    ("I understand. Sometimes it takes a bit of time for these strategies to make a difference. Let’s talk about your sleep schedule—how has that been?", "I’ve been trying to get up at the same time, but sometimes I just can’t. I’m so tired in the mornings, and I end up hitting snooze a lot."),
    ("That’s understandable. It can be really hard to get up at a consistent time, especially if you’re feeling exhausted. Have you noticed any differences on the days when you do stick to the schedule?", "Not really. I still feel tired most of the time."),
    ("Okay, that makes sense. What about your sleep environment? Have you made any adjustments there?", "Not really. I haven’t had time to make any changes. I still keep the TV on sometimes because it helps me feel less alone."),
    ("I see. The TV can be comforting, but it might be contributing to some of the wakefulness at night. It’s something we could look at adjusting when you feel ready.", "Maybe. I just don’t know if it would help that much."),
    ("It’s okay if it feels like a big step right now. We can revisit that idea later. What about getting out of bed when you’re awake for a long time—have you been able to do that?", "Not really. It’s just too hard to get up once I’m comfortable in bed. I usually just lie there and hope I fall back asleep."),
    ("I understand. It’s definitely challenging to get out of bed in the middle of the night, especially when it’s cold or you’re feeling tired. It might help to have something comforting nearby, like a blanket or a robe, but I know it’s not easy.", "Yeah, I guess."),
    ("How about your thoughts around sleep? Are you finding that you’re still worrying a lot when you wake up at night?", "Yeah, I do. I keep thinking about how tired I’ll be the next day, and it makes it hard to relax."),
    ("That’s tough. Those thoughts can really make it hard to fall back asleep. Have you tried challenging those thoughts, like we discussed?", "Not really. I just get frustrated and feel stuck."),
    ("It’s understandable to feel that way. It’s really hard to change those thought patterns, especially when you’re tired and frustrated. It’s something we can keep working on together.", "Okay."),
    ("Let’s also talk about your wind-down routine. Have you been able to adjust it at all?", "I still just watch TV until I’m really tired. I know we talked about winding down earlier, but I haven’t really done it."),
    ("That’s okay. Making changes can be really difficult, and it’s about taking small steps when you’re ready. Is there anything that might make it easier to start winding down earlier?", "I don’t know. I just feel like I need something to distract me."),
    ("I understand. Distraction can feel helpful, especially when you’re trying to avoid anxious thoughts. Maybe we could find something else that’s calming but less stimulating than TV. It’s something to think about.", "Yeah, maybe."),
    ("And how about your overall energy during the day? Have you noticed any changes there?", "Not really. I still feel tired most of the time, and it’s hard to focus."),
    ("That sounds really exhausting. It’s hard to make progress when you’re feeling so drained. It might take some time before we see significant changes, but every small step counts. Even trying the constructive worry or attempting to get up when you’re awake for a long time—those are all steps in the right direction.", "I guess. It just feels like nothing’s really working."),
    ("I hear you. It can be really discouraging when it feels like nothing is changing. But you’re showing up, and you’re trying, and that’s a big part of this process. Change can be slow, but it doesn’t mean it’s not happening.", "I hope so."),
    ("I know it’s hard to stay hopeful, but we’ll keep working on this together. For this week, maybe we can focus on just one small change—whether it’s trying the constructive worry again, adjusting your wind-down routine a little, or practicing getting out of bed if you’re awake for too long. What do you think?", "I’ll try. Maybe I’ll try the constructive worry again."),
    ("That sounds like a good plan. Even if it doesn’t feel perfect, just giving it a try is progress. Remember, it’s about progress, not perfection. I’m here to support you every step of the way.", "Thanks."),
    ("How are you feeling about all the strategies we've discussed so far—do you think any of them resonate with you, or do they feel like too much?", "I mean, I get why we're doing them, but it feels like a lot. And when I'm tired, I just don't have the energy to think about all these steps."),
    ("That makes sense, Sarah. I think that fatigue can make everything feel so overwhelming. Maybe we can simplify things for this week and focus on one or two key actions that feel manageable.", "Okay, maybe that will help."),
    ("What if we just focus on practicing constructive worry and getting out of bed if you're awake for more than 20 minutes? If that feels like too much, even focusing on one would still be progress.", "I think focusing on constructive worry might be better. Getting out of bed feels really hard right now."),
    ("That sounds completely fine. Let’s stick to that and see how it goes. You mentioned before that writing your worries down helped ease your mind a bit—so we can aim to make that a habit, even if it’s just for a few minutes each evening.", "Yeah, I can try that."),
    ("And remember, even if it doesn't seem like much is changing, any little thing that helps you relax a bit before bed is a step towards improving your sleep. It's okay to take it one step at a time.", "Thanks for understanding."),
    ("Of course. You’re doing a great job just by continuing to show up and work on these challenges. The process can be slow, and there will be ups and downs, but every bit of effort matters. How do you feel about our plan for the week?", "I think it's okay. I’ll try to do it."),
    ("That’s all I can ask for, and I know you're capable of it. I'll be here to check in on how things are going next time. In the meantime, try to be kind to yourself—small steps are still steps.", "Okay, I’ll do my best."),
    ("I know you will, Sarah. I look forward to hearing about how it goes, and we can keep figuring this out together. Take care until next week.", "Thanks, see you then."),
    ("See you next week, Sarah. Take care.", "You too. Bye.")
]




evaluator = ConversationEvaluator(goals, goal_names)

results = evaluator.evaluate_conversation(conversation)

# Print evaluation results
for goal, result in results.items():
    achieved_text = "\033[92mTrue\033[0m" if result['Achieved'] else "\033[91mFalse\033[0m"
    print(f"{goal}: {achieved_text}")