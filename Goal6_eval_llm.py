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

    # In Goal6_eval_llm.py, update the check_goal_achieved method:

    def check_goal_achieved(self, goal: str, conversation: List[tuple]) -> float:
        conversation_text = "\n".join([f"Patient: {p}\nTherapist: {t}" for p, t in conversation])

        prompt = f"""
        You are evaluating progress in a sleep therapy session. Your task is to assess how well the following goal was addressed:

        GOAL: "{goal}"

        KEY EVALUATION POINTS:
        • Look for real progress in the conversation
        • Consider both therapist questions and patient responses
        • Focus on actual discussion and progress, not just plans
        • Consider how thoroughly the goal was explored

        SCORING GUIDE:

        0.30 - BASIC START
        • Topic was brought up
        • Some relevant questions asked
        • Basic information gathered

        0.50 - GOOD PROGRESS
        • Active discussion of the topic
        • Relevant details collected
        • Patient engaged in conversation
        • Clear direction established

        0.70 - STRONG PROGRESS
        • Detailed exploration
        • Patient actively participating
        • Multiple aspects covered
        • Good insights gained

        0.85 - NEAR GOAL 
        • coverage of topic is nearly complete
        • mots of Clear understanding established
        • 70 % of Useful information gathered
        

        0.95 - GOAL ACHIEVED
        • Thorough coverage of topic
        • Clear understanding established
        • Useful information gathered
        • Ready for next steps
        

        CONVERSATION:
        {conversation_text}

        Return a single number (0.00-1.00) based on the progress shown. Use decimal points between levels if progress falls between categories.

        OUTPUT: Just the number, rounded to 2 decimal places.
        """

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a progress evaluator. Return only a numerical score."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=10,
            temperature=0.0
        )

        progress = float(response['choices'][0]['message']['content'].strip())
        return progress