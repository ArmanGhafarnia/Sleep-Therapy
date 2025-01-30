from typing import List, Dict
import openai
import time


class ConversationEvaluator:
    def __init__(self, goals: List[str], goal_names: List[str], model="gpt-4o"):
        self.goals = goals
        self.goal_names = goal_names
        self.model = model
        self.MAX_RETRIES = 3
        self.BASE_DELAY = 0.1
        openai.api_key = "your-api-key-here"

    def _make_api_call_with_retry(self, messages, max_tokens=10, temperature=0.0):
        for attempt in range(self.MAX_RETRIES):
            try:
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                return response
            except openai.error.RateLimitError as e:
                if attempt < self.MAX_RETRIES - 1:
                    wait_time = self.BASE_DELAY
                    if hasattr(e, 'headers') and 'retry-after' in e.headers:
                        wait_time = float(e.headers['retry-after'])
                    elif 'Please try again in' in str(e):
                        try:
                            wait_time = float(str(e).split('in ')[1].split('ms')[0]) / 1000
                        except:
                            pass
                    time.sleep(wait_time + 0.05)
                    continue
                raise
            except Exception as e:
                print(f"Error making API call: {str(e)}")
                raise

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
        • Coverage of topic is nearly complete
        • 70% of Useful information gathered

        0.95 - GOAL ACHIEVED
        • Thorough coverage of topic
        • Most of Clear understanding established
        • Useful information gathered
        • Ready for next steps

        CONVERSATION:
        {conversation_text}

        Return a single number (0.00-1.00) based on the progress shown.
        OUTPUT: Just the number, rounded to 2 decimal places.
        """

        messages = [
            {"role": "system", "content": "You are a progress evaluator. Return only a numerical score."},
            {"role": "user", "content": prompt}
        ]

        response = self._make_api_call_with_retry(messages)
        progress = float(response['choices'][0]['message']['content'].strip())
        return progress

    def evaluate_conversation(self, conversation: List[tuple]) -> Dict[str, Dict[str, object]]:
        evaluation_results = {}

        for goal, goal_name in zip(self.goals, self.goal_names):
            try:
                achievement_score = self.check_goal_achieved(goal, conversation)
                evaluation_results[goal_name] = {
                    "Description": goal,
                    "Achieved": achievement_score
                }
                time.sleep(0.1)
            except Exception as e:
                print(f"Error evaluating goal {goal_name}: {str(e)}")
                evaluation_results[goal_name] = {
                    "Description": goal,
                    "Achieved": 0.0
                }

        return evaluation_results