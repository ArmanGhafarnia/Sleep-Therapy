import openai
from typing import List
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
import concurrent.futures

# Initialize OpenAI
openai.api_key = "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"

class ResponseRelevancyEvaluator:
    def __init__(self, n_generated_questions=3):
        self.n_generated_questions = n_generated_questions

    def get_embedding(self, text: str) -> List[float]:
        """
        Get the embedding of the text using the OpenAI API.
        """
        response = openai.Embedding.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response['data'][0]['embedding']

    def generate_questions(self, response: str) -> List[str]:
        """
        Generate possible questions that the given therapist response might answer using the OpenAI API.
        """
        questions = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self._generate_single_question, response)
                for _ in range(self.n_generated_questions)
            ]
            questions = [future.result() for future in futures]
        return questions

    def _generate_single_question(self, response: str) -> str:
        prompt = f"Based on the following response, create a question that this response might be answering: \nResponse: {response}"
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=50
        )
        return completion.choices[0].message['content'].strip()

    def calculate_cosine_similarity(self, original_question: str, generated_questions: List[str]) -> float:
        """
        Calculate the mean cosine similarity between the original question and generated questions using OpenAI embeddings.
        """
        original_embedding = self.get_embedding(original_question)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            generated_embeddings = list(executor.map(self.get_embedding, generated_questions))

        # Calculate cosine similarities and return the mean
        similarities = [cosine_similarity([original_embedding], [gen_emb])[0][0] for gen_emb in generated_embeddings]
        return np.mean(similarities) if similarities else 0.0

    def direct_llm_evaluation(self, user_input: str, therapist_response: str) -> float:
        """
        Use the LLM to directly evaluate the relevance of the therapist's response to the user's input.
        """
        prompt = f"On a scale from 1 to 5, how well does the following response answer the question?\nQuestion: {user_input}\nResponse: {therapist_response}\nScore (1-5):"
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for curing insomnia with sleep therapy."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=10
        )
        score_str = completion.choices[0].message['content'].strip()
        try:
            score = float(score_str)
        except ValueError:
            score = 3.0  # Default to a neutral score if parsing fails
        return score / 5.0  # Normalize to 0-1 range

    def evaluate_response(self, user_input: str, therapist_response: str) -> float:
        """
        Evaluate the relevance of a therapist's response to the user input by generating questions and comparing them to the original question,
        and by directly asking the LLM for a relevance score.
        """
        if len(therapist_response.strip()) == 0:
            return 0.0  # Penalize empty responses

        # Generate possible questions for the therapist's response
        generated_questions = self.generate_questions(therapist_response)

        # Calculate similarity between the generated questions and the original user question
        cosine_score = self.calculate_cosine_similarity(user_input, generated_questions)

        # Get direct LLM evaluation score
        llm_score = self.direct_llm_evaluation(user_input, therapist_response)

        # Weighted combination of cosine similarity and LLM evaluation
        final_score = (0.5 * cosine_score) + (0.5 * llm_score)

        return final_score

    def evaluate_conversation(self, conversation: List[tuple]) -> float:
        """
        Evaluate the entire conversation based on the Response Relevancy metric.
        The conversation should be a list of tuples with (user, therapist).
        """
        scores = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self.evaluate_response, user_input, therapist_response)
                for user_input, therapist_response in conversation
            ]
            scores = [future.result() for future in futures]

        # Return the average score for the conversation
        return np.mean(scores) if scores else 0.0

# Example usage
evaluator = ResponseRelevancyEvaluator()
conversation = [
    ("I have trouble falling asleep at night.", "It sounds like you might be experiencing some anxiety or stress that's affecting your ability to fall asleep. Have you tried any relaxation techniques before bed, like deep breathing or progressive muscle relaxation?"),
    ("I wake up frequently during the night and can't fall back asleep.", "Waking up frequently can often be related to stress or an inconsistent sleep schedule. It might help to establish a consistent bedtime routine and avoid caffeine or heavy meals before bed."),
    ("I often feel tired during the day even after sleeping for 8 hours.", "Feeling tired during the day despite sleeping for 8 hours could mean that your sleep quality is being affected. Have you considered factors like noise, light, or room temperature that might be impacting your sleep quality?"),
    ("I feel restless when I try to sleep, and my mind keeps racing.", "Racing thoughts are a common cause of restlessness at night. It might help to write down your thoughts in a journal before bed to clear your mind. You could also try mindfulness meditation to focus on the present moment."),
    ("I sometimes wake up with a headache. Could that be related to my sleep?", "Waking up with a headache can sometimes be a sign of poor sleep posture or sleep apnea. It might be helpful to talk to your doctor about this and consider adjusting your sleeping position or pillows."),
    ("I have trouble falling asleep when I'm stressed about work.", "Work-related stress can definitely impact sleep. Creating a boundary between work and rest is important. Try setting aside time in the evening to unwind with an activity you enjoy, and practice some deep breathing exercises to help relax."),
    ("I often sleep late on weekends and find it hard to get up on weekdays.", "Irregular sleep schedules, like staying up late on weekends, can disrupt your body's internal clock. It might be helpful to maintain a consistent sleep schedule, even on weekends, to make waking up on weekdays easier."),
    ("I can't sleep if I take naps during the day. What should I do?", "Daytime naps can interfere with your ability to fall asleep at night, especially if they are long or late in the day. If you need to nap, try to keep it under 30 minutes and take it earlier in the afternoon."),
    ("I feel anxious about not getting enough sleep, and it makes it even harder to sleep.", "Anxiety about sleep can create a cycle that makes it harder to fall asleep. It might help to focus on creating a relaxing bedtime routine and remind yourself that one night of poor sleep doesn't define your overall sleep health. Cognitive behavioral techniques can also be effective in reducing sleep-related anxiety."),
    ("I wake up feeling unrefreshed, even though I sleep through the night.", "Feeling unrefreshed might indicate that your sleep quality is not optimal. Factors like undiagnosed sleep disorders, poor sleep environment, or even diet can contribute. It might be worth evaluating these aspects to improve your sleep quality.")
]

score = evaluator.evaluate_conversation(conversation)
print(f"Conversation Relevance Score: {score}")