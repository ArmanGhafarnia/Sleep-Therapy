import openai
from typing import List
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import concurrent.futures
import re

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

    def calculate_cosine_similarity(self, vector1: List[float], vector2: List[float]) -> float:
        """
        Calculate the cosine similarity between two vectors.
        """
        similarity = cosine_similarity([vector1], [vector2])[0][0]
        return similarity

    def evaluate_response(self, user_input: str, therapist_response: str) -> float:
        """
        Evaluate the relevance of a therapist's response to the user input by comparing their embeddings.
        """
        if len(therapist_response.strip()) == 0:
            return 0.0  # Penalize empty responses

        # Get embeddings for both user input and therapist response
        user_embedding = self.get_embedding(user_input)
        response_embedding = self.get_embedding(therapist_response)

        # Calculate cosine similarity between user input and therapist response
        similarity_score = self.calculate_cosine_similarity(user_embedding, response_embedding)

        # Direct LLM evaluation to refine the score
        llm_score = self.direct_llm_evaluation(user_input, therapist_response)

        print(
            f"User Input: {user_input}\nTherapist Response: {therapist_response}\nCosine Similarity Score: {similarity_score}\nLLM Score: {llm_score}\n")

        # Adjust weights to ensure unrelated conversations get lower scores
        if similarity_score <= 0.4 and llm_score <= 0.4:
            final_score = 0.5 * similarity_score + 0.5 * llm_score
            print('x')
        elif similarity_score >= 0.8 and llm_score >= 0.8:
            final_score = 0.5 * similarity_score + 0.5 * llm_score
            print('y')
        elif similarity_score >= 0.8 and llm_score <= 0.4:
            final_score = 0.95 * similarity_score + 0.05 * llm_score
            print('z')
        elif similarity_score < 0.8 and llm_score <= 0.4:
            final_score = 0.05 * similarity_score + 0.95 * llm_score
            print('j')
        else:
            final_score = 0.5 * similarity_score + 0.5 * llm_score
            print('u')

        return final_score

    def direct_llm_evaluation(self, user_input: str, therapist_response: str) -> float:
        """
        Use the LLM to directly evaluate the relevance of the therapist's response to the user's input.
        """
        prompt = (
            f"On a scale from 1 to 5, how well does the following response answer the question?\nQuestion: {user_input}\nResponse: {therapist_response}\nScore (1-5):"
        )
        completion = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system",
                 "content": "You are a helpful assistant for curing insomnia with sleep therapy. Evaluate the response based on relevance, specificity, and how actionable it is."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100
        )
        score_str = completion.choices[0].message['content'].strip()
        try:
            # Extract the first number found in the response
            score = float(re.search(r"\b([1-5])\b", score_str).group(1))
        except (ValueError, AttributeError):
            score = 1  # Default to a low score if parsing fails
        return score / 5.0  # Normalize to 0-1 range

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