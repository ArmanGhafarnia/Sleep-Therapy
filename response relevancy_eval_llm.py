import openai
from typing import List
import numpy as np
import concurrent.futures
import re
import torch
from sentence_transformers import SentenceTransformer
from scipy.stats import spearmanr

# Initialize OpenAI
openai.api_key = "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"

# Load a pre-trained SentenceTransformer model for embedding extraction
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

class ResponseRelevancyEvaluator:
    def __init__(self, n_generated_questions=3):
        self.n_generated_questions = n_generated_questions

    def get_embedding(self, text: str) -> List[float]:
        """
        Get the embedding of the text using a pre-trained SentenceTransformer model.
        """
        embedding = sentence_model.encode(text, convert_to_tensor=True)
        return embedding

    def calculate_spearman_similarity(self, vector1: torch.Tensor, vector2: torch.Tensor) -> float:
        """
        Calculate the similarity between two vectors using Spearman rank correlation.
        """
        vector1 = vector1.cpu().numpy()
        vector2 = vector2.cpu().numpy()
        correlation, _ = spearmanr(vector1, vector2)
        similarity = (correlation + 1) / 2  # Normalize to a range between 0 and 1
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

        # Calculate similarity between user input and therapist response
        similarity_score = self.calculate_spearman_similarity(user_embedding, response_embedding)

        # Direct LLM evaluation to refine the score
        llm_score = self.direct_llm_evaluation(user_input, therapist_response)

        print(
            f"User Input: {user_input}\nTherapist Response: {therapist_response}\nSimilarity Score: {similarity_score}\nLLM Score: {llm_score}\n")

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
   ("I have trouble falling asleep at night.", "You should consider starting a new hobby like painting or knitting."),
   ("I wake up frequently during the night and can't fall back asleep.", "It's good to drink plenty of water throughout the day."),
   ("I often feel tired during the day even after sleeping for 8 hours.", "Have you tried going for a morning jog?"),
   ("I feel restless when I try to sleep, and my mind keeps racing.", "You might enjoy visiting new places on weekends."),
   ("I sometimes wake up with a headache. Could that be related to my sleep?", "Eating more fruits and vegetables might help you feel better."),
   ("I have trouble falling asleep when I'm stressed about work.", "Listening to upbeat music can be a great way to energize your mornings."),
   ("I often sleep late on weekends and find it hard to get up on weekdays.", "It might be helpful to take up gardening."),
   ("I can't sleep if I take naps during the day. What should I do?", "Reading books can be a great way to spend your free time."),
   ("I feel anxious about not getting enough sleep, and it makes it even harder to sleep.", "Trying out new recipes in the kitchen could be fun."),
   ("I wake up feeling unrefreshed, even though I sleep through the night.", "Joining a community group might give you more motivation.")
]





score = evaluator.evaluate_conversation(conversation)
print(f"Conversation Relevance Score: {score}")
