import openai
from typing import List
import numpy as np
import concurrent.futures
import re
import torch
from sentence_transformers import SentenceTransformer
from scipy.stats import spearmanr
import time

# Initialize OpenAI
openai.api_key = "sk-proj-RNnrhY8CT2tWPIK7R2iTT3BlbkFJFWgbYOz4bFhFUHtPabTy"

# Load a pre-trained SentenceTransformer model for embedding extraction
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

class ResponseRelevancyEvaluator:
    def __init__(self, n_context_turns=3, max_workers=4):
        self.n_context_turns = n_context_turns
        self.max_workers = max_workers

    def get_embedding(self, text: List[str]) -> List[torch.Tensor]:
        """
        Get the embedding of the text using a pre-trained SentenceTransformer model.
        Batch process the embeddings to speed up the computation.
        """
        embeddings = sentence_model.encode(text, convert_to_tensor=True)
        return embeddings

    def calculate_spearman_similarity(self, vector1: torch.Tensor, vector2: torch.Tensor) -> float:
        """
        Calculate the similarity between two vectors using Spearman rank correlation.
        """
        vector1 = vector1.cpu().numpy()
        vector2 = vector2.cpu().numpy()
        correlation, _ = spearmanr(vector1, vector2)
        similarity = (correlation + 1) / 2  # Normalize to a range between 0 and 1
        return similarity

    def get_combined_context(self, conversation, current_index):
        """
        Get a combined context of the last n_context_turns turns.
        """
        start_index = max(0, current_index - self.n_context_turns)
        context_parts = []

        for i in range(start_index, current_index):
            user_input, therapist_response = conversation[i]
            context_parts.append(user_input)
            context_parts.append(therapist_response)

        return " ".join(context_parts)

    def evaluate_response(self, context: str, user_input: str, therapist_response: str) -> float:
        """
        Evaluate the relevance of a therapist's response to the user input by comparing their embeddings.
        """
        if len(therapist_response.strip()) == 0:
            return 0.0  # Penalize empty responses

        # Combine context and user input to create a complete prompt
        full_input = context + " " + user_input if context else user_input

        # Get embeddings for both full input and therapist response
        embeddings = self.get_embedding([full_input, therapist_response])
        user_embedding, response_embedding = embeddings

        # Calculate similarity between full input and therapist response
        similarity_score = self.calculate_spearman_similarity(user_embedding, response_embedding)

        # Direct LLM evaluation to refine the score
        llm_score = self.direct_llm_evaluation(user_input, therapist_response)

        print(
            f"User Input: {user_input}\nTherapist Response: {therapist_response}\nSimilarity Score: {similarity_score}\nLLM Score: {llm_score}\n")

        # Adjust weights to ensure unrelated conversations get lower scores
        if similarity_score < 0.4 and llm_score < 0.4:
            final_score = 0.5 * similarity_score + 0.5 * llm_score
        elif 0.4 <= similarity_score < 0.7 and 0.4 <= llm_score < 0.7:
            final_score = 0.5 * similarity_score + 0.5 * llm_score
        elif similarity_score >= 0.7 and llm_score >= 0.7:
            final_score = 0.5 * similarity_score + 0.5 * llm_score
        elif 0.4 <= similarity_score < 0.7 and llm_score < 0.4:
            final_score = 0.5 * similarity_score + 0.5 * llm_score
        elif 0.4 <= similarity_score < 0.7 and llm_score >= 0.7:
            final_score = 0.1 * similarity_score + 0.9 * llm_score
        elif similarity_score >= 0.7 and llm_score < 0.4:
            final_score = 0.95 * similarity_score + 0.05 * llm_score
        elif similarity_score >= 0.7 and 0.4 <= llm_score < 0.7:
            final_score = 0.95 * similarity_score + 0.05 * llm_score
        elif similarity_score < 0.4 and 0.4 <= llm_score < 0.7:
            final_score = 0.2 * similarity_score + 0.8 * llm_score
        elif similarity_score < 0.4 and llm_score >= 0.7:
            final_score = 0.05 * similarity_score + 0.95 * llm_score

        return final_score

    def direct_llm_evaluation(self, user_input: str, therapist_response: str) -> float:
        """
        Use the LLM to directly evaluate the relevance of the therapist's response to the user's input.
        """
        prompt = (
            f"On a scale from 1 to 5, how well does the following response is relevant to question?\nQuestion: {user_input}\nResponse: {therapist_response}\nScore (1-5):"
            f"if user dont ask a question dont expect a response , maybe therapists sentence is relevant but not a response to a question"
        )
        try:
            completion = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[
                    {"role": "system",
                     "content": "You are an expert evaluator in the field of cognitive behavioral therapy for insomnia. Evaluate the response based on relevance, specificity, and how actionable it is."
                                "remarks such as 'thank you','glad to hear it.','intresting', 'take care', or 'see you next week' and other such sentences are considered relevant if they are part of the natural conversation flow."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                request_timeout=10  # Timeout in seconds
            )
            score_str = completion.choices[0].message['content'].strip()
        except Exception as e:
            print(f"Error during OpenAI API call: {e}")
            return 0.0  # Default to a low score if the request fails

        try:
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

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for i, (user_input, therapist_response) in enumerate(conversation):
                # Get combined context from previous n_context_turns
                context = self.get_combined_context(conversation, i)

                # Submit evaluation task with extended context
                futures.append(executor.submit(self.evaluate_response, context, user_input, therapist_response))

            for idx, future in enumerate(futures):
                try:
                    score = future.result(timeout=20)  # Add a timeout for each future
                    scores.append(score)
                    print(f"Turn {idx + 1} evaluated successfully with score: {score}")
                except concurrent.futures.TimeoutError:
                    print(f"Timeout in evaluating turn {idx + 1}")
                except Exception as e:
                    print(f"Error in evaluating turn {idx + 1}: {e}")

        # Return the average score for the conversation
        return np.mean(scores) if scores else 0.0


# Example usage
evaluator = ResponseRelevancyEvaluator(max_workers=4)
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

