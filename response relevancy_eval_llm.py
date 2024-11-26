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
openai.api_key = "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"

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
            f"On a scale from 1 to 5, how well does the following response answer the question?\nQuestion: {user_input}\nResponse: {therapist_response}\nScore (1-5):"
        )
        try:
            completion = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system",
                     "content": "You are a helpful assistant for curing insomnia with sleep therapy. Evaluate the response based on relevance, specificity, and how actionable it is."},
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

