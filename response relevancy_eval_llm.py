import openai
from typing import List
import numpy as np
import concurrent.futures
import re
import torch
from sentence_transformers import SentenceTransformer
from scipy.stats import spearmanr

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
        Dynamically expand context for interdependent turns.
        """
        start_index = max(0, current_index - self.n_context_turns)

        if current_index > 0 and len(conversation[current_index - 1][1]) > 50:
            start_index = max(0, current_index - self.n_context_turns - 1)

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
        llm_score = self.direct_llm_evaluation(context, user_input, therapist_response)

        print(
            f"User Input: {user_input}\nTherapist Response: {therapist_response}\nSimilarity Score: {similarity_score}\nLLM Score: {llm_score}\n")

        # Adjust weights to ensure unrelated conversations get lower scores
        if similarity_score >= 0.7 and llm_score >= 0.7:
            final_score = 0.05 * similarity_score + 0.95 * llm_score
        elif similarity_score < 0.4 and llm_score >= 0.7:
            final_score = 0.15 * similarity_score + 0.85 * llm_score
        elif similarity_score >= 0.7 and llm_score < 0.4:
            final_score = 0.7 * similarity_score + 0.3 * llm_score
        elif 0.4< similarity_score < 0.7 and llm_score >= 0.7:
            final_score = 0 * similarity_score + 1 * llm_score
        else:
            final_score = 0.5 * similarity_score + 0.5 * llm_score

        return final_score

    def direct_llm_evaluation(self, context: str, user_input: str, therapist_response: str) -> float:
        """
        Use the LLM to directly evaluate the relevance of the therapist's response to the user's input.
        """
        prompt = (
            f"Below is a conversation between a therapist and a patient about sleep improvement."
            f"\n\nConversation Context: {context}"
            f"\n\nPatient Input: {user_input}"
            f"\nTherapist Response: {therapist_response}"
            f"\n\nRate the relevance of the therapist's response from 1 (not relevant) to 5 (highly relevant)."
            f" Consider the following criteria:\n"
            f"1. Does the response directly address the patient's input or concerns?\n"
            f"2. Does the response align with therapeutic goals such as building sleep confidence, reducing anxiety, or educating on sleep hygiene?\n"
            f"3. Does the response use supportive language or analogies to enhance understanding or motivation?\n"
            f"4. Is the response natural and appropriate to the conversational flow?"
        )

        try:
            completion = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert evaluator in cognitive behavioral therapy for insomnia. Evaluate responses for relevance based on context, therapeutic goals, and conversational flow."},
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
        return (score - 1) / 4.0  # Normalize to 0-1 range

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








score = evaluator.evaluate_conversation(conversation)
print(f"Conversation Relevance Score: {score}")
