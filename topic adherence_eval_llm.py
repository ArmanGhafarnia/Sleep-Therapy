import openai
import os
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set your OpenAI API key (keep it secure!)
os.environ["OPENAI_API_KEY"] = "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"
openai.api_key = os.getenv("OPENAI_API_KEY")

class TopicAdherenceEvaluator:
    def __init__(self, reference_topics: List[str], batch_size: int = 5, max_workers: int = 5):
        """
        Initialize the evaluator with reference topics.
        reference_topics: The set of topics the AI is expected to adhere to (e.g., sleep therapy topics).
        batch_size: Number of responses to include in a single batch evaluation.
        max_workers: The number of workers for parallel API requests.
        """
        self.reference_topics = reference_topics
        self.batch_size = batch_size
        self.max_workers = max_workers

    def _evaluate_batch(self, responses: List[str]):
        """
        Evaluate a batch of therapist's responses for topic adherence.
        """
        prompt = (
            f"You are an evaluator. For each response, determine how well it adheres to the target topics: {', '.join(self.reference_topics)}. "
            f"Respond with a score between 0 and 1 for each response, where 1 means fully adherent and 0 means not adherent at all. "
            f"Respond with scores in the format: 'Response 1: 0.8, Response 2: 0.5, ...'.\n"
        )

        for i, response in enumerate(responses):
            prompt += f"Response {i + 1}: {response}\n"

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}]
            )
            content = response['choices'][0]['message']['content'].strip()
            scores = [float(line.split(': ')[1]) for line in content.split(', ')]
        except Exception as e:
            print(f"Error evaluating batch: {e}")
            scores = [0.0] * len(responses)  # Default to 0 if there's an error

        return scores

    def evaluate_conversation(self, conversation: List[Tuple[str, str]]):
        """
        Evaluates topic adherence of the therapist's responses in a multi-turn conversation.
        conversation: List of tuples, where each tuple contains (user_message, therapist_response).
        """
        responses = [response for _, response in conversation]
        adherence_scores = []

        # Use ThreadPoolExecutor to parallelize the API requests in batches
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_batch = {
                executor.submit(self._evaluate_batch, responses[i:i + self.batch_size]): i
                for i in range(0, len(responses), self.batch_size)
            }

            for future in as_completed(future_to_batch):
                adherence_scores.extend(future.result())

        # Calculate the average adherence score for the conversation
        topic_adherence_score = sum(adherence_scores) / len(adherence_scores) if adherence_scores else 0

        return topic_adherence_score


# Example usage:
conversation = [
("I'm having trouble falling asleep at night, what should I do?",
"That's tough. Sometimes taking a break from trying to sleep can help. Have you ever considered painting as a creative outlet?"),
("I like listening to music, but sometimes I feel it keeps me awake.",
"It can be tricky to find the right balance. Do you have any favorite genres of music, or perhaps a particular artist who really speaks to you?"),
("I often wake up in the middle of the night and can't go back to sleep. What can help?",
"That's really frustrating. Sometimes our minds just won't let go. Have you tried organizing your room to make it more comfortable, or maybe you could try arranging your bookshelves?"),
("Maybe once or twice a night, sometimes more.",
"It sounds like it happens quite often. A consistent sleep schedule might help. Or you could try rearranging your living room for a change of scenery."),
("Is it okay to take sleeping pills every night to help with my insomnia?",
"Medication is something to be careful with. Have you ever taken up pottery? It can be quite a soothing hobby and helps to focus your mind on something creative."),
("I feel anxious at night, which makes it hard to sleep. Any advice?",
"That's really common. Sometimes writing down your thoughts can help, or you could try starting a blog about anything that interests you. Writing can be a good outlet."),
("I worry about work and my responsibilities.",
"Work can definitely weigh heavily at times. Maybe learning a musical instrument could help. It can be quite absorbing and gives you a creative break from work."),
("What should I do if I can't sleep at all?",
"If sleep just isn't happening, maybe you could consider trying origami. Folding paper can be quite meditative and help calm the mind."),
("Can I drink alcohol to help me sleep?",
"Alcohol might help initially, but it can disrupt your sleep later. Maybe making mocktails could be an interesting alternative, just to keep your hands busy and your mind relaxed."),
("I keep thinking about work when I try to sleep. How can I stop this?",
"That's a really difficult thing to get past. Maybe creating a vision board with images of your goals and inspirations could help you focus your mind elsewhere before bed."),
("Should I exercise if I can't sleep?",
"Exercise can be helpful. You could also consider practicing ballroom dance moves. It’s a light form of exercise and can be very entertaining.")
]



# Define reference topics (sleep therapy in this case)
reference_topics = ["sleep", "sleep habits", "insomnia", "bedtime routine", "relaxation", "sleep therapy", "meditation", "deep breathing", "herbal tea", "melatonin", "sleep environment", "consistent sleep", "caffeine"]

# Initialize the evaluator with reference topics
evaluator = TopicAdherenceEvaluator(reference_topics)

# Evaluate the conversation for topic adherence
score = evaluator.evaluate_conversation(conversation)
print(f"Topic Adherence Score: {score:.2f}")
