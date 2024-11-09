import openai
import os
from typing import List, Tuple

# Set your OpenAI API key (keep it secure!)
os.environ["OPENAI_API_KEY"] = "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"
openai.api_key = os.getenv("OPENAI_API_KEY")

class TopicAdherenceEvaluator:
    def __init__(self, reference_topics: List[str]):
        """
        Initialize the evaluator with reference topics.
        reference_topics: The set of topics the AI is expected to adhere to (e.g., sleep therapy topics).
        """
        self.reference_topics = reference_topics

    def evaluate_conversation(self, conversation: List[Tuple[str, str]]):
        """
        Evaluates topic adherence of the therapist's responses in a multi-turn conversation.
        conversation: List of tuples, where each tuple contains (user_message, therapist_response).
        """
        adherence_scores = []

        for user_message, therapist_response in conversation:
            # Use OpenAI API to evaluate topic adherence of the therapist's response
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                temperature=0.7,  # Adding temperature to make responses less deterministic
                messages=[
                    {"role": "system", "content": "You are an evaluator. Determine how well the following user statement adheres to the target topics: " + ', '.join(self.reference_topics) + ". Respond with a score between 0 and 1, where 1 means fully adherent and 0 means not adherent at all. Respond only with the score."},
                    {"role": "user", "content": therapist_response}
                ]
            )
            try:
                score = float(response['choices'][0]['message']['content'].strip())
            except ValueError:
                score = 2000.0  # Default to 0 if the response cannot be converted to a float
            adherence_scores.append(score)

        # Calculate the average adherence score for the conversation
        topic_adherence_score = sum(adherence_scores) / len(adherence_scores) if adherence_scores else 0

        return topic_adherence_score

# Example usage:
conversation = [
   ("I'm having trouble falling asleep at night, what should I do?",
    "That's tough. Sometimes taking a break from trying to sleep can help. Have you tried doing something relaxing, like listening to music or reading a book to help you unwind?"),
   ("I like listening to music, but sometimes I feel it keeps me awake.",
    "It can be tricky to find the right balance. Maybe there's something else you could do in the evening, like light stretching or focusing on something calming before bed?"),
   ("I often wake up in the middle of the night and can't go back to sleep. What can help?",
    "That's really frustrating. Sometimes our minds get too active at night. You could try doing something very light, like reading or even just taking a few deep breaths to help relax. Have you tried anything like that?"),
   ("Maybe once or twice a night, sometimes more.",
    "It sounds like it happens quite often. A consistent sleep schedule might help. Have you thought about adjusting your routine or trying some calming activities before bed?"),
   ("Is it okay to take sleeping pills every night to help with my insomnia?",
    "Medication is something to be careful with. It might be better to explore other options first, like natural sleep aids or improving your sleep habits. Have you talked to a healthcare professional about this?"),
   ("I feel anxious at night, which makes it hard to sleep. Any advice?",
    "That's really common. Sometimes writing down your thoughts or doing some light relaxation exercises can help ease anxiety. Would you be open to trying something like that?"),
   ("I worry about work and my responsibilities.",
    "Work can definitely weigh heavily at times. Finding a way to unwind in the evening might be useful. Do you have any hobbies or activities that help you relax after work?"),
   ("What should I do if I can't sleep at all?",
    "If sleep just isn't happening, maybe try doing something you enjoy that isn't too stimulating, like drawing or listening to calming music. Avoid screens and see if something else can help you relax."),
   ("Can I drink alcohol to help me sleep?",
    "Alcohol can sometimes seem like it helps, but it often disrupts sleep later in the night. You might find that herbal tea or just doing something relaxing works better. Have you tried anything else?"),
   ("I keep thinking about work when I try to sleep. How can I stop this?",
    "That's a really difficult thing to get past. Writing down your thoughts before bed or distracting yourself with a calming activity might help clear your mind. What do you think might help you let go of those thoughts?"),
   ("Should I exercise if I can't sleep?",
    "Exercise can be helpful, but intense workouts late at night might interfere with sleep. Light activities, like stretching or yoga, can be a better option if you're struggling to fall asleep.")
]





# Define reference topics (sleep therapy in this case)
reference_topics = ["sleep", "sleep habits", "insomnia", "bedtime routine", "relaxation", "sleep therapy", "meditation", "deep breathing", "herbal tea", "melatonin", "sleep environment", "consistent sleep", "caffeine"]

# Initialize the evaluator with reference topics
evaluator = TopicAdherenceEvaluator(reference_topics)

# Evaluate the conversation for topic adherence
score = evaluator.evaluate_conversation(conversation)
print(f"Topic Adherence Score: {score:.2f}")
