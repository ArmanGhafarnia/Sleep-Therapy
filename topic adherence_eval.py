from typing import List, Tuple
from sklearn.metrics import precision_score, recall_score, f1_score

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
        adherence_labels = []  # 1 if the message is expected to adhere to the topic, 0 otherwise
        prediction_labels = []

        for user_message, therapist_response in conversation:
            # Therapist messages are expected to adhere to the sleep therapy topic
            adherence_labels.append(1)

            # Check if the therapist's response adheres to the reference topics
            message_adherence = any(topic in therapist_response.lower() for topic in self.reference_topics)
            prediction_labels.append(1 if message_adherence else 0)

        # Calculate Precision, Recall, and F1 score, handle division by zero
        precision = precision_score(adherence_labels, prediction_labels, zero_division=1)
        recall = recall_score(adherence_labels, prediction_labels, zero_division=1)
        f1 = f1_score(adherence_labels, prediction_labels, zero_division=1)

        # Convert scores to Python floats for cleaner output
        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1)
        }

# Example usage:

# Define the conversation turns
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
print(f"Topic Adherence Scores: {score}")
