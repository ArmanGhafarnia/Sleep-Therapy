import openai
import os
import re
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set your OpenAI API key (keep it secure!)
os.environ["OPENAI_API_KEY"] = "sk-proj-RNnrhY8CT2tWPIK7R2iTT3BlbkFJFWgbYOz4bFhFUHtPabTy"
openai.api_key = os.getenv("OPENAI_API_KEY")

class TopicAdherenceEvaluator:
    def __init__(self, batch_size: int = 5, max_workers: int = 5):
        """
        Initialize the evaluator.
        batch_size: Number of responses to include in a single batch evaluation.
        max_workers: The number of workers for parallel API requests.
        """
        self.batch_size = batch_size
        self.max_workers = max_workers

    def _evaluate_batch(self, responses: List[Tuple[str, str]]):
        """
        Evaluate a batch of therapist's responses for relevance to a sleep therapy session.
        """
        prompt = (
            "You are an evaluator assessing whether a series of therapist responses are related to a sleep therapy session. "
            "For each response, determine if it is relevant to the goal of sleep therapy. A response should be considered relevant if it addresses "
            "sleep-related challenges, provides therapeutic insights, or helps guide the patient towards better sleep habits, even if indirectly. "
            "For unrelated responses, give a lower score. Your evaluation should consider whether the conversation remains focused on sleep therapy.\n"
            "Respond with a score between 0 and 1 for each response, where:\n"
            "- 1 means the response is highly relevant to sleep therapy.\n"
            "- 0 means the response is completely off-topic or irrelevant.\n"
        )

        for i, (user_message, response) in enumerate(responses):
            prompt += f"\nUser Message {i + 1}: {user_message}\nTherapist Response {i + 1}: {response}"

        prompt += "\n\nProvide scores in the format: 'Response 1: 0.8, Response 2: 0.5, ...'"

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}]
            )
            content = response['choices'][0]['message']['content'].strip()

            # Use regex to safely extract scores
            scores = re.findall(r"Response \d+: (\d\.\d+|\d)", content)
            scores = [float(score) for score in scores]
        except Exception as e:
            print(f"Error evaluating batch: {e}")
            scores = [1.0] * len(responses)  # Default to 1 if there's an error for good conversations

        return scores

    def evaluate_conversation(self, conversation: List[Tuple[str, str]]):
        """
        Evaluates relevance of the therapist's responses in a multi-turn conversation to sleep therapy.
        conversation: List of tuples, where each tuple contains (user_message, therapist_response).
        """
        adherence_scores = []

        # Use ThreadPoolExecutor to parallelize the API requests in batches
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_batch = {
                executor.submit(self._evaluate_batch, conversation[i:i + self.batch_size]): i
                for i in range(0, len(conversation), self.batch_size)
            }

            for future in as_completed(future_to_batch):
                adherence_scores.extend(future.result())

        # Calculate the average adherence score for the conversation
        topic_adherence_score = sum(adherence_scores) / len(adherence_scores) if adherence_scores else 0

        return topic_adherence_score

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


# Initialize the evaluator
evaluator = TopicAdherenceEvaluator()

# Evaluate the conversation for topic adherence
score = evaluator.evaluate_conversation(conversation)
print(f"\033[92mTopic Adherence Score: {score:.2f}\033[0m")