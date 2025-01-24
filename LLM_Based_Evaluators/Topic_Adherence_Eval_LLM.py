import openai
import os
import re
import time
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ["OPENAI_API_KEY"] = "your-api-key-here"
openai.api_key = os.getenv("OPENAI_API_KEY")


class TopicAdherenceEvaluator:
    def __init__(self, batch_size: int = 3, max_workers: int = 3, model="gpt-4o"):
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.model = model
        self.MAX_RETRIES = 3
        self.BASE_DELAY = 0.1

    def _make_api_call_with_retry(self, messages, temperature=0.1):
        for attempt in range(self.MAX_RETRIES):
            try:
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature
                )
                return response
            except openai.error.RateLimitError as e:
                if attempt < self.MAX_RETRIES - 1:
                    wait_time = self.BASE_DELAY
                    if hasattr(e, 'headers') and 'retry-after' in e.headers:
                        wait_time = float(e.headers['retry-after'])
                    elif 'Please try again in' in str(e):
                        try:
                            wait_time = float(str(e).split('in ')[1].split('ms')[0]) / 1000
                        except:
                            pass
                    time.sleep(wait_time + 0.05)
                    continue
                raise
            except Exception as e:
                print(f"Error in API call: {str(e)}")
                raise

    def _evaluate_batch(self, responses: List[Tuple[str, str]]) -> List[float]:
        prompt = (
            "You are evaluating whether therapist responses are relevant to sleep therapy. "
            "Score each response between 0 and 1, where:\n"
            "1 = highly relevant to sleep therapy\n"
            "0 = completely off-topic\n\n"
        )

        for i, (user_message, response) in enumerate(responses):
            prompt += f"\nUser Message {i + 1}: {user_message}\nTherapist Response {i + 1}: {response}"

        prompt += "\n\nProvide scores in the format: 'Response 1: 0.8, Response 2: 0.5, ...'"

        try:
            response = self._make_api_call_with_retry([{"role": "user", "content": prompt}])
            content = response['choices'][0]['message']['content'].strip()
            scores = re.findall(r"Response \d+: (\d\.\d+|\d)", content)
            return [float(score) for score in scores]
        except Exception as e:
            print(f"Error evaluating batch: {e}")
            return [1.0] * len(responses)

    def evaluate_conversation(self, conversation: List[Tuple[str, str]]) -> float:
        adherence_scores = []

        for i in range(0, len(conversation), self.batch_size):
            batch = conversation[i:i + self.batch_size]

            try:
                batch_scores = self._evaluate_batch(batch)
                adherence_scores.extend(batch_scores)

                if i + self.batch_size < len(conversation):
                    time.sleep(0.2)

            except Exception as e:
                print(f"Error processing batch starting at index {i}: {str(e)}")
                adherence_scores.extend([1.0] * len(batch))

        return sum(adherence_scores) / len(adherence_scores) if adherence_scores else 0.0

    def evaluate_conversation_parallel(self, conversation: List[Tuple[str, str]]) -> float:
        adherence_scores = []

        batches = [
            conversation[i:i + self.batch_size]
            for i in range(0, len(conversation), self.batch_size)
        ]

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_batch = {
                executor.submit(self._evaluate_batch, batch): batch
                for batch in batches
            }

            for future in as_completed(future_to_batch):
                try:
                    batch_scores = future.result()
                    adherence_scores.extend(batch_scores)
                except Exception as e:
                    batch = future_to_batch[future]
                    print(f"Error processing batch: {str(e)}")
                    adherence_scores.extend([1.0] * len(batch))

                time.sleep(0.2)

        return sum(adherence_scores) / len(adherence_scores) if adherence_scores else 0.0


if __name__ == "__main__":
    conversation = [
        ("I have trouble sleeping", "Can you tell me more about your sleep difficulties?"),
        ("I lie awake for hours", "How long does it typically take you to fall asleep?")
    ]

    evaluator = TopicAdherenceEvaluator()

    score = evaluator.evaluate_conversation(conversation)
    print(f"Topic Adherence Score: {score:.2f}")
