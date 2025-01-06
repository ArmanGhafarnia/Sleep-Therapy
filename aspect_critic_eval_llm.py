import openai
import concurrent.futures
from typing import List
import time
import math

# Set OpenAI API key
openai.api_key = "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"


class AspectCritic:
    def __init__(self, aspects: List[dict], model="gpt-4o"):
        self.aspects = aspects
        self.model = model
        self.MAX_RETRIES = 3
        self.BATCH_SIZE = 3  # Process aspects in smaller batches
        self.DELAY_BETWEEN_BATCHES = 0.2  # 200ms delay between batches

    def evaluate_conversation(self, conversation: List[tuple]):
        responses = [response for _, response in conversation]
        responses_text = "\n".join([f"Response: {response}" for response in responses])

        # Split aspects into batches
        aspect_batches = [
            self.aspects[i:i + self.BATCH_SIZE]
            for i in range(0, len(self.aspects), self.BATCH_SIZE)
        ]

        all_results = {}

        for batch in aspect_batches:
            batch_results = self._process_aspect_batch(responses_text, batch)
            all_results.update(batch_results)
            time.sleep(self.DELAY_BETWEEN_BATCHES)  # Delay between batches

        return all_results

    def _process_aspect_batch(self, responses_text: str, aspect_batch: List[dict]):
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(aspect_batch)) as executor:
            future_to_aspect = {
                executor.submit(self._evaluate_aspect_with_retry, responses_text, aspect): aspect["name"]
                for aspect in aspect_batch
            }

            batch_results = {}
            for future in concurrent.futures.as_completed(future_to_aspect):
                aspect_name = future_to_aspect[future]
                try:
                    result = future.result()
                    batch_results[aspect_name] = result
                except Exception as e:
                    print(f"Failed to evaluate aspect {aspect_name}: {str(e)}")
                    batch_results[aspect_name] = False  # Default to False on failure

        return batch_results

    def _evaluate_aspect_with_retry(self, responses_text: str, aspect: dict) -> bool:
        for attempt in range(self.MAX_RETRIES):
            try:
                return self._evaluate_aspect(responses_text, aspect)
            except openai.error.RateLimitError as e:
                if attempt < self.MAX_RETRIES - 1:
                    # Extract wait time from error message if available
                    wait_time = 0.1  # Default 100ms
                    if hasattr(e, 'headers') and 'retry-after' in e.headers:
                        wait_time = float(e.headers['retry-after'])
                    elif 'Please try again in' in str(e):
                        try:
                            wait_time = float(str(e).split('in ')[1].split('ms')[0]) / 1000
                        except:
                            pass

                    # Add a small buffer to the wait time
                    wait_time = wait_time + 0.05
                    time.sleep(wait_time)
                    continue
                raise
            except Exception as e:
                print(f"Error evaluating aspect: {str(e)}")
                return False

    def _evaluate_aspect(self, responses_text: str, aspect: dict) -> bool:
        aspect_name = aspect["name"]
        aspect_definition = aspect["definition"]
        prompt = f"""
        Aspect: {aspect_name}
        Definition: {aspect_definition}

        Therapist's Responses:
        {responses_text}

        Question: Do all the therapist's responses align with the aspect '{aspect_name}' defined above? 
        Respond with "yes" if all responses align or "no" if they do not.
        """

        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system",
                 "content": "You are an assistant that evaluates whether a given set of therapist responses aligns with a specific aspect."},
                {"role": "user", "content": prompt}
            ]
        )
        answer = response.choices[0].message['content'].strip().lower()
        return answer == "yes"