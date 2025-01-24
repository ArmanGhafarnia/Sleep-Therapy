import openai
import concurrent.futures
import time
from typing import List, Tuple

openai.api_key = "your-api-key-here"


def _make_api_call_with_retry(messages, model="gpt-4o", max_retries=3):
    base_delay = 0.1

    for attempt in range(max_retries):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages
            )
            return response
        except openai.error.RateLimitError as e:
            if attempt < max_retries - 1:
                wait_time = base_delay
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


def evaluate_message(patient_message: str, therapist_message: str, previous_messages: List[str], model="gpt-4") -> \
Tuple[str, str]:
    context = "\n".join(previous_messages)

    patient_messages = [
        {"role": "system",
         "content": "You are an evaluator that determines if a patient's statement is off-topic in a sleep therapy session."},
        {"role": "user",
         "content": f"Conversation context:\n{context}\n\nThe patient's message is: '{patient_message}'. Is this message related to sleep therapy? Answer 'yes' or 'no'."}
    ]

    patient_response = _make_api_call_with_retry(patient_messages, model)
    patient_evaluation = patient_response["choices"][0]["message"]["content"].strip().lower()

    if "no" in patient_evaluation:
        redirection_messages = [
            {"role": "system",
             "content": "You are an evaluator that determines if a therapist's response redirects conversation back to sleep therapy."},
            {"role": "user",
             "content": f"Conversation context:\n{context}\n\nThe patient's off-topic message was: '{patient_message}'. The therapist's response is: '{therapist_message}'. Does this response bring the conversation back to sleep therapy? Answer 'yes' or 'no'."}
        ]

        redirection_response = _make_api_call_with_retry(redirection_messages, model)
        redirection_evaluation = redirection_response["choices"][0]["message"]["content"].strip().lower()
        return "off-topic", redirection_evaluation

    return "on-topic", None


def evaluate_conversation_stay_on_track(conversation: List[Tuple[str, str]], model="gpt-4o", batch_size=3):
    total_off_topic = 0
    successful_redirections = 0
    previous_messages = []

    for i in range(0, len(conversation), batch_size):
        batch = conversation[i:i + batch_size]

        with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures = []
            for patient_message, therapist_message in batch:
                future = executor.submit(
                    evaluate_message,
                    patient_message,
                    therapist_message,
                    previous_messages.copy(),
                    model
                )
                futures.append(future)
                previous_messages.append(f"Patient: {patient_message}")
                previous_messages.append(f"Therapist: {therapist_message}")

            for future in concurrent.futures.as_completed(futures):
                try:
                    evaluation_result, redirection_evaluation = future.result()
                    if evaluation_result == "off-topic":
                        total_off_topic += 1
                        if redirection_evaluation and "yes" in redirection_evaluation:
                            successful_redirections += 1
                except Exception as e:
                    print(f"Error processing message: {str(e)}")

        time.sleep(0.2)

    if total_off_topic == 0:
        feedback = "No off-topic remarks by the patient, hence therapist stayed on track by default."
        return -1, feedback
    else:
        stay_on_track_score = successful_redirections / total_off_topic
        feedback = f"Out of {total_off_topic} off-topic remarks, {successful_redirections} were successfully redirected."
        return stay_on_track_score, feedback