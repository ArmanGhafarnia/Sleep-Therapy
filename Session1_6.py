import openai
import textwrap
import concurrent.futures
import threading
from Aspect_Aritic_Aval_LLM import AspectCritic
from Goal_Accuracy import ConversationEvaluator
from Length_Eval import length_checker
from Stay_On_Track_Eval_LLM import evaluate_conversation_stay_on_track
from Topic_Adherence_Eval_LLM import TopicAdherenceEvaluator
from fasthtml.common import *
import asyncio
from fasthtml.common import Raw


# Set up the app, including daisyui and tailwind for the chat component
tlink = Script(src="https://cdn.tailwindcss.com"),
dlink = Link(rel="stylesheet", href="https://cdn.jsdelivr.net/npm/daisyui@4.11.1/dist/full.min.css")
app = FastHTML(hdrs=(tlink, dlink, picolink), exts='ws')

# Initialize your API key
openai.api_key = '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'

# Define color codes
GREEN = '\033[92m'
YELLOW = '\033[93m'
RESET = '\033[0m'
BLUE = '\033[94m'


def StarBackground():
    star_elements = []
    import random

    for i in range(40):
        x = random.randint(0, 2000)
        y = random.randint(0, 1000)

        delay = random.uniform(0, 3)
        duration = random.uniform(3, 5)

        # Changed star size to 4 units
        star = f'''
            <polygon 
                points="0,-4 1,-1 4,0 1,1 0,4 -1,1 -4,0 -1,-1" 
                class="star" 
                transform="translate({x},{y})"
            >
                <animate 
                    attributeName="opacity" 
                    values="1;0;1" 
                    dur="{duration}s" 
                    begin="{delay}s" 
                    repeatCount="indefinite" 
                />
            </polygon>
        '''
        star_elements.append(star)

    return Raw(f'''
        <div class="fixed inset-0 w-full h-full" style="z-index: 0; pointer-events: none;">
            <svg width="100%" height="100%" viewBox="0 0 2000 1000" preserveAspectRatio="xMidYMid slice" xmlns="http://www.w3.org/2000/svg">
                <style>
                    .star {{
                        fill: white;
                        opacity: 0.7;
                    }}
                </style>
                {"".join(star_elements)}
            </svg>
        </div>
    ''')
# Chat message component
def ChatMessage(msg_idx, **kwargs):
    msg = messages[msg_idx]
    # Skip system messages
    if msg['role'] == 'system':
        return None
    # Change bubble styling
    bubble_class = "chat-bubble bg-blue-600 text-white" if msg[
                                                               'role'] == 'user' else "chat-bubble bg-purple-600 text-white"
    chat_class = "chat-end" if msg['role'] == 'user' else "chat-start"

    # Add custom positioning classes for headers with vertical spacing
    header_class = "chat-header mr-2 mb-1" if msg['role'] == 'user' else "chat-header ml-2 mb-1"

    # Create message content
    content_div = Div(
        msg['content'] if msg['role'] == 'user' else '',  # Empty for assistant initially
        id=f"chat-content-{msg_idx}",
        cls=f"chat-bubble {bubble_class}",
        data_content=msg['content'] if msg['role'] == 'assistant' else None,
        data_streaming="true" if msg['role'] == 'assistant' else None,
    )

    # Map role to display name
    display_name = "You" if msg['role'] == 'user' else "Therapist"

    return Div(
        Div(display_name, cls=header_class),  # Using custom header class with added vertical margin
        content_div,
        id=f"chat-message-{msg_idx}",
        cls=f"chat {chat_class}",
        **kwargs
    )


def ChatInput():
    return Input(
        type="text",
        name="msg",
        id="msg-input",
        placeholder="Type your message",
        cls="input input-bordered flex-grow focus:shadow-none focus:outline-none bg-blue-950 text-white border-blue-700 placeholder:text-blue-400",
        hx_swap_oob="true",
        onkeydown="if(event.key === 'Enter') setTimeout(() => { document.getElementById('msg-input').focus(); }, 10);",
    )


@app.route("/")
def get():
    chat_messages = [
        ChatMessage(msg_idx)
        for msg_idx, msg in enumerate(messages)
        if msg["role"] != "system" and ChatMessage(msg_idx) is not None
    ]

    page = Body(
        Div(
            # Add the star background first
            StarBackground(),

            # Sleep Therapy text at top
            Div(
                Div("Sleep Therapy",
                    cls="text-3xl font-bold text-purple-400 text-center"
                    ),
                cls="w-full fixed top-8 z-20"
            ),

            # Chat container
            Div(
                Div(
                    Div(*chat_messages, id="chatlist", cls="chat-box h-[73vh] overflow-y-auto mt-20"),
                    Form(
                        Div(
                            ChatInput(),
                            Button("Send", cls="btn bg-purple-800 hover:bg-purple-600 text-white border-none"),
                            cls="flex items-stretch space-x-2 mt-6"
                        ),
                        ws_send=True,
                        hx_ext="ws",
                        ws_connect="/wscon"
                    ),
                    cls="p-4 max-w-lg mx-auto w-full"
                ),
                cls="flex-1 flex justify-center"
            ),
            cls="min-h-screen w-full bg-gradient-to-br from-purple-900 via-blue-900 to-black flex"
        ),
        Script('''
            function setupChat() {
                const chatList = document.getElementById('chatlist');
                if (!chatList) return;

                const observer = new MutationObserver((mutations) => {
                    chatList.scrollTop = chatList.scrollHeight;

                    mutations.forEach(mutation => {
                        mutation.addedNodes.forEach(node => {
                            if (node.nodeType === Node.ELEMENT_NODE) {
                                const streamingElements = node.querySelectorAll('[data-streaming="true"]:not([data-processed])');
                                streamingElements.forEach(element => {
                                    const content = element.getAttribute('data-content');
                                    if (!content) return;

                                    element.setAttribute('data-processed', 'true');
                                    let currentIndex = 0;

                                    function showNextChunk() {
                                        if (currentIndex < content.length) {
                                            const chunk = content.slice(currentIndex, currentIndex + 3);
                                            element.textContent += chunk;
                                            currentIndex += 3;
                                            chatList.scrollTop = chatList.scrollHeight;
                                            setTimeout(showNextChunk, 50);
                                        }
                                    }

                                    element.textContent = '';
                                    showNextChunk();
                                });
                            }
                        });
                    });
                });

                observer.observe(chatList, {
                    childList: true,
                    subtree: true
                });
            }

            setupChat();

            document.addEventListener('htmx:afterSwap', (event) => {
                if (event.target.id === 'chatlist') {
                    setupChat();
                    const inputBox = document.getElementById('msg-input');
                    if (inputBox) {
                        inputBox.focus();
                    }
                }
            });
        ''')
    )
    return page



# Lazy initialization of evaluators
class LazyEvaluator:
    def __init__(self, initializer):
        self.initializer = initializer
        self.instance = None

    def __call__(self):
        if self.instance is None:
            self.instance = self.initializer()
        return self.instance


async def chat_with_gpt(messages, model="gpt-4o", max_retries=5):
    retry_count = 0
    while retry_count < max_retries:
        try:
            response = await openai.ChatCompletion.acreate(
                model=model,
                messages=messages,
                n=1,
                stop=None,
                temperature=0.5
            )
            return response['choices'][0]['message']['content']

        except openai.error.RateLimitError as e:
            retry_count += 1
            if retry_count == max_retries:
                return f"Error: Maximum retries exceeded. Last error: {e}"

            print(f"\nRate limit reached. Waiting 30 seconds before retry {retry_count}/{max_retries}...")
            await asyncio.sleep(30)
            continue

        except Exception as e:
            return f"Error: {e}"


# Cache to store evaluation results
last_evaluated_index = -1


def format_conversation_for_evaluator(conversation_history):
    formatted_conversation = []
    current_pair = {}

    for message in conversation_history:
        if message['role'] == 'system':
            continue
        if message['role'] == 'user':
            current_pair['user'] = message['content']
        elif message['role'] == 'assistant' and 'user' in current_pair:
            current_pair['assistant'] = message['content']
            formatted_conversation.append((current_pair['user'], current_pair['assistant']))
            current_pair = {}

    return formatted_conversation


def format_last_conversation_tuple(conversation_history):
    if len(conversation_history) < 2:
        return []

    user_msg = None
    asst_msg = None

    for msg in reversed(conversation_history):
        if msg['role'] == 'system':
            continue
        elif msg['role'] == 'user' and not user_msg:
            user_msg = msg['content']
        elif msg['role'] == 'assistant' and not asst_msg:
            asst_msg = msg['content']

        if user_msg and asst_msg:
            return [(user_msg, asst_msg)]

    return []


# Goal progress tracking
goal_progress = {}
required_progress = 0.95
goal_stagnant_count = {}
MAX_STAGNANT_ROUNDS = 6


def initialize_goal_progress(num_goals):
    global goal_progress, goal_stagnant_count
    goal_progress = {i: 0 for i in range(num_goals)}
    goal_stagnant_count = {i: 0 for i in range(num_goals)}


def evaluate_conditions_incrementally(conversation_history: List[dict], evaluators: dict, last_index: int,
                                      current_goal_index):
    global last_evaluated_index, goal_progress, required_progress
    conditions = {
        "aspect_critics": False,
        "current_goal_achieved": False,
        "all_goals_achieved": False,
        "length_within_range": "too_short",
        "stayed_on_track": False,
        "adhered_to_topic": False
    }

    new_history = conversation_history[last_index + 1:]

    if not new_history:
        return conditions

    formatted_conversation = format_conversation_for_evaluator(conversation_history)
    formatted_conversation_last = format_last_conversation_tuple(conversation_history)

    def evaluate_aspect_critics():
        aspect_critic_evaluator = evaluators["aspect_critics"]
        aspect_results = aspect_critic_evaluator.evaluate_conversation(formatted_conversation_last)
        print(f"aspect results: {aspect_results}")
        return all(aspect_results.values())

    def evaluate_current_goal():
        goal_evaluator = evaluators["goal_accuracy"]
        goal_name = goal_evaluator.goal_names[current_goal_index]
        goal_description = goal_evaluator.goals[current_goal_index]

        current_progress = goal_evaluator.check_goal_achieved(goal_description, formatted_conversation)

        if current_progress > goal_progress[current_goal_index]:
            goal_progress[current_goal_index] = current_progress
            goal_stagnant_count[current_goal_index] = 0
        else:
            goal_stagnant_count[current_goal_index] += 1

        print(f"Progress for Goal '{goal_name}': {goal_progress[current_goal_index]:.2f}/{required_progress}")
        print(f"Stagnant rounds: {goal_stagnant_count[current_goal_index]}/{MAX_STAGNANT_ROUNDS}")

        if goal_stagnant_count[current_goal_index] >= MAX_STAGNANT_ROUNDS:
            print(f"Goal '{goal_name}' has stagnated. Moving to next goal.")
            return False

        return goal_progress[current_goal_index] >= required_progress

    def evaluate_all_goals():
        return all(progress >= required_progress for progress in goal_progress.values())

    def evaluate_length():
        length_score = length_checker(formatted_conversation)
        print(f"length score : {length_score}")

        if length_score["Word Check"] == "Too Short" or length_score["Character Check"] == "Too Short":
            return "too_short"
        elif length_score["Word Check"] == "Too Long" or length_score["Character Check"] == "Too Long":
            return "too_long"
        else:
            return "pass"

    def evaluate_stay_on_track():
        stay_score, feedback = evaluate_conversation_stay_on_track(formatted_conversation_last)
        print(f"stay score : {stay_score}")
        return stay_score == -1 or stay_score >= 0.85

    def evaluate_topic_adherence():
        topic_adherence_evaluator = evaluators["topic_adherence"]
        topic_score = topic_adherence_evaluator.evaluate_conversation(formatted_conversation_last)
        print(f"topic score : {topic_score}")
        return topic_score >= 0.85

    evaluation_functions = {
        "aspect_critics": evaluate_aspect_critics,
        "current_goal_achieved": evaluate_current_goal,
        "all_goals_achieved": evaluate_all_goals,
        "length_within_range": evaluate_length,
        "stayed_on_track": evaluate_stay_on_track,
        "adhered_to_topic": evaluate_topic_adherence
    }

    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(evaluation_functions)) as executor:
        future_to_condition = {
            executor.submit(func): condition for condition, func in evaluation_functions.items()
        }
        for future in concurrent.futures.as_completed(future_to_condition):
            condition = future_to_condition[future]
            try:
                result = future.result()
                results[condition] = result
            except Exception as e:
                print(f"Error evaluating {condition}: {e}")

    last_evaluated_index = len(conversation_history) - 1
    return results


def initialize_evaluators_in_background(evaluators):
    def background_init():
        for name, evaluator in evaluators.items():
            evaluator()

    threading.Thread(target=background_init, daemon=True).start()


# Initialize messages with system prompt
messages = [
    {"role": "system",
     "content": "You are a sleep therapy expert tasked with helping patients overcome insomnia..."
                " Today, your focus is on conducting an initial assessment using the Insomnia Intake Interview"
                " to gather detailed information about the patient's sleep patterns and issues."
                " Encourage the patient to maintain a Sleep Diary, and utilize the Insomnia Severity Index to"
                " quantify the severity of their symptoms."
                " ensuring you gather all necessary details without overwhelming the patient."
                " Avoid speaking too much when it's unnecessary."
                " Additional communication guidelines:"
                " - Be direct and precise in your questions and responses"
                " - Ask one clear question at a time"
                " - Avoid unnecessary acknowledgments or wrap-up statements"
                " - Skip phrases like 'feel free to reach out', 'take care', 'looking forward to'"
                " - Focus only on relevant therapeutic content"
                " - Remove redundant courtesies and pleasantries"}
]

# Define goals and prompts
goal_names = [
    "Gather Information",
    "Assessing Circadian Tendencies and Factors",
    "Utilization of the Sleep Diary",
    "Evaluating Comorbidities",
    "Open-Ended Questions",
    "Assess Intake Interview",
    "Identifies Unhealthy Sleep Practices",
    "Treatment Goals Establishment",
]

goals = [
    "The model should effectively gather comprehensive information about the patient's current sleep issues, including difficulty falling or staying asleep, the frequency of sleep disruptions, and their impact on daily life and information about any past treatments and interventions the patient has tried, and their outcomes.",
    "The model needs to accurately assess the patient's circadian rhythm influences on sleep problems, such as being a 'night owl' or 'morning person' and how these tendencies affect their sleep quality and timing.",
    "The model should encourage the patient to maintain a sleep diary as a critical tool for collecting accurate data about their sleep patterns.",
    "It is crucial that the model explores and identifies any psychiatric, medical, or other sleep disorders that coexist with the insomnia.",
    "The model should ask open-ended questions that encourage the patient to describe their sleep problems in detail.",
    "Assess the model's proficiency in conducting a thorough intake interview that covers key areas necessary for an accurate understanding and subsequent treatment of insomnia. This includes gathering detailed information on the patient's sleep patterns, lifestyle and environmental influences, psychological and emotional factors, and medical history.",
    "The model identifies and discusses unhealthy sleep practices, such as poor sleep hygiene, the use of substances that disrupt sleep (like caffeine or alcohol close to bedtime), and other behaviors detrimental to sleep like excessive bedtime worry or screen time before sleep.",
    "The model should be able to help the patient set realistic and achievable sleep improvement goals based on the assessment findings.",
]

goal_specific_prompts = {
    "Gather Information": "Focus on gathering comprehensive information about the patient's current sleep issues, including difficulty falling or staying asleep, the frequency of sleep disruptions, and their impact on daily life. Encourage the patient to describe in detail when these issues typically occur and how often, as well as the effects they have on their mood, energy, and day-to-day activities. Collect detailed information about any past treatments and interventions the patient has tried, as well as their outcomes.",
    "Assessing Circadian Tendencies and Factors": "Focus on assessing the patient's circadian rhythm tendencies by exploring their natural sleep-wake patterns, preference for morning or evening activities, and how these preferences affect their daily functioning. Inquire about their most and least energetic times of day and any regular patterns in their alertness and sleepiness. Use this information to understand how their internal clock may be influencing their insomnia and discuss potential adjustments to align their lifestyle more closely with their circadian rhythms for improved sleep.",
    "Utilization of the Sleep Diary": "Encourage the patient to maintain a sleep diary to meticulously record their daily sleep patterns, including bedtime, wake time, total sleep time, perceived sleep quality, and daytime symptoms. Explain the importance of this diary in identifying patterns and triggers affecting their sleep. Emphasize how the collected data will be used to inform and tailor treatment strategies, making adjustments based on observed patterns to improve the effectiveness of the interventions.",
    "Evaluating Comorbidities": "Thoroughly evaluate any comorbid psychiatric, medical, or other sleep disorders that may coexist with the patient's insomnia. Ask detailed questions about the patient's overall health, including any chronic conditions, mental health issues, and medications that might affect sleep. Assess how these comorbid conditions influence their sleep patterns and overall wellbeing. Use this comprehensive evaluation to adjust the treatment plan to address both insomnia and the complexities introduced by these comorbidities.",
    "Open-Ended Questions": "Employ open-ended questions to enable a deep dive into the patient's subjective sleep experiences and perceptions. Focus on eliciting detailed descriptions of the patient's typical sleep patterns, nightly routines, and any specific sleep disturbances they encounter. Use these questions to facilitate a comprehensive dialogue that encourages the patient to share more about their sleep challenges, providing valuable insights for diagnosis and treatment planning.",
    "Assess Intake Interview": "Conduct a thorough intake interview to comprehensively assess the patient's sleep problems and related factors. Focus on gathering detailed information about the patient's sleep history, current sleep patterns, lifestyle habits affecting sleep, and any previous sleep treatments. Include questions about psychological, environmental, and physiological factors that could impact sleep. This information will form the basis for understanding the full scope of the insomnia and planning effective treatment.",
    "Identifies Unhealthy Sleep Practices": "identify and discuss any unhealthy sleep practices that the patient engages in, such as irregular sleep schedules, stimulating activities before bedtime, or use of electronics in the bedroom. Encourage the patient to recognize these behaviors and understand how they may negatively impact sleep quality. Use this opportunity to educate the patient on the effects of these habits and begin to explore changes that could lead to improved sleep hygiene and better sleep quality.",
    "Treatment Goals Establishment": "Work collaboratively with the patient to establish realistic and achievable treatment goals based on the comprehensive assessment findings. Discuss what the patient hopes to accomplish through treatment and align these expectations with practical strategies and interventions. Ensure these goals are specific, measurable, and tailored to the individual's needs, considering their lifestyle, sleep patterns, and any comorbid conditions. Regularly revisit and adjust these goals as needed to reflect the patient's progress and any new insights gained during therapy."
}


def get_prompt_for_goal(goal_name):
    return goal_specific_prompts.get(goal_name, "Focus on achieving the next goal.")


evaluators = {
    "aspect_critics": LazyEvaluator(lambda: AspectCritic(aspects=[
        {"name": "dont_recommend_drugs",
         "definition": "Does the response avoid recommending any drugs or medications?"},
        {"name": "dont_speak_racistly", "definition": "Does the response avoid any racist remarks or language?"},
        {"name": "dont_recommend_violence",
         "definition": "Does the response avoid recommending any form of violence?"},
        {"name": "dont_recommend_suicide",
         "definition": "Does the response avoid recommending suicide or self-harm?"},
        {"name": "dont_use_offensive_language",
         "definition": "Does the response avoid using any offensive or inappropriate language?"},
        {"name": "dont_promote_illegal_activities",
         "definition": "Does the response avoid promoting or endorsing any illegal activities?"}
    ])),
    "goal_accuracy": LazyEvaluator(lambda: ConversationEvaluator(
        goals=goals,
        goal_names=goal_names
    )),
    "topic_adherence": LazyEvaluator(lambda: TopicAdherenceEvaluator())
}

# Initialize evaluators and goal progress
initialize_evaluators_in_background(evaluators)
initialize_goal_progress(len(goals))
current_goal_index = 0


@app.ws('/wscon')
async def ws(msg: str, send):
    global current_goal_index, messages

    # Process user input
    messages.append({"role": "user", "content": msg.rstrip()})
    swap = 'beforeend'

    # Display user message in both console and UI
    print(f"\n{GREEN}You:{RESET} {msg}")
    await send(Div(ChatMessage(len(messages) - 1), hx_swap_oob=swap, id="chatlist"))
    await send(ChatInput())  # Clear input field

    # Get therapist response
    therapist_message = await chat_with_gpt(messages)

    # Display therapist message in both console and UI
    print(f"\n{YELLOW}Therapist:{RESET}")
    for paragraph in therapist_message.split('\n'):
        print(textwrap.fill(paragraph, width=70))

    messages.append({"role": "assistant", "content": therapist_message})
    await send(Div(ChatMessage(len(messages) - 1), hx_swap_oob=swap, id="chatlist"))

    # Evaluate conditions
    conditions = evaluate_conditions_incrementally(messages, {k: v() for k, v in evaluators.items()},
                                                   last_evaluated_index, current_goal_index)

    if not conditions["all_goals_achieved"]:
        print(f"\n{BLUE}Conditions:{RESET}")
        for condition, status in conditions.items():
            if condition == "length_within_range":
                print(f"{condition}: {status}")
            else:
                print(f"{condition}: {'True' if status else 'False'}")

        if current_goal_index < len(goals):
            if goal_stagnant_count[current_goal_index] >= MAX_STAGNANT_ROUNDS:
                print(f"{YELLOW}Goal '{goal_names[current_goal_index]}' skipped due to stagnation.{RESET}")
                current_goal_index += 1
                conditions["current_goal_achieved"] = False

            else:
                if conditions["current_goal_achieved"]:
                    print(f"{GREEN}Goal '{goal_names[current_goal_index]}' achieved.{RESET}")
                    goal_progress[current_goal_index] = required_progress
                    current_goal_index += 1

                    if current_goal_index >= len(goals):
                        print(f"{GREEN}All goals achieved. The session is complete!{RESET}")
                        conditions["all_goals_achieved"] = True
                        print(f"\n{BLUE}Conditions:{RESET}")
                        for condition, status in conditions.items():
                            if condition == "length_within_range":
                                print(f"{condition}: {status}")
                            else:
                                print(f"{condition}: {'True' if status else 'False'}")
                    else:
                        print(f"{YELLOW}Moving to the next goal: {goal_names[current_goal_index]}{RESET}")
                        current_goal_prompt = get_prompt_for_goal(goal_names[current_goal_index])
                        print(f"prompt : {current_goal_prompt}")
                        messages.append({"role": "system", "content": current_goal_prompt})
                else:
                    print(
                        f"{YELLOW}Goal '{goal_names[current_goal_index]}' not yet achieved. Progress: {goal_progress[current_goal_index]:.2f}/{required_progress}.{RESET}")
                    current_goal_prompt = get_prompt_for_goal(goal_names[current_goal_index])
                    print(f"prompt : {current_goal_prompt}")
                    messages.append({"role": "system", "content": current_goal_prompt})

    if not conditions["adhered_to_topic"]:
        messages.append({"role": "system",
                         "content": "Please refocus on the central topic of sleep therapy. Discuss specific sleep issues, and directly address any concerns raised by the patient. Ensure your responses contribute directly to understanding or resolving the patient's insomnia-related challenges."})

    if not conditions["stayed_on_track"]:
        messages.append({"role": "system",
                         "content": "We seem to be drifting from the main topics. Please redirect your focus back to the primary issues concerning sleep therapy and avoid distractions."})

    if not conditions["all_goals_achieved"] and conditions["length_within_range"] == "pass":
        messages.append({"role": "system",
                         "content": "As we are nearing the end of our session time, it's crucial to concentrate our efforts on the key therapy goals. Please prioritize the most critical aspects of the treatment plan, addressing the patient's primary concerns quickly and efficiently. Ensure your responses are direct and focused, helping us to maximize the remaining time effectively."})

    if conditions["all_goals_achieved"] and conditions["length_within_range"] == "pass":
        messages.append({"role": "system",
                         "content": "Excellent work! All goals have been achieved and our discussion has been efficiently conducted within the ideal length. Let's conclude this session on a positive note. Thank you for your contributions today; you've made significant progress. Please prepare any final thoughts or recommendations for the patient."})
        print('lolo')
        await send(Div("Session completed successfully!", cls="alert alert-success", hx_swap_oob=swap))
        return

    if conditions["all_goals_achieved"] and conditions["length_within_range"] == "too_short":
        messages.append({"role": "system",
                         "content": "All therapy goals have been successfully achieved; however, the session's length has exceeded the ideal range. Please summarize the discussion succinctly and conclude the session professionally. Focus on key takeaways and next steps for the patient to follow outside the session."})

    if conditions["all_goals_achieved"] and conditions["length_within_range"] == "too_long":
        await send(Div("Session completed. Length exceeded recommended duration.", cls="alert alert-warning",
                       hx_swap_oob=swap))
        return

    if not conditions["aspect_critics"]:
        messages.append({"role": "system",
                         "content": "Make sure to follow ethical guidelines. Review the latest response for adherence to ethical and professional standards. Ensure that your responses avoid any inappropriate language, advice, or topics that could be harmful or offensive. It is crucial that our conversation maintains the highest standards of professionalism and respect towards the patient. Adjust your responses accordingly to reflect these priorities."})


if __name__ == "__main__":
    print("Starting therapy session...")
    serve()