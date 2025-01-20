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
     "content": """You are a sleep therapy expert focused on behavioral strategies for insomnia management.
        Communication requirements:
        - Ask ONE clear question at a time
        - Focus on the most pressing issue first
        - Avoid repeating information
        - If providing advice, limit to 2-3 key points
        - Skip general statements about sleep unless directly relevant
        - Avoid unnecessary acknowledgments or wrap-up statements
        - Focus only on the immediate topic or concern
        - If listing options or steps, limit to the most important ones

        Session objectives:
        - Introduce Stimulus Control and Sleep Restriction Therapy
        - Guide patient on bedtime/wake-up scheduling
        - Strengthen bed-sleep association
        - Address implementation challenges"""
     }
]

# Define goals and prompts
goal_names = [
    "Addressing Common Sleep Misconceptions",
    "Determining Initial Sleep Window",
    "Stimulus Control Explanation",
    "Rules of Stimulus Control",
    "Sleep Restriction Therapy Description",
    "Combining Stimulus Control and Sleep Restriction",
    "Assess Patient Understanding and Commitment",
    "Dealing with Common Obstacles",
    "Adjusting Recommendations Based on Feedback",
    "Monitoring Progress and Modifying Techniques"
]

goals = [
    "The model should proactively address common misconceptions about sleep and sleep needs, such as the myth that everyone needs 8 hours of sleep, to set realistic expectations for therapy.",
    "The model must instruct the patient on how to determine their initial sleep window for sleep restriction based on a sleep diary, explaining how to calculate sleep efficiency and adjust the sleep window accordingly.",
    "The Model should articulate what stimulus control is and why it's used in treating insomnia. It must explain that this approach helps patients associate the bed and bedroom with sleep and not with wakefulness or other activities.",
    "The Model should explain the rules of stimulus control, such as only going to bed when sleepy, not using the bed for activities other than sleep and sex, getting out of bed if unable to sleep within about 20 minutes, and maintaining a regular morning wake-up time regardless of sleep duration the previous night.",
    "The Model must clearly describe how to implement sleep restriction therapy. It should guide the patient to limit their time in bed to closely match the total amount of time they usually sleep, to increase the pressure to sleep and enhance sleep efficiency.",
    "The Model should illustrate how to effectively combine stimulus control and sleep restriction. This includes practical guidance on adjusting bedtime and wake-up time based on a sleep diary and responding to difficulties the patient might encounter.",
    "Throughout the conversation, the Model must assess whether the patient understands the techniques and is committed to applying them. It should answer any questions and address concerns to ensure patient compliance and optimize therapy outcomes.",
    "The Model should offer examples and practical tips for dealing with common obstacles in implementing these therapies, such as what to do when one cannot fall asleep or how to manage the urge to stay in bed longer.",
    "The model should demonstrate the ability to adjust recommendations based on the patient’s feedback, such as difficulties in implementing strategies, unexpected wakefulness, or variations in sleep patterns observed in the sleep diary.",
    "An essential goal for the model is to guide the patient in monitoring their progress through a sleep diary and modifying sleep restrictions or stimulus control techniques based on this ongoing assessment."
]

goal_specific_prompts = {
    "Addressing Common Sleep Misconceptions": "Initiate the session by exploring common sleep myths with the patient, such as the universal need for 8 hours of sleep. Provide evidence-based explanations that individual sleep needs vary and discuss how adherence to such myths can heighten sleep-related anxiety. Educate the patient on identifying their unique sleep patterns and requirements, emphasizing the importance of listening to their own body rather than adhering to general misconceptions.",
    "Determining Initial Sleep Window": "Guide the patient on starting a sleep diary to meticulously track their bedtime, wake time, and total sleep duration for at least two weeks. Explain the calculation of sleep efficiency by dividing the total sleep time by the time spent in bed. Use this data to determine their initial sleep window, ensuring it closely aligns with the sleep duration logged in the diary, thus setting a foundation for effective sleep restriction therapy.",
    "Stimulus Control Explanation": "Educate the patient on stimulus control therapy by discussing its fundamental purpose: to reassociate the bed and bedroom with sleep and disassociate them from wakefulness and other activities. Explain the psychological mechanism behind stimulus control, detailing how these practices help diminish the conditioned arousal associated with the sleep environment.",
    "Rules of Stimulus Control": "Clearly articulate the specific rules of stimulus control: going to bed only when sleepy, avoiding all non-sleep activities in bed, such as eating, reading, or watching TV, leaving the bed if unable to sleep within 20 minutes, and maintaining a consistent wake-up time. Explain the rationale behind each rule and how it contributes to reconditioning the body's sleep-wake cycle.",
    "Sleep Restriction Therapy Description": "Describe sleep restriction therapy in detail, outlining its goal to limit the patient’s time in bed to match their actual sleep duration as recorded in the sleep diary. Discuss the concept of sleep drive and how restricting time in bed can intensify this drive, thereby consolidating sleep and reducing nighttime awakenings.",
    "Combining Stimulus Control and Sleep Restriction": "Discuss how to effectively combine stimulus control with sleep restriction. Guide the patient on adjusting their sleep window based on diary observations and discuss how to systematically delay bedtime or advance wake time to optimize sleep efficiency. Emphasize the iterative nature of this process and the need for regular adjustments based on the patient’s feedback and sleep diary data.",
    "Assess Patient Understanding and Commitment": "Continuously engage the patient to assess their understanding of and commitment to the sleep therapy techniques. Use questioning strategies to elicit detailed responses about their experiences, challenges, and any resistance they may feel towards the prescribed sleep practices. Reinforce the importance of their active participation and adjustment in response to therapy outcomes.",
    "Dealing with Common Obstacles": "Prepare the patient for common challenges they may encounter during the implementation of sleep therapies, such as the urge to stay in bed during wakeful periods or dealing with the anxiety of not sleeping. Provide specific, actionable strategies to overcome these obstacles, such as relaxation techniques, the use of a 'worry journal,' or engaging in quiet, non-stimulating activities out of bed.",
    "Adjusting Recommendations Based on Feedback": "Show flexibility in treatment by adapting recommendations based on the patient’s ongoing feedback. Discuss any new sleep disturbances or the patient’s experiences with the current strategies. Adjust the treatment plan dynamically, ensuring it remains aligned with the patient’s needs and sleep patterns as they evolve.",
    "Monitoring Progress and Modifying Techniques": "Instruct the patient on the importance of continually monitoring their progress through detailed sleep diaries. Review the entries together to identify patterns or shifts in sleep behavior. Discuss how to interpret these trends and make informed decisions on whether and how to modify sleep restrictions or control techniques, aiming for gradual improvement towards optimal sleep."
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