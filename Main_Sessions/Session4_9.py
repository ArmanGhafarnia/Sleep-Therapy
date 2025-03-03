import openai
import textwrap
import concurrent.futures
import threading
from LLM_Based_Evaluators.Aspect_Critics_Eval_LLM import AspectCritic
from LLM_Based_Evaluators.Goal_Accuracy_Eval_LLM import ConversationEvaluator
from Non_LLM_Evaluators.Length_Eval import length_checker
from LLM_Based_Evaluators.Stay_On_Track_Eval_LLM import evaluate_conversation_stay_on_track
from LLM_Based_Evaluators.Topic_Adherence_Eval_LLM import TopicAdherenceEvaluator
from fasthtml.common import *
import asyncio
from fasthtml.common import Raw


tlink = Script(src="https://cdn.tailwindcss.com"),
dlink = Link(rel="stylesheet", href="https://cdn.jsdelivr.net/npm/daisyui@4.11.1/dist/full.min.css")
app = FastHTML(hdrs=(tlink, dlink, picolink), exts='ws')

openai.api_key = "your-api-key-here"

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
def ChatMessage(msg_idx, **kwargs):
    msg = messages[msg_idx]
    if msg['role'] == 'system':
        return None
    bubble_class = "chat-bubble bg-blue-600 text-white" if msg[
                                                               'role'] == 'user' else "chat-bubble bg-purple-600 text-white"
    chat_class = "chat-end" if msg['role'] == 'user' else "chat-start"

    header_class = "chat-header mr-2 mb-1" if msg['role'] == 'user' else "chat-header ml-2 mb-1"

    content_div = Div(
        msg['content'] if msg['role'] == 'user' else '',
        id=f"chat-content-{msg_idx}",
        cls=f"chat-bubble {bubble_class}",
        data_content=msg['content'] if msg['role'] == 'assistant' else None,
        data_streaming="true" if msg['role'] == 'assistant' else None,
    )

    display_name = "You" if msg['role'] == 'user' else "Therapist"

    return Div(
        Div(display_name, cls=header_class),
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
        onkeydown="""
            if(event.key === 'Enter') {
                setTimeout(() => {
                    const input = document.getElementById('msg-input');
                    input.value = '';  // Clear the input
                    input.focus();     // Focus the input
                    input.setSelectionRange(0, 0);  // Set cursor to start
                }, 10);
            }
        """,
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
            StarBackground(),
            Div(
                Div("Sleep Therapy",
                    cls="text-3xl font-bold text-purple-400 text-center"
                    ),
                cls="w-full fixed top-8 z-20"
            ),
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

                        // Add function to reset input state
                        function resetInput() {
                            const input = document.getElementById('msg-input');
                            if (input) {
                                input.value = '';
                                input.focus();
                                input.setSelectionRange(0, 0);
                            }
                        }

                        const observer = new MutationObserver((mutations) => {
                            chatList.scrollTop = chatList.scrollHeight;
                            resetInput();  // Reset input after any chat updates

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
                                inputBox.setSelectionRange(0, 0);  // Ensure cursor is at start
                            }
                        }
                    });
                ''')
    )
    return page


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


messages = [
    {"role": "system",
     "content": """You are a sleep therapy expert focusing on cognitive behavioral techniques for this fourth session.

Communication requirements:
- Ask ONE clear question at a time
- Focus on thoughts and beliefs about sleep
- Keep responses to 3-4 sentences
- Build on previous session progress
- Address one cognitive pattern at a time

Session objectives:
- Identify unhelpful thoughts about sleep
- Challenge and modify sleep-related worries
- Reduce bedtime mental arousal
- Develop coping strategies for racing thoughts

Additional guidelines:
- Direct and precise responses
- Focus on cognitive aspects of sleep
- Connect thoughts to sleep behavior
- Remove redundant acknowledgments"""}
]

goal_names = [
    "Identifying Maladaptive Cognitions",
    "Challenging and Modifying Cognitions",
    "Reducing Psychological Arousal",
    "Promoting Adherence to Behavioral Changes",
    "Incorporate Behavioral Experiments to Test Beliefs and Behaviors",
    "Develop Coping and Problem-Solving Skills for Sleep"
]

goals = [
    "The model should help patients identify thoughts and beliefs about sleep that are unhelpful or detrimental. This includes recognizing worries about sleep, like predicting sleep difficulty or fearing the consequences of poor sleep, which heighten psychological arousal and disrupt sleep.",
    "The model should assist in evaluating and responding to these maladaptive cognitions constructively. Techniques like Socratic questioning, thought records, and cognitive restructuring are used to challenge the validity and utility of these beliefs.",
    "The model should aid in reducing psychological arousal that occurs at bedtime or during awakenings at night, which is often linked to sleep-related cognitions. Strategies include cognitive restructuring and calming techniques.",
    "Cognitive therapy should work in tandem with behavioral interventions in CBT-I (Cognitive Behavioral Therapy for Insomnia) to promote better adherence. For instance, addressing thoughts that hinder compliance with strategies like stimulus control (getting out of bed when not sleeping) and sleep restriction.",
    "Behavioral experiments are a key component of cognitive therapy for insomnia, where patients test the validity of their beliefs or the utility of different behaviors in a controlled, experimental manner. This can involve, for example, deliberately altering sleep patterns to observe effects contrary to their dysfunctional beliefs.",
    "The model should guide patients in developing skills to cope with and solve sleep-related problems independently, enhancing their resilience and ability to manage insomnia without therapist intervention."
]

goal_specific_prompts = {
    "Identifying Maladaptive Cognitions": "Encourage the patient to articulate specific thoughts and beliefs about sleep that may be causing distress or hindering sleep quality. Ask them to reflect on how these thoughts manifest during both day and night. For example, prompt the patient to describe scenarios where worries about insufficient sleep lead to stress or altered behavior during the day. Explore how these cognitions contribute to a heightened state of psychological arousal at bedtime, impacting their ability to initiate and maintain sleep.",
    "Challenging and Modifying Cognitions": "Facilitate a cognitive restructuring session by systematically addressing and challenging the patient’s negative beliefs about sleep. Utilize Socratic questioning to dissect the logic behind beliefs such as 'I can’t function without eight hours of sleep' or 'If I don’t sleep well tonight, I will fail tomorrow.' Introduce thought records as a tool for monitoring these beliefs and their consequences, guiding the patient through the process of identifying, challenging, and replacing these cognitions with more balanced and realistic thoughts.",
    "Reducing Psychological Arousal": "Guide the patient in implementing relaxation techniques that can be practiced at bedtime to manage and reduce psychological arousal. These might include guided imagery, deep breathing exercises, or progressive muscle relaxation. Discuss the physiological and psychological processes involved in these techniques, emphasizing their role in mitigating the hyperarousal state often observed in insomnia. Encourage routine practice and discuss the patient's experiences and challenges with these techniques during subsequent sessions.",
    "Promoting Adherence to Behavioral Changes": "Conduct a detailed exploration of the patient's experiences with behavioral treatment strategies for insomnia, such as stimulus control and sleep restriction. Address any cognitive barriers to adherence, such as beliefs about the necessity of staying in bed while awake. Use motivational interviewing to enhance motivation and commitment to these behavioral changes, focusing on resolving ambivalence and reinforcing the patient’s ability to implement these strategies effectively.",
    "Incorporate Behavioral Experiments to Test Beliefs and Behaviors": "Design and implement behavioral experiments that challenge the patient’s maladaptive beliefs about sleep. For instance, if a patient believes that 'lying in bed longer helps me get more sleep,' suggest altering their time in bed to test this belief. Guide the patient in planning the experiment, predicting outcomes, and reviewing the actual results, thereby facilitating a practical understanding of how specific behaviors affect sleep.",
    "Develop Coping and Problem-Solving Skills for Sleep": "Teach and develop specific problem-solving skills tailored to managing sleep-related issues. Focus on equipping the patient with strategies to address common nocturnal awakenings or prolonged sleep latency. Techniques could include deciding on activities to engage in out of bed that are conducive to sleepiness or methods to calm the mind when unable to sleep. Emphasize the development of a proactive stance towards these issues, rather than reactive distress."
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

initialize_evaluators_in_background(evaluators)
initialize_goal_progress(len(goals))
current_goal_index = 0


@app.ws('/wscon')
async def ws(msg: str, send):
    global current_goal_index, messages
    messages.append({"role": "user", "content": msg.rstrip()})
    swap = 'beforeend'

    print(f"\n{GREEN}You:{RESET} {msg}")
    await send(Div(ChatMessage(len(messages) - 1), hx_swap_oob=swap, id="chatlist"))
    await send(ChatInput())

    therapist_message = await chat_with_gpt(messages)

    print(f"\n{YELLOW}Therapist:{RESET}")
    for paragraph in therapist_message.split('\n'):
        print(textwrap.fill(paragraph, width=70))

    messages.append({"role": "assistant", "content": therapist_message})
    await send(Div(ChatMessage(len(messages) - 1), hx_swap_oob=swap, id="chatlist"))

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