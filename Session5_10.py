import openai
import textwrap
import concurrent.futures
import threading
from Aspect_Critics_Eval_LLM import AspectCritic
from Goal_Accuracy_Eval_LLM import ConversationEvaluator
from Length_Eval import length_checker
from Stay_On_Track_Eval_LLM import evaluate_conversation_stay_on_track
from Topic_Adherence_Eval_LLM import TopicAdherenceEvaluator
from fasthtml.common import *
import asyncio
from fasthtml.common import Raw


tlink = Script(src="https://cdn.tailwindcss.com"),
dlink = Link(rel="stylesheet", href="https://cdn.jsdelivr.net/npm/daisyui@4.11.1/dist/full.min.css")
app = FastHTML(hdrs=(tlink, dlink, picolink), exts='ws')

openai.api_key = "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"

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
required_progress = 1.00
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
     "content": """You are a sleep therapy expert conducting the final consolidation session.

Response requirements:
- Ask ONE clear question at a time
- Keep responses to 2-3 sentences
- Focus on maintenance strategies
- Address specific challenges
- Guide toward independence

Session goals:
- Review key improvements
- Fine-tune strategies
- Build self-management skills
- Address remaining issues
- Plan for long-term success

Guidelines:
- Direct and focused responses
- No lengthy explanations
- Practical, actionable advice
- Work toward closure"""}
]

goal_names = [
    "Assessment of Treatment Readiness",
    "Detailed Case Conceptualization",
    "Case Conceptualization Form Simulation",
    "Understanding Comorbidities",
    "Therapeutic Component Selection",
    "Flexibility in Treatment Application",
    "Individual Tailoring",
    "Anticipation of Adherence Challenges",
    "Sequential Treatment Implementation",
    "Evaluation of Treatment Effectiveness"
]

goals = [
    "The therapy session should assess the patient’s readiness for change, determining their willingness to adopt new sleep behaviors. This is critical for effectively timing and implementing interventions.",
    "Beyond the use of a form, the therapy should involve a detailed conceptualization that considers factors like life stressors, environmental influences, and personal habits that affect sleep.",
    "The session should simulate the use of a case conceptualization form to systematically organize and guide the treatment process, considering sleep habits, comorbidities, and behavioral factors.",
    "The model should demonstrate understanding of the impact of various comorbidities on insomnia and incorporate this knowledge into therapy suggestions.",
    "The session should reflect thoughtful selection of CBT-I components that are most appropriate to the patient’s case, considering readiness for change, obstacles, and the patient’s sleep environment.",
    "The therapy should adjust the standard CBT-I protocol to fit the patient’s specific situation, such as modifying techniques for comorbid conditions like anxiety or depression.",
    "The treatment should be tailored to the individual patient’s symptoms and history, using patient-specific information to guide the conversation and interventions.",
    "The model should anticipate potential adherence challenges, discussing strategies to overcome these barriers, motivating the patient, and setting realistic expectations.",
    "It should effectively sequence treatment interventions, ensuring that each component builds on the previous one and corresponds to the patient’s evolving therapeutic needs.",
    "The model should evaluate the effectiveness of implemented treatments, incorporating patient feedback and adjusting the plan as necessary to ensure optimal outcomes."
]

goal_specific_prompts = {
    "Assessment of Treatment Readiness": "Begin the session by discussing the patient's past experiences and current perceptions about sleep and treatment. Assess their willingness and readiness to engage in therapy by exploring their motivations, hesitations, and previous attempts at managing their sleep issues. Explain how their commitment and active participation are crucial for the success of the therapy. Provide clear examples of how behavioral changes can positively impact sleep and ask for their thoughts on making such changes.",
    "Detailed Case Conceptualization": "Use a structured approach to delve into the patient's personal history, sleep patterns, lifestyle choices, and environmental factors that may be influencing their sleep. This should include a discussion of stress levels, work-life balance, bedtime routines, and any other relevant psychosocial factors. Guide the patient through a detailed mapping of these elements and their interconnections to build a comprehensive understanding of their sleep disturbances. This conceptualization helps in identifying specific targets for intervention.",
    "Case Conceptualization Form Simulation": "Introduce and collaboratively fill out a case conceptualization form, explaining each section such as sleep habits, comorbidities, emotional stressors, and behavioral factors. Engage the patient in a step-by-step discussion, encouraging them to provide input and reflect on how each aspect of their life contributes to their sleep issues. This exercise aims to make the patient an active participant in their treatment planning, enhancing their understanding and ownership of the therapeutic process.",
    "Understanding Comorbidities": "Discuss in detail the comorbidities that might be impacting the patient's insomnia, such as anxiety, depression, chronic pain, or other medical conditions. Explore how these comorbidities interact with their sleep problems and how treating insomnia might also help manage these conditions. Explain the bidirectional nature of sleep and health issues to help the patient see the holistic importance of sleep improvement.",
    "Therapeutic Component Selection": "Present and discuss various components of Cognitive Behavioral Therapy for Insomnia (CBT-I), such as stimulus control, sleep restriction, and cognitive restructuring. Explain how each component works and its benefits, and use the patient’s specific sleep issues and preferences to decide together which components to prioritize in the treatment plan. This personalized selection process helps ensure that the therapy is tailored to the patient's unique needs.",
    "Flexibility in Treatment Application": "Prepare to adapt therapy techniques based on ongoing assessments and the patient’s evolving needs. Discuss potential adjustments like modifying sleep schedules or introducing relaxation techniques, explaining why and how these changes might help. This approach emphasizes flexibility and responsiveness, which are key in managing complex or changing sleep issues.",
    "Individual Tailoring": "Focus on tailoring the therapy to fit the patient’s specific circumstances, including their daily schedule, personal life stresses, and sleep environment. Discuss how personalized interventions, such as adjusting the bedroom environment or customizing sleep/wake times, can make therapy more effective. Encourage the patient to give feedback on what feels most relevant and manageable for them.",
    "Anticipation of Adherence Challenges": "Proactively discuss potential challenges that might impede adherence to the treatment plan, such as inconsistent schedules, motivation dips, or external stressors. Explore strategies to overcome these barriers, such as setting reminders, engaging support from family members, or adjusting goals to be more achievable. This discussion helps prepare the patient to handle difficulties throughout the treatment process.",
    "Sequential Treatment Implementation": "Outline the planned sequence of therapeutic interventions, explaining how each step builds on the previous one to progressively improve sleep. Discuss the rationale behind the sequence, such as starting with sleep hygiene improvements and gradually introducing more intensive interventions like sleep restriction therapy. This methodical approach helps the patient understand the progression and purpose of each phase of therapy.",
    "Evaluation of Treatment Effectiveness": "Regularly evaluate the effectiveness of the treatment through discussions with the patient about their sleep diary, symptom changes, and overall satisfaction with the progress. Use this feedback to make informed adjustments to the treatment plan, ensuring that it remains aligned with the patient's needs and goals. This ongoing evaluation fosters a dynamic and responsive therapeutic environment."
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