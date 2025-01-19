from fasthtml.common import *
import asyncio

# Set up the app, including daisyui and tailwind for the chat component
tlink = Script(src="https://cdn.tailwindcss.com"),
dlink = Link(rel="stylesheet", href="https://cdn.jsdelivr.net/npm/daisyui@4.11.1/dist/full.min.css")
app = FastHTML(hdrs=(tlink, dlink, picolink), exts='ws')

# List to store messages
messages = []


# Chat message component (renders a chat bubble)
def ChatMessage(msg_idx, **kwargs):
    msg = messages[msg_idx]
    bubble_class = "chat-bubble-primary" if msg['role'] == 'user' else 'chat-bubble-secondary'
    chat_class = "chat-end" if msg['role'] == 'user' else 'chat-start'
    return Div(Div(msg['role'], cls="chat-header"),
               Div(msg['content'],
                   id=f"chat-content-{msg_idx}",  # Target if updating the content
                   cls=f"chat-bubble {bubble_class}"),
               id=f"chat-message-{msg_idx}",  # Target if replacing the whole message
               cls=f"chat {chat_class}", **kwargs)


# The input field for the user message
def ChatInput():
    return Input(type="text", name='msg', id='msg-input',
                 placeholder="Type a message",
                 cls="input input-bordered w-full", hx_swap_oob='true')


# The main screen
@app.route("/")
def get():
    page = Body(H1('Chat Display Demo'),
                Div(*[ChatMessage(msg_idx) for msg_idx, msg in enumerate(messages)],
                    id="chatlist", cls="chat-box h-[73vh] overflow-y-auto"),
                Form(Group(ChatInput(), Button("Send", cls="btn btn-primary")),
                     ws_send=True, hx_ext="ws", ws_connect="/wscon",
                     cls="flex space-x-2 mt-2"),
                cls="p-4 max-w-lg mx-auto")
    return Title('Chat Display Demo'), page


@app.ws('/wscon')
async def ws(msg: str, send):
    # Add user message to messages list
    messages.append({"role": "user", "content": msg.rstrip()})
    swap = 'beforeend'

    # Send the user message to the UI
    await send(Div(ChatMessage(len(messages) - 1), hx_swap_oob=swap, id="chatlist"))

    # Clear the input field
    await send(ChatInput())

    # Add static response
    messages.append({"role": "assistant", "content": "This is a static response message!"})

    # Send the static response to the UI
    await send(Div(ChatMessage(len(messages) - 1), hx_swap_oob=swap, id="chatlist"))


serve()