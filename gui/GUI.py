import gradio as gr
import pandas as pd
import uuid
import random
import time

from openai import OpenAI

from util.ConfigLoader import ConfigLoader
from communicator.LLMCommunicator import LLMCommunicator

openai_client = None

const_values = {
    "you": "YOU:", 
    "user_msg_hint": "Press \"Enter\" to send message, press \"Shift\" + \"Enter\" to create a new line.", 
    "new": "New", 
    "clear": "Clear", 
    "send": "Send"
}

theme = gr.themes.Default(
    primary_hue="blue",
    secondary_hue="blue",
)

def init_session(session):
    if len(session) == 0:
        session.append({
            'counter': 0, 
            'selected_session': 0, 
            'chat_histories': [[]],
            'busy': False
        })
        
    return session

def handle_new_session(session, df, history):
    config = ConfigLoader().get()
    init_session(session)
    
    # handle selected row
    session[0]['selected_session'] = 0
    
    # global counter
    # counter += 1
    session[0]['counter'] += 1
    counter = session[0]['counter']
    
    # Check if DataFrame is empty or only contains one empty row
    if df.empty or (len(df) == 1 and df.iloc[0, 0] == ""):
        df.loc[0, "session_name"] = f"new chat {counter}"
        
        # update session
        session[0]['chat_histories'] = [[]]
    else:
        # Current logic: add a new row
        df.index = df.index + 1
        df.loc[0, "session_name"] = f"new chat {counter}"
        df = df.sort_index()
        
        # update session
        session[0]['chat_histories'] = [[]] + session[0]['chat_histories']
        
    if config["gui_log"]: print(f"chat_histories = {session[0]['chat_histories']}")
        
    return [session, df, []]

def handle_clear_selected_row(session, df, history):
    config = ConfigLoader().get()
    init_session(session)
    if config["gui_log"]: print(f"session[0] = {session[0]}")
    
    selected_index = session[0]['selected_session']
    if config["gui_log"]: print(f"selected_index = {selected_index}")
    df = df.drop([selected_index])
    df = df.reset_index(drop=True)
    session[0]['selected_session'] = -1
    
    # update session
    session[0]['chat_histories'].pop(selected_index)
    if config["gui_log"]: print(f"chat_histories = {session[0]['chat_histories']}")
        
    # set select session = 0, if there no other sessions, create a new one
    if len(df) == 0:
        return handle_new_session(session, df, history)
    else:
        [session, history] = _handle_select_row(session, df, {
            'value': 'NA',
            'index': [0, 0]
        }, history)
    
        return [session, df, history]

def handle_select_row(session, df, history, selectData: gr.SelectData):
    init_session(session)
    
    if session[0]['busy']:
        return session, history
    else:
        return _handle_select_row(session, df, {
            'value': selectData.value,
            'index': selectData.index
        }, history)

def _handle_select_row(session, df, selectData, history):
    config = ConfigLoader().get()
    init_session(session)
    
    if config["gui_log"]: print(f"You selected {selectData['value']} at {selectData['index']}")
    if config["gui_log"]: print(f"session[0] = {session[0]}")
    session[0]['selected_session'] = selectData['index'][0]
    
    history = session[0]['chat_histories'][selectData['index'][0]]
    
    return [session, history]

def update_selected_row_with_new_session(session, df, history):
    config = ConfigLoader().get()
    init_session(session)
    
    selected_session = session[0]['selected_session']
    if config["gui_log"]: print(f"update_selected_row_with_new_session called, selected_session = {selected_session}")
    if config["gui_log"]: print(f"latest user msg = {history[-1][0]}")
        
    if len(history) == 1:
        df.loc[0, "session_name"] = history[-1][0]
    
    session[0]['selected_session'] = selected_session
    return [session, df]

def update_chat_history(session, history):
    config = ConfigLoader().get()
    init_session(session)
    
    selected_index = session[0]['selected_session']
    session[0]['chat_histories'][selected_index] = history
    
    if config["gui_log"]: print(f"chat_histories = {session[0]['chat_histories']}")
    
def build_messages(history):
    messages = []
    
    for i in range(len(history)):
        messages.append({"role": "user", "content": history[i][0]})
        
        if i < len(history) - 1:
            messages.append({"role": "assistant", "content": history[i][1]})
            
    return messages

def handle_user_msg(user_message, history):
    return "", history + [[user_message, None]]
    
def handle_bot_msg(history):
    global openai_client
    
    config = ConfigLoader().get()
    model_config = config['model_config']
    default_completion_config = model_config['default_completion_config']
    stream_batch_size = config['stream_batch_size']
    
    messages = build_messages(history)
    if config["gui_log"]: print(f"messages = {messages}")
    
    # res = openai_client.chat.completions.create(
    #     model='LOCAL-LLM-SERVER',
    #     messages=messages,
    #     stream=True, 
    #     temperature=0.0
    # )
    llm = LLMCommunicator.get()
    
    # stream mode
    # response_stream = res
    
    default_max_tokens = default_completion_config['max_tokens']
    default_temperature = default_completion_config['temperature']
    default_repeat_penalty = default_completion_config['repeat_penalty']
    default_echo = default_completion_config['echo']
    
    # print(f"default_max_tokens = {default_max_tokens}")
    # print(f"default_temperature = {default_temperature}")
    # print(f"default_repeat_penalty = {default_repeat_penalty}")
    # print(f"default_echo = {default_echo}")
    
    response_stream = llm.complete_messages(
        messages, 
        max_tokens=default_max_tokens,
        temperature=default_temperature,
        repeat_penalty=default_repeat_penalty,
        echo=default_echo,
        stream=True
    )
    
    def generate(batch_size=stream_batch_size):
        prev_item = next(response_stream, None)  # Get the first item
        bulk_text = ""
        current_batch_size = 0
        first_item = True

        for item in response_stream:
            # option 1: directly yeild the current value
            # yield get_response_json(prev_item.choices[0].delta.content)

            # option 2: yield values in bulk
            # bulk_text += prev_item.choices[0].delta.content
            bulk_text += prev_item['choices'][0]['text']
            current_batch_size += 1

            if current_batch_size == batch_size:
                if first_item:
                    yield bulk_text
                    first_item = False
                else:
                    yield bulk_text

                bulk_text = ""
                current_batch_size = 0

            prev_item = item

        # Handle the last item
        if prev_item is not None:
            # option 1: directly yeild the current value
            if config["gui_log"]: print(f"token[END] = {item['choices'][0]['delta']['content']}")

            # option 2: yield values in bulk
            bulk_text += prev_item['choices'][0]['text']
            yield bulk_text
    
    completion = generate()
    history[-1][1] = ""
    for chunk in completion:
        history[-1][1] += chunk
        yield history
        
def freeze_ui(session):
    init_session(session)
    session[0]['busy'] = True
    return [
        session, 
        gr.Button(interactive=False), 
        gr.Button(interactive=False), 
        gr.Button(interactive=False)
    ]

def unfreeze_ui(session):
    init_session(session)
    session[0]['busy'] = False
    return [
        session, 
        gr.Button(interactive=True), 
        gr.Button(interactive=True), 
        gr.Button(interactive=True)
    ]
        

css = """
#left-panel {
    border-right: 1px solid #efefef;
    padding-right: 16px;
}

#right-panel { 
}

#df_sessions table thead {
    display: none;
}

#df_sessions table td span {
}

.full_height {
    height: calc(100vh - 32px);
}

.centered h1 {
    text-align: center;
    display:block;
}

.small_button {
    min-width: min(80px, 100%);
}

.chat_area {
    border: 0px;
    border-style: none !important;
    box-shadow: none !important;
}

.chat_area .message-bubble-border {
    border-radius: var(--radius-lg);
}

.chat_area .message-bubble-border.user {
    border-bottom-right-radius: 0px;
}

.chat_area .message-bubble-border.bot {
    border-bottom-left-radius: 0px;
}

.user_input_msg textarea {
    height: 80px !important;
    resize: none;
}

footer {
    display:none !important
}
"""

def start_gradio(share=False):
    config = ConfigLoader().get()
    
    global openai_client
    openai_client = OpenAI(
        api_key=config['llm_secret'],
        base_url=f"http://127.0.0.1:{config['port']}/v1"
    )
    
    with gr.Blocks(css=css, theme=theme) as demo:
        session = gr.State([])
        with gr.Row():
            with gr.Column(elem_id="left-panel", elem_classes=["full_height"], scale=0):
                lbl_logo = gr.Markdown("""
                ![LOCAL-LLM-SERVER](https://xiaohan-tian.github.io/res/lls-banner.png "LOCAL-LLM-SERVER")
                """, elem_classes=["centered"])
                # btn_settings = gr.Button("Settings")
                df_sessions = gr.Dataframe(
                    elem_id="df_sessions",
                    headers=["session_name"],
                    datatype=["str"], 
                    col_count=(1, "fixed"),
                    interactive=False, 
                    height=300
                )
                with gr.Row():
                    btn_new_session = gr.Button(const_values["new"], elem_classes=["small_button"])
                    btn_clear = gr.Button(const_values["clear"], elem_classes=["small_button"])
            with gr.Column(elem_id="right-panel", scale=2):
                cb_main = gr.Chatbot(height='calc(100vh - 237px)', elem_classes=["chat_area"], show_label=False)
                tb_user_msg = gr.Textbox(elem_id="user_msg", elem_classes=["user_input_msg"], label=const_values["you"], placeholder=const_values["user_msg_hint"]) # lines=3, max_lines=3
                btn_send = gr.Button(const_values["send"], variant="primary")

        btn_new_session.click(handle_new_session, inputs=[session, df_sessions, cb_main], outputs=[session, df_sessions, cb_main])
        btn_clear.click(handle_clear_selected_row, inputs=[session, df_sessions, cb_main], outputs=[session, df_sessions, cb_main])
        df_sessions.select(handle_select_row, inputs=[session, df_sessions, cb_main], outputs=[session, cb_main])
        tb_user_msg.submit(
            freeze_ui, [session], [session, btn_new_session, btn_clear, btn_send]
        ).then(
            handle_user_msg, [tb_user_msg, cb_main], [tb_user_msg, cb_main], queue=False
        ).then(
            update_selected_row_with_new_session, [session, df_sessions, cb_main], [session, df_sessions]
        ).then(
            handle_bot_msg, cb_main, cb_main
        ).then(
            update_chat_history, [session, cb_main]
        ).then(
            unfreeze_ui, [session], [session, btn_new_session, btn_clear, btn_send]
        )
        btn_send.click(
            freeze_ui, [session], [session, btn_new_session, btn_clear, btn_send]
        ).then(
            handle_user_msg, [tb_user_msg, cb_main], [tb_user_msg, cb_main], queue=False
        ).then(
            update_selected_row_with_new_session, [session, df_sessions, cb_main], [session, df_sessions]
        ).then(
            handle_bot_msg, cb_main, cb_main
        ).then(
            update_chat_history, [session, cb_main]
        ).then(
            unfreeze_ui, [session], [session, btn_new_session, btn_clear, btn_send]
        )

    demo.launch(share=share, prevent_thread_lock=True)

    
