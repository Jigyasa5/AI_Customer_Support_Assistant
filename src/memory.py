chat_history = []


def add_to_history(
    user_query,
    assistant_answer
):

    chat_history.append({

        "user": user_query,

        "assistant": assistant_answer
    })


def get_history():

    return chat_history


def format_history():

    history_text = ""

    for message in chat_history:

        history_text += (
            f"User: {message['user']}\n"
            f"Assistant: {message['assistant']}\n"
        )

    return history_text


def clear_history():

    chat_history.clear()