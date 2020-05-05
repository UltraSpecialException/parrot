import json
import os
import os.path as osp
import argparse
from typing import List, Dict


decode = {
    "â\x80\x99": "'"
}


def open_json(path: str) -> Dict[str, str]:
    """
    Shortcut to open a json instead of having to call json.load(open(path))
    every time
    """
    assert isinstance(path, str), "Argument <path> need to be of type string"
    with open(path) as data:
        return json.load(data)


def extract_thread(paths: List[str]) -> List[Dict[str, str]]:
    """
    Extract the message thread if the conversation is a regular one. Raise an
    error if it isn't.
    """
    thread = []
    for path in paths:
        data_file = open_json(path)
        if data_file["thread_type"] == "Regular":
            thread.extend(data_file["messages"])

        else:
            raise ValueError(f"Path {path} contains non-regular conversation.")

    return thread


def extract_messages(thread: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Concatenate sequentially sent messages by the same user and assemble the
    messages into a chronologically ordered list of messages.
    """
    if len(thread) <= 1:
        return []

    # the messages provided by Facebook are ordered chronologically where
    # the first message is the most recently sent message

    # since this is the case, we reverse the list to get the first message sent
    # at the beginning of the list
    thread.reverse()

    data = []

    # the first message is the "you're not connected..." so we skip it
    current_sender = thread[1]["sender_name"]
    current_message = {
        "sender_name": current_sender,
        "message": []
    }
    for i, message in enumerate(thread[1:]):
        # we only deal with normal text messages, stickers, media, photos, etc.
        # are their own problems, maybe another time...
        if "content" not in message.keys():
            continue

        # we want to concatenate the messages of the same sender that are sent
        # consecutively together, so once we detect that the sender of this
        # current message <message> is not the same as the current sender,
        # we clear out the <current_message> variable and update the
        # <current_sender> variable accordingly
        if message["sender_name"] != current_sender:
            current_message["message"] = " ".join(current_message["message"])
            for code in decode:
                encoded = current_message["message"]
                current_message["message"] = encoded.replace(code, decode[code])

            pre_unicode_decoded = ascii(current_message["message"])[1:-1]
            decoded = str.encode(pre_unicode_decoded).decode("unicode-escape")
            current_message["message"] = decoded.encode("latin1").decode("utf8")

            data.append(current_message)
            current_sender = message["sender_name"]
            current_message = {
                "sender_name": current_sender,
                "message": [message["content"]]
            }

        # if the sender is the same then we continue building up the message
        else:
            current_message["message"].append(message["content"])

    return data


def assemble_data(data: List[Dict[str, str]], target_sender: str) -> \
        List[Dict[str, Dict[str, str]]]:
    """
    Assemble data into pairs of input message and expected response where the
    messages sent by <target_sender> are the expected responses.
    """
    start = 0
    if data[0]["sender_name"] == target_sender:
        start = 1

    end = len(data)
    if len(data[start:]) % 2:
        end = len(data) - 1

    assembled_data = []

    for i in range(start, end, 2):
        input_message = data[i]
        expected_response = data[i + 1]
        assembled_data.append(
            {
                "input": input_message,
                "target": expected_response
            }
        )

    return assembled_data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("-dir", "-d",
                        help="Directory extracted from the FB messenger data "
                             "zip file",
                        type=str,
                        required=True)

    parser.add_argument("-name", "-n",
                        help="Name of the target sender",
                        type=str,
                        required=True)

    parser.add_argument("-save_dir", "--save_dir", "-s",
                        help="Directory to save all the output data",
                        default=".")

    args = parser.parse_args()

    dir_path = osp.abspath(args.dir)
    print(f"[{__file__}]:", f"Directory given: {dir_path}")

    inbox_path = dir_path + "/inbox"
    for subdir in os.listdir(inbox_path):
        print(f"[{__file__}]:", f"Parsing: {subdir}")
        messages_path = inbox_path + f"/{subdir}"
        data_file_paths = []
        for filename in os.listdir(messages_path):
            if filename.endswith(".json"):
                data_file_paths.append(messages_path + f"/{filename}")

        try:
            message_thread = extract_thread(data_file_paths)
            messages = extract_messages(message_thread)

            if messages:
                data = assemble_data(messages, args.name)

                file_name = f"{subdir[:subdir.rfind('_')]}.json"
                save_path = f"{args.save_dir}/{file_name}"
                with open(save_path, "w+") as save_file:
                    json.dump(data, save_file)

        except ValueError:
            continue
