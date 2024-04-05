import os
from pprint import pprint
from datetime import datetime
from typing import Optional, Dict
from warnings import warn

import openai
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion

# Housekeeping Variables
_generation_config = {"temperature": 0.9,
                      "max_tokens": 1024,
                      "top_p": 1.0,
                      "frequency_penalty": 0.0,
                      "presence_penalty": 0.0}
_client: Optional[OpenAI] = None
_target_majors = [1]
__version__: str = '0.2.2'


def _version_checker(target_majors: list) -> bool:
    minimal_version = min(target_majors)
    # Check if compatible with openai package
    major = int(openai.__version__.split('.')[0])
    if major not in target_majors:
        warn(f"OpenAI({major}.x) has not been tested on this version yet. Please use with caution.")
    return int(major) >= minimal_version


def _get_response(message: list,
                  model_name: str = "gpt-3.5-turbo",
                  debug_log_path: str = None,
                  client: OpenAI = _client,
                  **kwargs) -> ChatCompletion:
    # Check is the utility is properly setup
    assert client is not None, "need to run init() first or provide client."
    # Prepare generation arguments
    args = {}
    for k in _generation_config:
        args[k] = kwargs[k] if k in kwargs else _generation_config[k]
    # Get response from OpenAI
    result = client.chat.completions.create(
        model=model_name,
        messages=message,
        **args
    )
    # Save output if debug path was given
    if debug_log_path is not None:
        # Create if the directory does not exist
        os.makedirs(debug_log_path, exist_ok=True)
        # Generate output file name
        output_file_name = datetime.now().strftime(model_name + "_log_%Y_%m_%d_%H_%M_%S_%f.json")
        with open(os.path.join(debug_log_path, output_file_name), "w") as fp:
            fp.write(result.model_dump_json(indent=2, exclude_unset=True))
    return result


def _append_question(context: list, question: str):
    # Append question to the existing dictionary
    context.append({"role": "user", "content": question})


def _append_response(context: list, response: str):
    # Append responds to the existing dictionary
    context.append({"role": "assistant", "content": response})


def _extract_raw_result(raw: ChatCompletion) -> str:
    # Return the striped text from the completion
    return raw.choices[0].message.content.strip()


def generate_explanation(init_context: str,
                         questions: list,
                         model_name: str = "gpt-3.5-turbo",
                         verbose: bool = False,
                         task_desc: str = None,
                         debug_log: str = None,
                         client: OpenAI = _client,
                         **kwargs) -> Dict[str, str]:
    # Check is the utility is properly setup
    assert client is not None, "need to run init() first or provide client."

    # Create list to hold chat histories
    context = list()

    # Add system prompt if provided
    if task_desc is not None:
        context.append({"role": "system", "content": task_desc})

    # Prepare the list for the first question
    _append_question(context, '\n\n'.join([init_context, questions[0]]).strip())

    # Create return dict object and record initial context
    output_dict = dict()
    output_dict['initial_context'] = init_context

    # Query the first question and append to the history and return dict
    curr_response = _extract_raw_result(_get_response(context, model_name, debug_log, client, **kwargs))
    _append_response(context, curr_response)
    output_dict[questions[0]] = curr_response

    # Loop through the rest questions
    for q in questions[1:]:
        _append_question(context, q)
        curr_response = _extract_raw_result(_get_response(context, model_name, debug_log, client, **kwargs))
        output_dict[q] = curr_response
        _append_response(context, curr_response)

    # If verbose was set, print the entire chat history
    if verbose:
        pprint(context)
        print()
    return output_dict


def generate_explanation_wrapper(arg_dict: dict) -> Dict[str, str]:
    # Initialize OpenAI Object
    client = OpenAI(**arg_dict['openai_args'])
    del arg_dict['openai_args']
    arg_dict['client'] = client
    try:
        return generate_explanation(**arg_dict)
    except Exception as e:
        print(e)
        return dict()


def init(api_key: str, gen_conf: dict = None, **kwargs):
    global _generation_config
    global _client

    # Check version compatibility
    assert _version_checker(_target_majors), \
        f"openai({openai.__version__}) is no longer supported, please upgrade first."
    # Initialize OpenAI Object
    if not kwargs:
        _client = OpenAI(api_key=api_key, max_retries=5)
    else:
        _client = OpenAI(api_key=api_key, **kwargs)
    # Override configuration files if needed
    if gen_conf:
        for k in list(gen_conf.keys()):
            if k in _generation_config:
                _generation_config[k] = gen_conf[k]
