import os
import re
import json
from pprint import pprint
from datetime import datetime
from typing import Optional, Dict, Union, List
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
_default_model: str = "gpt-3.5-turbo" \
    if os.environ.get('GU_DEFAULT_MODEL') is None \
    else os.environ.get('GU_DEFAULT_MODEL')
_target_majors = [1]
_image_special_token = "{IMAGE_PLH}"
__version__: str = '0.3.3'


def _version_checker(target_majors: list) -> bool:
    """
    Check if the current version of OpenAI is compatible with the target major versions
    :param target_majors: List of target major versions
    :type target_majors: list

    :return: True if OpenAI version is larger or equal to the minimal version, False otherwise
    :rtype: bool
    """
    minimal_version = min(target_majors)
    # Check if compatible with openai package
    major = int(openai.__version__.split('.')[0])
    if major not in target_majors:
        warn(f"OpenAI({major}.x) has not been tested on this version yet. Please use with caution.")
    return int(major) >= minimal_version


def _get_response(message: list,
                  model_name: str = _default_model,
                  debug_log_path: str = None,
                  client: OpenAI = _client,
                  **kwargs) -> ChatCompletion:
    """
    Wrapper function to get response from OpenAI
    :param message: List of messages in the format defined in OpenAI API
    :type message: list

    :param model_name: Name for the OpenAI model to use, defaults to _default_model
    :type model_name: str(, optional)

    :param debug_log_path: Path to save the debug log, defaults to None
    :type debug_log_path: str(, optional)

    :param client: OpenAI client object, defaults to `_client` if not provided
    :type client: OpenAI(, optional)

    :param kwargs: Generation parameters (e.g. top_p, presence_penalty, etc.). Possible keys can be found at
        https://platform.openai.com/docs/api-reference/chat/create, defaults to _generation_config if not set
    :type kwargs: dict(, optional)

    :return: ChatCompletion object from OpenAI
    :rtype: ChatCompletion
    """
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
            # Dump the model to dict and add message
            result_dict = result.model_dump(exclude_unset=True)
            result_dict['message'] = message
            # Write the result to the file
            fp.write(json.dumps(result_dict, indent=2))
    return result


def _image_verification(image_url: str) -> bool:
    """
    Verify if the image provided has a valid format
    :param image_url: image in base64 format or a URL
    :type image_url: str

    :return: True if the image_url is valid, False otherwise
    :rtype: bool
    """
    # Check if the image URL is a base64 encoded image
    if image_url.startswith("data:image/jpeg;base64,"):
        return True
    # Check if the image URL is a valid http/https URL
    if re.match(r'^https?://', image_url):
        return True
    return False


def _vision_question_verification(question: Dict[str, Union[str, List[str]]]) -> bool:
    """
    Verify if the dictionary provided has a valid format
    :param question: Dictionary with 'text' and 'image' keys. 'text' is a string and 'image' is a list of strings.
    :type question: Dict[str, Union[str, List[str]]]

    :return: True if the dictionary is valid, False otherwise
    :rtype: bool
    """
    # Check if the dictionary has the required keys
    if 'text' not in question:
        return False
    if 'image' not in question:
        return False
    # Check if the 'text' key is a string
    if not isinstance(question['text'], str):
        return False
    # Check if the 'image' key is a list
    if not isinstance(question['image'], list):
        return False
    # Check if the 'image' list contains only strings and describe a valid image (url or base64)
    for img in question['image']:
        if not isinstance(img, str):
            return False
        if not _image_verification(img):
            return False
    # Check if the number of images is equal to the number of image placeholders in the text
    text_segs = question['text'].split(_image_special_token)
    return len(text_segs) == len(question['image']) + 1


def _append_question(context: list, question: Union[str, Dict[str, Union[str, List[str]]]]):
    """
    Append a question to the context list

    :param context: List where the question will be appended
    :type context: list

    :param question: Question to append, for question with image, provide a dictionary with 'text' and 'image' keys.
        Multiple images can be provided as a list of strings under 'image' key.
    :type question: Union[str, Dict[str, Union[str, List[str]]]]

    :return: None
    """
    # For questions with images
    if isinstance(question, dict):
        # Check if the dictionary has the required keys and format
        # Need: 'text' and 'image' keys. 'text' is a string and 'image' is a list of strings.
        # Number of images should be equal to the number of image placeholders in the text
        if not _vision_question_verification(question):
            raise ValueError("Invalid question dictionary, expected a dictionary with 'text' and 'image' keys.")
        # Prepare the content list for the current message
        content_list = []
        # Split the text by the image placeholder
        text_segs = question['text'].split(_image_special_token)
        # Append the text and image segments to the content list
        for i, seg in enumerate(text_segs):
            # If text segment is not empty, append it to the content list
            if seg:
                content_list.append({"type": "text", "text": seg})
            # In case if there is more text after the last image otherwise append the image
            if i < len(question['image']):
                content_list.append({"type": "image", "image_url": {"url": question['image'][i]}})
    else:
        # For questions without images (text-only)
        context.append({"role": "user", "content": question})


def _append_response(context: list, response: str):
    """
    Append a response from GPT to the context list

    :param context: List where the response will be appended
    :type context: list

    :param response: Response to append
    :type response: str

    :return: None
    """
    context.append({"role": "assistant", "content": response})


def _extract_raw_result(raw: ChatCompletion) -> str:
    """
    Extract the response from the ChatCompletion object

    :param raw: ChatCompletion object from OpenAI, contains the response
    :type raw: ChatCompletion

    :return: Extracted response from the ChatCompletion object
    :rtype: str
    """
    return raw.choices[0].message.content.strip()


def generate_explanation(init_context: str,
                         questions: list,
                         model_name: str = _default_model,
                         verbose: bool = False,
                         task_desc: str = None,
                         debug_log: str = None,
                         client: OpenAI = _client,
                         **kwargs) -> Dict[str, str]:
    """
    Given a list of questions for a given context, generate responses for each question step by step.
    :param init_context: Part of the first question, will be included as `initial_context` in the return dict
    :type init_context: str

    :param questions: List of questions to ask, each question can be a string or a dictionary with 'text' and 'image'
    :type questions: List[Union[str, Dict[str, List[str]]]]

    :param model_name: Name for the OpenAI model to use, defaults to _default_model
    :type model_name: str(, optional)

    :param verbose: Print chat history if set to True, defaults to False
    :type verbose: bool(, optional)

    :param task_desc: System prompt, defaults to None
    :type task_desc: str(, optional)

    :param debug_log: Path to save the debug log, defaults to None
    :type debug_log: str(, optional)

    :param client: OpenAI client object, defaults to `_client` if not provided
    :type client: OpenAI(, optional)

    :param kwargs: Generation parameters (e.g. top_p, presence_penalty, etc.). Possible keys can be found at
        https://platform.openai.com/docs/api-reference/chat/create, defaults to _generation_config if not set
    :type kwargs: dict(, optional)

    :return: Dictionary with responses for each question. Each key is a question and the value is the response to the
        question.
    :rtype: dict
    """
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
    # Record the response to the first question; if the first question is a dictionary, record the text part
    if isinstance(questions[0], dict):
        output_dict[questions[0]['text']] = curr_response
    else:
        output_dict[questions[0]] = curr_response

    # Loop through the rest questions
    for q in questions[1:]:
        _append_question(context, q)
        curr_response = _extract_raw_result(_get_response(context, model_name, debug_log, client, **kwargs))
        # Record the response to the question; if the question is a dictionary, record the text part
        if isinstance(q, dict):
            output_dict[q['text']] = curr_response
        else:
            output_dict[q] = curr_response
        _append_response(context, curr_response)

    # If verbose was set, print the entire chat history
    if verbose:
        pprint(context)
        print()
    return output_dict


def generate_explanation_wrapper(arg_dict: dict) -> Dict[str, str]:
    """
    Wrapper function of `generate_explanation` for `GPTMPHandler` to use in multiprocessing.

    :param arg_dict: Argument dictionary for `generate_explanation`
    :type arg_dict: dict

    :return: Dictionary with responses for each question. Each key is a question and the value is the response to the
        question.
    :rtype: dict
    """
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
    """
    Initialize the OpenAI utility with the given API key and generation configuration.
    :param api_key: OpenAI API Key
    :type api_key: str

    :param gen_conf: Generation configuration to override the default configuration, defaults to None
    :type gen_conf: dict(, optional)

    :param kwargs: Other arguments to pass to OpenAI client, defaults to None
    :type kwargs: dict

    :return: None
    """
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
