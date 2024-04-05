from tqdm.contrib.concurrent import process_map
from typing import List, Dict
from warnings import warn
from os import cpu_count, environ
from . import gpt_util
import inspect


class GPTMPHandler:
    def __init__(self, api_key: str = environ.get('OPENAI_API_KEY'),
                 num_worker: int = min(32, cpu_count() + 4),
                 gen_conf: dict = None,
                 max_retries: int = 2,
                 **kwargs):
        """
        Initialize the GPT Multi-Processing Handler

        :param api_key: OpenAI API Key as a string, defaults to environment variable OPENAI_API_KEY
        :type api_key: str(, optional)

        :param num_worker: Number of worker threads to use, defaults to CPU count + 4 or 32, whichever is smaller
        :type num_worker: int(, optional)

        :param gen_conf: Generation parameters (e.g. top_p, presence_penalty, etc.). Possible keys can be found at
         https://platform.openai.com/docs/api-reference/chat/create, defaults to gpt_util._generation_config if not set
        :type gen_conf: dict(, optional)

        :param max_retries: When processing batch, how many times to retry failed instances, defaults to 2
        :type max_retries: int(, optional)

        :param kwargs: Additional arguments to pass to OpenAI when initializing the OpenAI client

        :raises AssertionError: If the OpenAI API Key is not provided
        """
        # Initialize backend utility
        assert api_key, ("OpenAI API Key is required. "
                         "Either pass as argument or set as environment variable OPENAI_API_KEY")
        self.api_key = api_key
        self.openai_args = kwargs
        self.gen_conf = gen_conf
        # Create and set necessary parameters
        self.num_worker: int = num_worker
        self.queue: List[dict] = list()
        self.max_retries: int = max_retries
        # Generate function signature mappings
        func_signature: inspect.Signature = inspect.signature(gpt_util.generate_explanation)
        func_arg_list: List[str] = list(func_signature.parameters.keys())
        func_arg_list.remove('kwargs')
        self.func_required_arg_list: List[str] = \
            [arg for arg in func_arg_list if func_signature.parameters[arg].default is inspect.Parameter.empty]
        self.func_args: Dict[str, type] = {arg: func_signature.parameters[arg].annotation for arg in func_arg_list}

    def _dict_verifier(self, input_dict: dict) -> bool:
        """
        Verify if the input dictionary is valid for processing

        :param input_dict: Individual instance that will be processed when process() is called
        :type input_dict: dict

        :return: True if the dictionary is valid, False otherwise
        """
        # Check if all required arguments are provided
        for k in self.func_required_arg_list:
            if k not in input_dict:
                warn(f"Invalid dictionary, key {k} is required")
                return False
        # Check if argument types
        for k, v in input_dict.items():
            if k not in self.func_args:
                continue
            if not isinstance(v, self.func_args[k]):
                warn(f"Invalid dictionary, item {k} is not of type {self.func_args[k]}")
                return False
        return True

    def add_batch(self, batch: List[dict]):
        """
        Add a batch of instances to the queue, the batch will be processed when process() is called.
        Each added batch will be verified to ensure all required arguments are provided and of correct type.

        :param batch: A list of dictionaries, each dictionary should contain all required arguments used in
        gpt_util.generate_explanation. If not an error will be raised.
        :type batch: List[dict]

        :raises AssertionError: If any dictionary in the batch has missing or invalid arguments
        """
        verification_result: List[bool] = process_map(self._dict_verifier,
                                                      batch,
                                                      max_workers=self.num_worker,
                                                      chunksize=max(1, len(batch) // self.num_worker),
                                                      desc="Verifying Batch")
        assert False not in verification_result, f"Verification failed"
        for b in batch:
            # Adding additional required arguments for multi-threading
            b['openai_args'] = dict()
            b['openai_args']['api_key'] = self.api_key
            b['openai_args'].update(self.openai_args)
            b.update(self.gen_conf)
            if 'max_retries' not in b['openai_args']:
                b['openai_args']['max_retries'] = 5
            self.queue.append(b)

    def process(self, rerun_on_error: bool = False) -> List[Dict[str, str]]:
        """
        Process the batch of instances added to the queue. The function will return a list of dictionaries, each
        dictionary contains the output of gpt_util.generate_explanation for each instance in the batch. If
        rerun_on_error is set to True, the function will retry failed instances up to self.max_retries number of times.
        Any instance that failed after self.max_retries will be returned as an empty dictionary.

        :param rerun_on_error: If set to True, the function will retry failed instances up to self.max_retries number of
        times, defaults to False
        :type rerun_on_error: bool

        :return: list of dictionaries, each dictionary contains the output of gpt_util.generate_explanation for each
        instance in the batch
        :rtype: List[Dict[str, str]]
        """
        # copy the queue to a local variable and clear the public facing queue
        queue = self.queue[:]
        self.queue = []
        # get the results with multi-threading
        # note: func process_map() ensures the output is in the original order
        results: List[Dict[str, str]] = process_map(gpt_util.generate_explanation_wrapper,
                                                    queue,
                                                    max_workers=min(len(queue), self.num_worker),
                                                    chunksize=max(1, len(queue) // self.num_worker),
                                                    desc="Processing Batch")
        # post-process the output and add failed instances back to the queue
        failed_indexes: List[int] = list()
        for queue_index, result in enumerate(results):
            # if empty dict was returned, add the request back to the queue
            if len(result) == 0:
                if rerun_on_error:
                    self.queue.append(queue[queue_index])
                failed_indexes.append(queue_index)
        # if not rerun on error or no error, simply return the results
        if not rerun_on_error or len(failed_indexes) == 0:
            return results
        # try self.max_retries number of time for failed examples
        for _ in range(self.max_retries):
            # sanity check to ensure each queue item have a corresponding index mapping
            assert len(self.queue) == len(failed_indexes)
            # re-run the failed examples
            retied_results: List[Dict[str, str]] = self.process(rerun_on_error=False)
            # go through all retried examples and add success ones back to results
            success_indexes: List[int] = list()  # record which index from failed_indexes to remove
            for retry_index, result in enumerate(retied_results):
                queue_index: int = failed_indexes[retry_index]
                if len(result) != 0:
                    results[queue_index] = result  # replace the failed result with new one
                    success_indexes.append(queue_index)
                else:
                    # if failed again, add back to the queue
                    self.queue.append(queue[queue_index])
            # remove all succeed instances from failed_indexes
            for success_index in success_indexes:
                failed_indexes.remove(success_index)
            del success_index
            del success_indexes
            # if no failed instance remaining, jump out of the retry loop
            if len(failed_indexes) == 0:
                break
        return results
