import os

os.sys.path.append("..")
from configs.template import get_config as default_config


def get_config():
    config = default_config()

    config.result_prefix = 'results/individual_qwen2.5-7b'

    config.tokenizer_path = "/home/comp/f2256768/JBShield/models/Qwen2.5-7B-Instruct"  # to be set
    config.model_path = "/home/comp/f2256768/JBShield/models/Qwen2.5-7B-Instruct"  # to be set
    config.conversation_template = 'qwen'

    config.vis_dict_path = "../../experiments/cache/scores_qwen2.5-7b.json"

    config.demo_suffix_template = "Of course! Here's my answer to your task:"

    config.gpt_eval = False

    return config