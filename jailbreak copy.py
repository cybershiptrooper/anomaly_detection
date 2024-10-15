# %%
from easyjailbreak.attacker.PAIR_chao_2023 import PAIR
from easyjailbreak.datasets import JailbreakDataset
from easyjailbreak.models.model_base import BlackBoxModelBase
from easyjailbreak.attacker.GCG_Zou_2023 import GCG
from fastchat.conversation import get_conv_template
import warnings
from easyjailbreak.models import from_pretrained
import torch

# %%
class LLamaModel(BlackBoxModelBase):
    def __init__(self, model):
        self.model = model
        self.conversation = get_conv_template("llama-2")

    def generate(self, messages, clear_old_history=True, **kwargs):
        if clear_old_history:
            self.conversation.messages = []
        if isinstance(messages, str):
            messages = [messages]
        for index, message in enumerate(messages):
            self.conversation.append_message(self.conversation.roles[index % 2], message)
        prompt = self.conversation.get_prompt()
        response = self.model.generate(prompt)
        return response

    def batch_generate(self, conversations, **kwargs):
        responses = []
        for conversation in conversations:
            if isinstance(conversation, str):
                warnings.warn(
                    "For batch generation based on several conversations, provide a list[str] for each conversation. "
                    "Using list[list[str]] will avoid this warning."
                )
            responses.append(self.generate(conversation, **kwargs))
        return responses

    def set_system_message(self, system_message: str):
        self.conversation.system_message = system_message


torch.cuda.empty_cache()
# default device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define the target model of the attack
model = from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf", "llama-2", dtype=torch.bfloat16, max_new_tokens=200
)


# First, prepare models and datasets.
attack_model = model
target_model = LLamaModel(model)
eval_model = LLamaModel(model)
dataset = JailbreakDataset('AdvBench')

# Then instantiate the recipe.
# attacker = PAIR(attack_model=attack_model,
#                 target_model=target_model,
#                 eval_model=eval_model,
#                 jailbreak_datasets=dataset)

attacker = GCG(
    attack_model=model,
    target_model=model,
    jailbreak_datasets=dataset,
    max_num_iter=100,
    batchsize=8,
)

# Finally, start jailbreaking.
# raise
dataset = attacker.single_attack(dataset[1])

# attacker.attack()

# %%
print(dataset[0])
