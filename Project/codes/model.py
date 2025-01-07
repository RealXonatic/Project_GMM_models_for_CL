import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM

class MiniGPT4Wrapper(nn.Module):
    def __init__(self, minigpt4_path, vicuna7b_path, device):
        super().__init__()
        self.device = device

        # Charger le tokenizer depuis Vicuna
        self.tokenizer = AutoTokenizer.from_pretrained(vicuna7b_path, use_fast=False)

        # Charger le modèle Vicuna en tant que base
        self.model = AutoModelForCausalLM.from_pretrained(vicuna7b_path).to(device)

        # Charger les poids MiniGPT-4 préentraînés
        state_dict = torch.load(minigpt4_path, map_location=device)
        self.model.load_state_dict(state_dict, strict=False)

    def generate_text(self, image_embedding):
        prompt = "Describe the following image:"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Générer du texte
        output = self.model.generate(
            input_ids=inputs.input_ids,
            max_length=50,
            num_beams=5,
            no_repeat_ngram_size=2
        )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)