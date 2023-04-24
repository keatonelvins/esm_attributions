from captum.attr import IntegratedGradients
import torch
from transformers import EsmTokenizer, EsmForSequenceClassification, EsmConfig

model_name = "facebook/esm2_t12_35M_UR50D"
tokenizer = EsmTokenizer.from_pretrained(model_name)

# Load the model from the finetuned checkpoint
config = EsmConfig.from_pretrained("esm_finetune/output/final_model")
model = EsmForSequenceClassification(config)
model_weights = torch.load("esm_finetune/output/final_model/pytorch_model.bin")
model.load_state_dict(model_weights)

# Define input sequence
sequence = ["DVLTFNSAAYNNK", "DVLTFNSAAYNNK"]
base_sequence = ["", "DDDDDDDDDDDDD"]

# Tokenize input sequence
sequence_tokenized = tokenizer(sequence, padding=True, truncation=True, max_length=512, return_tensors="pt")
base_sequence_tokenized = tokenizer(base_sequence, padding=True, truncation=True, max_length=512, return_tensors="pt")

# Define Integrated Gradients
ig = IntegratedGradients(model)

# Compute Integrated Gradients
attributions, delta = ig.attribute(inputs=(sequence_tokenized["input_ids"]),
                                baselines=(base_sequence_tokenized["input_ids"]),
                                return_convergence_delta=True)

# Print attributions
print(attributions)
# Print delta
print(delta)


