from typing import Any

import torch
from transformers import T5Tokenizer
from transformers.data.data_collator import DataCollatorForSeq2Seq


def get_model_name() -> str:
    """Method to consistently use the same model everywhere.

    Returns:
        str: Returns the model name to be used with hugging face libraries.
    """
    return "t5-small"


# To get the contextualised token embeddings from the pre-trained model
tokenizer = T5Tokenizer.from_pretrained(get_model_name())
# Maximum length to handle for input (digesting the article). Truncates after.
max_input_length = 512
# Maximum length for output (generating summary).
max_target_length = 128


def get_tokenizer():
    """Return the tokenizer instance needed for the main loop.
    """
    return tokenizer


def preprocess_function(instance) -> dict[str, Any]:
    """Given a single dataset instance, the function pads them and breaks them into tokens
    needed for training.

    Args:
        instance (Dataset): A single dataset item to be converted.

    Returns:
        dict[str, Any]: The processed input tokens for the input and output sequences.
    """
    input_text = "summarize: " + instance["article"]
    target_text = instance["highlights"]

    model_inputs = tokenizer(
        input_text,
        max_length=max_input_length,
        truncation=True,
        padding="max_length"
    )

    labels = tokenizer(
        target_text,
        max_length=max_target_length,
        truncation=True,
        padding="max_length"
    ).input_ids

    # Replace padding token in the labels by -100 so they're ignored by the loss
    labels = [label if label !=
              tokenizer.pad_token_id else -100 for label in labels]
    model_inputs["labels"] = labels

    return model_inputs


def get_data_collator(model: torch.nn.Module) -> DataCollatorForSeq2Seq:
    """The module that collects examples and converts them into tokens in order to convert
    into a standard batch to be provided to the actual model for training or evaluation.

    Args:
        model (torch.nn.Module): The actual model to be supplied to.

    Returns:
        DataCollatorForSeq2Seq: Returns the DataCollator instance.
    """
    return DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
