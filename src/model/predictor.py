import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.utils.iterate import chunks

class BertTransformer:
    def __init__(self, name) -> None:
        """Creates the model class that handles the actual predictions.

        Args:
            name (str): name of the huggingface repo.
        """
        self.name = name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tokenizer = None
        self._model = None
        self._max_length = None

    @property
    def tokenizer(self):
        if not self._tokenizer:
            self._tokenizer = AutoTokenizer.from_pretrained("luerhard/PopBERT")
        return self._tokenizer

    @property
    def model(self):
        if not self._model:
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.name,
            ).to(
                self.device,
            )
        return self._model

    def tokenize(self, batch):
        encodings = self.tokenizer(
            batch,
            is_split_into_words=False,
            truncation=True,
            padding=True,
            return_tensors="pt",
            max_length=512,
        )
        return encodings

    def _get_probas(self, out):
        probs = torch.nn.functional.sigmoid(out.logits)
        probs = probs.detach().cpu().numpy()
        return probs

    def predict(self, text: list[str] | str, chunksize=32):
        """Predict populism dimensions of an already tokenized sentence."""
        # ensure correct tokens-batch format
        if isinstance(text, str):
            text = [text]

        results = []
        for batch in chunks(text, chunksize):
            encodings = self.tokenize(batch)
            encodings = encodings.to(self.device)

            with torch.inference_mode():
                out = self.model(**encodings)

            probas = self._get_probas(out)
            results.extend(probas)

        return results
