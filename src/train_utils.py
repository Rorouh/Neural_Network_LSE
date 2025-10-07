import torch
from torch.nn import CrossEntropyLoss, nn
from transformers import Trainer


class WeightedTrainer(Trainer):
    """
    Repondera la pérdida de los tokens de destino que igualan hash_token_id (normalmente '#').
    - Si no se pasa hash_token_id, cae al comportamiento estándar.
    - Compatible con HF >= 4.42: acepta num_items_in_batch y **kwargs.
    """

    def __init__(self, *args, hash_token_id=None, hash_weight=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.hash_token_id = hash_token_id
        self.hash_weight = float(hash_weight)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        # Si no queremos ponderar '#', usa la compute_loss por defecto
        if self.hash_token_id is None:
            return super().compute_loss(
                model, inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
                **kwargs
            )

        labels = inputs.get("labels")
        # Necesitamos logits para cálculo personalizado
        outputs = model(**inputs)
        logits = outputs.get("logits", None)
        if logits is None:
            # fallback: por si el modelo no devuelve logits, usa el loss estándar
            return super().compute_loss(
                model, inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
                **kwargs
            )

        # Cross-entropy por-token (ignorando -100)
        vocab_size = logits.size(-1)
        loss_fct = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)

        loss_per_token = loss_fct(
            logits.view(-1, vocab_size),
            labels.view(-1)
        )  # shape: (batch*tgt_len,)

        # pesos: 1.0 para todo, hash_weight cuando el label == hash_token_id
        weights = torch.ones_like(loss_per_token, dtype=loss_per_token.dtype, device=loss_per_token.device)

        if self.hash_token_id is not None:
            labels_flat = labels.view(-1)
            hash_mask = (labels_flat == int(self.hash_token_id))
            weights = torch.where(hash_mask, torch.as_tensor(self.hash_weight, device=weights.device, dtype=weights.dtype), weights)

            # no contribuir con posiciones ignoradas
            valid_mask = (labels_flat != -100)
            weights = weights * valid_mask.to(weights.dtype)

        # normalización: media ponderada solo sobre válidos
        denom = torch.clamp(weights.sum(), min=1.0)
        loss = (loss_per_token * weights).sum() / denom

        if return_outputs:
            return loss, outputs
        return loss