import math

import torch


def _tokens_to_feature_map(tokens):
    if tokens.dim() != 3:
        raise ValueError(f"Expected prompted patch tokens [B,N,C], got {tuple(tokens.shape)}")

    num_tokens = tokens.shape[1]
    grid = int(math.sqrt(num_tokens))
    if grid * grid != num_tokens:
        patch_tokens = num_tokens - 1
        patch_grid = int(math.sqrt(patch_tokens))
        if patch_grid * patch_grid != patch_tokens:
            raise ValueError(
                "Prompted patch token count must be square after class-token removal, "
                f"got N={num_tokens}"
            )
        tokens = tokens[:, 1:, :]
        grid = patch_grid

    batch, num_patches, channels = tokens.shape
    if grid * grid != num_patches:
        raise ValueError(f"Prompted patch token count must be square, got N={num_patches}")
    return tokens.transpose(1, 2).reshape(batch, channels, grid, grid).contiguous()


def _extract_prompted_features(self, images, layers=None, return_projected=False):
    """
    Return prompted visual patch features for selected ViT blocks.

    layers are 1-indexed transformer block numbers. return_projected=False returns
    projection-pre patch tokens from encode_image. return_projected=True returns the
    patch_token_layer outputs and is kept for later experiments.
    """
    layers = list(layers or self.output_layers)
    if not layers:
        raise ValueError("layers must contain at least one 1-indexed transformer block number.")

    previous_output_layers = self.output_layers
    self.output_layers = layers
    try:
        if 'D' in self.prompting_type:
            self.generate_and_set_dynamic_promtps(images)

        if self.enable_visual_prompt:
            image_features, patch_tokens, _ = self.encode_image(images)
        else:
            with torch.no_grad():
                image_features, patch_tokens, _ = self.encode_image(images)

        if len(patch_tokens) != len(layers):
            raise RuntimeError(
                f"Expected {len(layers)} prompted feature levels, got {len(patch_tokens)}. "
                f"layers={layers}"
            )

        if return_projected:
            _, patch_tokens = self.proj_visual_tokens(image_features, patch_tokens)

        return {
            f"layer{layer}": _tokens_to_feature_map(tokens)
            for layer, tokens in zip(layers, patch_tokens)
        }
    finally:
        self.output_layers = previous_output_layers


def install_prompted_feature_api(adaclip_cls):
    if not hasattr(adaclip_cls, "extract_prompted_features"):
        adaclip_cls.extract_prompted_features = _extract_prompted_features
    return adaclip_cls
