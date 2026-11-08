# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace

from .data2vec.data.modality import Modality


d2v2_config = SimpleNamespace(
    **{'_name': 'data2vec_multi', 'loss_beta': 0.0, 'loss_scale': None, 'depth': 8, 'start_drop_path_rate': 0.0, 'end_drop_path_rate': 0.0, 'num_heads': 12,
       'norm_eps': 1e-05, 'norm_affine': True, 'encoder_dropout': 0.1, 'post_mlp_drop': 0.1, 'attention_dropout': 0.1, 'activation_dropout': 0.0,
       'dropout_input': 0.0, 'layerdrop': 0.05, 'embed_dim': 768, 'mlp_ratio': 4.0, 'layer_norm_first': False, 'average_top_k_layers': 8,
       'end_of_block_targets': False, 'clone_batch': 8, 'layer_norm_target_layer': False, 'batch_norm_target_layer': False, 'instance_norm_target_layer': True,
       'instance_norm_targets': False, 'layer_norm_targets': False, 'ema_decay': 0.999, 'ema_same_dtype': True, 'log_norms': True, 'ema_end_decay': 0.99999,
       'ema_anneal_end_step': 75000, 'ema_encoder_only': False, 'max_update': 400000, 'modalities': SimpleNamespace(**{'_name': None, 'audio': SimpleNamespace(
            **{'type': Modality.AUDIO, 'prenet_depth': 4, 'prenet_layerdrop': 0.05, 'prenet_dropout': 0.1, 'start_drop_path_rate': 0.0,
               'end_drop_path_rate': 0.0, 'num_extra_tokens': 0, 'init_extra_token_zero': True, 'mask_noise_std': 0.01, 'mask_prob_min': None, 'mask_prob': 0.5,
               'inverse_mask': False, 'mask_prob_adjust': 0.05, 'keep_masked_pct': 0.0, 'mask_length': 5, 'add_masks': False, 'remove_masks': False,
               'mask_dropout': 0.0, 'encoder_zero_mask': True, 'mask_channel_prob': 0.0, 'mask_channel_length': 64, 'ema_local_encoder': False,
               'local_grad_mult': 1.0, 'use_alibi_encoder': True, 'alibi_scale': 1.0, 'learned_alibi': False, 'alibi_max_pos': None,
               'learned_alibi_scale': True, 'learned_alibi_scale_per_head': True, 'learned_alibi_scale_per_layer': False, 'num_alibi_heads': 12,
               'model_depth': 8, 'decoder': SimpleNamespace(
                    **{'decoder_dim': 384, 'decoder_groups': 16, 'decoder_kernel': 7, 'decoder_layers': 4, 'input_dropout': 0.1, 'add_positions_masked': False,
                       'add_positions_all': False, 'decoder_residual': True, 'projection_layers': 1, 'projection_ratio': 2.0,
                       'channel_mult': [1, 0.5, 0.25, 0.25, 0.25], 'decoder_transformer_layers': 4}), 'extractor_mode': 'layer_norm',
               'feature_encoder_spec': '[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] + [(512,2,2)]', 'conv_pos_width': 95, 'conv_pos_groups': 16,
               'conv_pos_depth': 5, 'conv_pos_pre_ln': False})}), 'shared_decoder': None, 'min_target_var': 0.1, 'min_pred_var': 0.01,
       'supported_modality': Modality.AUDIO, 'mae_init': False, 'seed': 1, 'skip_ema': False, 'cls_loss': 0.0, 'recon_loss': 0.0, 'd2v_loss': 1.0,
       'decoder_group': False}
)
