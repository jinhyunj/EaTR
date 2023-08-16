import torch

from models.model import EaTR
from models.transformer import Transformer
from models.matcher import build_matcher, build_event_matcher
from models.position_encoding import build_position_encoding
from models.criterion import SetCriterion


def build_model(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/moment_detr/issues/108#issuecomment-650269223
    device = torch.device(args.device)
    position_embedding, txt_position_embedding = build_position_encoding(args)

    transformer = Transformer(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        return_intermediate_dec=True,
        query_dim = args.query_dim,
        num_queries=args.num_queries,
        num_iteration=args.num_slot_iter,
    )
    model = EaTR(
        transformer,
        position_embedding,
        txt_position_embedding,
        txt_dim=args.t_feat_dim,
        vid_dim=args.v_feat_dim,
        num_queries=args.num_queries,
        input_dropout=args.input_dropout,
        aux_loss=args.aux_loss,
        contrastive_align_loss=args.contrastive_align_loss,
        contrastive_hdim=args.contrastive_hdim,
        span_loss_type=args.span_loss_type,
        use_txt_pos=args.use_txt_pos,
        n_input_proj=args.n_input_proj,
        query_dim=args.query_dim,
    )

    matcher = build_matcher(args)
    event_matcher = build_event_matcher(args)
    weight_dict = {"loss_span": args.span_loss_coef,
                "loss_giou": args.giou_loss_coef,
                "loss_label": args.label_loss_coef,
                "loss_saliency": args.lw_saliency,
                "loss_event_span": args.event_coef*args.span_loss_coef,
                "loss_event_giou": args.event_coef*args.giou_loss_coef,
                }
    if args.contrastive_align_loss:
        weight_dict["loss_contrastive_align"] = args.contrastive_align_loss_coef

    if args.aux_loss:
        aux_weight_dict = {}
        weight_dict.update(aux_weight_dict)
        for i in range(args.dec_layers - 1):
            loss = ["loss_span", "loss_giou", "loss_label"]
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items() if k in loss})
        weight_dict.update(aux_weight_dict)

    losses = ['spans', 'labels', 'saliency', 'event_spans']
    if args.contrastive_align_loss:
        losses += ["contrastive_align"]
    criterion = SetCriterion(
        matcher=matcher, event_matcher=event_matcher, weight_dict=weight_dict, losses=losses,
        eos_coef=args.eos_coef, temperature=args.temperature,
        span_loss_type=args.span_loss_type, max_v_l=args.max_v_l,
        saliency_margin=args.saliency_margin,
        )

    criterion.to(device)
    return model, criterion
