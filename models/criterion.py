import torch
import torch.nn.functional as F
from torch import nn

from utils.span_utils import generalized_temporal_iou, generalized_temporal_iou_, span_cxw_to_xx
from misc import accuracy


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, matcher, weight_dict, eos_coef, losses, temperature, span_loss_type, max_v_l,
                 saliency_margin=1, event_matcher=None):
        """ Create the criterion.
        Parameters:
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            temperature: float, temperature for NCE loss
            span_loss_type: str, [l1, ce]
            max_v_l: int,
            saliency_margin: float
        """
        super().__init__()
        self.matcher = matcher
        self.event_matcher = event_matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.temperature = temperature
        self.span_loss_type = span_loss_type
        self.max_v_l = max_v_l
        self.saliency_margin = saliency_margin

        # foreground and background classification
        self.foreground_label = 0
        self.background_label = 1
        self.eos_coef = eos_coef
        empty_weight = torch.ones(2)
        empty_weight[-1] = self.eos_coef  # lower weight for background (index 1, foreground index 0)
        self.register_buffer('empty_weight', empty_weight)

    def loss_spans(self, outputs, targets, indices):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "spans" containing a tensor of dim [nb_tgt_spans, 2]
           The target spans are expected in format (center_x, w), normalized by the image size.
        """
        assert 'pred_spans' in outputs
        targets = targets["span_labels"]
        idx = self._get_src_permutation_idx(indices)
        src_spans = outputs['pred_spans'][idx]  # (#spans, max_v_l * 2)
        tgt_spans = torch.cat([t['spans'][i] for t, (_, i) in zip(targets, indices)], dim=0)  # (#spans, 2)
        if self.span_loss_type == "l1":
            loss_span = F.l1_loss(src_spans, tgt_spans, reduction='none')
            loss_giou = 1 - torch.diag(generalized_temporal_iou(span_cxw_to_xx(src_spans), span_cxw_to_xx(tgt_spans)))
        else:  # ce
            n_spans = src_spans.shape[0]
            src_spans = src_spans.view(n_spans, 2, self.max_v_l).transpose(1, 2)
            loss_span = F.cross_entropy(src_spans, tgt_spans, reduction='none')
            loss_giou = loss_span.new_zeros([1])

        losses = {}
        losses['loss_span'] = loss_span.mean()
        losses['loss_giou'] = loss_giou.mean()
        return losses

    def loss_event_spans(self, outputs, targets, indices):
        assert 'pred_event_spans' in outputs
        ## boundary span prediction
        idx = self._get_src_permutation_idx(indices)
        src_spans = outputs['pred_event_spans'][idx]
        tgt_spans = torch.cat([t[i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_event_span = F.l1_loss(src_spans, tgt_spans, reduction='none')
        loss_event_giou = 1 - torch.diag(generalized_temporal_iou_(span_cxw_to_xx(src_spans), span_cxw_to_xx(tgt_spans)))
        return {
            'loss_event_span': loss_event_span.mean(),
            'loss_event_giou': loss_event_giou.mean(),
        }

    def loss_labels(self, outputs, targets, indices, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        # TODO add foreground and background classifier.  use all non-matched as background.
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']  # (batch_size, #queries, #classes=2)
        # idx is a tuple of two 1D tensors (batch_idx, src_idx), of the same length == #objects in batch
        idx = self._get_src_permutation_idx(indices)
        target_classes = torch.full(src_logits.shape[:2], self.background_label,
                                    dtype=torch.int64, device=src_logits.device)  # (batch_size, #queries)
        target_classes[idx] = self.foreground_label

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight, reduction="none")
        losses = {'loss_label': loss_ce.mean()}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], self.foreground_label)[0]
        return losses

    def loss_saliency(self, outputs, targets, indices, log=True):
        """higher scores for positive clips"""
        if "saliency_pos_labels" not in targets:
            return {"loss_saliency": 0}
        saliency_scores = outputs["saliency_scores"]  # (N, L)
        pos_indices = targets["saliency_pos_labels"]  # (N, #pairs)
        neg_indices = targets["saliency_neg_labels"]  # (N, #pairs)
        num_pairs = pos_indices.shape[1]  # typically 2 or 4
        batch_indices = torch.arange(len(saliency_scores)).to(saliency_scores.device)
        pos_scores = torch.stack(
            [saliency_scores[batch_indices, pos_indices[:, col_idx]] for col_idx in range(num_pairs)], dim=1)
        neg_scores = torch.stack(
            [saliency_scores[batch_indices, neg_indices[:, col_idx]] for col_idx in range(num_pairs)], dim=1)
        loss_saliency = torch.clamp(self.saliency_margin + neg_scores - pos_scores, min=0).sum() \
            / (len(pos_scores) * num_pairs) * 2  # * 2 to keep the loss the same scale
        return {"loss_saliency": loss_saliency}

    def loss_contrastive_align(self, outputs, targets, indices, log=True):
        """encourage higher scores between matched query span and input text"""
        normalized_text_embed = outputs["proj_txt_mem"]  # (bsz, #tokens, d)  text tokens
        normalized_img_embed = outputs["proj_queries"]  # (bsz, #queries, d)
        logits = torch.einsum(
            "bmd,bnd->bmn", normalized_img_embed, normalized_text_embed)  # (bsz, #queries, #tokens)
        logits = logits.sum(2) / self.temperature  # (bsz, #queries)
        idx = self._get_src_permutation_idx(indices)
        positive_map = torch.zeros_like(logits, dtype=torch.bool)
        positive_map[idx] = True
        positive_logits = logits.masked_fill(~positive_map, 0)

        pos_term = positive_logits.sum(1)  # (bsz, )
        num_pos = positive_map.sum(1)  # (bsz, )
        neg_term = logits.logsumexp(1)  # (bsz, )
        loss_nce = - pos_term / num_pos + neg_term  # (bsz, )
        losses = {"loss_contrastive_align": loss_nce.mean()}
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx  # two 1D tensors of the same length

    def get_loss(self, loss, outputs, targets, indices, **kwargs):
        loss_map = {
            "spans": self.loss_spans,
            "labels": self.loss_labels,
            "contrastive_align": self.loss_contrastive_align,
            "saliency": self.loss_saliency,
            "event_spans": self.loss_event_spans,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, **kwargs)

    def forward(self, outputs, targets, epoch_i):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        self.epoch_i = epoch_i

        outputs_without_aux = {k: v for k, v in outputs.items() if 'aux_outputs' not in k}

        # Retrieve the matching between the outputs of the last layer and the targets
        # list(tuples), each tuple is (pred_span_indices, tgt_span_indices)
        moment_indices = self.matcher(outputs_without_aux, targets)
        event_indices = self.event_matcher(outputs['pred_event_spans'], outputs['pseudo_event_spans'])

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            if loss == 'event_spans':
                indices_in = event_indices
                targets_in = outputs['pseudo_event_spans']
            else:
                indices_in = moment_indices
                targets_in = targets
            losses.update(self.get_loss(loss, outputs, targets_in, indices_in))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss in ['saliency', 'event_spans']:
                        continue
                    kwargs = {}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses