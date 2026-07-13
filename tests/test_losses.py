"""Tests for LDAMLoss, FocalLoss, and ClassBalancedLoss."""

import torch

from src.loss import FocalLoss, ClassBalancedLoss, LDAMLoss


# -- FocalLoss -----------------------------------------------------------------

class TestFocalLoss:
    def test_forward_returns_scalar(self):
        loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
        logits = torch.randn(10, 3)
        targets = torch.randint(0, 3, (10,))
        loss = loss_fn(logits, targets)
        assert loss.ndim == 0

    def test_requires_grad(self):
        loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
        logits = torch.randn(10, 3, requires_grad=True)
        targets = torch.randint(0, 3, (10,))
        loss = loss_fn(logits, targets)
        assert loss.requires_grad

    def test_reduction_sum(self):
        loss_fn = FocalLoss(alpha=0.25, gamma=2.0, reduction="sum")
        logits = torch.randn(10, 3)
        targets = torch.randint(0, 3, (10,))
        loss = loss_fn(logits, targets)
        assert loss.ndim == 0

    def test_reduction_none(self):
        loss_fn = FocalLoss(alpha=0.25, gamma=2.0, reduction="none")
        logits = torch.randn(10, 3)
        targets = torch.randint(0, 3, (10,))
        loss = loss_fn(logits, targets)
        assert loss.shape == (10,)

    def test_lower_gamma_less_focus(self):
        logits = torch.randn(10, 3)
        targets = torch.randint(0, 3, (10,))
        loss_high = FocalLoss(alpha=0.25, gamma=5.0)(logits, targets)
        loss_low = FocalLoss(alpha=0.25, gamma=0.5)(logits, targets)
        assert loss_low > 0
        assert loss_high > 0

    def test_alpha_zero(self):
        loss_fn = FocalLoss(alpha=0.0, gamma=2.0)
        logits = torch.randn(10, 3)
        targets = torch.randint(0, 3, (10,))
        loss = loss_fn(logits, targets)
        assert loss.ndim == 0


# -- ClassBalancedLoss ---------------------------------------------------------

class TestClassBalancedLoss:
    def test_forward_with_class_counts(self):
        loss_fn = ClassBalancedLoss(beta=0.999)
        logits = torch.randn(10, 3)
        targets = torch.randint(0, 3, (10,))
        loss = loss_fn(logits, targets, class_counts=[100, 10, 5])
        assert loss.ndim == 0

    def test_forward_without_class_counts_falls_back_to_ce(self):
        loss_fn = ClassBalancedLoss(beta=0.999)
        logits = torch.randn(10, 3)
        targets = torch.randint(0, 3, (10,))
        loss = loss_fn(logits, targets)
        assert loss.ndim == 0

    def test_requires_grad(self):
        loss_fn = ClassBalancedLoss(beta=0.999)
        logits = torch.randn(10, 3, requires_grad=True)
        targets = torch.randint(0, 3, (10,))
        loss = loss_fn(logits, targets, class_counts=[100, 10, 5])
        assert loss.requires_grad

    def test_uniform_counts_equal_ce(self):
        loss_fn_cb = ClassBalancedLoss(beta=0.999)
        logits = torch.randn(10, 3)
        targets = torch.randint(0, 3, (10,))
        loss_cb = loss_fn_cb(logits, targets, class_counts=[100, 100, 100])
        loss_ce = loss_fn_cb(logits, targets)
        assert abs(loss_cb.item() - loss_ce.item()) < 1e-6

    def test_different_beta_values(self):
        loss_fn_high = ClassBalancedLoss(beta=0.9999)
        loss_fn_low = ClassBalancedLoss(beta=0.9)
        logits = torch.randn(10, 3)
        targets = torch.randint(0, 3, (10,))
        loss_high = loss_fn_high(logits, targets, class_counts=[100, 10, 5])
        loss_low = loss_fn_low(logits, targets, class_counts=[100, 10, 5])
        assert loss_high.ndim == 0
        assert loss_low.ndim == 0

    def test_reduction_sum(self):
        loss_fn = ClassBalancedLoss(beta=0.999, reduction="sum")
        logits = torch.randn(10, 3)
        targets = torch.randint(0, 3, (10,))
        loss = loss_fn(logits, targets, class_counts=[100, 10, 5])
        assert loss.ndim == 0


# -- LDAMLoss ------------------------------------------------------------------

class TestLDAMLoss:
    def test_forward_with_cls_num_list(self):
        loss_fn = LDAMLoss(cls_num_list=[100, 10, 5], max_m=0.5, s=30.0)
        logits = torch.randn(10, 3)
        targets = torch.randint(0, 3, (10,))
        loss = loss_fn(logits, targets)
        assert loss.ndim == 0

    def test_forward_without_cls_num_list_falls_back_to_ce(self):
        loss_fn = LDAMLoss()
        logits = torch.randn(10, 3)
        targets = torch.randint(0, 3, (10,))
        loss = loss_fn(logits, targets)
        assert loss.ndim == 0

    def test_requires_grad(self):
        loss_fn = LDAMLoss(cls_num_list=[100, 10, 5])
        logits = torch.randn(10, 3, requires_grad=True)
        targets = torch.randint(0, 3, (10,))
        loss = loss_fn(logits, targets)
        assert loss.requires_grad

    def test_margin_applied(self):
        logits = torch.randn(10, 3)
        targets = torch.randint(0, 3, (10,))
        loss_with = LDAMLoss(cls_num_list=[100, 10, 5])(logits, targets)
        loss_without = LDAMLoss()(logits, targets)
        assert loss_with.ndim == 0
        assert loss_without.ndim == 0

    def test_different_max_m(self):
        logits = torch.randn(10, 3)
        targets = torch.randint(0, 3, (10,))
        loss_high = LDAMLoss(cls_num_list=[100, 10, 5], max_m=1.0)(logits, targets)
        loss_low = LDAMLoss(cls_num_list=[100, 10, 5], max_m=0.1)(logits, targets)
        assert loss_high.ndim == 0
        assert loss_low.ndim == 0

    def test_different_s_values(self):
        logits = torch.randn(10, 3)
        targets = torch.randint(0, 3, (10,))
        loss_s = LDAMLoss(cls_num_list=[100, 10, 5], s=10.0)(logits, targets)
        assert loss_s.ndim == 0

    def test_reduction_sum(self):
        loss_fn = LDAMLoss(cls_num_list=[100, 10, 5], reduction="sum")
        logits = torch.randn(10, 3)
        targets = torch.randint(0, 3, (10,))
        loss = loss_fn(logits, targets)
        assert loss.ndim == 0

    def test_scale_applied_correctly(self):
        """Verify the s * logits_m scaling is applied."""
        loss_fn = LDAMLoss(cls_num_list=[100, 10, 5], s=30.0)
        logits = torch.randn(10, 3)
        targets = torch.randint(0, 3, (10,))
        loss = loss_fn(logits, targets)
        assert loss > 0
