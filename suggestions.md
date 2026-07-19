Several critical issues surfaced. Let me now read the data, bank core, methods, model, and PL module.

Continuing to read the bank core and methods.

Let me read the methods, ResNet, and the PL module.

Now let me read the configs.

Now I have the complete picture. Let me identify the issues:

🚨 Critical bugs found
Issue 1: CIFAR-100 evaluation is WRONG (severe — invalidates all reported results)
File: studies/runner/cifar100/metrics.py:6-9 and run.py:251-260

In a Class-IL split-CIFAR-100 evaluation, the standard protocol evaluates on all seen tasks after each new task, producing matrix R[i,j] = accuracy on task j after training task i. The runner builds this matrix correctly, but average_accuracy returns the mean of the last ROW (only the final task).

The standard Split-CIFAR-100 metric is AA_k = (1/k) Σ_{j=1..k} R_{k,j} (mean of last row) — so average_accuracy is actually correct IF the cell stores the accuracy on task j evaluated at the end of task i, but the way R[i,j] is built in run.py does row-by-row updates with task_test_loader(prev_task) calling trainer.test(pl_module, dataloaders=...). There's an important subtlety here:

The metric in metrics.py final_row = acc_matrix[-1, :] is the last row, mean of final-row, which is correct as AA_k (only after the final task). Good.

But forgetting and backward_transfer are computed with _to_rect's nanmean-safe logic. Let me re-check these.

Actually there's a more serious problem: the test loop uses trainer.test(), and the test dataloader receives the same pl_module which calls on_test_start() that resets test_preds. Good.

But — trainer.test() needs a dataloader. Let me check for an evaluation bug: the test step accumulates test_preds and test_labels separately, BUT we're calling trainer.test(...) once per previous task. After each call, on_test_epoch_end clears the lists. Then we collect test_results[0]["test/acc"], which is the acc.mean() over the batch. That should be correct... let me verify in pl_module.py:118-120.

OK, the test_step logs accumulated test/acc as mean across batches. PL averages them via on_epoch=True. Let me verify this gives correct accuracy.

Actually — test/acc is logged as float (preds == y).float().mean() per batch, with on_epoch=True means PL averages it. So test_results[0]["test/acc"] is mean-of-means. This is incorrect for accuracy — it gives equal weight to each batch, not each sample. For balanced test sets it's fine, but across tasks with different sizes it's slightly biased. For Split-CIFAR-100 (uniform 1000 per task), it's fine.

Issue 2: Bank stores RAW pre-transform tensors — but the trainer sees augmented tensors
Look at StaticBankMethod.compute_loss:38:

x = torch.cat([x] + bank_x)
x (the batch) is on GPU after T.Normalize and RandomCrop and ConvertImageDtype. The bank items stored are (x[i], y[i]) — these were the already-transformed/normalized/augmented tensors. When stitched into x, they're recomputed through pl_module(x) again — and concatenated with the new batch's normalized tensors. Both are float32-normalized, so they can be concatenated.

BUT — augmentations like RandomCrop(32, padding=4) and RandomHorizontalFlip() are stochastic. The bank stores one augmented view, replays that exact view repeatedly on every iteration. This is wrong — replay should use the original raw images, not augmented ones. In standard replay buffers (e.g., iCaRL, BiR), you store the pre-transform image and apply a train-time augmentation at retrieval time.

This is a real bug — it makes replay effectively see "frozen snapshots" but still WRONG-DISTRIBUTION data because augmentation at query depends on global RNG state that doesn't apply to stored tensors. The model will overfit to those exact replay tensors.

Issue 3: Type mismatch — bank stores uint8 images but model expects float32
In dataset.py:

train_images are loaded as the tensor saved by ingest, which is uint8 NHWC per ingest.py:182-183. ✓
__getitem__ returns torch.tensor(label) and img from self._images[index] — but img is uint8 NHWC, then permuted to NCHW only if img.shape[0] not in (1, 3) — wait, the saved tensor is torch.from_numpy(np.stack(...)).permute(0, 2, 3, 1) which makes it NHWC — so shape is [N, 32, 32, 3]. After __getitem__ returns this NHWC tensor (shape [32, 32, 3]), pl_module receives it. Then _train_transform runs RandomCrop(32, padding=4) — but augmentations need NCHW input. Let me check torchvision behavior... T.RandomCrop expects [..., H, W] and broadcasts batch dimensions including any leading dims.
Then ConvertImageDtype(torch.float32) converts uint8 to float32.

These transforms run on [H, W, C] = NHWC. But Image-class torchvision transforms usually expect [C, H, W]. Let me check ConvertImageDtype:

It accepts PIL or tensor. For tensor it expects [..., H, W] with arbitrary leading dims representing batch. It WILL work on NHWC, but Normalize then expects mean/std to match the channel dimension. With NHWC, mean shape [3] broadcasts as channel-last, so mean=[0.5071,...] would normalize wrong.
Actually T.Normalize for tensors expects shape [C, 1, 1] (broadcasts against NCHW). With NHWC, broadcasting fails because mean shape is (3,) and tensor shape is (H, W, C). We need to transpose first, or accept the broadcasting issue.

Actually checking torchvision source: Normalize calls (tensor - mean) / std. With mean.shape=(3,) and tensor.shape=(H, W, 3), broadcasting makes the last dim match — treats tensor as having channel last. So normalization behaves correctly even for NHWC! It would output (H, W, C) after ConvertImageDtype and Normalize.

But then the ResNet model expects NCHW. Look at the forward: out = F.relu(self.bn1(self.conv1(x))) — conv1 is _conv3x3(3, 64), expects NCHW. The [H, W, C] input would fail.

So somewhere we need NCHW. Let me re-read the transform:

T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T.ConvertImageDtype(torch.float32),
    T.Normalize(mean=mean, std=std),
])
None of these convert NHWC → NCHW. So either: (a) there's a bug, or (b) the saved tensor is NCHW.

Re-reading ingest.py lines 169-183:

train_imgs = torch.from_numpy(np.stack([...]))
# Images from PIL are uint8 NHWC already.
if train_imgs.ndim == 4 and train_imgs.shape[-1] not in (1, 3):
    train_imgs = train_imgs.permute(0, 2, 3, 1)
The condition train_imgs.shape[-1] not in (1, 3) would mean "if NOT already in (N, H, W, C), permute to NHWC". After PIL→numpy, images are (H, W, C), then stacked as (N, H, W, C). So shape[-1] = 3 IS in (1, 3), condition is False, no permutation → tensor is left as (N, H, W, C) (NHWC) ✓.

Pickle path line 370-373:

images = torch.from_numpy(data).reshape(-1, 3, 32, 32).permute(0, 2, 3, 1).contiguous()
CIFAR pickle data is (N, 3072) reshaped to (N, 3, 32, 32) then permuted to (N, 32, 32, 3) NHWC. ✓.

So images are saved as NHWC [N, 32, 32, 3] uint8. The dataset __getitem__ returns NHWC. Augmentations don't transpose to NCHW. ResNet needs NCHW.

Wait, but this run was working — got non-zero accuracy. So either the transform handles it correctly (somehow), or there's a fallback. Let me check what T.Normalize does on a [32, 32, 3] tensor with mean=[3]. Broadcasting (3,) - (3,) would actually work — broadcasting aligns from the right. So tensor - mean where tensor.shape=(32, 32, 3) and mean.shape=(3,) → broadcasts the mean across the last dim only. This means with NHWC, per-channel normalization is correctly applied. OK.

But then the model conv1 expects (N, C, H, W). We're feeding (32, 32, 3) — which would actually work too because Conv2d accepts [..., in_channels, H, W] where leading dims are batch-like. So (H=32, W=32, C=3) would be interpreted as (in_channels=32, H=32, W=3) — that's WRONG.

So either: (a) the model output was nonsense due to NHWC being misinterpreted, or (b) something transposes it. Let me look at this more carefully — the run produced 76-83% on task 9 (the only one trained and tested at task 9), but 0% on tasks 0-8 (forgetting) which is expected for all methods in CIL. So the model IS learning something.

Looking more carefully: _train_transform runs T.RandomCrop(32, padding=4) on [H=32, W=32, C=3] — RandomCrop takes any shape as long as it has at least 2 dims, treats last two as H, W. ✓. RandomHorizontalFlip — same story. Then ConvertImageDtype(torch.float32) — ok. Then Normalize — broadcasts fine. Output is float32 of shape [32, 32, 3].

Now batched, it becomes [B, 32, 32, 3] (since DataLoader stacks). Then conv1 sees [B, C=32, H=32, W=3] — wrong.

Hmm, but the model is learning. Perhaps PyTorch's Conv2d raises an error or maybe the values just collapse? Actually, looking at Python: in_channels=64 for the first conv, input must have 3 channels but we provide 32. nn.Conv2d(in_planes=3, out_planes=64) checks F.conv2d's channels: with input [B, 32, 32, 3], it expects in_channels=32 — error.

Wait, perhaps RandomCrop changes the shape. With padding=4, the input becomes [40, 40, 3] (after padding with zeros) — that's still NHWC but RandomCrop then crops to [32, 32, 3]. With padding_mode='constant' (default), zeros are added — but padding in RandomCrop is applied ALONG H/W dims only, not channel dim. So result has same shape.

Let me search the implementation more carefully — perhaps there's a T.PILToTensor() or transpose I'm missing.

Look at dataset.py:28-29! There IS a permute:

if img.dim() == 3 and img.shape[0] not in (1, 3):
    img = img.permute(2, 0, 1)
This says "if 3-dim image, the first dim is not 1 or 3, permute (2,0,1) → NHWC → NCHW". When NHWC is [32,32,3], img.shape[0] = 32, which is not in (1,3) — so permute runs → [3, 32, 32] ✓.

BUT — T.RandomCrop, RandomHorizontalFlip, etc., come AFTER this permute and operate on the now-NCHW tensor (C=3, H=32, W=32). So RandomCrop(32, padding=4) pads it to [3, 40, 40] then crops to [3, 32, 32] ✓. Normalize(mean=[3]) broadcasts per the first dim = channels — mean=[0.5071, ...] broadcasts across 3 channels ✓. Output is [3, 32, 32] float32 ✓. ResNet consumes it ✓.

But then RandomCrop(32, padding=4) — with padding on a 3-channel image, the padding is filled with zeros. Out of 10 classes per task, the model sees a lot of black borders on first conv → slight signal leak but standard CIFAR practice.

OK the model is correct. NHWC storage is correct. Byte-valued uint8 to Normalize→float32 is correct.

Issue 4: Bank stores uint8 vs receives normalized batch — CRITICAL
Let me look again very carefully at the bank storage and replay:

def compute_loss(self, batch, pl_module, bank=None):
    x, y = batch
    # x here is post-transform, normalized float32  shape [B, 3, 32, 32]
    if bank is not None:
        bank.store([(x[i], y[i]) for i in range(len(y))])  # stores normalized float32
        ...
        if pl_module.global_step >= self.warmup_steps:
            for bx, by in bank.query(budget=self.retrieval_budget):
                bank_x.append(bx.unsqueeze(0))
                bank_y.append(by.unsqueeze(0))

        if bank_x:
            x = torch.cat([x] + bank_x)  # cat normalized batch with stored normalized samples
            y = torch.cat([y] + bank_y)

    return F.cross_entropy(pl_module(x), y)
This concatenates normalized-float32-batch with stored normalized-float32-samples. Both come from the same augmentation pipeline (since they were stored from the SAME training pipeline). They have the same mean/std applied. Good for distribution, but the replay samples are the augmented versions, not raw.

Standard practice: store raw uint8 or normalized-float32 raw images, apply augmentation at query time. Right now augmentation is applied once at storage time, replayed forever. This means:

Less effective data augmentation during replay (each frozen replay view is shown again and again in the same pose/flip).
Mild overfitting problem.
This is a deviation from the standard rep. STANDARD replay baselines (iCaRL, BiR, ER, etc.) apply augmentation each time, not once-augmented-frozen. The proper fix: store raw images, augment at retrieval time.

Issue 5: Bank stores CPU tensors but model is GPU
bank.store([(x[i], y[i])]) — x[i] is on GPU. When concatenating back via torch.cat([x] + bank_x), the bank tensors must be on the same device. The first iteration x[i].unsqueeze(0) → still on GPU ✓. But on batch N, git-batch stores GPU tensor but query reconstructs via rng.choices(pool, k=count) which produces a list of stored examples — they're GPU. ✓ but — over many tasks, GPU memory keeps growing. Replay items stay on GPU until process death. This is memory-bad but not correctness-bad.

Issue 6: In PIDGBMethod._compute_per_class_loss — leak of non-training data evaluation
Line 71: with torch.no_grad(): logits = pl_module(x). The PID controller evaluates the model using x from batch. But then on line 95 return F.cross_entropy(pl_module(x), y) — this is called AGAIN with grad enabled (the whole compute_loss is wrapped by Lightning inside the training step, so this is the actual gradient loss). So x first gets evaluated for PID (no grad), then again for loss (with grad). This is double forward pass — expensive but correct.

The bank-probe forward (line 126) for absent classes also uses torch.no_grad() on bank_logits. Good.

Issue 7: PID bank's allocation uses raw_loss, not any normalization
Line 75 of pid_controller.py: raw_debt.append(max(0.0, self.class_weights[c] * (p + i + d))). This is unbounded. For high losses, debt grows large. Then allocate_by_debt proportionally allocates budget. Allocation can saturate to single class. This may cause drift — feedback explosion.

Standard practice: target allocation (e.g., target_per_class like EDGBMethod does, instead of debt).

Issue 8: Standard CIL baselines are not "standard" — the published baselines (BiR, iCaRL, PODNet, etc.) have very specific implementations
Looking at this:

BaselineMethod — just standard CE fine-tuning. NOT "Lower Bound" LowerBound standard. Lower bound / upper bound differentiation is fine.
StaticBankMethod — uniform random replay (a.k.a. Experience Replay / ER). This is standard ✓ but missing "herding selection" (iCaRL-style selection of exemplars closest to mean). ER uses ring buffer random, so this is fine.
EDGBMethod — exposure-debt (custom method). ✓ if we claim "our method".
PIDGBMethod — PID-controlled (custom method). ✓ if we claim "our method".
Missing standard CIL baselines for a paper:

LwF (Learning without Forgetting, distillation)
iCaRL (herding-based exemplar + distillation + NCM)
BiR (Bias Remover) or EWC (regularization-based)
DER (Dark Experience Replay) — logit-level replay
Pure BN bias correction
For a publication-worthy paper, 4 methods (baseline + 3 ours) is minimal. Ideally add at least EWC, LwF, iCaRL, DER. But for the publish-as-is scope, "vanilla lower/upper bound" definition: baseline = naive CE (finetuning) and our 3 banks. Static bank = uniformly random sampling from previous tasks (a decent "naive replay" baseline).

Issue 9: CIFAR-100 protocol hyperparameter mismatch
Standard Split-CIFAR-100 protocol (per Hou et al. 2019 "Learning a Unified Classifier Incrementally via Unbalanced Knowledge Distillation"):

10 tasks × 10 classes ✓
Incremental training starting from scratch ✓ (model.expand_head)
Initial teacher inference for KD methods — N/A here, no KD
Standard optimizer: SGD with momentum=0.9, weight_decay=5e-4 ✓
Learning rate: usually 0.1 with cosine decay ✓
Epochs per task: usually 70 (some papers use 100, some 30) ✓
This is acceptable.

Issue 10: Standard practice: reset optimizer between tasks (for fair comparison)
In class-IL, do we reset the optimizer at each task (a common practice that isolates incremental learning performance from optimizer annealing)? Standard: NO reset, keep optimizer running, let LR scheduler span all tasks.

Looking at the code, the LR scheduler is CosineAnnealingLR(max_epochs=70) per task, which is wrong — cosine should span across many epochs of all tasks, not reset at each task boundary. Each task only sees 70 epochs and the LR decays to 0 within task 1, then resets at task 2. This means the model trains at LR 0 most of the time after task 1.

Actually looking again at PL's CosineAnnealingLR with T_max = trainer.max_epochs per task. Since trainer.max_epochs=70, the scheduler runs cosine over 70 epochs within each task and resets between tasks (because we make a new trainer per task).

This means: Task 0 starts at LR=0.1, decays to 0 over 70 epochs. Task 1 starts at 0.1 again. This is wrong for continuous learning — usually T_max = total_epochs_across_tasks. The standard protocol uses continuous cosine schedule across all 10×70=700 epochs.

Actually, the standard Split-CIFAR-100 protocol typically keeps LR constant during each task (no within-task decay). The protocol from Buzzega et al. (Dark Experience Replay, "Mammoth" framework) uses constant LR per task or step decay at task boundaries, NOT cosine-per-task.

This is a real issue.

Issue 11: Early stopping on val/acc — but val is test set
The val loader is dm.get_task_test_loader(task_id) — i.e., the CURRENT task's test set. So val/acc is current-task test accuracy. Early stopping on current-task val/acc stops training when the current task saturates, which is legitimate for current task detection but not standard. Standard CIL: fixed epochs, no early stopping.

But more crucially: using the test set as validation is a major protocol violation. In standard ML, you never validate on the test set. CIFAR-100 has no validation set — you would hold out some training samples as validation. The current code doesn't hold out any, and the val/ep acc is actually current-task test acc → selection bias in early stopping.

Issue 12: No test/acc_class_j reported via PL — but logged in pl_module. PL version concern
Let me check the PL version being used (we have lightning calls like pl.Trainer.fit(..., train_dataloaders=..., val_dataloaders=...) which is older PL 1.x syntax). Modern PL 2.x uses train_dataloaders (alias) but both work. Compatibility note: PL 1.5-2.x seems OK.

Let me verify PL version:

PL 2.6.5. OK. Functions used: EarlyStopping, CSVLogger, TQDMProgressBar, trainer.test(...), trainer.fit(...) — all compatible.

Issue 13: metrics.py — forgetting has an off-by-one
Look at metrics.py:13-21:

n_tasks = acc_matrix.shape[1]
forget_vals = []
for i in range(n_tasks):
    col = acc_matrix[: i + 1, i]
    peak = float(np.nanmax(col))
    final = float(col[-1])
    forget_vals.append(peak - final)
For i = 0: col = acc_matrix[:1, 0] = [R[0,0]]. Peak = R[0,0], final = R[0,0] → 0. For i = 9: col = acc_matrix[:10, 9] = [R[0,9], ..., R[9,9]] — but row 9 is the FINAL row, its col-length is i+1=10. But matrix is rectangular only at row 9 (the last). For rows 0..8, the col would have fewer entries than i+1. Actually _to_rect pads each row to max_cols = 10 with np.nan. So acc_matrix[:i+1, i] is [nan, ..., nan, R[i,i], R[i+1,i], ..., R[9,i]]. nanmax ignores nan. Peak is correctly max of all available values.

Wait — acc_matrix.shape — n_tasks = acc_matrix.shape[1] = 10. But shape[0] = 10. ✓. ✓.

OK metrics.py is correct. BUT — the definition of forgetting here is: per-task f_j = max_{i<=j} R[i,j] - R[T,j]. This is "Total Forgetting" (a.k.a. Δ_acc per task, normalized to 0 if never learned). Standard: yes, this is correct for "average forgetting metric".

backward_transfer: BWT = (1/(T-1)) Σ_{i<j} (R[T,j] - R[j,j]) = mean_j final_j - first_j. Same as forgetting but signed: final - first. Forgetting = -BWT (peak-first, where peak ≈ first when model improves monotonically; final ≈ is final state).

So forgetting = peak - final, backward_transfer = final - first. These are correlated but not identical due to potential rise-then-decline. Standard CIL reporting: report both. ✓.

Issue 14: Task 0 has forgetting=0, BWT=0 — but standard papers compute these only on tasks 1..T-1
forget = forgetting(acc_matrix) if num_tasks > 1 else 0.0  # = mean of [0,...] = excludes task 0's 0 vs full
Wait — for i=0, forget_vals[0] = 0 is INCLUDED. Then mean([0, x1, ..., x9]) = average includes the 0. Standard practice: exclude task 0. The result is "biased downward" by including this zero. Fix: skip task 0.

Issue 15: create_pl_module ignores cfg.training.lr_scheduler for the GPU-test logic
Actually cfg.training.lr_scheduler=cosine, which is passed to PL module but… trainer.max_epochs = cfg.runner.get("epochs_per_task", 70) = 70. CosineAnnealingLR(T_max=70) per task. This is wrong.

Issue 16: EDGBMethod: target_per_class estimation divided by len(acc)
In ed_gb/method.py:50:

target = pl_module.global_step * x.size(0) / len(acc)
target_per_class = [target] * len(acc)
pl_module.global_step = current trainer global step within task (0-indexed). x.size(0) = batch size (128). len(acc) = number of classes incl. all prior tasks. This treats "global_step * batch_size" as a count of total samples seen across all tasks — but pl_module.global_step only counts within this task. So global_step counts steps within task. So target_per_class says "if exposure equal across classes, each class should have seen global_step * batch_size / num_classes samples". This is incorrect for multi-task — at task 5, classes 0..49 are 5 tasks old, but global_step only counts task 5's steps. So target_per_class is wrong: it treats ALL classes as new. Classes learned in task 0 are showing large "debt" forever.

Compute valid: total_seen = sum of (global_step over all completed tasks) * batch_size. This becomes highly inaccurate.

Issue 17: PID class_weights initialization sets default too low
In base_runner.py:181-185:

if class_counts is not None and method.use_class_weights:
    max_count = max(class_counts)
    method.class_weights = [
        (max_count / c) ** 0.5 if c > 0 else 1.0
        for c in class_counts
    ]
But create_method is called with class_counts=None. So method.class_weights = []. Inside the pl_module, getattr(method, "class_weights", None) = [] (empty list, not None) → passed into PIDController(class_weights=None) — list [] is "falsy" but class_weights=[] or [1.0]*num_classes = [1.0]*num_classes ✓. Actually PIDController.__init__: self.class_weights = class_weights or [1.0] * num_classes — [] is falsy ✓.

But for the create_method(class_counts=...) path, when class_counts=None is passed (which is what happens), else branch:

method.class_weights = [1.0] * len(class_counts) if class_counts else []
This is [1.0] * 0 = [] for class_counts=None. OK that's fine. No issue here.

Summary of BUGS and DEFICIENCIES
#	Severity	Issue	Impact
1	🔴 Critical	Bank stores augmented/normalized tensors; random bias & overfitting vs standard raw replay	Invalidates headline result
2	🔴 Critical	Val loader = test loader → selection bias, no early stopping should run	Methodological violation
3	🔴 Critical	Early stopping on val/acc mixes metric with test set	Selection bias
4	🟠 Major	global_step-only counting in EDGB exposure target ignores prior tasks	EDG bugs
5	🟠 Major	exclude_classes default in banks — exclude_classes=None → {0} (silent class 0 skip)	Risk of class 0 being skipped when caller passes None
6	🟠 Major	NHWC storage + permute-correct augmentation flow, but augmentation applied once at storage and frozen for replay is the actual replay issue	Replay correctness
7	🟡 Moderate	Forgetting includes task 0's 0	Mild numerical bias
8	🟡 Moderate	Per-task CosineAnnealingLR with max_epochs=70 — standard uses continuous or constant	Suboptimal training
9	🟡 Moderate	test/acc is mean-of-batch-means (correct for uniform batch sizes but ambiguous)	Minor — fine for CIFAR
10	🟡 Minor	cumulative_step for ED-GB target math should be tracked properly	Paper comparison
11	🟡 Minor	PID debt unbounded; risk of runaway allocation	EDG stability
12	🟢 Minor	NaN-loss issue from earlier run could recur without gradient clipping stability (already have gradient_clip_val: 1.0)	OK
Now I need to FIX these. Let me start coding.

Let me also check the test files, output manager, other supporting code.
Let me also check the test files, output manager, other supporting code.

Good. Now I have full context. Let me make a careful fix plan.

Fix 1: Standard CIFAR-100 Protocol Design
Based on the standard "Class-IL Split CIFAR-100" protocol (Rebuffi et al., iCaRL 2017; Buzzega et al., DER/Mammoth 2020):

Standard practice	Current implementation	Action
Split 10 tasks × 10 classes	✓	Keep
Train fixed 70 epochs per task with fixed LR 0.1 (no per-task cosine decay)	Per-task cosine annealing	FIX: disable per-task scheduler decay
Held-out validation from training or fixed epochs (no early stopping on test)	Val = test set + early stopping	FIX: held-out validation split
Replay buffer stores PRE-AUGMENTATION raw images, augmented at query time	Stores post-augmentation tensors	FIX: store raw uint8
Reservoir or ring buffer sampling	Uniform random with replacement	OK
No head expansion; logits head fixed-size, masked unreachable classes at test	Head expands	OK for some papers (e.g., BiR); standard iCaRL uses fixed-size head
SGD with momentum=0.9, weight_decay=5e-4	✓	Keep
LR=0.1 (default)	✓	Keep
Mean-of-means accuracy is fine for uniform split	✓	OK
CIL metrics: AA_k, AF, BWT (forgetting only on tasks 1..T-1)	Includes task 0	FIX: exclude task 0
Let me now write all the fixes. I'll start with the most important ones.

Fix the bank to store raw images, augment at query
The cleanest approach: store the raw uint8 image (NHWC [32, 32, 3] uint8), and at query time, apply a configurable augmentation transform that matches train-time. But the method doesn't have access to the train-time transform. So I'll either: (a) pass the augmentation function into the method/bank, or (b) use the standard transform the datamodule applies.

The cleanest fix: transform is applied once at the storage boundary. Storage stores the post-transform tensor, but the augmentation is re-applied at retrieval time with a separate image. Wait — that requires storing raw image AND the augmentation function in the method.

Pragmatic approach: store the raw uint8 image. At retrieval time, the method applies its OWN T.Compose(random_crop_flip+convert_dtype+normalize) to the retrieved raw samples, using fresh random state (a separate torch.Generator). This way the bank items are augmented differently each time they're used.

Let me implement this.

from . import transforms as cifar_transforms
Hmm — let me revert that change. I'll define the transforms as a function in the dataset.py module instead of a separate file, since that's the cleanest approach. Let me revert first:

$ git checkout src/data/cifar100/__init__.py 2>&1

git : Updated 1 path from the index
At line:1 char:1
+ git checkout src/data/cifar100/__init__.py 2>&1
+ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : NotSpecified: (Updated 1 path from the index:String) [], RemoteException
    + FullyQualifiedErrorId : NativeCommandError
 
Now I'll create the augmentation transforms as a small helper module that both the dataset and the bank can use. The transforms are pure functions so this is clean.

Wait — I need to think more carefully. The issue with generator is that torchvision.transforms.RandomCrop uses its own internal RNG, which reads from torch.default_generator. We can't easily inject a generator per-call. So if the bank runs augmentation on query, it will drift the training RNG.

Let me simplify: use a self-contained augmentation function that uses its own RNG key. For the bank's replay path, we want augmentation to be deterministic per call. The cleanest approach is to refactor to run the augmentation on the raw images at call time with a per-call seeded RNG.

Actually, the simplest and standard-correct approach: the bank stores the raw uint8 image (NHWC, the dataset's stored format). At query time, the method applies a train-consistent augmentation via its own augmentation transform. The issue is that sharing one transform's RNG between training pipeline and replay pipeline creates correlation.

Pragmatic correct approach: apply augmentation with a different seed. Specifically, when the method is created, also create a separate self._augment_rng = torch.Generator(device='cpu').manual_seed(seed)or just use the bank's RNG that already exists. The bank's existingrandom.Random(seed)for selection. We can add atorch.Generator` for augmentation.

Let me redesign cleanly: the dataset loads raw uint8 images + applies train transform → model. The bank stores raw uint8 images (no augmentation). At query time, the method applies a fresh training transform to each retrieved sample, using the bank's separate RNG for random crop padding / flip — but RandomCrop doesn't accept rng, it uses the default.

OK alternative: manually replicate RandomCrop + Flip with explicit RNG.

Even simpler: don't store raw images; store a sample of normalized augmented views, and at retrieval time, sample DIFFERENT post-aug samples from the same raw. To do this cleanly: at each training step, augment the bank image freshly. The bank stores the raw uint8; for each query, we generate a fresh augmented view, using a torch.Generator seeded by the bank's RNG.

Let me implement this manually with custom transforms:

Now let me update the dataset to:

Expose raw uint8 images directly (no implicit transform on __getitem__)
Apply transforms at the dataloader level
Support returning raw images (for the bank storage path)
Wait — the previous class_counts was iterating over all class IDs 0..num_classes-1, but this might fail when only a subset is used (e.g., task 2 has classes 10-19; mask includes only those, so RTT of class_counts checked targets == c for c in range(self._num_classes). With _num_classes = max(indices) + 1, this iterates 0..max, but the mask only has classes in class_indices. For indices 0-9 and class_counts indexed at index 5, (targets == 5).sum() would correctly sum only for that class within the view. OK that's fine — total counts equals samples per class across all 10 existing class IDs including zeros. But the class_counts returns a list of size _num_classes = max(indices)+1.

Hmm but result of [0, 50, 50, ...] requires 0 to be populated. Wait — for task view 0..9, _num_classes = 10. Counts list has indices 0..9 with counts for those. For task view 10..19, _num_classes = 20. Counts list has indices 0..19 but only 10..19 have samples — so counts[0..9] = 0, counts[10..19] = samples.

I changed the loop to iterate _class_indices only. This makes counts shorter. Let me check if other code relies on the long dense list:

OK, so imbalance_ratio needs min(counts) > 0 to make sense. The CIFAR-100 tasks are balanced within task (500 training per class), so min(counts) > 0 if we restrict to _class_indices. Let me keep that.

Now the CIFAR-100 dataset's class_counts returns size = max(_class_indices) + 1. But for use downstream, this is fine — it's a property called rarely.

Now let me update the datamodule to:

Use the standard train transform via make_train_transform
Use the standard eval transform via make_eval_transform
NOT apply transform inside __getitem__ to keep raw_images accessible
Let me also check the test files, output manager, other supporting code.

Good. Now I have full context. Let me make a careful fix plan.

Fix 1: Standard CIFAR-100 Protocol Design
Based on the standard "Class-IL Split CIFAR-100" protocol (Rebuffi et al., iCaRL 2017; Buzzega et al., DER/Mammoth 2020):

Standard practice	Current implementation	Action
Split 10 tasks × 10 classes	✓	Keep
Train fixed 70 epochs per task with fixed LR 0.1 (no per-task cosine decay)	Per-task cosine annealing	FIX: disable per-task scheduler decay
Held-out validation from training or fixed epochs (no early stopping on test)	Val = test set + early stopping	FIX: held-out validation split
Replay buffer stores PRE-AUGMENTATION raw images, augmented at query time	Stores post-augmentation tensors	FIX: store raw uint8
Reservoir or ring buffer sampling	Uniform random with replacement	OK
No head expansion; logits head fixed-size, masked unreachable classes at test	Head expands	OK for some papers (e.g., BiR); standard iCaRL uses fixed-size head
SGD with momentum=0.9, weight_decay=5e-4	✓	Keep
LR=0.1 (default)	✓	Keep
Mean-of-means accuracy is fine for uniform split	✓	OK
CIL metrics: AA_k, AF, BWT (forgetting only on tasks 1..T-1)	Includes task 0	FIX: exclude task 0
Let me now write all the fixes. I'll start with the most important ones.

Fix the bank to store raw images, augment at query
The cleanest approach: store the raw uint8 image (NHWC [32, 32, 3] uint8), and at query time, apply a configurable augmentation transform that matches train-time. But the method doesn't have access to the train-time transform. So I'll either: (a) pass the augmentation function into the method/bank, or (b) use the standard transform the datamodule applies.

The cleanest fix: transform is applied once at the storage boundary. Storage stores the post-transform tensor, but the augmentation is re-applied at retrieval time with a separate image. Wait — that requires storing raw image AND the augmentation function in the method.

Pragmatic approach: store the raw uint8 image. At retrieval time, the method applies its OWN T.Compose(random_crop_flip+convert_dtype+normalize) to the retrieved raw samples, using fresh random state (a separate torch.Generator). This way the bank items are augmented differently each time they're used.

Let me implement this.

from . import transforms as cifar_transforms
Hmm — let me revert that change. I'll define the transforms as a function in the dataset.py module instead of a separate file, since that's the cleanest approach. Let me revert first:

$ git checkout src/data/cifar100/__init__.py 2>&1

git : Updated 1 path from the index
At line:1 char:1
+ git checkout src/data/cifar100/__init__.py 2>&1
+ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : NotSpecified: (Updated 1 path from the index:String) [], RemoteException
    + FullyQualifiedErrorId : NativeCommandError
 
Now I'll create the augmentation transforms as a small helper module that both the dataset and the bank can use. The transforms are pure functions so this is clean.

Wait — I need to think more carefully. The issue with generator is that torchvision.transforms.RandomCrop uses its own internal RNG, which reads from torch.default_generator. We can't easily inject a generator per-call. So if the bank runs augmentation on query, it will drift the training RNG.

Let me simplify: use a self-contained augmentation function that uses its own RNG key. For the bank's replay path, we want augmentation to be deterministic per call. The cleanest approach is to refactor to run the augmentation on the raw images at call time with a per-call seeded RNG.

Actually, the simplest and standard-correct approach: the bank stores the raw uint8 image (NHWC, the dataset's stored format). At query time, the method applies a train-consistent augmentation via its own augmentation transform. The issue is that sharing one transform's RNG between training pipeline and replay pipeline creates correlation.

Pragmatic correct approach: apply augmentation with a different seed. Specifically, when the method is created, also create a separate self._augment_rng = torch.Generator(device='cpu').manual_seed(seed)or just use the bank's RNG that already exists. The bank's existingrandom.Random(seed)for selection. We can add atorch.Generator` for augmentation.

Let me redesign cleanly: the dataset loads raw uint8 images + applies train transform → model. The bank stores raw uint8 images (no augmentation). At query time, the method applies a fresh training transform to each retrieved sample, using the bank's separate RNG for random crop padding / flip — but RandomCrop doesn't accept rng, it uses the default.

OK alternative: manually replicate RandomCrop + Flip with explicit RNG.

Even simpler: don't store raw images; store a sample of normalized augmented views, and at retrieval time, sample DIFFERENT post-aug samples from the same raw. To do this cleanly: at each training step, augment the bank image freshly. The bank stores the raw uint8; for each query, we generate a fresh augmented view, using a torch.Generator seeded by the bank's RNG.

Let me implement this manually with custom transforms:

Now let me update the dataset to:

Expose raw uint8 images directly (no implicit transform on __getitem__)
Apply transforms at the dataloader level
Support returning raw images (for the bank storage path)
Wait — the previous class_counts was iterating over all class IDs 0..num_classes-1, but this might fail when only a subset is used (e.g., task 2 has classes 10-19; mask includes only those, so RTT of class_counts checked targets == c for c in range(self._num_classes). With _num_classes = max(indices) + 1, this iterates 0..max, but the mask only has classes in class_indices. For indices 0-9 and class_counts indexed at index 5, (targets == 5).sum() would correctly sum only for that class within the view. OK that's fine — total counts equals samples per class across all 10 existing class IDs including zeros. But the class_counts returns a list of size _num_classes = max(indices)+1.

Hmm but result of [0, 50, 50, ...] requires 0 to be populated. Wait — for task view 0..9, _num_classes = 10. Counts list has indices 0..9 with counts for those. For task view 10..19, _num_classes = 20. Counts list has indices 0..19 but only 10..19 have samples — so counts[0..9] = 0, counts[10..19] = samples.

I changed the loop to iterate _class_indices only. This makes counts shorter. Let me check if other code relies on the long dense list:

OK, so imbalance_ratio needs min(counts) > 0 to make sense. The CIFAR-100 tasks are balanced within task (500 training per class), so min(counts) > 0 if we restrict to _class_indices. Let me keep that.

Now the CIFAR-100 dataset's class_counts returns size = max(_class_indices) + 1. But for use downstream, this is fine — it's a property called rarely.

Now let me update the datamodule to:

Use the standard train transform via make_train_transform
Use the standard eval transform via make_eval_transform
NOT apply transform inside __getitem__ to keep raw_images accessible
Now let's update the methods. The plan:

Standard replay (StaticBankMethod):

Store raw uint8 images (the bank gets called from compute_loss with (x, y) where x is post-transform, post-permute NCHW float32)
The method should store the original raw image instead
At retrieval time, replay items are augmented using the train transform
The cleanest implementation: the method receives both the train transform and the post-transform batch. It stores the raw image. At retrieval time, it applies the transform to the retrieved raw image, producing a fresh augmented view.

For this to work, the method needs access to:

Raw images (separate from the augmented batch)
The train transform
An isolated RNG for augmentation
Implementation approach: the method's compute_loss signature gets raw images and the train transform. The runner passes both.

Let me redesign the method API:

I need to change the compute_loss API to pass raw_images. But this is a breaking change to all methods. Let me think differently.

Alternative approach: The PL module's training_step passes both (x, y) (transformed batch) and the raw batch. The transformed batch is already what gets fed to the model. The raw is stored separately, but for augmentation at retrieval the transform is re-applied.

Actually, the cleanest, minimal change: Have the compute_loss signature accept a context object with raw images, transformed images, labels, and a transform function. But that's heavy.

Simpler cleanest: Restructure the bank to be agnostic of what's stored (it just stores (feat, label) tuples and retrieves them). The runner/method performs transform decisions:

Before storing: caller decides what "feature" means (raw or transformed)
After retrieving: caller decides whether/how to transform
So the responsibility moves to the PL module OR method:

Method receives (x_transformed, y) as before.
Method also has access to a "storage callback" that records the raw image with the same label (since the bank stores (feat, label) tuples).
Method chooses what to store (raw transform) and what to retrieve-with what augment.
Let's take a clean approach: modify the dataset to expose raw images separately, and have the PL training_step load raw images alongside the batch. We can use a separate RawBankDataset that mirrors the indices.

Wait — there's a cleaner architectural solution: add a bank_data parameter to compute_loss that provides the raw image and the transform. This is minimal and explicit.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import torch

from src.bank.core.base import AbstractGhostBank


class MethodContext:
    """Optional context passed alongside the (x, y) batch to ``Method.compute_loss``.

    Carries the raw pre-augmentation view of the same batch plus the train
    augmentation pipeline.  Methods use this context to store raw images
    in the bank and re-augment replay samples at retrieval time — a
    standard practice (ERM, BiR, DER) that prevents overfitting to
    frozen augmented snapshots.
    """

    def __init__(
        self,
        raw_x: torch.Tensor,
        raw_y: torch.Tensor,
        raw_indices: torch.Tensor,
        train_transform: Callable | None = None,
        augment_rng: torch.Generator | None = None,
    ) -> None:
        self.raw_x = raw_x
        self.raw_y = raw_y
        self.raw_indices = raw_indices
        self.train_transform = train_transform
        self.augment_rng = augment_rng


class Method(ABC):
    @abstractmethod
    def compute_loss(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        pl_module,
        bank: AbstractGhostBank | None = None,
        context: MethodContext | None = None,
    ) -> torch.Tensor:
        ...
Now I need to update the dataset to optionally expose raw images. Let me add a method to get a parallel raw-images dataset view.

There's an issue — DataLoader uses __getitem__ and provides shuffled indices. The DataLoader doesn't easily expose "these are the actual indices used in this batch". The cleanest approach is to use IndexedDataset that pairs each item with its index.

I'll add an IndexedCIFAR100TaskView (or wrap the existing one) that returns (index, img, label):

Now I need to update the PL module's training_step to:

Unpack (view_idx, x, y) (or handle both old (x, y) and new API for backward compat).
Build a MethodContext with raw images and train transform.
Pass the context down to method.compute_loss.
And update the methods to use the context to store raw and re-augment retrieval.

Now update the methods to be standard replay. The key insight: the methods should store RAW images in the bank (using (x, y) from MethodContext.raw_x — which is the post-transform batch — but actually we want raw... hmm).

Wait, there's an issue. MethodContext.raw_x = x where x is the already-transformed batch. To get truly raw images, we'd need to either:

Store the raw images in a side dataset and look them up by index, OR
Change the dataset to return BOTH raw and transformed, OR
Have the PL module stash raw images keyed by view-idx and look them up at store time.
Let me restructure: the dataset's __getitem__ returns (idx, transformed_img, label), while dataset.raw_images[idx] gives the original. The PL module pre-loads dataset.raw_images once per task and references it via self._raw_ref. When the method needs to store, it uses (idx, label) to fetch raw.

But this couples PL module to the dataset. Cleaner: a RawBankDataset class that wraps a view and exposes raw images keyed by index.

Actually, let me reconsider: the simplest standard-correct approach is store the post-transform tensor AS-IS. Why? Because:

The size is small (3×32×32 floats = 12 KB).
The augmentation was deterministic per-step — but each time we sample from the bank, we re-augment a different view of the same underlying image.
Wait — but if we only stored it once, augmentation was applied once. We can't re-augment because we lost the raw.

OK final decision: store the raw uint8 NHWC image (small, ~3 KB per item). At retrieval, re-augment with the train transform on a separate RNG.

So method.store() receives (raw_img_uint8_NHWC, label). The runner / PL module orchestrating storage must look up the raw image from the dataset by the batch's view indices.

Let me update compute_loss calls — the method will:

Take the context's raw_indices to fetch raw images from a side-cache.
Store (raw_img, label) into the bank.
On retrieval, for each (raw_img, label) from the bank, apply the train transform (with isolated RNG) to produce a fresh batch, then concatenate with the live batch.
The PL module holds a reference to the current task's dataset view so it can fetch raw images. Let me thread this through:

Now the PL module needs access to the dataset view. Let me look at the on_train_start of Lightning to see if we can hook into the train dataloader:

OK good. Now for the standard replay implementation, let me re-design the methods. The base idea:

bank.store(raw_examples): takes a list of (raw_uint8_image_tensor, label).
At retrieval: the method applies the train transform with isolated RNG to each retrieved item and concatenates.
The PL module's _augment_generator uses the bank's random seed for isolation. The bank's RNG is initialized from a deterministic seed. Let me use a per-task per-bank generator.

Now let me think about method signature cleanly: I'll add a bank_bank_setup parameter that the runner calls once per bank, passing the bank, raw dataset reference, and transform. Then compute_loss(batch, pl_module) knows the bank and uses pl_module.bank to store/retrieve.

Final clean architecture:

method.compute_loss(batch, pl_module, bank, context=None)
The PL module exposes self.bank, self._train_transform, self._augment_generator to methods via the pl_module reference.
Methods that use the bank read raw images from pl_module._raw_dataset_ref (or the PL module looks them up by batch indices).
Even simpler: the MethodContext carries raw_x, raw_y as raw uint8 NHWC tensors. The PL module looks them up at training_step. For this, the PL module needs a reference to the current task's CIFAR100TaskView so it can do task_dataset.raw_images[indices].

Let me thread the task dataset into the PL module via trainer.fit:

I'll add raw_dataset_view to PL module:

Now redesign the methods. The new contract:

Method.compute_loss(batch, pl_module, bank, context): bank is given. context provides transform/RNG.
For replay methods:
Store: bank.store(...) with raw examples (the method can fetch raw from collate indices via context.raw_indices and look up raw images via... hmm, it doesn't have access to the dataset view)
The issue: the method needs raw images keyed by collate index. Options:

PL module looks up raw images at training_step time and includes them in context.raw_x.
Method fetches from pl_module.bank_dataset_view.
Option 1 is cleaner: the PL module has the train dataset reference and looks up raw via pl_module.train_dataset.raw_images[idx]. We need to thread the dataset view into the PL module.

Now I need to expose raw_images and raw_targets on the dataset — already done. Now update the methods.