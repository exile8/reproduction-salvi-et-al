import random
import math
from torch.utils.data import Sampler

class BalancedBatchSampler(Sampler):

    def __init__(self, labels, batch_size, seed=42, drop_last=False, strict_labels=True):
        if batch_size % 2 != 0:
            raise ValueError(f"batch_size must be even, got {batch_size}")

        self.batch_size = batch_size
        self.samples_per_class = batch_size // 2
        self.base_seed = int(seed)
        self.drop_last = bool(drop_last)
        self.strict_labels = bool(strict_labels)
        self.epoch = 0

        bon, spf = [], []
        for idx, label in enumerate(labels):
            if hasattr(label, "item"):
                label = label.item()
            label = int(label)

            if self.strict_labels and label not in (0, 1):
                raise ValueError(f"Unexpected label={label} at idx={idx}. Expected 0/1.")

            (bon if label == 1 else spf).append(idx)

        if not bon or not spf:
            raise ValueError(f"One of the classes is empty: bonafide={len(bon)}, spoof={len(spf)}")

        self.bonafide_indices = bon
        self.spoof_indices = spf

        self.bon_size = len(bon)
        self.spf_size = len(spf)
        self.majority_size = max(self.bon_size, self.spf_size)

        self.num_full_batches = self.majority_size // self.samples_per_class
        self.remainder = self.majority_size % self.samples_per_class

        if self.drop_last:
            self.num_batches = self.num_full_batches
        else:
            self.num_batches = self.num_full_batches + (1 if self.remainder > 0 else 0)

        minority_size = min(self.bon_size, self.spf_size)
        minority_repeats_est = math.ceil(self.majority_size / minority_size)
        print("BalancedBatchSampler (no majority repeats):")
        print(f"  Bonafide: {self.bon_size:,}")
        print(f"  Spoof:    {self.spf_size:,}")
        print(f"  Batch size: {batch_size} ({self.samples_per_class} per class)")
        print(f"  drop_last: {self.drop_last}")
        print(f"  Batches/epoch: {self.num_batches:,} "
              f"(full={self.num_full_batches:,}, remainder={self.remainder})")
        print(f"  Minority repeats (approx): ~{minority_repeats_est}x")
        if not self.drop_last and self.remainder:
            print(f"  Last batch will be size {2*self.remainder} (balanced)")

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __iter__(self):
        rng = random.Random(self.base_seed + self.epoch)

        bon = self.bonafide_indices.copy()
        spf = self.spoof_indices.copy()
        rng.shuffle(bon)
        rng.shuffle(spf)

        if self.bon_size >= self.spf_size:
            majority, minority = bon, spf
        else:
            majority, minority = spf, bon

        # minority replacement through cyclic buffer
        minority_buf = minority.copy()
        rng.shuffle(minority_buf)
        min_ptr = 0

        maj_ptr = 0

        # 1) full batches
        for _ in range(self.num_full_batches):
            batch = []

            # strictly unique majority (no repeats)
            batch.extend(majority[maj_ptr:maj_ptr + self.samples_per_class])
            maj_ptr += self.samples_per_class

            # the same amount of minority (replacement)
            for _ in range(self.samples_per_class):
                if min_ptr >= len(minority_buf):
                    minority_buf = minority.copy()
                    rng.shuffle(minority_buf)
                    min_ptr = 0
                batch.append(minority_buf[min_ptr])
                min_ptr += 1

            rng.shuffle(batch)
            yield batch

        # 2) majority tail (only if drop_last=False)
        if (not self.drop_last) and self.remainder > 0:
            batch = []

            # remainder unique majority
            batch.extend(majority[maj_ptr:maj_ptr + self.remainder])

            # the same amount of minority (replacement)
            for _ in range(self.remainder):
                if min_ptr >= len(minority_buf):
                    minority_buf = minority.copy()
                    rng.shuffle(minority_buf)
                    min_ptr = 0
                batch.append(minority_buf[min_ptr])
                min_ptr += 1

            rng.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_batches