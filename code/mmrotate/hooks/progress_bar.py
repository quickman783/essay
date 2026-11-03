# Copyright (c) OpenMMLab. All rights reserved.
"""Keras 风格的实时进度条。

每个 iteration 都在同一行刷新，显示：

    Epoch 1/12 [=========>----------] 137/813 16s 118ms/step loss: 1.2841 loss_cls: 0.8132 lr: 1.00e-04 eta: 1:19:20

用法（写在配置里）：

    custom_imports = dict(imports=["mmrotate.hooks.progress_bar"])
    custom_hooks = [dict(type="ProgressBarHook", bar_width=24)]

说明：
  * 只往终端写，不进日志文件；日志文件由 LoggerHook 在每个 epoch 结束时写一行；
  * 终端不支持回车刷新（例如被管道捕获）时，自动退化成每 N 次迭代打印一行；
  * 优先级设为 VERY_LOW，保证在其他 hook 更新完 loss 之后再读取数据。
"""

import sys
import time

from mmengine.hooks import Hook
from mmrotate.registry import HOOKS

# 这些标量是耗时/内存统计，不算进 loss 统计
_NON_LOSS_KEYS = {"time", "data_time", "memory", "lr", "momentum", "grad_norm"}


@HOOKS.register_module()
class ProgressBarHook(Hook):
    """在终端用单行进度条实时显示训练进度。

    Args:
        bar_width (int): 进度条方块的字符宽度，默认 24。
        loss_keys (list[str], optional): 需要显示的 loss 项，默认显示
            ``loss`` 以及以 ``loss_`` 开头的项。
        log_interval (int): 终端不支持回车刷新时，每多少次迭代打印一行。
        show_eta (bool): 是否显示预计剩余时间。
    """

    priority = "VERY_LOW"

    def __init__(self,
                 bar_width: int = 24,
                 loss_keys=None,
                 log_interval: int = 10,
                 show_eta: bool = True) -> None:
        self.bar_width = max(int(bar_width), 5)
        self.loss_keys = list(loss_keys) if loss_keys is not None else None
        self.log_interval = max(int(log_interval), 1)
        self.show_eta = show_eta

        self._interactive = True
        self._epoch_start = 0.0
        self._iter_count = 0
        self._loss_sums = {}
        self._epoch_max_epochs = 0
        self._line_ended = True

    # ---------------------------------------------------------------- 工具方法
    @staticmethod
    def _fmt_seconds(seconds: float) -> str:
        seconds = max(int(seconds), 0)
        hours, remainder = divmod(seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours:
            return f"{hours}:{minutes:02d}:{secs:02d}"
        return f"{minutes}:{secs:02d}"

    def _render_bar(self, done: int, total: int) -> str:
        ratio = done / total if total else 0.0
        filled = int(round(self.bar_width * ratio))
        if filled >= self.bar_width:
            return "[" + "=" * self.bar_width + "]"
        return "[" + "=" * filled + ">" + "-" * (self.bar_width - filled - 1) + "]"

    def _collect_loss_text(self) -> str:
        if not self._loss_sums or not self._iter_count:
            return ""
        names = self.loss_keys
        if names is None:
            names = [k for k in self._loss_sums if k == "loss" or k.startswith("loss_")]
        parts = []
        for name in names:
            if name in self._loss_sums:
                parts.append(f"{name}: {self._loss_sums[name] / self._iter_count:.4f}")
        return " ".join(parts)

    def _current_lr(self, runner) -> str:
        log_scalars = runner.message_hub.log_scalars
        for key, buffer in log_scalars.items():
            if key.endswith("/lr"):
                try:
                    return f"{buffer.current():.2e}"
                except Exception:
                    return ""
        return ""

    def _write(self, text: str, newline: bool = False) -> None:
        if self._interactive:
            sys.stdout.write("\r" + text)
            if newline:
                sys.stdout.write("\n")
        else:
            sys.stdout.write(text + "\n")
        sys.stdout.flush()
        self._line_ended = newline

    # ------------------------------------------------------------------ 钩子
    def before_run(self, runner) -> None:
        # 输出被重定向到管道/文件时，回车刷新会变成一坨，改用逐行打印
        try:
            self._interactive = bool(sys.stdout.isatty())
        except Exception:
            self._interactive = False

    def before_train_epoch(self, runner) -> None:
        self._epoch_start = time.time()
        self._iter_count = 0
        self._loss_sums = {}
        self._epoch_max_epochs = runner.train_loop.max_epochs

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None) -> None:
        if outputs:
            for key, value in outputs.items():
                if key in _NON_LOSS_KEYS:
                    continue
                try:
                    self._loss_sums[key] = self._loss_sums.get(key, 0.0) + float(value)
                except (TypeError, ValueError):
                    continue

        self._iter_count += 1
        done = batch_idx + 1
        total = len(runner.train_dataloader)
        elapsed = time.time() - self._epoch_start
        ms_per_step = elapsed / done * 1000 if done else 0.0

        eta = ""
        if self.show_eta and done:
            remaining = (total - done) * (elapsed / done)
            eta = f" eta: {self._fmt_seconds(remaining)}"

        loss_text = self._collect_loss_text()
        lr_text = self._current_lr(runner)

        line = (f"Epoch {runner.epoch + 1}/{self._epoch_max_epochs} "
                f"{self._render_bar(done, total)} "
                f"{done}/{total} "
                f"{self._fmt_seconds(elapsed)} {ms_per_step:.0f}ms/step")
        if loss_text:
            line += " " + loss_text
        if lr_text:
            line += f" lr: {lr_text}"
        line += eta

        if self._interactive:
            # 最后一个 iteration 之后换行，好让 LoggerHook 的整轮汇总另起一行
            self._write(line, newline=(done >= total))
        elif done % self.log_interval == 0 or done >= total:
            self._write(line, newline=True)

    def before_val_epoch(self, runner) -> None:
        # 上一轮训练的最后一次迭代已经换过行，这里不重复换行
        if not self._line_ended:
            sys.stdout.write("\n")
            sys.stdout.flush()
            self._line_ended = True

    def after_val_epoch(self, runner, metrics=None) -> None:
        if not metrics:
            return
        parts = []
        for key in ("mAP", "mAP_50", "AP50"):
            if key in metrics:
                parts.append(f"{key}: {float(metrics[key]):.4f}")
        if not parts:
            parts = [f"{k}: {v:.4f}" for k, v in metrics.items()
                     if isinstance(v, (int, float))]
        if not parts:
            return
        # 验证发生在训练 epoch 结束之后，此时 runner.epoch 已经加 1，
        # 与训练进度条显示的是同一个 epoch 序号
        text = (f"Epoch {runner.epoch}/{self._epoch_max_epochs} "
                f"验证结果 -> " + "  ".join(parts)
                + f"   验证完成于 {time.strftime('%H:%M:%S')}")
        sys.stdout.write(text + "\n")
        sys.stdout.flush()
        self._line_ended = True

    def after_train_epoch(self, runner) -> None:
        if not self._line_ended:
            sys.stdout.write("\n")
            sys.stdout.flush()
            self._line_ended = True


def _unused():  # pragma: no cover - 保持 math 的显式引用，避免 lint 报未使用
    return math.isfinite(0.0)
