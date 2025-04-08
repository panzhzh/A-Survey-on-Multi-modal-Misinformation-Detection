#!/usr/bin/env python3
# monitor.py
# -*- coding: utf-8 -*-

import time
import threading
import torch
import wandb
import numpy as np
import pynvml  # 直接导入 pynvml，不再做可用性检查

from sklearn.metrics import precision_recall_curve, roc_curve

# 在模块导入时就进行 W&B 登录
wandb.login(key="07cbc531f83e7b149d1cd6e3f7ca3b71dc686136")


class GPUUsageMonitor:
    """
    后台线程，定期查询 GPU 利用率和显存使用率并保存。
    使用 pynvml 获取信息，需要先:
      pip install pynvml
    """
    def __init__(self, gpu_index=0, interval=1.0):
        """
        Args:
            gpu_index: 要监控的 GPU 编号，默认为 0
            interval:  采样间隔，单位秒
        """
        self.gpu_index = gpu_index
        self.interval = interval
        self.keep_running = False
        self.thread = None

        # 采样记录
        self.gpu_usage_list = []  # GPU 核心利用率(%)序列
        self.mem_usage_list = []  # GPU 显存使用率(%)序列

        # 初始化 pynvml
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)

    def start(self):
        """启动后台线程，循环采集 GPU 利用率。"""
        self.keep_running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        """停止后台线程，并释放 NVML 句柄。"""
        self.keep_running = False
        if self.thread is not None:
            self.thread.join()
        pynvml.nvmlShutdown()

    def _run(self):
        """后台线程循环，每隔 interval 秒采一次。"""
        while self.keep_running:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                gpu_util_percent = util.gpu  # 单位：%
                self.gpu_usage_list.append(gpu_util_percent)

                # 显存使用率
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                used_mb = mem_info.used / (1024 ** 2)
                total_mb = mem_info.total / (1024 ** 2)
                mem_util_percent = 100.0 * used_mb / total_mb
                self.mem_usage_list.append(mem_util_percent)

            except:
                pass
            time.sleep(self.interval)

    @property
    def avg_gpu_usage(self):
        """采样到的 GPU 核心平均利用率(%)"""
        if len(self.gpu_usage_list) == 0:
            return 0.0
        return float(np.mean(self.gpu_usage_list))

    @property
    def avg_mem_usage(self):
        """采样到的 GPU 显存平均使用率(%)"""
        if len(self.mem_usage_list) == 0:
            return 0.0
        return float(np.mean(self.mem_usage_list))


class TrainingMonitor:
    """
    用于训练过程的统一监控：
      - GPU 利用率 & 显存使用率
      - 样本处理速率 (SPS)
      - Weights & Biases 日志记录 (loss / acc / lr / PR曲线 / ROC曲线 等)
    """

    def __init__(
        self,
        project_name="DefaultProject",
        run_name="DefaultRun",
        gpu_index=0,
        enable_gpu_monitor=True,
        wandb_config=None
    ):
        """
        Args:
            project_name: W&B 项目名称
            run_name:     当前运行名称
            gpu_index:    要监控的 GPU ID
            enable_gpu_monitor: 是否启用 GPU 监控
            wandb_config: 传给 wandb.init 的配置，如超参
        """
        self.project_name = project_name
        self.run_name = run_name
        self.gpu_index = gpu_index
        self.enable_gpu_monitor = enable_gpu_monitor

        # 初始化 W&B
        wandb.init(
            project=self.project_name,
            name=self.run_name,
            config=wandb_config if wandb_config else {}
        )

        # GPU 监控线程
        self.gpu_monitor = None
        if enable_gpu_monitor:
            self.gpu_monitor = GPUUsageMonitor(gpu_index=self.gpu_index)
            self.gpu_monitor.start()

        # SPS 相关
        self.last_time = None
        self.last_count = 0
        self.sample_counter = 0
        self.sps_list = []  # 新增：记录每次的 sps

        # 避免多次 stop
        self._stopped = False

    def stop(self):
        """
        手动停止 GPU 监控线程，并将平均 GPU 核心和显存利用率记到 W&B，然后结束 wandb。
        """
        if self._stopped:
            return
        self._stopped = True

        if self.gpu_monitor:
            self.gpu_monitor.stop()
            avg_gpu_usage = self.gpu_monitor.avg_gpu_usage
            avg_gpu_mem   = self.gpu_monitor.avg_mem_usage

            print(f"[GPU Monitor] Average GPU usage = {avg_gpu_usage:.2f}%")
            print(f"[GPU Monitor] Average GPU memory usage = {avg_gpu_mem:.2f}%")

            wandb.log({
                "avg_gpu_usage": avg_gpu_usage,
                "avg_gpu_mem_usage": avg_gpu_mem
            }, commit=True)

        wandb.finish()

    def log_metrics(self, metrics_dict, step=None):
        """
        向 W&B 记录任意指标，如 {"train_loss":0.12, "train_acc":0.95}。
        Args:
            metrics_dict: 指标字典
            step:         步数(可选)
        """
        if step is not None:
            wandb.log(metrics_dict, step=step)
        else:
            wandb.log(metrics_dict)

    def log_lr(self, optimizer, step=None):
        """记录当前学习率到 W&B。"""
        if len(optimizer.param_groups) > 0:
            lr = optimizer.param_groups[0]["lr"]
            self.log_metrics({"learning_rate": lr}, step=step)

    def update_sps(self, batch_size, step=None):
        """
        记录 Samples Per Second (SPS)。
        在每个训练 step 调用，传入本 step 的 batch_size。
        """
        now = time.time()
        self.sample_counter += batch_size

        if self.last_time is None:
            self.last_time = now
            self.last_count = self.sample_counter
            return

        elapsed = now - self.last_time
        if elapsed > 0:
            sps = (self.sample_counter - self.last_count) / elapsed
            # 记录 sps
            self.sps_list.append(sps)
            self.log_metrics({"sps": sps}, step=step)

            self.last_time = now
            self.last_count = self.sample_counter

    def log_pr_curve(self, y_true, y_probs, class_names=None, step=None):
        """
        记录多分类/二分类的 PR 曲线到 W&B。
        y_true:  [N], int
        y_probs: [N, C]
        class_names: 可选
        step:    当前步数
        """
        num_classes = y_probs.shape[1]
        if class_names is None:
            class_names = [f"Class_{i}" for i in range(num_classes)]

        for c in range(num_classes):
            y_true_c = (y_true == c).astype(int)
            y_prob_c = y_probs[:, c]
            precision, recall, _ = precision_recall_curve(y_true_c, y_prob_c)
            wandb.log({
                f"pr_curve/{class_names[c]}": wandb.plot.line_series(
                    xs=recall,
                    ys=[precision],
                    keys=[f"Precision_{class_names[c]}"],
                    title=f"PR Curve - {class_names[c]}",
                    xname="Recall"
                )
            }, step=step)

    def log_roc_curve(self, y_true, y_probs, class_names=None, step=None):
        """
        记录多分类/二分类的 ROC 曲线到 W&B。
        y_true:  [N], int
        y_probs: [N, C], 每个类别的概率
        class_names: 可选
        step:    当前步数
        """
        num_classes = y_probs.shape[1]
        if class_names is None:
            class_names = [f"Class_{i}" for i in range(num_classes)]

        for c in range(num_classes):
            y_true_c = (y_true == c).astype(int)
            y_prob_c = y_probs[:, c]
            fpr, tpr, _ = roc_curve(y_true_c, y_prob_c)
            wandb.log({
                f"roc_curve/{class_names[c]}": wandb.plot.line_series(
                    xs=fpr,
                    ys=[tpr],
                    keys=[f"TPR_{class_names[c]}"],
                    title=f"ROC Curve - {class_names[c]}",
                    xname="FPR"
                )
            }, step=step)

    def get_avg_gpu_util(self):
        """
        返回当前监控到的平均 GPU 利用率(%)。
        如果未启用 GPU 监控或没有采集到数据，返回 0.0。
        """
        if self.gpu_monitor is not None:
            return self.gpu_monitor.avg_gpu_usage
        return 0.0

    def get_avg_gpu_mem(self):
        """
        返回当前监控到的平均 GPU 显存使用率(%)。
        如果未启用 GPU 监控或没有采集到数据，返回 0.0。
        """
        if self.gpu_monitor is not None:
            return self.gpu_monitor.avg_mem_usage
        return 0.0

    def get_avg_sps(self):
        """
        返回当前记录到的平均 SPS (Samples Per Second)。
        如果尚未采集到数据，返回 0.0。
        """
        if len(self.sps_list) == 0:
            return 0.0
        return float(np.mean(self.sps_list))
