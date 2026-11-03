"""环境自检：显卡 / CUDA / mmcv 全家桶 / 编译工具链。

用法：在 VS Code 里打开本文件，点右上角 ▶ 运行即可。
只做检查，不安装任何东西。

怎么看结果：
  * CUDA 是否可用 = True                  -> 显卡能参与计算
  * mmengine / mmcv / mmdet / mmrotate 都在 -> 训练框架齐全
  * selective_scan_cuda_* 与 nvcc 都缺失   -> 用的是纯 PyTorch 实现，速度慢
"""

import importlib
import importlib.util
import os
import shutil
import subprocess
import sys

# 避免控制台代码页导致的乱码
for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

PACKAGES = [
    "torch",
    "mmengine",
    "mmcv",
    "mmdet",
    "mmrotate",
    "timm",
    "einops",
    "fvcore",
    "selective_scan_cuda",
    "selective_scan_cuda_core",
    "selective_scan_cuda_oflex",
    "triton",
]

CONDA_PYTHON = r"C:\Users\feng1\anaconda3\envs\conda\python.exe"


def section(title):
    print()
    print("=" * 62)
    print(title)
    print("=" * 62)


def check_gpu():
    section("1) 显卡驱动")
    if shutil.which("nvidia-smi"):
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, encoding="utf-8", errors="replace")
        print(out.stdout.strip() or out.stderr.strip())
    else:
        print("nvidia-smi 未找到 —— 没有 NVIDIA 显卡或驱动未安装")


def check_python():
    section("2) Python 与框架")
    print("python :", sys.version.split()[0])
    print("路径   :", sys.executable)
    if os.path.abspath(sys.executable).lower() != os.path.abspath(CONDA_PYTHON).lower():
        print(f"注意   : 当前解释器不是 {CONDA_PYTHON}，包可能不全")

    for name in PACKAGES:
        try:
            available = importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            available = name in sys.modules
        if not available:
            print(f"  {name:<26} 缺失")
            continue
        try:
            module = importlib.import_module(name)
            version = getattr(module, "__version__", "已安装")
        except Exception as exc:                       # 导入失败也算不可用
            print(f"  {name:<26} 导入失败：{type(exc).__name__}: {exc}")
            continue
        print(f"  {name:<26} {version}")

    section("3) CUDA 可用性")
    try:
        import torch
        print("torch 版本      :", torch.__version__)
        print("torch 编译 CUDA :", torch.version.cuda)
        print("CUDA 是否可用   :", torch.cuda.is_available())
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            print("显卡            :", torch.cuda.get_device_name(0))
            print("显存            : %.1f GB" % (props.total_memory / 1024 ** 3))
            print("计算能力        : SM %d.%d" % (props.major, props.minor))
            # 真跑一次矩阵乘，确认 GPU 不只是标志位为真
            a = torch.randn(256, 256, device="cuda")
            print("GPU 矩阵乘测试  :", float((a @ a).sum()) != 0.0)
    except Exception as exc:
        print("torch 检查失败：", type(exc).__name__, exc)


def check_toolchain():
    section("4) 编译工具链（编译 selective_scan 需要）")
    nvcc = shutil.which("nvcc")
    if nvcc:
        out = subprocess.run(["nvcc", "--version"], capture_output=True, text=True,
                             encoding="utf-8", errors="replace")
        print("nvcc   :", nvcc)
        print((out.stdout or "").strip().splitlines()[-1] if out.stdout else "")
    else:
        print("nvcc   : 未安装（无 CUDA Toolkit）")

    for tool in ("cl", "clang", "clang-cl"):
        path = shutil.which(tool)
        print(f"{tool:<7}: {path if path else '未找到'}")

    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    print("CUDA_HOME:", cuda_home if cuda_home else "未设置")


def conclusion():
    section("结论")
    has_framework = all(
        importlib.util.find_spec(name) is not None
        for name in ("mmengine", "mmcv", "mmdet", "mmrotate")
    )
    has_kernel = importlib.util.find_spec("selective_scan_cuda_oflex") is not None
    has_compiler = bool(shutil.which("nvcc") and (shutil.which("cl") or shutil.which("clang")))

    print("训练框架（mmcv 全家桶）:", "齐全，可以训练" if has_framework else "不齐，缺包")
    if has_kernel:
        print("VMamba CUDA 算子       : 已安装，使用官方快速实现")
    else:
        print("VMamba CUDA 算子       : 未安装，当前使用纯 PyTorch 并行扫描（较慢）")
    print("能否自行编译 CUDA 算子 :", "可以（nvcc 与编译器齐全）" if has_compiler
          else "不可以（缺少 nvcc 或 MSVC/clang）")


if __name__ == "__main__":
    check_gpu()
    check_python()
    check_toolchain()
    conclusion()
    print()
