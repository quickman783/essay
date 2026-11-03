@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
title DMM / MM Rotate env setup (py3.10 + torch2.1.2 cu118)

set CONDA_EXE=C:\Users\feng1\anaconda3\Scripts\conda.exe
set ENV_NAME=dmm
set PY=C:\Users\feng1\anaconda3\envs\%ENV_NAME%\python.exe
set PT_INDEX=https://download.pytorch.org/whl/cu118
set MMCV_INDEX=https://download.openmmlab.com/mmcv/dist/cu118/torch2.1.0/index.html
set PYPI=-i https://pypi.tuna.tsinghua.edu.cn/simple

echo ============================================================
echo  Step 1/7  create conda env "%ENV_NAME%" (python 3.10)
echo ============================================================
if exist "%PY%" (
    echo     env already exists, skip.
) else (
    call "%CONDA_EXE%" create -n %ENV_NAME% python=3.10 -y
)
if not exist "%PY%" goto :fail

echo.
echo ============================================================
echo  Step 2/7  upgrade pip to 24.0
echo ============================================================
"%PY%" -m pip install --upgrade "pip==24.0" %PYPI%
if errorlevel 1 goto :fail

echo.
echo ============================================================
echo  Step 3/7  install torch 2.1.2 + cu118  (~2.5 GB, slow)
echo ============================================================
"%PY%" -m pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url %PT_INDEX%
if errorlevel 1 goto :fail

echo.
echo ============================================================
echo  Step 4/7  check CUDA
echo ============================================================
"%PY%" -c "import torch;print('  torch     :',torch.__version__);print('  cuda build:',torch.version.cuda);print('  available :',torch.cuda.is_available());print('  gpu       :',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
if errorlevel 1 goto :fail

echo.
echo ============================================================
echo  Step 5/7  install mmengine / mmcv / mmdet (order matters!)
echo ============================================================
"%PY%" -m pip install mmengine==0.10.5 %PYPI%
if errorlevel 1 goto :fail
"%PY%" -m pip install mmcv==2.1.0 -f %MMCV_INDEX%
if errorlevel 1 goto :fail
"%PY%" -m pip install mmdet==3.3.0 %PYPI%
if errorlevel 1 goto :fail
"%PY%" -m pip install numpy==1.26.4 %PYPI%
if errorlevel 1 goto :fail

echo.
echo ============================================================
echo  Step 6/7  other deps (timm / einops / fvcore ...)
echo ============================================================
"%PY%" -m pip install timm==0.9.16 einops fvcore scikit-learn scipy shapely pycocotools terminaltables six matplotlib %PYPI%
if errorlevel 1 goto :fail

echo.
echo ============================================================
echo  Step 7/7  install project itself (editable, no deps)
echo ============================================================
"%PY%" -m pip install -e . --no-deps
if errorlevel 1 goto :fail

echo.
echo ============================================================
echo  Verify
echo ============================================================
"%PY%" -c "import mmcv,mmengine,mmdet,numpy,timm,mmrotate;print('  mmcv    :',mmcv.__version__);print('  mmengine:',mmengine.__version__);print('  mmdet   :',mmdet.__version__);print('  numpy   :',numpy.__version__);print('  timm    :',timm.__version__);print('  mmrotate:',mmrotate.__version__)"
"%PY%" -c "from mmcv.ops import nms;print('  mmcv CUDA ops: OK')"
"%PY%" -c "import sys;sys.path.insert(0,'.');from mmrotate.models.backbones.vmamba.vmamba import DEFAULT_SELECTIVE_SCAN as D;print('  selective_scan impl:',D.__name__)"

echo.
echo ============================================================
echo  DONE.  Activate with:  conda activate %ENV_NAME%
echo ============================================================
pause
exit /b 0

:fail
echo.
echo [ERROR] something failed above. scroll up and read the red text.
pause
exit /b 1
