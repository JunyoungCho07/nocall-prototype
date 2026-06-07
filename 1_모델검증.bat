@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ================================================
echo    NoCall 모델 검증 (웹캠 없이 숫자로 확인)
echo ================================================
echo.
echo 처음 실행은 준비에 몇 분 걸릴 수 있어요. 기다려 주세요...
echo.
call uv sync
echo.
call uv run python train/verify.py
echo.
echo ================================================
echo  위 표에서 5개 항목이 모두 PASS 면 정상입니다.
echo  결과 그림은 runs\verify 폴더에서 볼 수 있어요.
echo ================================================
pause
