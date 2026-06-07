@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ================================================
echo    NoCall 스마트카트 데모 (웹캠 실시간 인식)
echo ================================================
echo.
echo [1/2] 필요한 프로그램을 준비합니다 (처음엔 몇 분 걸려요)...
call uv sync
echo.
echo [2/2] 잠시 후 인터넷 창이 자동으로 열립니다.
echo       웹캠 사용 허용을 눌러 주세요.
echo.
echo  * 끝내려면: 이 검은 창을 클릭한 뒤 Ctrl 키와 C 키를 함께 누르세요.
echo.
timeout /t 3 >nul
start "" http://localhost:8000
call uv run python run.py
pause
