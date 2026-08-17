@echo off
rem ================================================================
rem Diagnose model loading: restart app, capture hilog to diag\ folder
rem Run this by double click. Output files land in 05_device_files\diag\
rem ================================================================
setlocal EnableDelayedExpansion

set "REAL_DIR=/data/app/el2/100/base/com.huawei.cannkit.llmengine/haps/entry/files"
set "BUNDLE=com.huawei.cannkit.llmengine"
set "SRC=%~dp0"
set "DIAG=%SRC%diag"
set "TMP_OUT=%TEMP%\cannkit_hdc_out.txt"
if not exist "%DIAG%" mkdir "%DIAG%"

rem ---- locate hdc ----
set "HDC="
where hdc >nul 2>&1
if %errorlevel%==0 set "HDC=hdc"
if not defined HDC if exist "C:\Program Files\Huawei\DevEco Studio\sdk\default\openharmony\toolchains\hdc.exe" set "HDC=C:\Program Files\Huawei\DevEco Studio\sdk\default\openharmony\toolchains\hdc.exe"
if not defined HDC (
    echo [FAIL] hdc not found.
    goto :fail
)

rem ---- device check with server-port fallback ----
"%HDC%" list targets > "%TMP_OUT%" 2>&1
for %%A in ("%TMP_OUT%") do if %%~zA equ 0 call :find_server_port
"%HDC%" list targets > "%TMP_OUT%" 2>&1
type "%TMP_OUT%"
for %%A in ("%TMP_OUT%") do if %%~zA equ 0 (
    echo [FAIL] no device connected. Unplug/replug USB, then rerun.
    goto :fail
)

echo == app info (bm dump) ==
"%HDC%" shell "bm dump -n %BUNDLE%" > "%DIAG%\bm_dump.txt" 2>&1
findstr /I /C:"appIndex" /C:"userId" /C:"installPath" /C:"codePath" "%DIAG%\bm_dump.txt"

echo == files and permissions ==
"%HDC%" shell "ls -laZ %REAL_DIR%" > "%DIAG%\files_ls.txt" 2>&1
type "%DIAG%\files_ls.txt"

echo == restarting app ==
"%HDC%" shell "aa force-stop %BUNDLE%" >nul 2>&1
"%HDC%" shell "sleep 1" >nul 2>&1
"%HDC%" shell "hilog -r" >nul 2>&1
"%HDC%" shell "aa start -a EntryAbility -b %BUNDLE%"
echo    waiting 30s for model load ...
timeout /t 30 /nobreak >nul

echo == app process ==
"%HDC%" shell "ps -ef | grep %BUNDLE%" > "%DIAG%\app_ps.txt" 2>&1
type "%DIAG%\app_ps.txt"

echo == dumping hilog ==
"%HDC%" shell "hilog -x" > "%DIAG%\hilog_dump.txt" 2>&1

echo.
echo ---- LLM related log lines ----
findstr /I /C:"LLM_DEMO" /C:"llm_engine" /C:"hiai" "%DIAG%\hilog_dump.txt" | more +0
echo.
echo [DONE] full logs saved to:
echo    %DIAG%\hilog_dump.txt
echo    %DIAG%\bm_dump.txt
echo    %DIAG%\files_ls.txt
echo    %DIAG%\app_ps.txt
goto :end

:find_server_port
set "HDC_PORT="
for /f "tokens=2 delims=," %%I in ('tasklist /FI "IMAGENAME eq hdc.exe" /FO CSV /NH 2^>nul') do (
    for /f "tokens=2" %%A in ('netstat -ano 2^>nul ^| findstr /C:"LISTENING" ^| findstr /C:"127.0.0.1:" ^| findstr /R /C:" %%~I$"') do (
        for /f "tokens=2 delims=:" %%Q in ("%%A") do if not defined HDC_PORT set "HDC_PORT=%%Q"
    )
)
if defined HDC_PORT set "HDC_SERVER_PORT=!HDC_PORT!"
exit /b 0

:fail
echo.
echo Diagnose aborted.
pause
exit /b 1

:end
pause
exit /b 0
