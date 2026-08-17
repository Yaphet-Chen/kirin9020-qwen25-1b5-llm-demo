@echo off
rem ================================================================
rem Route B (HarmonyOS NEXT): push 7 model/config files into app sandbox
rem Generic version: works in BOTH delivery dirs (05_device_files_base\
rem and 05_device_files_instruct\) - it auto-detects the *.omc name
rem (Qwen25_1b5_Base_*.omc / Qwen25_1b5_Instruct_*.omc) and the
rem embedding file names in its own directory. Keep this file inside
rem the delivery dir (it locates files by its own path).
rem NOTE 1: hdc v3.x prints nothing and exits 0 when it fails, so every
rem         step below checks the captured output instead of exit codes.
rem NOTE 2: DevEco Studio may run its hdc server on a non-default port
rem         (e.g. 7035). If the default port sees no device, we reuse the
rem         port of an already-running hdc server, otherwise our own server
rem         cannot grab the USB device from it.
rem NOTE 3: switching demo models (base <-> instruct) requires pushing the
rem         FULL 7-file set: SubGraph_0.weight has a fixed name shared by
rem         both models; a partial push would mix old/new files.
rem ================================================================
setlocal EnableDelayedExpansion

set "REAL_DIR=/data/app/el2/100/base/com.huawei.cannkit.llmengine/haps/entry/files"
set "SRC=%~dp0"
set "TMP_OUT=%TEMP%\cannkit_hdc_out.txt"

rem ---- auto-detect omc / embedding file names in this dir ----
set "OMC_FILE="
for %%F in ("%SRC%*.omc") do if not defined OMC_FILE set "OMC_FILE=%%~nxF"
set "EMB_W="
for %%F in ("%SRC%*.embedding_weights") do if not defined EMB_W set "EMB_W=%%~nxF"
set "EMB_S="
for %%F in ("%SRC%*.embedding_dequant_scale") do if not defined EMB_S set "EMB_S=%%~nxF"
if not defined OMC_FILE (
    echo [FAIL] no *.omc found in %SRC%. Run pack first ^(pipeline.sh ^<base^|instruct^> pack^).
    goto :fail
)
if not defined EMB_W (
    echo [FAIL] no *.embedding_weights found in %SRC%. Run pack first.
    goto :fail
)
if not defined EMB_S (
    echo [FAIL] no *.embedding_dequant_scale found in %SRC%. Run pack first.
    goto :fail
)
set "MODEL_TAG=BASE"
echo !OMC_FILE! | findstr /C:"Instruct" >nul && set "MODEL_TAG=INSTRUCT"
echo ^>^>^>^>^> pushing model: [!MODEL_TAG!] !OMC_FILE!

rem ---- locate hdc: PATH first, then DevEco Studio default install ----
set "HDC="
where hdc >nul 2>&1
if %errorlevel%==0 set "HDC=hdc"
if not defined HDC if exist "C:\Program Files\Huawei\DevEco Studio\sdk\default\openharmony\toolchains\hdc.exe" set "HDC=C:\Program Files\Huawei\DevEco Studio\sdk\default\openharmony\toolchains\hdc.exe"
if not defined HDC (
    echo [FAIL] hdc not found. Install DevEco Studio or add its toolchains dir to PATH.
    goto :fail
)
echo [OK] hdc: %HDC%

rem ---- check connected device: empty output means no device ----
echo ---- connected devices ----
"%HDC%" list targets > "%TMP_OUT%" 2>&1
for %%A in ("%TMP_OUT%") do if %%~zA equ 0 (
    echo ... nothing on default port, looking for an existing hdc server ...
    call :find_server_port
)
if defined HDC_PORT (
    echo [OK] reusing hdc server port !HDC_PORT!
    set "HDC=%HDC%"
    "%HDC%" list targets > "%TMP_OUT%" 2>&1
)
type "%TMP_OUT%"
for %%A in ("%TMP_OUT%") do if %%~zA equ 0 (
    echo [FAIL] no device connected. Plug in the phone, allow USB debugging,
    echo        or unplug/replug the USB cable, then rerun.
    goto :fail
)

rem ---- check app installed: sandbox dir must exist on device ----
"%HDC%" shell "ls -d %REAL_DIR%" > "%TMP_OUT%" 2>&1
for %%A in ("%TMP_OUT%") do if %%~zA equ 0 (
    echo [FAIL] %REAL_DIR% not found on device.
    echo        Install CANNLLMEngineDemoNext via DevEco Studio first, then rerun.
    echo        Also check: hdc shell bm dump -n com.huawei.cannkit.llmengine
    goto :fail
)

rem ---- push 7 files: local file -> device file name ----
call :push "!OMC_FILE!"                      "!OMC_FILE!"
call :push "SubGraph_0.weight"               "SubGraph_0.weight"
call :push "!EMB_W!"                         "!EMB_W!"
call :push "!EMB_S!"                         "!EMB_S!"
call :push "tokenizer.json"                  "tokenizer.json"
call :push "context.json"                  "context.json"
call :push "executor.json"                   "executor.json"
if defined PUSH_FAILED goto :fail

rem ---- fix permissions ----
"%HDC%" shell "chmod -R 755 %REAL_DIR%" >nul 2>&1

rem ---- final verification: all 7 files must appear in device dir ----
"%HDC%" shell "ls -lh %REAL_DIR%" > "%TMP_OUT%" 2>&1
echo ---- files on device ----
type "%TMP_OUT%"
set "MISSING="
for %%N in ("!OMC_FILE!" SubGraph_0.weight "!EMB_W!" "!EMB_S!" tokenizer.json context.json executor.json) do (
    findstr /C:"%%~N" "%TMP_OUT%" >nul || set "MISSING=!MISSING! %%~N"
)
if defined MISSING (
    echo [FAIL] missing on device:!MISSING!
    goto :fail
)

echo.
echo [DONE] [!MODEL_TAG!] All 7 files verified. App reads sandbox path /data/storage/el2/base/haps/entry/files/
echo        Fully close and restart the app to load the model. To demo the OTHER
echo        model, run the push script inside the other delivery dir (full 7-file set).
goto :end

rem ---- find listening port of an already-running hdc server ----
:find_server_port
set "HDC_PORT="
for /f "tokens=2 delims=," %%I in ('tasklist /FI "IMAGENAME eq hdc.exe" /FO CSV /NH 2^>nul') do (
    for /f "tokens=2" %%A in ('netstat -ano 2^>nul ^| findstr /C:"LISTENING" ^| findstr /C:"127.0.0.1:" ^| findstr /R /C:" %%~I$"') do (
        for /f "tokens=2 delims=:" %%Q in ("%%A") do if not defined HDC_PORT set "HDC_PORT=%%Q"
    )
)
if defined HDC_PORT set "HDC_SERVER_PORT=!HDC_PORT!"
exit /b 0

:push
echo pushing %~2 ...
if not exist "%SRC%%~1" (
    echo [FAIL] local file missing: %SRC%%~1
    set "PUSH_FAILED=1"
    exit /b 1
)
"%HDC%" file send "%SRC%%~1" "%REAL_DIR%/%~2" > "%TMP_OUT%" 2>&1
for %%A in ("%TMP_OUT%") do if %%~zA equ 0 (
    echo [FAIL] hdc file send produced no output: %~2
    set "PUSH_FAILED=1"
    exit /b 1
)
type "%TMP_OUT%"
exit /b 0

:fail
echo.
echo Push aborted.
pause
exit /b 1

:end
pause
exit /b 0
