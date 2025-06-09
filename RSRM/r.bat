@echo off
setlocal enabledelayedexpansion

:: Default values
set "TASK=SR/s"
set "CONFIG=ecology.json"
set "NUM_TEST=1"
set "GROUP= "

:: Loop through arguments
for %%A in (%*) do (
    set "arg=%%~A"
    set "last4=!arg:~-4!"  &rem Get last 4 chars (assumes .json is lowercase)
    set "last5=!arg:~-5!"  &rem Get last 5 chars (accounts for .JSON)
    echo("%%~A"|findstr "^[\"][-][1-9][0-9]*[\"]$ ^[\"][1-9][0-9]*[\"]$ ^[\"]0[\"]$">nul && set "numeric=1" || set "numeric=0"

    :: Check if the argument ends with .json (case-insensitive)
    if /i "!last5!"==".json" (
        set "CONFIG=%%~A"
    ) else if /i "!last4!"==".json" (
        set "CONFIG=%%~A"
    ) else if /i "!arg!"=="g" (
        set "GROUP=--fit A,T --split Archipelago,species"
    ) else if "!numeric!" == "1" (
        set "NUM_TEST=%%~A"
    ) else (
        set "TASK=%%~A"
    )
)

echo TASK: %TASK%
echo CONFIG: config/%CONFIG%
echo NUM_TEST: %NUM_TEST%

python main.py --task %TASK% --num_test %NUM_TEST% --json_path config/%CONFIG% --threshold 1e-10 %GROUP%