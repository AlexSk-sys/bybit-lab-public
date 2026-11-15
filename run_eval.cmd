@echo off
cd /d "C:\Users\AgentPC\source\repos\bybit-lab"
set PY=.venv\Scripts\python.exe
set COLLECT=micro\collect_20261014T175001Z.json
set CALLS=micro\agent_calls_20261014T175001Z.json
echo ==== run %DATE% %TIME% ==== >> micro\eval_log.txt
"%PY%" microstructure.py evaluate "%COLLECT%" "%CALLS%" >> micro\eval_log.txt 2>&1
echo ==== exit %ERRORLEVEL% ==== >> micro\eval_log.txt
