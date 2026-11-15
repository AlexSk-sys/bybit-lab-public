@echo off
cd /d "C:\Users\AgentPC\source\repos\bybit-lab"
set PY=.venv\Scripts\python.exe
set COLLECT=micro2\collect_20261014T185712Z.json
set CALLS=micro2\agent_calls_20261014T190100Z.json
echo ==== run %DATE% %TIME% ==== >> micro2\eval_log.txt
"%PY%" microstructure_v2.py evaluate "%COLLECT%" "%CALLS%" --desktop micro2 >> micro2\eval_log.txt 2>&1
set RC=%ERRORLEVEL%
echo ==== exit %RC% ==== >> micro2\eval_log.txt
exit /b %RC%
