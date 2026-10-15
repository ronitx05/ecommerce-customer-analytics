@echo off
REM Windows version of run_pipeline.sh: raw -> Bronze -> Silver -> Gold -> MLlib predictions
setlocal
cd /d "%~dp0src"

REM Spark on Windows needs winutils.exe; use C:\hadoop if HADOOP_HOME is not already set
if not defined HADOOP_HOME if exist "C:\hadoop\bin\winutils.exe" set "HADOOP_HOME=C:\hadoop"
if defined HADOOP_HOME set "PATH=%HADOOP_HOME%\bin;%PATH%"

REM Make Spark workers use the same Python as this script
set "PYSPARK_PYTHON=python"
set "PYSPARK_DRIVER_PYTHON=python"

for %%S in (01_ingest_bronze 02_silver 03_gold 04_ml) do (
    echo ===== Running %%S =====
    python %%S.py
    if errorlevel 1 (
        echo Step %%S failed.
        exit /b 1
    )
)
echo Pipeline complete. Launch the dashboard with: streamlit run dashboard\app.py
endlocal
