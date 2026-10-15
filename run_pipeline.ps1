# Windows (PowerShell) version of run_pipeline.sh: raw -> Bronze -> Silver -> Gold -> MLlib predictions
# Run with:  powershell -ExecutionPolicy Bypass -File .\run_pipeline.ps1
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "src")

# Spark on Windows needs winutils.exe; use C:\hadoop if HADOOP_HOME is not already set
if (-not $env:HADOOP_HOME -and (Test-Path "C:\hadoop\bin\winutils.exe")) { $env:HADOOP_HOME = "C:\hadoop" }
if ($env:HADOOP_HOME) { $env:PATH = "$env:HADOOP_HOME\bin;$env:PATH" }

# Make Spark workers use the same Python as this script
$env:PYSPARK_PYTHON = "python"
$env:PYSPARK_DRIVER_PYTHON = "python"

foreach ($step in "01_ingest_bronze", "02_silver", "03_gold", "04_ml") {
    Write-Host "===== Running $step ====="
    python "$step.py"
    if ($LASTEXITCODE -ne 0) { Write-Host "Step $step failed." -ForegroundColor Red; exit 1 }
}
Write-Host "Pipeline complete. Launch the dashboard with: streamlit run dashboard\app.py"
