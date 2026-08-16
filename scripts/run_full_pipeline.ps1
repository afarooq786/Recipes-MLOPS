<#
scripts/run_full_pipeline.ps1

Brings up the full Recipe MLOps stack end-to-end:
host MLflow, model training/registration (optional), Docker Compose
(API + Airflow + Postgres), monitoring/validation pipeline, and tests.

Usage:

    .\scripts\run_full_pipeline.ps1

    .\scripts\run_full_pipeline.ps1 -SkipTraining

    .\scripts\run_full_pipeline.ps1 -SkipTests

    .\scripts\run_full_pipeline.ps1 -SkipTraining -SkipTests

Prerequisites:
    - Docker Desktop running
    - Python environment activated
    - requirements.txt + requirements-dev.txt installed
    - MLflow installed
    - Port 5000 available

Notes on the MLflow startup below:
    - We launch MLflow via `python -m mlflow` rather than the `mlflow.exe`
      console-script launcher. The launcher has been observed on Windows to
      mis-glob arguments (e.g. turning a bare "*" into a listing of the
      current directory's files), so going through the interpreter avoids
      that class of bug entirely.
    - We bind to 0.0.0.0 (not 127.0.0.1) so the API/Airflow containers can
      reach this server via host.docker.internal. Binding to 127.0.0.1
      only accepts connections from this machine and will silently break
      container-to-host calls.
    - We pass --disable-security-middleware because MLflow 3.5+ ships a
      Host-header validation layer that, in local testing, rejected
      requests with Host: host.docker.internal:5000 even when that exact
      value was listed in --allowed-hosts. Since this server is only ever
      reachable from your own machine's Docker Desktop VM (never the
      public internet or a browser from another origin), disabling it is
      an acceptable tradeoff for local dev. Revisit this if you ever
      expose this server beyond localhost.

Notes on Stage 2 (training) below:
    - train_final_candidates and register_best_model specifically run
      inside a Linux container (via `docker compose run api ...`), not
      native Windows Python. MLflow bakes the relative artifact path into
      the model manifest using whatever OS ran the logging step -- on
      native Windows that means backslashes, which break model loading
      inside the Linux API container later at serve time ("No such file
      or directory: ...\char_logistic.joblib"). The earlier candidate-
      comparison scripts (train_logistic, train_xgboost, train_challengers,
      ensemble_stability) only log experiment-tracking metrics/artifacts
      that are never deployed, so they're safe to leave on native Windows.
#>

param(
    [switch]$SkipTraining,
    [switch]$SkipTests
)

$ErrorActionPreference = "Continue"

$MLFLOW_PID_FILE = ".mlflow.pid"
$MLFLOW_URL = "http://localhost:5000"
$API_URL = "http://localhost:8000"
$AIRFLOW_URL = "http://localhost:8080"

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

function Info {
    param([string]$Message)

    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Ok {
    param([string]$Message)

    Write-Host "  OK: $Message" -ForegroundColor Green
}

function Fail {
    param([string]$Message)

    Write-Host "  FAILED: $Message" -ForegroundColor Red
}

function Die {
    param([string]$Message)

    Fail $Message

    Write-Host ""
    Write-Host "Stopping here -- fix the above, then re-run this script." -ForegroundColor Yellow
    Write-Host "See README.md for additional troubleshooting information." -ForegroundColor Yellow

    exit 1
}

function Test-CommandExists {
    param([string]$CommandName)

    return $null -ne (Get-Command $CommandName -ErrorAction SilentlyContinue)
}

function Wait-ForHttp {
    param(
        [string]$Url,
        [string]$Label,
        [int]$MaxAttempts = 30
    )

    for ($Attempt = 1; $Attempt -le $MaxAttempts; $Attempt++) {

        try {
            $Response = Invoke-WebRequest `
                -Uri $Url `
                -UseBasicParsing `
                -TimeoutSec 5 `
                -ErrorAction Stop

            if ($Response.StatusCode -ge 200 -and $Response.StatusCode -lt 500) {
                Ok "$Label is up (attempt $Attempt/$MaxAttempts)"
                return $true
            }
        }
        catch {
            # Service is not ready yet.
        }

        Start-Sleep -Seconds 10
    }

    return $false
}

function Get-Port5000Listeners {
    # Returns owning process IDs for anything LISTENING on port 5000,
    # whether or not that PID corresponds to a live process. A listener
    # with no matching process ("ghost" socket) has been observed on
    # Windows/WSL2 setups and is not fixable by Stop-Process -- it needs
    # a `wsl --shutdown` + Docker Desktop restart, or a full reboot.
    try {
        $Connections = Get-NetTCPConnection -LocalPort 5000 -State Listen -ErrorAction Stop
        return $Connections | Select-Object -ExpandProperty OwningProcess -Unique
    }
    catch {
        return @()
    }
}

function Test-PortInUse {
    param([int]$Port)

    try {
        $Connection = Get-NetTCPConnection `
            -LocalPort $Port `
            -State Listen `
            -ErrorAction Stop

        return $null -ne $Connection
    }
    catch {
        return $false
    }
}

function Stop-ExistingMLflow {
    if (-not (Test-Path $MLFLOW_PID_FILE)) {
        return
    }

    try {
        $StoredPid = [int](Get-Content $MLFLOW_PID_FILE -ErrorAction Stop)

        $Process = Get-Process -Id $StoredPid -ErrorAction SilentlyContinue

        if ($null -ne $Process) {
            Stop-Process -Id $StoredPid -Force
            Write-Host "Stopped MLflow process $StoredPid."
        }

        Remove-Item $MLFLOW_PID_FILE -Force -ErrorAction SilentlyContinue
    }
    catch {
        Write-Host "Could not clean up MLflow PID file."
    }
}

# ---------------------------------------------------------------------
# Stage 0: Preflight checks
# ---------------------------------------------------------------------

Info "Stage 0: Preflight checks"

if (-not (Test-CommandExists "docker")) {
    Die "docker not found. Install Docker Desktop first."
}

try {
    docker info *> $null

    if ($LASTEXITCODE -ne 0) {
        Die "Docker daemon not reachable. Is Docker Desktop running?"
    }
}
catch {
    Die "Docker daemon not reachable. Is Docker Desktop running?"
}

Ok "Docker is running"

if (-not (Test-CommandExists "python")) {
    Die "python not found. Make sure Python is installed and your virtual environment is activated."
}

Ok "Python available"

python -m mlflow --version *> $null

if ($LASTEXITCODE -ne 0) {
    Die "mlflow not importable via 'python -m mlflow'. Run: pip install -r requirements.txt -r requirements-dev.txt"
}

Ok "MLflow available (python -m mlflow)"

# Check port 5000. If something is listening but isn't actually serving
# HTTP, and it isn't a process we can identify/stop, this is almost
# certainly a stale/orphaned socket rather than a real MLflow instance --
# bail out with a specific fix instead of a generic "port in use" error.
if (Test-PortInUse 5000) {

    try {
        $Response = Invoke-WebRequest `
            -Uri $MLFLOW_URL `
            -UseBasicParsing `
            -TimeoutSec 3 `
            -ErrorAction Stop

        Ok "Port 5000 is already serving HTTP -- assuming it's MLflow."
    }
    catch {

        $Listeners = Get-Port5000Listeners
        $LiveProcesses = $Listeners | ForEach-Object { Get-Process -Id $_ -ErrorAction SilentlyContinue }

        if ($null -eq $LiveProcesses -or $LiveProcesses.Count -eq 0) {
            Die @"
Port 5000 has a listener that doesn't respond to HTTP and doesn't match
any running process (PIDs seen: $($Listeners -join ', ')).

This is a stale/orphaned socket, not a real MLflow server. Stop-Process
will not fix this. Fix:

    1. wsl --shutdown
    2. Quit and restart Docker Desktop completely (tray icon -> Quit)
    3. Re-check:  netstat -ano | findstr :5000   (should be empty)
    4. If still not empty, reboot Windows.

Then re-run this script.
"@
        }
        else {
            Die "Port 5000 is in use by another process ($($LiveProcesses.Id -join ', ')) that isn't serving HTTP. Stop it or change the MLflow port."
        }
    }
}

# ---------------------------------------------------------------------
# Stage 1: Start MLflow on the host
# ---------------------------------------------------------------------

Info "Stage 1: Starting MLflow on the host"

try {
    $Response = Invoke-WebRequest `
        -Uri $MLFLOW_URL `
        -UseBasicParsing `
        -TimeoutSec 3 `
        -ErrorAction Stop

    Ok "MLflow already running at $MLFLOW_URL -- reusing it"
}
catch {

    Write-Host "Starting MLflow..."

    $Arguments = @(
        "-m", "mlflow"
        "server"
        "--host", "0.0.0.0"
        "--port", "5000"
        "--backend-store-uri", "sqlite:///mlflow.db"
        "--default-artifact-root", "./mlartifacts"
        "--disable-security-middleware"
    )

    try {

        $MLflowStdout = Join-Path (Get-Location) "mlflow_stdout.log"
        $MLflowStderr = Join-Path (Get-Location) "mlflow_stderr.log"

        $MLflowProcess = Start-Process `
            -FilePath "python" `
            -ArgumentList $Arguments `
            -RedirectStandardOutput $MLflowStdout `
            -RedirectStandardError $MLflowStderr `
            -PassThru `
            -WindowStyle Hidden

        $MLflowProcess.Id | Out-File -FilePath $MLFLOW_PID_FILE -Encoding ascii

        Info "MLflow starting in background (PID $($MLflowProcess.Id))..."
        Info "MLflow stdout: $MLflowStdout"
        Info "MLflow stderr: $MLflowStderr"

        if (-not (Wait-ForHttp $MLFLOW_URL "MLflow" 12)) {
            Die "MLflow did not come up after 2 minutes. Check mlflow_stdout.log and mlflow_stderr.log for errors."
        }
    }
    catch {
        Die "Failed to start MLflow: $($_.Exception.Message)"
    }
}

# ---------------------------------------------------------------------
# Stage 2: Train + register champion model
# ---------------------------------------------------------------------

if ($SkipTraining) {

    Info "Stage 2: Skipped (-SkipTraining). Assuming a champion model is already registered."

}
else {

    Info "Stage 2: Training and registering the champion model"

    python -m models.train_logistic

    if ($LASTEXITCODE -ne 0) {
        Die "train_logistic failed"
    }

    python -m models.train_xgboost

    if ($LASTEXITCODE -ne 0) {
        Die "train_xgboost failed"
    }

    python -m models.train_challengers

    if ($LASTEXITCODE -ne 0) {
        Die "train_challengers failed"
    }

    python -m models.ensemble_stability

    if ($LASTEXITCODE -ne 0) {
        Die "ensemble_stability failed"
    }

    # train_final_candidates and register_best_model produce/register the
    # actual deployed champion model. Run these two specifically inside a
    # Linux container so MLflow bakes forward-slash artifact paths into
    # the model manifest instead of Windows backslashes -- see the note
    # at the top of this script for why that matters.

    Info "Building the api image (needed to train inside Linux)..."

    docker compose build api

    if ($LASTEXITCODE -ne 0) {
        Die "docker compose build api failed"
    }

    docker compose run --rm `
        -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 `
        -v "${PWD}/data:/app/data" `
        api python -m models.train_final_candidates

    if ($LASTEXITCODE -ne 0) {
        Die "train_final_candidates failed (containerized)"
    }

    docker compose run --rm `
        -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 `
        api python -m models.register_best_model

    if ($LASTEXITCODE -ne 0) {
        Die "register_best_model failed (containerized)"
    }

    Ok "Champion model registered (trained inside Linux container -- no Windows path bug)"
}

# ---------------------------------------------------------------------
# Stage 3: Docker Compose
# ---------------------------------------------------------------------

Info "Stage 3: Building and starting the Docker Compose stack"

docker compose up -d --build

if ($LASTEXITCODE -ne 0) {
    Die "docker compose up failed -- see output above."
}

Info "Waiting for the API to report healthy..."

if (-not (Wait-ForHttp "$API_URL/health" "API" 12)) {
    Die "API did not come up. Run: docker compose logs api"
}

try {

    $HealthResponse = Invoke-WebRequest `
        -Uri "$API_URL/health" `
        -UseBasicParsing `
        -TimeoutSec 5 `
        -ErrorAction Stop

    $HealthJson = $HealthResponse.Content

}
catch {
    Die "Could not retrieve API health response."
}

if ($HealthJson -match '"model_loaded"\s*:\s*true') {

    Ok "API is healthy and model_loaded=true"

}
else {

    Write-Host ""
    Write-Host "API health response:" -ForegroundColor Yellow
    Write-Host $HealthJson

    Die @"
API is up but model_loaded=false.

Most likely causes:

1. MLflow does not have a champion model registered yet.
   Re-run without -SkipTraining.

2. The API container can't reach MLflow at host.docker.internal:5000.
   Test with:
       docker exec recipe-api python -c "import requests; r=requests.get('http://host.docker.internal:5000'); print(r.status_code)"
   A 403 here usually means MLflow's security middleware is rejecting the
   request -- confirm this script's MLflow invocation still includes
   --disable-security-middleware. A connection error usually means
   MLflow isn't bound to 0.0.0.0, or Docker Desktop's networking needs
   a restart (wsl --shutdown, then restart Docker Desktop).

3. Dockerfile.api Python version does not match the training environment.
   Check:
       python --version

   and compare it with the FROM line in Dockerfile.api.

See README.md for additional details.
"@
}

Info "Waiting for Airflow webserver..."

if (-not (Wait-ForHttp "$AIRFLOW_URL/health" "Airflow webserver" 18)) {
    Die "Airflow webserver did not come up. Run: docker compose logs airflow-webserver"
}

# ---------------------------------------------------------------------
# Stage 4: Monitoring pipeline
# ---------------------------------------------------------------------

Info "Stage 4: Running the monitoring pipeline"

python -m monitoring.drift_simulation

if ($LASTEXITCODE -ne 0) {
    Die "drift_simulation failed"
}

python -m monitoring.verify_alerts

if ($LASTEXITCODE -ne 0) {
    Die "verify_alerts failed"
}

python -m monitoring.production_validation --api-url $API_URL

if ($LASTEXITCODE -ne 0) {
    Die "production_validation failed"
}

Ok "Monitoring pipeline complete -- see monitoring/reports/ for evidence"

# ---------------------------------------------------------------------
# Stage 5: Test suite
# ---------------------------------------------------------------------

if ($SkipTests) {

    Info "Stage 5: Skipped (-SkipTests)"

}
else {

    Info "Stage 5: Running the test suite"

    python -m pytest tests/ -v

    if ($LASTEXITCODE -ne 0) {
        Die "Test suite failed -- see output above."
    }

    Ok "All tests passed"
}

# ---------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------

Info "Pipeline complete."

Write-Host ""
Write-Host "  MLflow:  $MLFLOW_URL"
Write-Host "  API:     $API_URL/docs"
Write-Host "  Airflow: $AIRFLOW_URL  (admin / admin)"
Write-Host ""
Write-Host "  Airflow DAGs still need to be triggered manually from the UI."
Write-Host ""
Write-Host "  To stop everything cleanly:"
Write-Host "      .\scripts\stop_pipeline.ps1"
Write-Host ""