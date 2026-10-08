<#
.SYNOPSIS
    Run the agent with THIS repo's virtualenv, whatever `python` is on PATH.

.DESCRIPTION
    The trap this avoids: `python -m agent` resolves to whichever interpreter
    PATH happens to point at, which may have different package versions than
    the venv the project was set up with. An agent that silently falls back to
    heuristics because of a stale SDK plays far worse than one that doesn't.

    All arguments are passed straight through to `python -m agent`.

.EXAMPLE
    .\play.ps1 --check-tournament

.EXAMPLE
    .\play.ps1 --tournament

.EXAMPLE
    .\play.ps1 --match --agent examples.smoke_agent
#>

$ErrorActionPreference = 'Stop'

$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'

if (-not (Test-Path $python)) {
    Write-Error @"
No virtualenv found at $python

Create one and install the project into it:
    py -m venv .venv
    .venv\Scripts\python.exe -m pip install -e ".[dev]"
"@
    exit 1
}

& $python -m agent @args
exit $LASTEXITCODE
