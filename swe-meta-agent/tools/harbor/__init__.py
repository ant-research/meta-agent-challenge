"""
Minimal Harbor compatibility package used inside the SWE meta-agent container.

The real evaluation is executed by an external Harbor deployment (cli_service host)
which has the official `harbor` package installed.

This stub exists so agent authors can write:
  from harbor.agents.base import BaseAgent
and have it import successfully in this offline build environment.
"""

