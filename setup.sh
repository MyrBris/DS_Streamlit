#!/bin/bash
pip install --upgrade pip
pip uninstall numpy -y  # Force la réinstallation
pip install --no-cache-dir --force-reinstall -r requirements.txt
