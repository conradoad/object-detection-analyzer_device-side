#!/bin/bash
export FLASK_ENV=development
export FLASK_APP=./server/app.py
python3 -m flask run --host 0.0.0.0
