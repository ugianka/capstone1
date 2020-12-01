#!/bin/sh

export FLASK_APP="./src/server.py"
# allow connections from all outside machines  
flask run --host 0.0.0.0
