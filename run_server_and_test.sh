#!/bin/bash

# Check if a JSON file was provided
if [ -z "$1" ]; then
  echo " Usage: ./run_server_and_test.sh path/to/sample_request.json"
  exit 1
fi

JSON_FILE=$1

# Start the Flask server in the background
FLASK_APP=app.py flask run --host=0.0.0.0 --port=5050 &
FLASK_PID=$!

# Wait a few seconds to let the server start
sleep 3

# Send the test request using curl and external JSON
echo "Sending request using $JSON_FILE"
curl -X POST http://127.0.0.1:5050/room_match \
  -H 'Content-Type: application/json' \
  -d @"$JSON_FILE"

# Kill the server process
kill $FLASK_PID

