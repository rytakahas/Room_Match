#!/bin/bash

# Check if a JSON file was provided
if [ -z "$1" ]; then
  echo "❗ Usage: ./run_server_and_test.sh path/to/sample_request.json"
  exit 1
fi

JSON_FILE=$1
LOG_FILE=flask_server.log

echo "🚀 Starting Flask server using python app.py on port 5050..."
python app.py > "$LOG_FILE" 2>&1 &
FLASK_PID=$!

# Wait for the server to be ready (up to 15 seconds)
echo "⏳ Waiting for Flask server to be ready..."
for i in {1..15}; do
  if nc -z 127.0.0.1 5050; then
    echo "✅ Flask server is ready!"
    break
  fi
  sleep 1
done

# If still not ready, print error and logs
if ! ps -p $FLASK_PID > /dev/null; then
  echo "❌ Flask server failed to start. Check the logs:"
  cat "$LOG_FILE"
  exit 1
fi

# Send the request using curl
echo "📤 Sending request using $JSON_FILE..."
if command -v jq > /dev/null; then
  curl -s -X POST http://127.0.0.1:5050/room_match \
    -H "Content-Type: application/json" \
    -d @"$JSON_FILE" | jq '.'
else
  curl -s -X POST http://127.0.0.1:5050/room_match \
    -H "Content-Type: application/json" \
    -d @"$JSON_FILE"
fi

# Stop the Flask server
echo "🛑 Stopping Flask server (PID: $FLASK_PID)..."
kill $FLASK_PID 2>/dev/null

# Clean up
rm -f "$LOG_FILE"

