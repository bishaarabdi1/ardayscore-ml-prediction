# ArdayScore client

This folder is the standalone web client for the Student Prediction API.

## Run it

1. Start the backend from the project root:

   ```powershell
   uvicorn main:app --reload --port 8000
   ```

2. Serve this folder with a static server. For example, with VS Code use
   Live Server, or with Python run this from the `client` folder:

   ```powershell
   python -m http.server 5173 --bind 127.0.0.1
   ```

3. Open `http://127.0.0.1:5173` in a browser.


The API URL is configured in `index.html` as `http://127.0.0.1:8000`.
