Using ngrok
To expose your local server to the internet (for example, to make a local web app accessible remotely), you can use ngrok.

Steps:
1. Download and install ngrok from here.

2. Open a terminal (Command Prompt or PowerShell).

3. Authenticate ngrok (only once):
   
     ngrok config add-authtoken YOUR_AUTHTOKEN
     
   (You can find your auth token in your ngrok dashboard.)

5. Start a tunnel to your local port (for example, port 8000):
   
  ngrok http 8000
  
7. ngrok will give you a public URL (like https://abc123.ngrok.io) that redirects to your local server.
