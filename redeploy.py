import os
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/webhook', methods=['POST'])
def webhook():
    # Pull the latest changes from Git
    repo_path = '/home/IgorVinson/Stock-Data-Analysis-and-Signal-Generator'  # Update this path
    os.system(f'cd {repo_path} && git pull origin main')  # Replace 'main' with your branch name

    # Restart the web app
    os.system('touch /var/www/igorvinson_pythonanywhere_com_wsgi.py')

    return jsonify({'success': True}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)