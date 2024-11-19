#!/bin/bash
source /var/lib/jenkins/workspace/assistant/venv/Script/activate
exec python3 /var/lib/jenkins/workspace/assistant/chatbot/manage.py runserver 192.168.88.60:41000