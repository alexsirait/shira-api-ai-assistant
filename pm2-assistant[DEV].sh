#!/bin/bash
source /var/www/assistant/venv/Script/activate
exec python3 /var/www/assistant/mysatnusa/manage.py runserver 192.168.88.60:41000