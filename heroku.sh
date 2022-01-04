#!/bin/bash
celery -A app.celery worker -l info --concurrency 2 &
gunicorn app:app