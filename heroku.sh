#!/bin/bash
celery -A app.celery worker -l info --concurrency 1 &
gunicorn app:app