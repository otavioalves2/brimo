#!/bin/bash
celery -A app.celery worker &
gunicorn app:app