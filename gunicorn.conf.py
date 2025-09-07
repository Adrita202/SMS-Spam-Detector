import multiprocessing
import os


bind = os.getenv("GUNICORN_BIND", "0.0.0.0:5000")
workers = int(os.getenv("GUNICORN_WORKERS", str(multiprocessing.cpu_count() * 2 + 1)))
threads = int(os.getenv("GUNICORN_THREADS", "2"))
worker_class = os.getenv("GUNICORN_WORKER_CLASS", "gthread")
timeout = int(os.getenv("GUNICORN_TIMEOUT", "30"))
graceful_timeout = int(os.getenv("GUNICORN_GRACEFUL_TIMEOUT", "30"))
loglevel = os.getenv("GUNICORN_LOGLEVEL", "info")
accesslog = "-"  # stdout
errorlog = "-"   # stderr

# Forwarded allow-from header support (container/ingress proxies)
forwarded_allow_ips = "*"
proxy_protocol = False


