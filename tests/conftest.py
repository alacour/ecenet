"""Put the repo root on sys.path so `import ecenet` (and the root train_ scripts)
resolve when tests are collected by pytest."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
