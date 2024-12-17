import multiprocessing

# Multiprocessing
MAX_PROCESSES = min(32, multiprocessing.cpu_count()) - 1


