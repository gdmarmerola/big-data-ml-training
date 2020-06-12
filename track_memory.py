# libs to help us track memory via sampling
import numpy as np
import tracemalloc
from time import sleep
import matplotlib.pyplot as plt

# sampling time in seconds
SAMPLING_TIME = 0.001

class MemoryMonitor:
    def __init__(self, close=True):
        
        # start tracemalloc and sets
        # measurement atribute to True
        tracemalloc.start()
        self.keep_measuring = True
        self.close = close
        
    def measure_usage(self):
        
        """
        Takes measurements of used memory on
        regular intevals determined by the 
        global SAMPLING_TIME constant
        """
        
        # list to store memory usage samples
        usage_list = []
        
        # keeps going until someone changes this parameter to false
        while self.keep_measuring:
            
            # takes a sample, stores it in the usage_list and sleeps
            current, peak = tracemalloc.get_traced_memory()
            usage_list.append(current/1e6)
            sleep(SAMPLING_TIME)
            
        # stop tracemalloc and returns list
        if self.close:
            tracemalloc.stop()
        return usage_list

# imports executor
from concurrent.futures import ThreadPoolExecutor
from functools import wraps

def plot_memory_use(history, fn_name, open_figure=True, offset=0, **kwargs):
    
    """Function to plot memory use from a history collected
        by the MemoryMonitor class
    """

    # getting times from counts and sampling time
    times = (offset + np.arange(len(history))) * SAMPLING_TIME
    
    # opening figure and plotting
    if open_figure:
        plt.figure(figsize=(10,3), dpi=120)
    plt.plot(times, history, 'k--', linewidth=1)
    plt.fill_between(times, history, alpha=0.5, **kwargs)
    
    # axes titles
    plt.ylabel('Memory usage [MB]')
    plt.xlabel('Time [seconds]')
    plt.title(f'{fn_name} memory usage over time')
    
    # legend
    plt.legend();

def track_memory_use(plot=True, close=True, return_history=False):
    
    def meta_wrapper(fn):
    
        """
        This function is meant to be used as a decorator
        that informs wrapped function memory usage
        """
        
        # decorator so we can retrieve original fn
        @wraps(fn)
        def wrapper(*args, **kwargs):

            """
            Starts wrapped function and holds a process 
            to sample memory usage while executing it
            """

            # context manager for executor
            with ThreadPoolExecutor() as executor:

                # start memory monitor
                monitor = MemoryMonitor(close=close)
                mem_thread = executor.submit(monitor.measure_usage)

                # start wrapped function and get its result
                try:
                    fn_thread = executor.submit(fn, *args, **kwargs)
                    fn_result = fn_thread.result()

                # when wrapped function ends, stop measuring
                finally:
                    monitor.keep_measuring = False
                    history = mem_thread.result()

                # inform results via prints and plot
                print(f'Current memory usage: {history[-1]:2f}')
                print(f'Peak memory usage: {max(history):2f}')
                if plot:
                    plot_memory_use(history, fn.__name__)
            if return_history:
                return fn_result, history
            else:
                return fn_result

        return wrapper
    
    return meta_wrapper