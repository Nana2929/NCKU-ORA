
#%%
import time
def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        print(f"Running {func.__name__} ...", end='\r')
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Args: {args}")
        print(f"{func.__name__} Done in {end - start:.2f} seconds")
        return result
    return wrapper

# %%
@timer
def func(x=1, y=2):
    print(x+y)

# %%
