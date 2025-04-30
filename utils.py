import jax
import inspect
import re
import jax.numpy as jnp

def compare_results(x, y):
    frame = inspect.currentframe().f_back
    # Find the line in the source code where compare_results is called
    lines, lineno = inspect.getsourcelines(frame)
    line = lines[frame.f_lineno - lineno - 1].strip()
    pos = line.find('(')
    line = line[pos+1:-1]
    pos = line.find(',')
    x_name = line[:pos].strip() if pos is not None else 'x'
    y_name = line[pos+1:].strip() if pos is not None else 'y'
    
    print(
f"""Info for first argument: 
    Name: {x_name},
    Class: {x.__class__},
    Shape: {x.shape}.""")
    print(
f"""Info for second argument:
    Name: {y_name},
    Class: {y.__class__},
    Shape: {y.shape}.""")
    print(f"PCC Score: {jnp.min(jnp.corrcoef(x.flatten(), y.flatten()))}")
    print(f"Max Difference: {jnp.max(jnp.abs(x - y))}")
    print("=" * 50)