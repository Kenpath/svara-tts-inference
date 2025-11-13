import time
import functools
import asyncio
import inspect
from typing import Optional, Dict, Callable
import logging

logger = logging.getLogger(__name__)

# Global stats tracker
_timing_stats: Dict[str, Dict[str, float]] = {}


def track_time(name: Optional[str] = None, log_level: str = "INFO"):
    """
    Decorator to track execution time of functions.
    
    Args:
        name: Custom name for the timing log (defaults to function name)
        log_level: Logging level (DEBUG, INFO, WARNING)
    
    Usage:
        @track_time()
        def my_function():
            ...
        
        @track_time("Custom Name")
        async def my_async_function():
            ...
    """
    def decorator(func: Callable) -> Callable:
        func_name = name or f"{func.__module__}.{func.__qualname__}"
        
        # Initialize stats
        if func_name not in _timing_stats:
            _timing_stats[func_name] = {
                "count": 0,
                "total_time": 0.0,
                "min_time": float('inf'),
                "max_time": 0.0,
            }
        
        # Handle async generators (async def with yield)
        if inspect.isasyncgenfunction(func):
            @functools.wraps(func)
            async def async_gen_wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                try:
                    async for item in func(*args, **kwargs):
                        yield item
                finally:
                    elapsed = time.perf_counter() - start_time
                    _log_timing(func_name, elapsed, log_level)
            return async_gen_wrapper
        # Handle regular async functions
        elif asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                try:
                    result = await func(*args, **kwargs)
                    return result
                finally:
                    elapsed = time.perf_counter() - start_time
                    _log_timing(func_name, elapsed, log_level)
            return async_wrapper
        # Handle sync generators (def with yield)
        elif inspect.isgeneratorfunction(func):
            @functools.wraps(func)
            def sync_gen_wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                try:
                    for item in func(*args, **kwargs):
                        yield item
                finally:
                    elapsed = time.perf_counter() - start_time
                    _log_timing(func_name, elapsed, log_level)
            return sync_gen_wrapper
        # Handle regular sync functions
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    elapsed = time.perf_counter() - start_time
                    _log_timing(func_name, elapsed, log_level)
            return sync_wrapper
    
    return decorator


def _log_timing(func_name: str, elapsed: float, log_level: str):
    """Update stats and log timing."""
    # Initialize stats if not present (defensive programming)
    if func_name not in _timing_stats:
        _timing_stats[func_name] = {
            "count": 0,
            "total_time": 0.0,
            "min_time": float('inf'),
            "max_time": 0.0,
        }
    
    stats = _timing_stats[func_name]
    stats["count"] += 1
    stats["total_time"] += elapsed
    stats["min_time"] = min(stats["min_time"], elapsed)
    stats["max_time"] = max(stats["max_time"], elapsed)
    
    avg_time = stats["total_time"] / stats["count"]
    
    log_msg = (
        f"⏱️  {func_name}: {elapsed*1000:.2f}ms "
        f"(avg: {avg_time*1000:.2f}ms, min: {stats['min_time']*1000:.2f}ms, "
        f"max: {stats['max_time']*1000:.2f}ms, count: {stats['count']})"
    )
    
    log_func = getattr(logger, log_level.lower(), logger.info)
    log_func(log_msg)


def get_timing_stats() -> Dict[str, Dict[str, float]]:
    """Get all timing statistics."""
    return _timing_stats.copy()


def reset_timing_stats():
    """Reset all timing statistics."""
    _timing_stats.clear()


def print_timing_summary():
    """Print a summary of all timing stats."""
    if not _timing_stats:
        print("No timing data collected.")
        return
    
    print("\n" + "="*80)
    print("TIMING SUMMARY")
    print("="*80)
    
    for func_name, stats in sorted(_timing_stats.items(), key=lambda x: x[1]["total_time"], reverse=True):
        avg = stats["total_time"] / stats["count"]
        print(f"\n{func_name}:")
        print(f"  Total time: {stats['total_time']*1000:.2f}ms")
        print(f"  Calls: {stats['count']}")
        print(f"  Average: {avg*1000:.2f}ms")
        print(f"  Min: {stats['min_time']*1000:.2f}ms")
        print(f"  Max: {stats['max_time']*1000:.2f}ms")
    
    print("="*80 + "\n")

