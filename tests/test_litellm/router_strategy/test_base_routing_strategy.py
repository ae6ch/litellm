import json
from typing import Any, Dict, List, Optional, Set, Union

import pytest


import asyncio
from unittest.mock import MagicMock, patch


from litellm.caching.caching import DualCache
from litellm.caching.redis_cache import RedisPipelineIncrementOperation
from litellm.router_strategy.base_routing_strategy import BaseRoutingStrategy


@pytest.fixture
async def mock_dual_cache():
    dual_cache = MagicMock(spec=DualCache)
    dual_cache.in_memory_cache = MagicMock()
    dual_cache.redis_cache = MagicMock()

    # Set up async method mocks to return coroutines
    future1: asyncio.Future[None] = asyncio.Future()
    future1.set_result(None)
    dual_cache.in_memory_cache.async_increment.return_value = future1

    future2: asyncio.Future[None] = asyncio.Future()
    future2.set_result(None)
    dual_cache.redis_cache.async_increment_pipeline.return_value = future2

    future3: asyncio.Future[None] = asyncio.Future()
    future3.set_result(None)
    dual_cache.in_memory_cache.async_set_cache.return_value = future3

    # Fix for async_batch_get_cache
    batch_future: asyncio.Future[Dict[str, str]] = asyncio.Future()
    batch_future.set_result({"key1": "10.0", "key2": "20.0"})
    dual_cache.redis_cache.async_batch_get_cache.return_value = batch_future

    return dual_cache


@pytest.fixture
async def base_strategy(mock_dual_cache):
    return BaseRoutingStrategy(
        dual_cache=mock_dual_cache,
        should_batch_redis_writes=False,
        default_sync_interval=1,
    )


@pytest.mark.asyncio
async def test_increment_value_in_current_window(base_strategy, mock_dual_cache):
    # Test incrementing value in current window
    key = "test_key"
    value = 10.0
    ttl = 3600

    await base_strategy._increment_value_in_current_window(key, value, ttl)

    # Verify in-memory cache was incremented
    mock_dual_cache.in_memory_cache.async_increment.assert_called_once_with(
        key=key, value=value, ttl=ttl
    )

    # Verify operation was queued for Redis
    assert len(base_strategy.redis_increment_operation_queue) == 1
    queued_op = base_strategy.redis_increment_operation_queue[0]
    assert isinstance(queued_op, dict)
    assert queued_op["key"] == key
    assert queued_op["increment_value"] == value
    assert queued_op["ttl"] == ttl


@pytest.mark.asyncio
async def test_push_in_memory_increments_to_redis(base_strategy, mock_dual_cache):
    # Add some operations to the queue
    base_strategy.redis_increment_operation_queue = [
        RedisPipelineIncrementOperation(key="key1", increment_value=10, ttl=3600),
        RedisPipelineIncrementOperation(key="key2", increment_value=20, ttl=3600),
    ]

    await base_strategy._push_in_memory_increments_to_redis()

    # Verify Redis pipeline was called
    mock_dual_cache.redis_cache.async_increment_pipeline.assert_called_once()
    # Verify queue was cleared
    assert len(base_strategy.redis_increment_operation_queue) == 0


@pytest.mark.asyncio
async def test_sync_in_memory_spend_with_redis(base_strategy, mock_dual_cache):
    from litellm.types.caching import RedisPipelineIncrementOperation

    # Setup test data
    base_strategy.in_memory_keys_to_update = {"key1"}
    base_strategy.redis_increment_operation_queue = [
        RedisPipelineIncrementOperation(key="key1", increment_value=10, ttl=3600),
    ]

    # Mock the in-memory cache batch get responses for before snapshot
    in_memory_before_future: asyncio.Future[List[str]] = asyncio.Future()
    in_memory_before_future.set_result(["5.0"])  # Initial values
    mock_dual_cache.in_memory_cache.async_batch_get_cache.return_value = (
        in_memory_before_future
    )

    # Mock Redis batch get response
    redis_future: asyncio.Future[Dict[str, str]] = asyncio.Future()
    redis_future.set_result([15.0])  # Redis values
    mock_dual_cache.redis_cache.async_increment_pipeline.return_value = redis_future

    # Mock in-memory get for after snapshot
    in_memory_after_future: asyncio.Future[Optional[str]] = asyncio.Future()
    in_memory_after_future.set_result("8.0")  # Value after potential updates
    mock_dual_cache.in_memory_cache.async_get_cache.return_value = (
        in_memory_after_future
    )

    await base_strategy._sync_in_memory_spend_with_redis()

    # Verify the final merged values
    set_cache_calls = mock_dual_cache.in_memory_cache.async_set_cache.call_args_list
    print(f"set_cache_calls: {set_cache_calls}")
    assert any(
        call.kwargs["key"] == "key1" and float(call.kwargs["value"]) == 18.0
        for call in set_cache_calls
    )

    # The sync DRAINS the dirty set. This assertion previously read `== 1`, which pinned
    # the leak: keys were read but never cleared, so the set grew for the lifetime of the
    # process. A key that still needs syncing is re-added by
    # _increment_value_in_current_window, so draining loses nothing.
    assert len(base_strategy.in_memory_keys_to_update) == 0


@pytest.mark.asyncio
async def test_cache_keys_management(base_strategy):
    # Test adding and getting cache keys
    base_strategy.add_to_in_memory_keys_to_update("key1")
    base_strategy.add_to_in_memory_keys_to_update("key2")
    base_strategy.add_to_in_memory_keys_to_update("key1")  # Duplicate should be ignored

    cache_keys = base_strategy.get_in_memory_keys_to_update()
    assert len(cache_keys) == 2
    assert "key1" in cache_keys
    assert "key2" in cache_keys

    # Test resetting cache keys
    base_strategy.reset_in_memory_keys_to_update()
    assert len(base_strategy.get_in_memory_keys_to_update()) == 0


@pytest.mark.asyncio
async def test_sync_does_not_accumulate_in_memory_keys(base_strategy, mock_dual_cache):
    """
    Regression: `in_memory_keys_to_update` must not grow without bound.

    `_sync_in_memory_spend_with_redis` used to read the dirty set via
    `get_in_memory_keys_to_update()` and never clear it, while
    `get_and_reset_in_memory_keys_to_update()` was defined but called nowhere in the
    package. Every key ever touched therefore stayed in the set for the lifetime of the
    process, and each tick re-walked all of them: a batch-get over every key plus a
    per-key merge. With `lowest_tpm_rpm_v2`'s hardcoded 0.1s interval that is 10 full
    walks per second, so CPU climbed with uptime rather than with load -- observed in
    production as a proxy reaching ~1 core over 2-3 days while serving <1 request/minute.

    This drives the real mutation path (`_increment_value_in_current_window`, the only
    caller of `add_to_in_memory_keys_to_update`) rather than assigning the set directly,
    so it pins the actual invariant: sync drains, and only genuinely-changed keys return.
    """

    def _done(value):
        fut: asyncio.Future = asyncio.Future()
        fut.set_result(value)
        return fut

    # Fresh future per call -- these mocks are hit repeatedly across ticks.
    mock_dual_cache.in_memory_cache.async_increment.side_effect = lambda *a, **k: _done(None)
    mock_dual_cache.in_memory_cache.async_batch_get_cache.side_effect = lambda *a, **k: _done(["5.0"])
    mock_dual_cache.in_memory_cache.async_get_cache.side_effect = lambda *a, **k: _done("8.0")
    mock_dual_cache.in_memory_cache.async_set_cache.side_effect = lambda *a, **k: _done(None)
    mock_dual_cache.redis_cache.async_increment_pipeline.side_effect = lambda *a, **k: _done([15.0])

    seen_sizes = []
    for i in range(5):
        # Each tick touches a DIFFERENT key, exactly as real traffic across many
        # deployments/providers does.
        await base_strategy._increment_value_in_current_window(
            key=f"key{i}", value=1.0, ttl=3600
        )
        assert base_strategy.in_memory_keys_to_update == {f"key{i}"}

        await base_strategy._sync_in_memory_spend_with_redis()
        seen_sizes.append(len(base_strategy.in_memory_keys_to_update))

    # Without the drain this is [1, 2, 3, 4, 5] and keeps climbing forever.
    assert seen_sizes == [0, 0, 0, 0, 0], (
        f"dirty set accumulated across sync ticks: {seen_sizes}"
    )
    assert base_strategy.in_memory_keys_to_update == set()
