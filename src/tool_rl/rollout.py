"""Explicit textual multi-turn controller shared by evaluation and GRPO."""

import copy
import json

from .protocol import format_tool_result, invalid_tool_feedback, make_prompt, parse_tool_action


def valid_call(call):
    """Compatibility check for the old structured fixture representation."""
    if not isinstance(call, dict) or not isinstance(call.get("function"), dict):
        return False
    function = call["function"]
    arguments = function.get("arguments")
    return (
        call.get("type") == "function"
        and function.get("name") == "execute_python"
        and isinstance(arguments, dict)
        and set(arguments) == {"code"}
        and isinstance(arguments["code"], str)
    )


def _decoded_result(value):
    if isinstance(value, dict):
        return value
    try:
        decoded = json.loads(value)
        return decoded if isinstance(decoded, dict) else {"ok": False, "error": "unstructured_tool_result"}
    except (json.JSONDecodeError, TypeError):
        return {"ok": False, "error": "unstructured_tool_result"}


def trajectory_metrics(messages):
    """Reconstruct protocol metrics from the stored public trajectory."""
    requested = valid = invalid = executed = successful = errors = timeouts = denials = 0
    turns = 0
    for message in messages:
        if message.get("role") == "assistant":
            turns += 1
            native_calls = message.get("tool_calls") or []
            if native_calls:
                requested += len(native_calls)
                native_valid = sum(bool(valid_call(call)) for call in native_calls)
                valid += native_valid
                invalid += len(native_calls) - native_valid
            else:
                action = parse_tool_action(message.get("content", ""))
                if action["kind"] == "tool_call":
                    requested += 1
                    valid += 1
                elif action["kind"] == "invalid":
                    requested += 1
                    invalid += 1
        event = message.get("tool_event")
        if not isinstance(event, dict):
            continue
        if event.get("kind") == "execution":
            executed += 1
            successful += int(event.get("ok") is True)
            errors += int(event.get("ok") is not True)
            timeouts += int(event.get("timed_out") is True)
        elif event.get("kind") == "budget_denial":
            denials += 1
    return {
        "requested_calls": requested,
        "valid_calls": valid,
        "invalid_calls": invalid,
        "executed_calls": executed,
        "successful_calls": successful,
        "tool_errors": errors,
        "timeouts": timeouts,
        "budget_denials": denials,
        "agent_turns": turns,
    }


def run_agent_episode(
    prompt,
    backend,
    environment,
    *,
    max_execute_calls,
    max_invalid_calls,
    max_agent_turns,
    tools_enabled=True,
    require_execute=False,
    entry_point=None,
):
    """Generate, parse, execute, observe, and recover under independent budgets."""
    for name, value in (
        ("max_execute_calls", max_execute_calls),
        ("max_invalid_calls", max_invalid_calls),
        ("max_agent_turns", max_agent_turns),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    environment.reset()
    # No native tool schema is passed. The fixed prefix defines the complete
    # textual interface and this controller owns every parse and execution.
    backend.start(copy.deepcopy(prompt), None)
    messages, generated = [], 0
    invalid_calls = execute_calls = 0
    truncated, error, finished = False, None, False

    for _ in range(max_agent_turns):
        turn = backend.next_turn()
        generated += turn.get("generated_tokens", 0)
        if turn.get("error"):
            error = turn["error"]
            truncated = error in {"token_budget", "context_budget"}
            break
        raw = turn.get("message", {}).get("content", "")
        message = {"role": "assistant", "content": raw}
        messages.append(message)
        if not turn.get("terminated"):
            truncated = True
            error = "unterminated_generation"
            break

        action = parse_tool_action(raw)
        if action["kind"] == "final":
            if tools_enabled and require_execute and execute_calls == 0:
                feedback = format_tool_result({
                    "ok": False,
                    "error": "tool_use_required",
                    "message": (
                        "This tools-enabled experiment requires at least one "
                        "execute_python(code: str) call before the final answer. "
                        "Call execute_python now, inspect its result, then return "
                        "the completed answer or solution."
                    ),
                })
                messages.append({
                    "role": "user",
                    "content": feedback,
                    "tool_event": {"kind": "requirement_feedback"},
                })
                backend.observe([{"role": "user", "content": feedback}])
                continue
            finished = True
            break
        if not tools_enabled:
            error = "tool_disabled"
            break

        if action["kind"] == "invalid":
            if invalid_calls >= max_invalid_calls:
                error = "invalid_tool_budget"
                break
            invalid_calls += 1
            feedback = invalid_tool_feedback(
                action, entry_point=entry_point, exhausted=invalid_calls == max_invalid_calls
            )
            observation = {
                "role": "user",
                "content": feedback,
                "tool_event": {"kind": "invalid_feedback", "error": action["error"]},
            }
            messages.append(observation)
            backend.observe([{"role": "user", "content": feedback}])
            continue

        if execute_calls >= max_execute_calls:
            denial = format_tool_result({
                "ok": False,
                "error": "execute_budget_exhausted",
                "message": "The execute_python budget is exhausted. Return the final answer or code now.",
            })
            observation = {
                "role": "user",
                "content": denial,
                "tool_event": {"kind": "budget_denial"},
            }
            messages.append(observation)
            backend.observe([{"role": "user", "content": denial}])
            continue

        encoded = environment.execute_python(action["code"])
        execute_calls += 1
        result = _decoded_result(encoded)
        content = format_tool_result(result)
        observation = {
            "role": "user",
            "content": content,
            "tool_event": {
                "kind": "execution",
                "ok": result.get("ok") is True,
                "timed_out": result.get("timed_out") is True,
                "exit_code": result.get("exit_code"),
                "error": result.get("error"),
            },
        }
        messages.append(observation)
        backend.observe([{"role": "user", "content": content}])

    if not finished and error is None:
        error = "agent_turn_budget"
    return {
        "prompt": prompt,
        "messages": messages,
        "generated_tokens": generated,
        "truncated": truncated,
        "error": error,
        "tool_metrics": trajectory_metrics(messages),
        "executions": list(environment._calls),
    }


def run_native_tool_episode(
    prompt,
    backend,
    environment,
    *,
    max_execute_calls,
    max_invalid_calls,
    max_agent_turns,
    tools_enabled=True,
):
    """Run the compact native Qwen tool loop with current independent budgets."""
    for name, value in (
        ("max_execute_calls", max_execute_calls),
        ("max_invalid_calls", max_invalid_calls),
        ("max_agent_turns", max_agent_turns),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not tools_enabled:
        raise ValueError("compact_math requires its historical native tool schema")

    environment.reset()
    backend.start(copy.deepcopy(prompt), [environment.execute_python])
    messages, generated = [], 0
    execute_calls = invalid_calls = 0
    truncated, error, finished = False, None, False

    for _ in range(max_agent_turns):
        turn = backend.next_turn()
        generated += turn.get("generated_tokens", 0)
        if turn.get("error"):
            error = turn["error"]
            truncated = error in {"token_budget", "context_budget"}
            break
        message = dict(turn.get("message") or {}, role="assistant")
        message.setdefault("content", "")
        messages.append(message)
        if not turn.get("terminated"):
            truncated = True
            error = "unterminated_generation"
            break

        calls = message.get("tool_calls") or []
        if not calls:
            finished = True
            break

        observations, stored_observations = [], []
        for call in calls:
            function = call.get("function", {}) if isinstance(call, dict) else {}
            if not isinstance(function, dict):
                function = {}
            name = function.get("name", "unknown")
            if not valid_call(call):
                invalid_calls += 1
                if invalid_calls > max_invalid_calls:
                    error = "invalid_tool_budget"
                    break
                parse_error = (
                    call.get("parse_error", "invalid_tool_call")
                    if isinstance(call, dict)
                    else "invalid_tool_call"
                )
                if name not in {"unknown", "execute_python"}:
                    detail = (
                        f"`{name}` is not an available tool. The only available "
                        "tool is execute_python(code: str)."
                    )
                else:
                    detail = (
                        "The native tool call was malformed. The only available "
                        "tool is execute_python(code: str)."
                    )
                content = json.dumps({
                    "ok": False,
                    "error": parse_error,
                    "message": detail,
                })
                event = {"kind": "invalid_feedback", "error": parse_error}
            elif execute_calls >= max_execute_calls:
                content = json.dumps({
                    "ok": False,
                    "error": "execute_budget_exhausted",
                    "message": "Return the final answer now.",
                })
                event = {"kind": "budget_denial"}
            else:
                content = environment.execute_python(function["arguments"]["code"])
                execute_calls += 1
                result = _decoded_result(content)
                event = {
                    "kind": "execution",
                    "ok": result.get("ok") is True,
                    "timed_out": result.get("timed_out") is True,
                    "exit_code": result.get("exit_code"),
                    "error": result.get("error"),
                }
            observation = {"role": "tool", "name": name, "content": content}
            observations.append(observation)
            stored_observations.append({**observation, "tool_event": event})
        messages.extend(stored_observations)
        if error is not None:
            break
        backend.observe(observations)

    if not finished and error is None:
        error = "agent_turn_budget"
    return {
        "prompt": prompt,
        "messages": messages,
        "generated_tokens": generated,
        "truncated": truncated,
        "error": error,
        "tool_metrics": trajectory_metrics(messages),
        "executions": list(environment._calls),
    }


def run_episode(problem, backend, environment, *, max_execute_calls=2, max_invalid_calls=2, max_agent_turns=6, tools_enabled=True, require_execute=False):
    """Run one math episode using the shared explicit controller."""
    prompt = make_prompt(problem, max_execute_calls, tools_enabled)
    return run_agent_episode(
        prompt,
        backend,
        environment,
        max_execute_calls=max_execute_calls,
        max_invalid_calls=max_invalid_calls,
        max_agent_turns=max_agent_turns,
        tools_enabled=tools_enabled,
        require_execute=require_execute,
    )
