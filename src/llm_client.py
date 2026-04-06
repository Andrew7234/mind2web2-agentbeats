import os
import logging

import backoff
import litellm

logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.getenv("AGENT_LLM", "openai/gpt-4o-mini")

litellm._turn_on_debug()

def _log_backoff(details):
    exc = details.get("exception")
    tries = details.get("tries")
    wait = details.get("wait")
    kwargs = details.get("kwargs") or {}
    model = kwargs.get("model")
    logger.warning(
        "LLM retry #%s after %.1fs (model=%s): %s: %s",
        tries, wait or 0, model, type(exc).__name__, exc,
    )


@backoff.on_exception(
    backoff.expo,
    (litellm.exceptions.ServiceUnavailableError,
     litellm.exceptions.RateLimitError,
     litellm.exceptions.Timeout,
     litellm.exceptions.APIConnectionError,
     litellm.exceptions.InternalServerError),
    max_tries=8,
    on_backoff=_log_backoff,
)
async def _acompletion_with_backoff(**kwargs):
    return await litellm.acompletion(**kwargs)


class LiteLLMClient:
    """Drop-in replacement for mind2web2's AsyncOpenAIClient using litellm."""

    async def async_response(self, count_token: bool = False, **kwargs):
        if "model" not in kwargs or not kwargs["model"]:
            kwargs["model"] = DEFAULT_MODEL

        response_format = kwargs.get("response_format")
        is_structured = (
            response_format is not None
            and isinstance(response_format, type)
            and hasattr(response_format, "model_validate_json")
        )

        if is_structured:
            kwargs["response_format"] = response_format
            resp = await _acompletion_with_backoff(**kwargs)
            content = resp.choices[0].message.content
            parsed = response_format.model_validate_json(content)
            if count_token:
                tokens = {
                    "input_tokens": resp.usage.prompt_tokens,
                    "output_tokens": resp.usage.completion_tokens,
                }
                return parsed, tokens
            return parsed

        resp = await _acompletion_with_backoff(**kwargs)
        tokens = {
            "input_tokens": resp.usage.prompt_tokens,
            "output_tokens": resp.usage.completion_tokens,
        }
        content = resp.choices[0].message.content
        if count_token:
            return content, tokens
        return content
