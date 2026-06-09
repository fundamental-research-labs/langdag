# Apple Foundation Models `fm serve`

Last tested: 2026-06-09 on `http://127.0.0.1:1976`.

This documents the local `fm serve` HTTP API from observed behavior because the
online documentation for the server subcommand is sparse. The interface is best
treated as an OpenAI Chat Completions-style subset for Apple Foundation Models,
not as a full OpenAI-compatible server.

Useful upstream references:

- Apple Foundation Models framework: https://developer.apple.com/documentation/foundationmodels
- Apple Intelligence resources: https://developer.apple.com/apple-intelligence/resources/
- WWDC26 "What's new in the Foundation Models framework": https://developer.apple.com/videos/play/wwdc2026/241/
- Google Gemini OpenAI compatibility: https://ai.google.dev/gemini-api/docs/openai
- Google Gemini native function calling: https://ai.google.dev/gemini-api/docs/function-calling
- OpenAI Chat Completions reference: https://developers.openai.com/api/reference/resources/chat

## Standards Note

Even if one of the backing models is Gemini-like internally, `fm serve` does not
expose Gemini's native `contents`, `functionDeclarations`, or function response
parts. Speak the local server's Chat Completions dialect instead.

The basic tool shape matches both OpenAI Chat Completions and Google's Gemini
OpenAI-compatibility layer:

```json
{
  "messages": [{ "role": "user", "content": "..." }],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "tool_name",
        "description": "...",
        "parameters": { "type": "object", "properties": {} }
      }
    }
  ],
  "tool_choice": "auto"
}
```

The local server is still only partially compatible with those standards. The
main observed differences are default streaming, unsupported `developer` role,
unsupported object-form forced tool choice, unreliable `tool_choice` enforcement,
and `json_schema`-only structured output.

## Server

Start:

```sh
fm serve
```

Observed CLI help:

```text
POST /v1/chat/completions  Chat completions (streaming & non-streaming)
GET  /v1/models            List available models
GET  /health               Health check
```

Models:

- `system`: on-device Apple Foundation Model. This is the default if `model` is omitted.
- `pcc`: Apple Foundation Model on Private Cloud Compute.

Health check:

```sh
curl -sS http://127.0.0.1:1976/health
```

Observed response:

```json
{
  "status": "fm serve is running",
  "models": [
    { "available": true, "name": "system" },
    { "available": true, "name": "pcc" }
  ]
}
```

List models:

```sh
curl -sS http://127.0.0.1:1976/v1/models
```

## Chat Basics

Always set `stream` explicitly. Unlike OpenAI's default, omitting `stream` was
observed to return server-sent events. Use `stream: false` when you want one JSON
object.

Minimal non-streaming request:

```sh
curl -sS http://127.0.0.1:1976/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "system",
    "stream": false,
    "max_tokens": 64,
    "messages": [
      { "role": "user", "content": "Answer briefly: what is 2 plus 2?" }
    ]
  }'
```

Response shape:

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1781039086,
  "model": "system",
  "choices": [
    {
      "index": 0,
      "finish_reason": "stop",
      "message": {
        "role": "assistant",
        "content": "4"
      }
    }
  ],
  "usage": {
    "prompt_tokens": 62,
    "completion_tokens": 3,
    "total_tokens": 65
  }
}
```

Supported roles tested:

- `system`: use this for system prompts or instructions.
- `user`: user input.
- `assistant`: prior assistant turns.
- `tool`: tool outputs, when paired with a prior assistant `tool_calls` item.

Rejected role tested:

- `developer`: returns a 400 decode error in this build.

Multi-turn history works by sending the full message list:

```json
{
  "model": "system",
  "stream": false,
  "messages": [
    { "role": "user", "content": "My code word is tangerine." },
    { "role": "assistant", "content": "Understood." },
    { "role": "user", "content": "What is my code word? Reply with one word." }
  ]
}
```

`content` can be a plain string. OpenAI-style text content arrays were also
accepted:

```json
{
  "role": "user",
  "content": [
    { "type": "text", "text": "Reply with CONTENT_ARRAY_OK only." }
  ]
}
```

## System Prompts

Use a `system` message at the front of `messages`:

```json
{
  "model": "pcc",
  "stream": false,
  "messages": [
    {
      "role": "system",
      "content": "For every request, reply with exactly BLUE_TOKEN and no other text."
    },
    { "role": "user", "content": "What is 2 plus 2?" }
  ]
}
```

Both `system` and `pcc` obeyed this in controlled tests.

Do not use top-level `instructions` with this server. It was accepted
syntactically but ignored by both models in Chat Completions requests.

## Streaming

Set `stream: true` to receive server-sent events:

```json
{
  "model": "system",
  "stream": true,
  "messages": [
    { "role": "user", "content": "Count from 1 to 3." }
  ]
}
```

Observed SSE shape:

```text
data: {"model":"system","id":"chatcmpl-...","choices":[{"delta":{"role":"assistant"}}]}

data: {"model":"system","id":"chatcmpl-...","choices":[{"delta":{"content":"1"}}]}

data: {"model":"system","id":"chatcmpl-...","choices":[{"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

`stream_options: { "include_usage": true }` was accepted, but no final usage
chunk was observed.

## Tool Calling

Tool calling uses OpenAI-style `tools` and `tool_calls`.

Request:

```json
{
  "model": "system",
  "stream": false,
  "messages": [
    {
      "role": "user",
      "content": "What is the weather in Paris? Use the get_weather tool."
    }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get current weather for a city.",
        "parameters": {
          "type": "object",
          "properties": {
            "location": { "type": "string" },
            "unit": {
              "type": "string",
              "enum": ["celsius", "fahrenheit"]
            }
          },
          "required": ["location", "unit"],
          "additionalProperties": false
        }
      }
    }
  ],
  "tool_choice": "auto",
  "max_tokens": 128
}
```

Observed tool-call response:

```json
{
  "choices": [
    {
      "index": 0,
      "finish_reason": "tool_calls",
      "message": {
        "role": "assistant",
        "tool_calls": [
          {
            "id": "37DDFABB-4B70-454D-8ED9-976E1EBFD6B2",
            "type": "function",
            "function": {
              "name": "get_weather",
              "arguments": "{\"unit\": \"celsius\", \"location\": \"Paris\"}"
            }
          }
        ]
      }
    }
  ]
}
```

Important details:

- `function.arguments` is a JSON string. Parse it before invoking your local tool.
- Multiple tool calls in one assistant message were observed with both `system`
  and `pcc`.
- Streaming tool calls use `delta.tool_calls`.
- JSON Schema features tested successfully in tool parameters: object,
  properties, required, string enum, and `additionalProperties: false`.
- Object tool parameter schemas must include a top-level `required` array in
  this build. Use `"required": []` when no fields are required; omitting it can
  return `Invalid tool definition`.

Tool-choice values tested:

- `auto`: works.
- `required`: accepted, and tool calls were produced when the prompt/tool setup
  clearly called for tools. It is not enforced in this build: with an unrelated
  "Say hello" prompt, both models answered normally instead of calling a tool.
- `none`: accepted but not reliable. If tools are present and the user asks for a
  tool, both models still emitted tool calls. To prohibit tool use, omit `tools`.
- Object-form forced choice is rejected with 400:

```json
{
  "tool_choice": {
    "type": "function",
    "function": { "name": "get_weather" }
  }
}
```

That object-form forced choice is part of OpenAI's documented Chat Completions
API, so this is a concrete compatibility gap.

## Sending Tool Results Back

After you execute a tool locally, append the assistant tool-call message and a
matching `role: "tool"` message. The `tool_call_id` must match the returned
tool-call `id`.

```json
{
  "model": "system",
  "stream": false,
  "messages": [
    {
      "role": "user",
      "content": "What is the weather in Paris? Use the get_weather tool."
    },
    {
      "role": "assistant",
      "tool_calls": [
        {
          "id": "C26CCD61-2786-4632-9611-0D36ADA30DA0",
          "type": "function",
          "function": {
            "name": "get_weather",
            "arguments": "{\"location\": \"Paris\"}"
          }
        }
      ]
    },
    {
      "role": "tool",
      "tool_call_id": "C26CCD61-2786-4632-9611-0D36ADA30DA0",
      "content": "{\"location\":\"Paris\",\"temperature\":18,\"unit\":\"celsius\",\"condition\":\"light rain\"}"
    }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get current weather for a city.",
        "parameters": {
          "type": "object",
          "properties": {
            "location": { "type": "string" }
          },
          "required": ["location"]
        }
      }
    }
  ],
  "max_tokens": 128
}
```

Observed final answer from both models:

```text
The weather in Paris is 18 C with light rain.
```

For multiple tool calls, send one `role: "tool"` message per returned
`tool_calls[]` item.

## Structured Output

`response_format: { "type": "json_object" }` is rejected:

```json
{
  "error": {
    "code": "400",
    "message": "response_format type 'json_object' is not supported. Use 'json_schema' instead.",
    "type": "invalid_request_error"
  }
}
```

Use `json_schema` instead:

```json
{
  "model": "pcc",
  "stream": false,
  "messages": [
    { "role": "user", "content": "Return whether two plus two equals four." }
  ],
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "math_check",
      "schema": {
        "type": "object",
        "properties": {
          "ok": { "type": "boolean" },
          "answer": { "type": "integer" }
        },
        "required": ["ok", "answer"],
        "additionalProperties": false
      },
      "strict": true
    }
  }
}
```

The model output is still returned as a JSON string in
`choices[0].message.content`. Parse it yourself.

## Reasoning, Thinking, And Sampling

No working reasoning/thinking/effort control was found for `fm serve`.

Fields tested and apparently ignored by both `system` and `pcc`:

- `reasoning_effort`
- `reasoning`
- `thinking_level`
- `thinking_budget`
- `thinking_config`
- `google.thinking_config`
- `extra_body.google.thinking_config`
- `context_options`
- `contextOptions`
- `reasoning_level`
- `reasoningLevel`
- `verbosity`

Invalid values for those fields, for example
`{ "reasoning_effort": "banana" }`, were accepted with HTTP 200. That strongly
suggests the server ignores them instead of validating or applying them.

No hidden chain-of-thought or thought-summary field was observed in response
messages, even with `include_thoughts: true` variants.

Sampling-related fields:

- `temperature` and `top_p` are decoded as numeric fields. Invalid string values
  returned 400 decode errors.
- Range validation was not observed: `temperature: 99` was accepted.
- `greedy` is not a reliable HTTP option. Both `greedy: true` and
  `greedy: "banana"` were accepted, which suggests the field is ignored by
  `fm serve` even though the `fm respond` CLI has a `--greedy` flag.

System-model CLI options did not appear to map to HTTP request fields:

- `use_case`, including invalid values, was accepted and appeared ignored.
- `guardrails`, including invalid values, was accepted and appeared ignored.
- `system_model_options` was accepted and appeared ignored.

## Other Compatibility Notes

Observed as accepted:

- `temperature`
- `top_p`
- `stream_options`, but `include_usage` did not emit usage.

Observed as accepted but ineffective or not consistently enforced in basic tests:

- `max_tokens`: accepted but ignored in all tested `system` and `pcc` requests.
- `max_completion_tokens`: the only budget-like field found. It constrained
  visible output for the `system` model in multiple tests, including a 1-to-50
  counting prompt and a repeated-word prompt. It was ignored by `pcc`.
  Tool-call flows need enough budget for the serialized tool call: with
  `system`, values 1, 5, and 10 returned empty assistant content for a simple
  weather tool request, while 20 returned the expected tool call.
- `n`: `n: 2` returned one choice in the tested request.
- `stop`: exact stop sequences were included in output instead of stopping
  generation.

Observed errors:

- Unknown model:

```json
{
  "error": {
    "code": "400",
    "message": "Unknown model 'bogus'. Available models: system, pcc",
    "type": "invalid_request_error"
  }
}
```

- Unsupported or mismatched request shapes usually return:

```json
{
  "error": {
    "code": "400",
    "message": "Invalid JSON: The data couldn't be read because it isn't in the correct format.",
    "type": "invalid_request_error"
  }
}
```

## `system` vs `pcc`

The wire interface appears the same for both models:

- Same endpoints.
- Same `model` field selection.
- Same message roles.
- Same non-streaming and streaming response shapes.
- Same tool-call and tool-result flow.
- Same observed compatibility gaps around `developer`, `tool_choice: "none"`,
  object-form forced tool choice, and default streaming.

Do not assume usage accounting is equivalent:

- `system` returned normal-looking `prompt_tokens`, `completion_tokens`, and
  `total_tokens`.
- `pcc` repeatedly returned `prompt_tokens: 0`, with `total_tokens` matching or
  resembling completion-only counts in the tested requests.

Treat `usage` as advisory, especially for `pcc`.

## Practical Client Guidance

For clients, use this pattern:

1. Send `stream: false` unless you explicitly want SSE.
2. Use `role: "system"` for instructions. Do not use `developer` or top-level
   `instructions`.
3. Use `tools` with OpenAI-style function tool definitions.
4. Accept assistant responses with either `message.content` or
   `message.tool_calls`.
5. Parse `tool_calls[].function.arguments` as JSON.
6. Execute tools locally.
7. Send tool outputs back as `role: "tool"` messages with matching
   `tool_call_id`.
8. Omit `tools` when you want to prevent tool use.
9. Use `max_completion_tokens` only as a `system`-model best-effort cap. Do not
   rely on `max_tokens`, and do not expect server-side output budgeting for
   `pcc`.
10. Treat `tool_choice`, `n`, and `stop` as advisory until your exact client path
   is tested.
11. Use `response_format.type: "json_schema"` for structured content.
