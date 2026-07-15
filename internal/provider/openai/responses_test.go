package openai

import (
	"encoding/json"
	"errors"
	"io"
	"strings"
	"testing"

	"langdag.com/langdag/types"
)

func TestBuildResponsesRequest_Basic(t *testing.T) {
	req := &types.CompletionRequest{
		Model:  "grok-3",
		System: "You are helpful.",
		Messages: []types.Message{
			{Role: "user", Content: json.RawMessage(`"Hello"`)},
		},
		MaxTokens:   1024,
		Temperature: 0.7,
	}

	body := buildResponsesRequest(req, false)
	var m map[string]interface{}
	if err := json.Unmarshal(body, &m); err != nil {
		t.Fatalf("failed to unmarshal request: %v", err)
	}

	if m["model"] != "grok-3" {
		t.Errorf("model = %v, want %q", m["model"], "grok-3")
	}
	if m["instructions"] != "You are helpful." {
		t.Errorf("instructions = %v, want %q", m["instructions"], "You are helpful.")
	}
	if m["stream"] != false {
		t.Errorf("stream = %v, want false", m["stream"])
	}
	if m["store"] != false {
		t.Errorf("store = %v, want false", m["store"])
	}
	if m["max_output_tokens"] != float64(1024) {
		t.Errorf("max_output_tokens = %v, want 1024", m["max_output_tokens"])
	}

	input, ok := m["input"].([]interface{})
	if !ok || len(input) != 1 {
		t.Fatalf("expected 1 input item, got %v", m["input"])
	}
	msg := input[0].(map[string]interface{})
	if msg["role"] != "user" {
		t.Errorf("input[0].role = %v, want %q", msg["role"], "user")
	}
	if msg["content"] != "Hello" {
		t.Errorf("input[0].content = %v, want %q", msg["content"], "Hello")
	}
}

func TestBuildResponsesRequest_WithTools(t *testing.T) {
	req := &types.CompletionRequest{
		Model: "grok-3",
		Messages: []types.Message{
			{Role: "user", Content: json.RawMessage(`"Search the web"`)},
		},
		Tools: []types.ToolDefinition{
			{Name: types.ServerToolWebSearch, Description: "Web search"},
			{
				Name:        "get_weather",
				Description: "Get weather",
				InputSchema: json.RawMessage(`{"type":"object","properties":{"location":{"type":"string"}}}`),
			},
		},
	}

	body := buildResponsesRequest(req, true)
	var m map[string]interface{}
	if err := json.Unmarshal(body, &m); err != nil {
		t.Fatalf("failed to unmarshal request: %v", err)
	}

	if m["stream"] != true {
		t.Errorf("stream = %v, want true", m["stream"])
	}

	tools, ok := m["tools"].([]interface{})
	if !ok || len(tools) != 2 {
		t.Fatalf("expected 2 tools, got %v", m["tools"])
	}

	// First tool: server-side web_search
	t0 := tools[0].(map[string]interface{})
	if t0["type"] != "web_search" {
		t.Errorf("tools[0].type = %v, want %q", t0["type"], "web_search")
	}

	// Second tool: function tool
	t1 := tools[1].(map[string]interface{})
	if t1["type"] != "function" {
		t.Errorf("tools[1].type = %v, want %q", t1["type"], "function")
	}
	if t1["name"] != "get_weather" {
		t.Errorf("tools[1].name = %v, want %q", t1["name"], "get_weather")
	}
}

func TestBuildOpenAIResponsesRequest_ReasoningFromThink(t *testing.T) {
	for _, tc := range []struct {
		name       string
		model      string
		think      bool
		wantEffort string
	}{
		{name: "gpt-5.5 enabled", model: "gpt-5.5", think: true, wantEffort: "medium"},
		{name: "gpt-5.5 disabled", model: "gpt-5.5", think: false, wantEffort: "none"},
		{name: "gpt-5.6 enabled", model: "gpt-5.6-sol", think: true, wantEffort: "medium"},
		{name: "gpt-5.6 disabled", model: "gpt-5.6-sol", think: false, wantEffort: "none"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			req := &types.CompletionRequest{
				Model: tc.model,
				Messages: []types.Message{
					{Role: "user", Content: json.RawMessage(`"Hello"`)},
				},
				Think: &tc.think,
			}

			body := buildOpenAIResponsesRequest(req, false)
			var m map[string]interface{}
			if err := json.Unmarshal(body, &m); err != nil {
				t.Fatalf("failed to unmarshal request: %v", err)
			}
			if _, ok := m["think"]; ok {
				t.Fatalf("request included non-OpenAI think field: %s", string(body))
			}
			reasoning, ok := m["reasoning"].(map[string]interface{})
			if !ok {
				t.Fatalf("reasoning = %#v, want object", m["reasoning"])
			}
			if reasoning["effort"] != tc.wantEffort {
				t.Fatalf("reasoning.effort = %v, want %q", reasoning["effort"], tc.wantEffort)
			}
		})
	}
}

func TestBuildOpenAIResponsesRequest_OmitsReasoningForNonReasoningModel(t *testing.T) {
	think := true
	req := &types.CompletionRequest{
		Model: "gpt-4.1",
		Messages: []types.Message{
			{Role: "user", Content: json.RawMessage(`"Hello"`)},
		},
		Think: &think,
	}

	body := buildOpenAIResponsesRequest(req, false)
	var m map[string]interface{}
	if err := json.Unmarshal(body, &m); err != nil {
		t.Fatalf("failed to unmarshal request: %v", err)
	}
	if _, ok := m["reasoning"]; ok {
		t.Fatalf("reasoning should be omitted for non-reasoning model: %s", string(body))
	}
	if _, ok := m["think"]; ok {
		t.Fatalf("request included non-OpenAI think field: %s", string(body))
	}
}

func TestBuildResponsesRequest_GrokDoesNotMapThinkToOpenAIReasoning(t *testing.T) {
	think := true
	req := &types.CompletionRequest{
		Model: "grok-3",
		Messages: []types.Message{
			{Role: "user", Content: json.RawMessage(`"Hello"`)},
		},
		Think: &think,
	}

	body := buildResponsesRequest(req, false)
	var m map[string]interface{}
	if err := json.Unmarshal(body, &m); err != nil {
		t.Fatalf("failed to unmarshal request: %v", err)
	}
	if _, ok := m["reasoning"]; ok {
		t.Fatalf("Grok request should not include OpenAI reasoning: %s", string(body))
	}
	if _, ok := m["think"]; ok {
		t.Fatalf("Responses request should not include legacy think field: %s", string(body))
	}
}

func TestConvertResponsesMessages_ToolUseAndResult(t *testing.T) {
	toolUseBlocks := []types.ContentBlock{
		{Type: "text", Text: "Let me check."},
		{Type: "tool_use", ID: "call_123", Name: "get_weather", Input: json.RawMessage(`{"location":"SF"}`)},
	}
	toolUseJSON, _ := json.Marshal(toolUseBlocks)

	toolResultBlocks := []types.ContentBlock{
		{Type: "tool_result", ToolUseID: "call_123", Content: "72°F"},
	}
	toolResultJSON, _ := json.Marshal(toolResultBlocks)

	messages := []types.Message{
		{Role: "user", Content: json.RawMessage(`"What is the weather?"`)},
		{Role: "assistant", Content: toolUseJSON},
		{Role: "user", Content: toolResultJSON},
	}

	_, input := convertResponsesMessages(messages, "")
	if len(input) < 4 {
		t.Fatalf("expected at least 4 input items, got %d", len(input))
	}

	// Item 0: user message
	msg0, ok := input[0].(responsesInputMessage)
	if !ok || msg0.Role != "user" {
		t.Errorf("input[0] should be user message, got %T %+v", input[0], input[0])
	}

	// Item 1: assistant text
	msg1, ok := input[1].(responsesInputMessage)
	if !ok || msg1.Role != "assistant" {
		t.Errorf("input[1] should be assistant message, got %T %+v", input[1], input[1])
	}
	msg1Content, ok := msg1.Content.([]interface{})
	if !ok || len(msg1Content) != 1 {
		t.Fatalf("input[1].content should be one assistant output part, got %#v", msg1.Content)
	}
	outputText, ok := msg1Content[0].(responsesOutputText)
	if !ok {
		t.Fatalf("input[1].content[0] should be output_text, got %T %#v", msg1Content[0], msg1Content[0])
	}
	if outputText.Type != "output_text" || outputText.Text != "Let me check." {
		t.Fatalf("assistant text part = %+v, want output_text Let me check.", outputText)
	}

	// Item 2: function call
	fc, ok := input[2].(responsesFunctionCallInput)
	if !ok {
		t.Fatalf("input[2] should be function call, got %T", input[2])
	}
	if fc.Type != "function_call" {
		t.Errorf("function_call.type = %q, want %q", fc.Type, "function_call")
	}
	if fc.CallID != "call_123" {
		t.Errorf("function_call.call_id = %q, want %q", fc.CallID, "call_123")
	}
	if fc.Name != "get_weather" {
		t.Errorf("function_call.name = %q, want %q", fc.Name, "get_weather")
	}

	// Item 3: function call output
	fco, ok := input[3].(responsesFunctionCallOutput)
	if !ok {
		t.Fatalf("input[3] should be function call output, got %T", input[3])
	}
	if fco.Type != "function_call_output" {
		t.Errorf("function_call_output.type = %q, want %q", fco.Type, "function_call_output")
	}
	if fco.CallID != "call_123" {
		t.Errorf("function_call_output.call_id = %q, want %q", fco.CallID, "call_123")
	}
	if fco.Output != "72°F" {
		t.Errorf("function_call_output.output = %q, want %q", fco.Output, "72°F")
	}
}

func TestConvertResponsesMessages_AssistantPlainTextUsesOutputText(t *testing.T) {
	messages := []types.Message{
		{Role: "assistant", Content: json.RawMessage(`"Done."`)},
	}

	_, input := convertResponsesMessages(messages, "")
	if len(input) != 1 {
		t.Fatalf("expected 1 input item, got %d", len(input))
	}
	msg, ok := input[0].(responsesInputMessage)
	if !ok || msg.Role != "assistant" {
		t.Fatalf("input[0] should be assistant message, got %T %+v", input[0], input[0])
	}
	parts, ok := msg.Content.([]interface{})
	if !ok || len(parts) != 1 {
		t.Fatalf("assistant content should be one output part, got %#v", msg.Content)
	}
	part, ok := parts[0].(responsesOutputText)
	if !ok {
		t.Fatalf("assistant content part should be output_text, got %T %#v", parts[0], parts[0])
	}
	if part.Type != "output_text" || part.Text != "Done." {
		t.Fatalf("assistant content part = %+v, want output_text Done.", part)
	}
}

func TestBuildOpenAIResponsesRequest_Debug20260520ToolResultHistory(t *testing.T) {
	firstToolUseBlocks := []types.ContentBlock{
		{Type: "tool_use", ID: "call_status", Name: "git", Input: json.RawMessage(`{"subcommand":"status","args":["--short","--branch"]}`)},
		{Type: "tool_use", ID: "call_log", Name: "git", Input: json.RawMessage(`{"subcommand":"log","args":["--oneline"]}`)},
	}
	firstToolUseJSON, _ := json.Marshal(firstToolUseBlocks)
	firstToolResultBlocks := []types.ContentBlock{
		{Type: "tool_result", ToolUseID: "call_status", Content: "## branch [ahead 2]\n"},
		{Type: "tool_result", ToolUseID: "call_log", Content: "ee1597f change\n7769421 change\n"},
	}
	firstToolResultJSON, _ := json.Marshal(firstToolResultBlocks)
	pushToolUseBlocks := []types.ContentBlock{
		{Type: "text", Text: "You are ahead by 2 commits. Push?"},
		{Type: "tool_use", ID: "call_push", Name: "git", Input: json.RawMessage(`{"subcommand":"push","args":["origin","branch"]}`)},
	}
	pushToolUseJSON, _ := json.Marshal(pushToolUseBlocks)
	pushToolResultBlocks := []types.ContentBlock{
		{Type: "tool_result", ToolUseID: "call_push", Content: "To github.com:aduermael/herm.git\n"},
	}
	pushToolResultJSON, _ := json.Marshal(pushToolResultBlocks)

	req := &types.CompletionRequest{
		Model: "gpt-5.5-2026-04-23",
		Messages: []types.Message{
			{Role: "user", Content: json.RawMessage(`"Can you push those 2 commits?"`)},
			{Role: "assistant", Content: firstToolUseJSON},
			{Role: "user", Content: firstToolResultJSON},
			{Role: "assistant", Content: pushToolUseJSON},
			{Role: "user", Content: pushToolResultJSON},
		},
	}

	body := buildOpenAIResponsesRequest(req, true)
	var payload map[string]interface{}
	if err := json.Unmarshal(body, &payload); err != nil {
		t.Fatalf("failed to unmarshal request: %v", err)
	}
	for _, key := range []string{"messages", "max_tokens", "think"} {
		if _, ok := payload[key]; ok {
			t.Fatalf("request included %q: %s", key, string(body))
		}
	}
	input, ok := payload["input"].([]interface{})
	if !ok || len(input) != 8 {
		t.Fatalf("input length = %d, want 8; body=%s", len(input), string(body))
	}
	assistant, ok := input[5].(map[string]interface{})
	if !ok {
		t.Fatalf("input[5] = %T, want assistant message", input[5])
	}
	if assistant["type"] != "message" || assistant["role"] != "assistant" {
		t.Fatalf("input[5] = %#v, want assistant message", assistant)
	}
	content, ok := assistant["content"].([]interface{})
	if !ok || len(content) != 1 {
		t.Fatalf("input[5].content = %#v, want one content part", assistant["content"])
	}
	textPart, ok := content[0].(map[string]interface{})
	if !ok {
		t.Fatalf("input[5].content[0] = %T, want object", content[0])
	}
	if textPart["type"] != "output_text" {
		t.Fatalf("input[5].content[0].type = %v, want output_text; body=%s", textPart["type"], string(body))
	}

	pushCall, ok := input[6].(map[string]interface{})
	if !ok {
		t.Fatalf("input[6] = %T, want function_call", input[6])
	}
	if pushCall["type"] != "function_call" || pushCall["call_id"] != "call_push" {
		t.Fatalf("input[6] = %#v, want function_call call_push", pushCall)
	}
	pushResult, ok := input[7].(map[string]interface{})
	if !ok {
		t.Fatalf("input[7] = %T, want function_call_output", input[7])
	}
	if pushResult["type"] != "function_call_output" || pushResult["call_id"] != "call_push" {
		t.Fatalf("input[7] = %#v, want function_call_output call_push", pushResult)
	}
}

// ---------------------------------------------------------------------------
// mapResponsesUsage — cache and reasoning token mapping
// ---------------------------------------------------------------------------

func TestMapResponsesUsage(t *testing.T) {
	u := &responsesUsage{
		InputTokens:  100,
		OutputTokens: 50,
		InputTokensDetails: &responsesInputTokensDetails{
			CachedTokens: 30,
		},
		OutputTokensDetails: &responsesOutputTokensDetails{
			ReasoningTokens: 10,
		},
	}

	result := mapResponsesUsage(u)

	if result.InputTokens != 70 {
		t.Errorf("expected InputTokens=70, got %d", result.InputTokens)
	}
	if result.OutputTokens != 50 {
		t.Errorf("expected OutputTokens=50, got %d", result.OutputTokens)
	}
	if result.CacheReadInputTokens != 30 {
		t.Errorf("expected CacheReadInputTokens=30, got %d", result.CacheReadInputTokens)
	}
	if result.ReasoningTokens != 10 {
		t.Errorf("expected ReasoningTokens=10, got %d", result.ReasoningTokens)
	}
}

func TestMapResponsesUsage_NoDetails(t *testing.T) {
	u := &responsesUsage{
		InputTokens:  100,
		OutputTokens: 50,
	}

	result := mapResponsesUsage(u)

	if result.CacheReadInputTokens != 0 {
		t.Errorf("expected CacheReadInputTokens=0, got %d", result.CacheReadInputTokens)
	}
	if result.ReasoningTokens != 0 {
		t.Errorf("expected ReasoningTokens=0, got %d", result.ReasoningTokens)
	}
}

// ---------------------------------------------------------------------------
// convertResponsesResult — verify cache tokens flow through
// ---------------------------------------------------------------------------

func TestConvertResponsesResult_CacheTokens(t *testing.T) {
	resp := &responsesResponse{
		ID:    "resp_cache",
		Model: "grok-3",
		Output: []responsesOutput{
			{
				Type: "message",
				Role: "assistant",
				Content: []responsesContentBlock{
					{Type: "output_text", Text: "Hello!"},
				},
			},
		},
		Usage: &responsesUsage{
			InputTokens:  100,
			OutputTokens: 20,
			InputTokensDetails: &responsesInputTokensDetails{
				CachedTokens: 80,
			},
			OutputTokensDetails: &responsesOutputTokensDetails{
				ReasoningTokens: 5,
			},
		},
		Status: "completed",
	}

	cr := convertResponsesResult(resp)
	if cr.Usage.InputTokens != 20 {
		t.Errorf("InputTokens = %d, want 20", cr.Usage.InputTokens)
	}
	if cr.Usage.CacheReadInputTokens != 80 {
		t.Errorf("CacheReadInputTokens = %d, want 80", cr.Usage.CacheReadInputTokens)
	}
	if cr.Usage.ReasoningTokens != 5 {
		t.Errorf("ReasoningTokens = %d, want 5", cr.Usage.ReasoningTokens)
	}
}

func TestConvertResponsesResult_TextOnly(t *testing.T) {
	resp := &responsesResponse{
		ID:    "resp_123",
		Model: "grok-3",
		Output: []responsesOutput{
			{
				Type: "message",
				Role: "assistant",
				Content: []responsesContentBlock{
					{Type: "output_text", Text: "Hello!"},
				},
			},
		},
		Usage: &responsesUsage{
			InputTokens:  10,
			OutputTokens: 5,
		},
		Status: "completed",
	}

	cr := convertResponsesResult(resp)
	if cr.ID != "resp_123" {
		t.Errorf("ID = %q, want %q", cr.ID, "resp_123")
	}
	if cr.Model != "grok-3" {
		t.Errorf("Model = %q, want %q", cr.Model, "grok-3")
	}
	if cr.StopReason != "stop" {
		t.Errorf("StopReason = %q, want %q", cr.StopReason, "stop")
	}
	if len(cr.Content) != 1 {
		t.Fatalf("expected 1 content block, got %d", len(cr.Content))
	}
	if cr.Content[0].Type != "text" {
		t.Errorf("content[0].type = %q, want %q", cr.Content[0].Type, "text")
	}
	if cr.Content[0].Text != "Hello!" {
		t.Errorf("content[0].text = %q, want %q", cr.Content[0].Text, "Hello!")
	}
	if cr.Usage.InputTokens != 10 || cr.Usage.OutputTokens != 5 {
		t.Errorf("usage = %+v, want in=10 out=5", cr.Usage)
	}
}

func TestConvertResponsesResult_WithFunctionCall(t *testing.T) {
	resp := &responsesResponse{
		ID:    "resp_456",
		Model: "grok-3",
		Output: []responsesOutput{
			{
				Type: "message",
				Role: "assistant",
				Content: []responsesContentBlock{
					{Type: "output_text", Text: "Let me check."},
				},
			},
			{
				Type:      "function_call",
				CallID:    "call_789",
				Name:      "get_weather",
				Arguments: `{"location":"SF"}`,
			},
		},
		Status: "completed",
	}

	cr := convertResponsesResult(resp)
	if cr.StopReason != "tool_calls" {
		t.Errorf("StopReason = %q, want %q", cr.StopReason, "tool_calls")
	}
	if len(cr.Content) != 2 {
		t.Fatalf("expected 2 content blocks, got %d", len(cr.Content))
	}
	if cr.Content[0].Type != "text" || cr.Content[0].Text != "Let me check." {
		t.Errorf("content[0] = %+v, want text block", cr.Content[0])
	}
	if cr.Content[1].Type != "tool_use" {
		t.Errorf("content[1].type = %q, want %q", cr.Content[1].Type, "tool_use")
	}
	if cr.Content[1].ID != "call_789" {
		t.Errorf("content[1].id = %q, want %q", cr.Content[1].ID, "call_789")
	}
	if cr.Content[1].Name != "get_weather" {
		t.Errorf("content[1].name = %q, want %q", cr.Content[1].Name, "get_weather")
	}
}

func TestConvertResponsesResult_IncompleteMaxOutputTokens(t *testing.T) {
	resp := &responsesResponse{
		ID:     "resp_max",
		Model:  "gpt-5.5",
		Status: "incomplete",
		IncompleteDetails: &responsesIncompleteDetails{
			Reason: "max_output_tokens",
		},
		Output: []responsesOutput{
			{
				Type: "message",
				Role: "assistant",
				Content: []responsesContentBlock{
					{Type: "output_text", Text: "Partial"},
				},
			},
		},
	}

	cr := convertResponsesResult(resp)
	if cr.StopReason != "max_tokens" {
		t.Fatalf("StopReason = %q, want max_tokens", cr.StopReason)
	}
	if len(cr.Content) != 1 || cr.Content[0].Text != "Partial" {
		t.Fatalf("Content = %+v, want partial text", cr.Content)
	}
}

func TestParseResponsesSSEStream_TextOnly(t *testing.T) {
	sse := strings.Join([]string{
		`data: {"type":"response.created","response":{"id":"resp_1","model":"grok-3","output":[],"status":"in_progress"}}`,
		`data: {"type":"response.output_item.added","output_index":0,"item":{"type":"message","role":"assistant"}}`,
		`data: {"type":"response.content_part.added","output_index":0,"content_index":0}`,
		`data: {"type":"response.output_text.delta","output_index":0,"content_index":0,"delta":"Hello"}`,
		`data: {"type":"response.output_text.delta","output_index":0,"content_index":0,"delta":" world"}`,
		`data: {"type":"response.output_text.done","output_index":0,"content_index":0}`,
		`data: {"type":"response.completed","response":{"id":"resp_1","model":"grok-3","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Hello world"}]}],"usage":{"input_tokens":5,"output_tokens":2},"status":"completed"}}`,
	}, "\n")

	events := make(chan types.StreamEvent, 100)
	go func() {
		defer close(events)
		parseResponsesSSEStream(strings.NewReader(sse), events)
	}()

	var deltas []string
	var doneResp *types.CompletionResponse
	for ev := range events {
		switch ev.Type {
		case types.StreamEventDelta:
			deltas = append(deltas, ev.Content)
		case types.StreamEventDone:
			doneResp = ev.Response
		}
	}

	if got := strings.Join(deltas, ""); got != "Hello world" {
		t.Errorf("accumulated deltas = %q, want %q", got, "Hello world")
	}
	if doneResp == nil {
		t.Fatal("expected done response")
	}
	if doneResp.StopReason != "stop" {
		t.Errorf("stop reason = %q, want %q", doneResp.StopReason, "stop")
	}
	if doneResp.Usage.InputTokens != 5 || doneResp.Usage.OutputTokens != 2 {
		t.Errorf("usage = %+v, want in=5 out=2", doneResp.Usage)
	}
}

func TestParseResponsesSSEStream_CacheTokens(t *testing.T) {
	sse := strings.Join([]string{
		`data: {"type":"response.output_text.delta","output_index":0,"content_index":0,"delta":"Hi"}`,
		`data: {"type":"response.completed","response":{"id":"resp_c","model":"grok-3","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Hi"}]}],"usage":{"input_tokens":100,"output_tokens":5,"input_tokens_details":{"cached_tokens":80},"output_tokens_details":{"reasoning_tokens":2}},"status":"completed"}}`,
	}, "\n")

	events := make(chan types.StreamEvent, 100)
	go func() {
		defer close(events)
		parseResponsesSSEStream(strings.NewReader(sse), events)
	}()

	var doneResp *types.CompletionResponse
	for ev := range events {
		if ev.Type == types.StreamEventDone {
			doneResp = ev.Response
		}
	}

	if doneResp == nil {
		t.Fatal("expected done response")
	}
	if doneResp.Usage.InputTokens != 20 {
		t.Errorf("InputTokens = %d, want 20", doneResp.Usage.InputTokens)
	}
	if doneResp.Usage.CacheReadInputTokens != 80 {
		t.Errorf("CacheReadInputTokens = %d, want 80", doneResp.Usage.CacheReadInputTokens)
	}
	if doneResp.Usage.ReasoningTokens != 2 {
		t.Errorf("ReasoningTokens = %d, want 2", doneResp.Usage.ReasoningTokens)
	}
}

func TestParseResponsesSSEStream_FunctionCall(t *testing.T) {
	sse := strings.Join([]string{
		`data: {"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","call_id":"call_abc","name":"get_weather","arguments":""}}`,
		`data: {"type":"response.function_call_arguments.delta","output_index":0,"delta":"{\"loc"}`,
		`data: {"type":"response.function_call_arguments.delta","output_index":0,"delta":"ation\":\"SF\"}"}`,
		`data: {"type":"response.function_call_arguments.done","output_index":0}`,
		`data: {"type":"response.completed","response":{"id":"resp_2","model":"grok-3","output":[{"type":"function_call","call_id":"call_abc","name":"get_weather","arguments":"{\"location\":\"SF\"}"}],"usage":{"input_tokens":10,"output_tokens":8},"status":"completed"}}`,
	}, "\n")

	events := make(chan types.StreamEvent, 100)
	go func() {
		defer close(events)
		parseResponsesSSEStream(strings.NewReader(sse), events)
	}()

	var contentDone *types.ContentBlock
	var doneResp *types.CompletionResponse
	for ev := range events {
		switch ev.Type {
		case types.StreamEventContentDone:
			contentDone = ev.ContentBlock
		case types.StreamEventDone:
			doneResp = ev.Response
		}
	}

	if contentDone == nil {
		t.Fatal("expected content_done event with function call")
	}
	if contentDone.Type != "tool_use" {
		t.Errorf("content_done.type = %q, want %q", contentDone.Type, "tool_use")
	}
	if contentDone.ID != "call_abc" {
		t.Errorf("content_done.id = %q, want %q", contentDone.ID, "call_abc")
	}
	if contentDone.Name != "get_weather" {
		t.Errorf("content_done.name = %q, want %q", contentDone.Name, "get_weather")
	}
	if string(contentDone.Input) != `{"location":"SF"}` {
		t.Errorf("content_done.input = %s, want %s", string(contentDone.Input), `{"location":"SF"}`)
	}

	if doneResp == nil {
		t.Fatal("expected done response")
	}
	if doneResp.StopReason != "tool_calls" {
		t.Errorf("stop reason = %q, want %q", doneResp.StopReason, "tool_calls")
	}
}

func TestParseResponsesSSEStream_FunctionCallDoneArguments(t *testing.T) {
	sse := strings.Join([]string{
		`data: {"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","call_id":"call_final","name":"search","arguments":""}}`,
		`data: {"type":"response.function_call_arguments.delta","output_index":0,"delta":"{\"q\":\"part"}`,
		`data: {"type":"response.function_call_arguments.done","output_index":0,"arguments":"{\"q\":\"final\"}"}`,
		`data: {"type":"response.completed","response":{"id":"resp_final","model":"gpt-5.5","output":[{"type":"function_call","call_id":"call_final","name":"search","arguments":"{\"q\":\"final\"}"}],"status":"completed"}}`,
	}, "\n")

	events := make(chan types.StreamEvent, 100)
	go func() {
		defer close(events)
		parseResponsesSSEStream(strings.NewReader(sse), events)
	}()

	var contentDone *types.ContentBlock
	for ev := range events {
		if ev.Type == types.StreamEventContentDone {
			contentDone = ev.ContentBlock
		}
	}

	if contentDone == nil {
		t.Fatal("expected content_done event with function call")
	}
	if string(contentDone.Input) != `{"q":"final"}` {
		t.Fatalf("content_done.input = %s, want final arguments", string(contentDone.Input))
	}
}

func TestParseResponsesSSEStream_ErrorEvent(t *testing.T) {
	sse := `data: {"type":"error","error":{"message":"bad request","type":"invalid_request_error","code":"invalid_value"}}`

	events := make(chan types.StreamEvent, 100)
	go func() {
		defer close(events)
		parseResponsesSSEStream(strings.NewReader(sse), events)
	}()

	var gotError error
	for ev := range events {
		if ev.Type == types.StreamEventError {
			gotError = ev.Error
		}
	}
	if gotError == nil {
		t.Fatal("expected stream error")
	}
	if !strings.Contains(gotError.Error(), "bad request") || !strings.Contains(gotError.Error(), "invalid_value") {
		t.Fatalf("error = %q, want message and code", gotError.Error())
	}
}

// ---------------------------------------------------------------------------
// convertResponsesMessages — error branch coverage
// ---------------------------------------------------------------------------

func TestConvertResponsesMessages_MalformedJSON(t *testing.T) {
	// Falls back to raw string content — no panic.
	messages := []types.Message{
		{Role: "user", Content: json.RawMessage(`{invalid}`)},
	}
	_, input := convertResponsesMessages(messages, "")
	if len(input) != 1 {
		t.Fatalf("expected 1 input, got %d", len(input))
	}
	msg, ok := input[0].(responsesInputMessage)
	if !ok {
		t.Fatalf("expected responsesInputMessage, got %T", input[0])
	}
	if msg.Content != `{invalid}` {
		t.Errorf("content = %v, want raw fallback", msg.Content)
	}
}

func TestConvertResponsesMessages_MalformedAssistantJSONUsesOutputText(t *testing.T) {
	messages := []types.Message{
		{Role: "assistant", Content: json.RawMessage(`{invalid}`)},
	}
	_, input := convertResponsesMessages(messages, "")
	if len(input) != 1 {
		t.Fatalf("expected 1 input, got %d", len(input))
	}
	msg, ok := input[0].(responsesInputMessage)
	if !ok {
		t.Fatalf("expected responsesInputMessage, got %T", input[0])
	}
	parts, ok := msg.Content.([]interface{})
	if !ok || len(parts) != 1 {
		t.Fatalf("assistant fallback content should be one output part, got %#v", msg.Content)
	}
	part, ok := parts[0].(responsesOutputText)
	if !ok {
		t.Fatalf("assistant fallback part should be output_text, got %T %#v", parts[0], parts[0])
	}
	if part.Type != "output_text" || part.Text != `{invalid}` {
		t.Fatalf("assistant fallback part = %+v, want output_text raw content", part)
	}
}

func TestConvertResponsesMessages_EmptyContentArray(t *testing.T) {
	// Empty block array → message with empty text.
	messages := []types.Message{
		{Role: "user", Content: json.RawMessage(`[]`)},
	}
	_, input := convertResponsesMessages(messages, "")
	if len(input) != 1 {
		t.Fatalf("expected 1 input, got %d", len(input))
	}
	msg, ok := input[0].(responsesInputMessage)
	if !ok {
		t.Fatalf("expected responsesInputMessage, got %T", input[0])
	}
	if msg.Content != "" {
		t.Errorf("content = %v, want empty", msg.Content)
	}
}

func TestConvertResponsesMessages_UnknownBlockTypes(t *testing.T) {
	// Unknown block types are silently skipped; only text is joined.
	blocks := []types.ContentBlock{
		{Type: "audio"},
		{Type: "text", Text: "Hello"},
	}
	content, _ := json.Marshal(blocks)
	messages := []types.Message{
		{Role: "user", Content: content},
	}
	_, input := convertResponsesMessages(messages, "")
	if len(input) != 1 {
		t.Fatalf("expected 1 input, got %d", len(input))
	}
	msg, ok := input[0].(responsesInputMessage)
	if !ok {
		t.Fatalf("expected responsesInputMessage, got %T", input[0])
	}
	if msg.Content != "Hello" {
		t.Errorf("content = %v, want %q", msg.Content, "Hello")
	}
}

func TestConvertResponsesMessages_ToolResultWithContentJSON(t *testing.T) {
	// When ContentJSON is set, it takes priority over Content.
	blocks := []types.ContentBlock{
		{Type: "tool_result", ToolUseID: "call_1", Content: "plain text", ContentJSON: json.RawMessage(`{"key":"value"}`)},
	}
	content, _ := json.Marshal(blocks)
	messages := []types.Message{
		{Role: "user", Content: content},
	}
	_, input := convertResponsesMessages(messages, "")
	if len(input) != 1 {
		t.Fatalf("expected 1 input, got %d", len(input))
	}
	fco, ok := input[0].(responsesFunctionCallOutput)
	if !ok {
		t.Fatalf("expected responsesFunctionCallOutput, got %T", input[0])
	}
	if fco.Output != `{"key":"value"}` {
		t.Errorf("output = %q, want %q", fco.Output, `{"key":"value"}`)
	}
}

// ---------------------------------------------------------------------------
// convertResponsesResult — error branch coverage
// ---------------------------------------------------------------------------

func TestConvertResponsesResult_EmptyOutput(t *testing.T) {
	// Empty output array → no content, stop reason "stop".
	resp := &responsesResponse{
		ID:     "resp_empty",
		Model:  "grok-3",
		Output: []responsesOutput{},
		Usage:  &responsesUsage{InputTokens: 5},
	}
	result := convertResponsesResult(resp)
	if len(result.Content) != 0 {
		t.Errorf("expected 0 content blocks, got %d", len(result.Content))
	}
	if result.StopReason != "stop" {
		t.Errorf("stop reason = %q, want %q", result.StopReason, "stop")
	}
}

func TestConvertResponsesResult_UnknownOutputType(t *testing.T) {
	// Unknown output types should be silently skipped.
	resp := &responsesResponse{
		ID:    "resp_unk",
		Model: "grok-3",
		Output: []responsesOutput{
			{Type: "web_search_result", Content: []responsesContentBlock{{Type: "output_text", Text: "search results"}}},
			{Type: "message", Role: "assistant", Content: []responsesContentBlock{{Type: "output_text", Text: "Here"}}},
		},
	}
	result := convertResponsesResult(resp)
	// Only "message" type should be processed
	if len(result.Content) != 1 {
		t.Fatalf("expected 1 content block, got %d", len(result.Content))
	}
	if result.Content[0].Text != "Here" {
		t.Errorf("text = %q, want %q", result.Content[0].Text, "Here")
	}
}

func TestConvertResponsesResult_NilUsage(t *testing.T) {
	resp := &responsesResponse{
		ID:     "resp_nousage",
		Model:  "grok-3",
		Output: []responsesOutput{{Type: "message", Content: []responsesContentBlock{{Type: "output_text", Text: "Hi"}}}},
	}
	result := convertResponsesResult(resp)
	if result.Usage.InputTokens != 0 {
		t.Errorf("expected 0 InputTokens with nil usage, got %d", result.Usage.InputTokens)
	}
}

func TestConvertResponsesResult_EmptyTextSkipped(t *testing.T) {
	// output_text blocks with empty text should be skipped.
	resp := &responsesResponse{
		ID:    "resp_empty_text",
		Model: "grok-3",
		Output: []responsesOutput{
			{Type: "message", Content: []responsesContentBlock{
				{Type: "output_text", Text: ""},
				{Type: "output_text", Text: "Hi"},
			}},
		},
	}
	result := convertResponsesResult(resp)
	if len(result.Content) != 1 {
		t.Fatalf("expected 1 content block (empty skipped), got %d", len(result.Content))
	}
	if result.Content[0].Text != "Hi" {
		t.Errorf("text = %q, want %q", result.Content[0].Text, "Hi")
	}
}

func TestConvertResponsesResult_FunctionCallNoArguments(t *testing.T) {
	// Function call with empty arguments string.
	resp := &responsesResponse{
		ID:    "resp_noargs",
		Model: "grok-3",
		Output: []responsesOutput{
			{Type: "function_call", CallID: "call_1", Name: "no_args"},
		},
	}
	result := convertResponsesResult(resp)
	if len(result.Content) != 1 {
		t.Fatalf("expected 1 content block, got %d", len(result.Content))
	}
	if result.Content[0].Name != "no_args" {
		t.Errorf("name = %q, want %q", result.Content[0].Name, "no_args")
	}
	if result.StopReason != "tool_calls" {
		t.Errorf("stop reason = %q, want %q", result.StopReason, "tool_calls")
	}
}

// ---------------------------------------------------------------------------
// parseResponsesSSEStream — error branch coverage
// ---------------------------------------------------------------------------

func TestParseResponsesSSEStream_MalformedJSON(t *testing.T) {
	// Malformed JSON data lines should be silently skipped.
	sse := strings.Join([]string{
		`data: {"type":"response.output_text.delta","delta":"Hello"}`,
		`data: {invalid json}`,
		`data: {"type":"response.output_text.delta","delta":" world"}`,
		`data: {"type":"response.completed","response":{"id":"r1","model":"grok-3","output":[{"type":"message","content":[{"type":"output_text","text":"Hello world"}]}],"usage":{"input_tokens":5,"output_tokens":2},"status":"completed"}}`,
	}, "\n")

	events := make(chan types.StreamEvent, 100)
	go func() {
		defer close(events)
		parseResponsesSSEStream(strings.NewReader(sse), events)
	}()

	var text string
	for ev := range events {
		if ev.Type == types.StreamEventDelta {
			text += ev.Content
		}
	}
	if text != "Hello world" {
		t.Errorf("text = %q, want %q", text, "Hello world")
	}
}

func TestParseResponsesSSEStream_EmptyStream(t *testing.T) {
	// Empty stream → fallback: start + done with empty response.
	events := make(chan types.StreamEvent, 20)
	go func() {
		defer close(events)
		parseResponsesSSEStream(strings.NewReader(""), events)
	}()

	var gotStart, gotDone bool
	for ev := range events {
		switch ev.Type {
		case types.StreamEventStart:
			gotStart = true
		case types.StreamEventDone:
			gotDone = true
		}
	}
	if !gotStart {
		t.Error("expected start event")
	}
	if !gotDone {
		t.Error("expected done event")
	}
}

func TestParseResponsesSSEStream_NoCompletedEvent(t *testing.T) {
	// Stream without response.completed → uses fallback from accumulated calls.
	sse := strings.Join([]string{
		`data: {"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","call_id":"call_1","name":"search"}}`,
		`data: {"type":"response.function_call_arguments.delta","output_index":0,"delta":"{\"q\":\"test\"}"}`,
		`data: {"type":"response.function_call_arguments.done","output_index":0}`,
	}, "\n")

	events := make(chan types.StreamEvent, 100)
	go func() {
		defer close(events)
		parseResponsesSSEStream(strings.NewReader(sse), events)
	}()

	var doneResp *types.CompletionResponse
	for ev := range events {
		if ev.Type == types.StreamEventDone {
			doneResp = ev.Response
		}
	}
	if doneResp == nil {
		t.Fatal("expected done response from fallback")
	}
	if doneResp.StopReason != "tool_calls" {
		t.Errorf("stop reason = %q, want %q", doneResp.StopReason, "tool_calls")
	}
	if len(doneResp.Content) != 1 {
		t.Fatalf("expected 1 content block, got %d", len(doneResp.Content))
	}
	if doneResp.Content[0].Name != "search" {
		t.Errorf("name = %q, want %q", doneResp.Content[0].Name, "search")
	}
}

func TestParseResponsesSSEStream_ArgsDeltaWithoutPriorItem(t *testing.T) {
	// arguments.delta event for an index not yet registered by output_item.added.
	// Should create a new entry, not panic.
	sse := strings.Join([]string{
		`data: {"type":"response.function_call_arguments.delta","output_index":5,"delta":"{\"x\":1}"}`,
		`data: {"type":"response.function_call_arguments.done","output_index":5}`,
	}, "\n")

	events := make(chan types.StreamEvent, 100)
	go func() {
		defer close(events)
		parseResponsesSSEStream(strings.NewReader(sse), events)
	}()

	var contentDone *types.ContentBlock
	for ev := range events {
		if ev.Type == types.StreamEventContentDone {
			contentDone = ev.ContentBlock
		}
	}
	if contentDone == nil {
		t.Fatal("expected content_done event")
	}
	if string(contentDone.Input) != `{"x":1}` {
		t.Errorf("input = %q, want %q", string(contentDone.Input), `{"x":1}`)
	}
}

func TestParseResponsesSSEStream_ReadError(t *testing.T) {
	pr, pw := io.Pipe()
	go func() {
		pw.Write([]byte("data: {\"type\":\"response.output_text.delta\",\"delta\":\"Hi\"}\n\n"))
		pw.CloseWithError(errors.New("connection reset"))
	}()

	events := make(chan types.StreamEvent, 100)
	go func() {
		defer close(events)
		parseResponsesSSEStream(pr, events)
	}()

	var gotError bool
	for ev := range events {
		if ev.Type == types.StreamEventError {
			gotError = true
			if !strings.Contains(ev.Error.Error(), "connection reset") {
				t.Errorf("error = %q, should contain 'connection reset'", ev.Error.Error())
			}
		}
	}
	if !gotError {
		t.Fatal("expected error event from connection reset")
	}
}

func TestParseResponsesSSEStream_DONESentinel(t *testing.T) {
	// [DONE] sentinel should stop processing and use fallback.
	sse := strings.Join([]string{
		`data: {"type":"response.output_text.delta","delta":"Hi"}`,
		`data: [DONE]`,
		`data: {"type":"response.output_text.delta","delta":" ignored"}`,
	}, "\n")

	events := make(chan types.StreamEvent, 100)
	go func() {
		defer close(events)
		parseResponsesSSEStream(strings.NewReader(sse), events)
	}()

	var text string
	for ev := range events {
		if ev.Type == types.StreamEventDelta {
			text += ev.Content
		}
	}
	// Only "Hi" should be received — " ignored" comes after [DONE]
	if text != "Hi" {
		t.Errorf("text = %q, want %q", text, "Hi")
	}
}

func TestParseResponsesSSEStream_CompletedNilResponse(t *testing.T) {
	// response.completed with nil response field → falls to fallback.
	sse := `data: {"type":"response.completed"}`

	events := make(chan types.StreamEvent, 100)
	go func() {
		defer close(events)
		parseResponsesSSEStream(strings.NewReader(sse), events)
	}()

	var gotDone bool
	for ev := range events {
		if ev.Type == types.StreamEventDone {
			gotDone = true
		}
	}
	if !gotDone {
		t.Error("expected done event from fallback")
	}
}

func TestParseResponsesSSEStream_NonDataLinesIgnored(t *testing.T) {
	sse := `: comment
event: something

data: {"type":"response.output_text.delta","delta":"Hi"}
data: {"type":"response.completed","response":{"id":"r","model":"m","output":[{"type":"message","content":[{"type":"output_text","text":"Hi"}]}],"status":"completed"}}
`
	events := make(chan types.StreamEvent, 100)
	go func() {
		defer close(events)
		parseResponsesSSEStream(strings.NewReader(sse), events)
	}()

	var text string
	for ev := range events {
		if ev.Type == types.StreamEventDelta {
			text += ev.Content
		}
	}
	if text != "Hi" {
		t.Errorf("text = %q, want %q", text, "Hi")
	}
}
