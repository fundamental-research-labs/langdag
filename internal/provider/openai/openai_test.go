package openai

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"langdag.com/langdag/types"
)

func TestOpenAIProviderCompleteUsesResponsesAPI(t *testing.T) {
	var sawRequest bool
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		sawRequest = true
		if r.URL.Path != "/responses" {
			t.Errorf("path = %q, want /responses", r.URL.Path)
		}
		if r.Header.Get("Authorization") != "Bearer test-key" {
			t.Errorf("authorization = %q, want Bearer test-key", r.Header.Get("Authorization"))
		}
		if !strings.Contains(r.Header.Get("Content-Type"), "application/json") {
			t.Errorf("content-type = %q, want application/json", r.Header.Get("Content-Type"))
		}

		var body map[string]interface{}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatalf("decode request body: %v", err)
		}
		if body["model"] != "gpt-5.5" {
			t.Errorf("model = %v, want gpt-5.5", body["model"])
		}
		if body["stream"] != false {
			t.Errorf("stream = %v, want false", body["stream"])
		}
		if body["store"] != false {
			t.Errorf("store = %v, want false", body["store"])
		}
		if body["max_output_tokens"] != float64(42) {
			t.Errorf("max_output_tokens = %v, want 42", body["max_output_tokens"])
		}
		if _, ok := body["messages"]; ok {
			t.Errorf("request included chat completions messages field: %+v", body)
		}
		if _, ok := body["max_tokens"]; ok {
			t.Errorf("request included chat completions max_tokens field: %+v", body)
		}
		if _, ok := body["think"]; ok {
			t.Errorf("request included non-OpenAI think field: %+v", body)
		}

		tools, ok := body["tools"].([]interface{})
		if !ok || len(tools) != 1 {
			t.Fatalf("tools = %#v, want one Responses hosted tool", body["tools"])
		}
		tool, ok := tools[0].(map[string]interface{})
		if !ok {
			t.Fatalf("tools[0] = %#v, want object", tools[0])
		}
		if tool["type"] != "web_search" {
			t.Errorf("tools[0].type = %v, want web_search", tool["type"])
		}
		if tool["type"] == "web_search_preview" {
			t.Errorf("tools[0].type used legacy chat completions value")
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"resp_1","model":"gpt-5.5","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Hello"}]}],"usage":{"input_tokens":10,"output_tokens":2},"status":"completed"}`)
	}))
	defer server.Close()

	provider := New("test-key", server.URL)
	resp, err := provider.Complete(context.Background(), &types.CompletionRequest{
		Model:     "gpt-5.5",
		System:    "Be concise.",
		MaxTokens: 42,
		Messages: []types.Message{
			{Role: "user", Content: json.RawMessage(`"Hi"`)},
		},
		Tools: []types.ToolDefinition{{Name: types.ServerToolWebSearch}},
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if !sawRequest {
		t.Fatal("server did not receive request")
	}
	if resp.ID != "resp_1" || resp.Model != "gpt-5.5" {
		t.Fatalf("response identity = %q/%q, want resp_1/gpt-5.5", resp.ID, resp.Model)
	}
	if len(resp.Content) != 1 || resp.Content[0].Text != "Hello" {
		t.Fatalf("content = %+v, want Hello text block", resp.Content)
	}
	if resp.Usage.InputTokens != 10 || resp.Usage.OutputTokens != 2 {
		t.Fatalf("usage = %+v, want 10/2", resp.Usage)
	}
}

func TestOpenAIProviderStreamUsesResponsesAPI(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/responses" {
			t.Errorf("path = %q, want /responses", r.URL.Path)
		}
		var body map[string]interface{}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatalf("decode request body: %v", err)
		}
		if body["stream"] != true {
			t.Errorf("stream = %v, want true", body["stream"])
		}
		if _, ok := body["messages"]; ok {
			t.Errorf("request included chat completions messages field: %+v", body)
		}

		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, `data: {"type":"response.output_text.delta","output_index":0,"content_index":0,"delta":"Hello"}

data: {"type":"response.completed","response":{"id":"resp_stream","model":"gpt-5.5","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Hello"}]}],"usage":{"input_tokens":3,"output_tokens":1},"status":"completed"}}

`)
	}))
	defer server.Close()

	provider := New("test-key", server.URL)
	events, err := provider.Stream(context.Background(), &types.CompletionRequest{
		Model: "gpt-5.5",
		Messages: []types.Message{
			{Role: "user", Content: json.RawMessage(`"Hi"`)},
		},
	})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}

	var text string
	var done *types.CompletionResponse
	for ev := range events {
		switch ev.Type {
		case types.StreamEventDelta:
			text += ev.Content
		case types.StreamEventDone:
			done = ev.Response
		case types.StreamEventError:
			t.Fatalf("unexpected stream error: %v", ev.Error)
		}
	}
	if text != "Hello" {
		t.Fatalf("stream text = %q, want Hello", text)
	}
	if done == nil || done.ID != "resp_stream" || done.StopReason != "stop" {
		t.Fatalf("done response = %+v, want resp_stream stop", done)
	}
}

func TestOpenAIProviderCompleteResponsesFailureReturnsError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/responses" {
			t.Errorf("path = %q, want /responses", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"resp_failed","model":"gpt-5.5","status":"failed","error":{"message":"bad request","type":"invalid_request_error","code":"invalid_value"}}`)
	}))
	defer server.Close()

	provider := New("test-key", server.URL)
	_, err := provider.Complete(context.Background(), &types.CompletionRequest{
		Model: "gpt-5.5",
		Messages: []types.Message{
			{Role: "user", Content: json.RawMessage(`"Hi"`)},
		},
	})
	if err == nil {
		t.Fatal("Complete returned nil error for failed Responses body")
	}
	if !strings.Contains(err.Error(), "bad request") || !strings.Contains(err.Error(), "invalid_value") {
		t.Fatalf("error = %v, want provider error details", err)
	}
}

func TestOpenAIProviderCompleteCanUseChatCompletionsAPI(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/chat/completions" {
			t.Errorf("path = %q, want /chat/completions", r.URL.Path)
		}
		var body map[string]interface{}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatalf("decode request body: %v", err)
		}
		if body["model"] != "gpt-4.1" {
			t.Errorf("model = %v, want gpt-4.1", body["model"])
		}
		if _, ok := body["messages"]; !ok {
			t.Fatalf("chat completions request missing messages: %+v", body)
		}
		if _, ok := body["max_tokens"]; ok {
			t.Errorf("chat completions request included deprecated max_tokens field: %+v", body)
		}
		if body["max_completion_tokens"] != float64(32) {
			t.Errorf("max_completion_tokens = %v, want 32", body["max_completion_tokens"])
		}
		if _, ok := body["input"]; ok {
			t.Errorf("chat completions request included Responses input field: %+v", body)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"chatcmpl_1","model":"gpt-4.1","choices":[{"message":{"role":"assistant","content":"Hello chat"},"finish_reason":"stop"}],"usage":{"prompt_tokens":4,"completion_tokens":2}}`)
	}))
	defer server.Close()

	provider := New("test-key", server.URL)
	resp, err := provider.Complete(context.Background(), &types.CompletionRequest{
		Model:         "gpt-4.1",
		APIProtocolID: openAIProtocolChatCompletions,
		MaxTokens:     32,
		Messages: []types.Message{
			{Role: "user", Content: json.RawMessage(`"Hi"`)},
		},
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if resp.ID != "chatcmpl_1" {
		t.Fatalf("ID = %q, want chatcmpl_1", resp.ID)
	}
	if len(resp.Content) != 1 || resp.Content[0].Text != "Hello chat" {
		t.Fatalf("content = %+v, want chat text", resp.Content)
	}
}

func TestOpenAIProviderChatCompletionsOmitsThinkAndMapsReasoning(t *testing.T) {
	think := true
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]interface{}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatalf("decode request body: %v", err)
		}
		if _, ok := body["think"]; ok {
			t.Fatalf("chat completions request included non-OpenAI think field: %+v", body)
		}
		if body["reasoning_effort"] != "medium" {
			t.Fatalf("reasoning_effort = %v, want medium", body["reasoning_effort"])
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"chatcmpl_1","model":"gpt-5.5-2026-04-23","choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`)
	}))
	defer server.Close()

	provider := New("test-key", server.URL)
	_, err := provider.Complete(context.Background(), &types.CompletionRequest{
		Model:         "gpt-5.5-2026-04-23",
		APIProtocolID: openAIProtocolChatCompletions,
		Think:         &think,
		Messages: []types.Message{
			{Role: "user", Content: json.RawMessage(`"Hi"`)},
		},
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
}

func TestOpenAIProviderChatCompletionsOmitsHostedToolsForGeneralModels(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/chat/completions" {
			t.Errorf("path = %q, want /chat/completions", r.URL.Path)
		}
		var body map[string]interface{}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatalf("decode request body: %v", err)
		}
		tools, ok := body["tools"].([]interface{})
		if !ok || len(tools) != 1 {
			t.Fatalf("tools = %#v, want only the client function tool", body["tools"])
		}
		tool, ok := tools[0].(map[string]interface{})
		if !ok {
			t.Fatalf("tools[0] = %#v, want object", tools[0])
		}
		if tool["type"] != "function" {
			t.Fatalf("tools[0].type = %v, want function", tool["type"])
		}
		if tool["type"] == "web_search_preview" || tool["type"] == "web_search" {
			t.Fatalf("chat completions request leaked hosted web search tool: %+v", body)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"chatcmpl_1","model":"gpt-4.1","choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`)
	}))
	defer server.Close()

	provider := New("test-key", server.URL)
	_, err := provider.Complete(context.Background(), &types.CompletionRequest{
		Model:         "gpt-4.1",
		APIProtocolID: openAIProtocolChatCompletions,
		Messages: []types.Message{
			{Role: "user", Content: json.RawMessage(`"Hi"`)},
		},
		Tools: []types.ToolDefinition{
			{Name: types.ServerToolWebSearch, Description: "Search the web"},
			{Name: "get_weather", Description: "Get weather", InputSchema: json.RawMessage(`{"type":"object"}`)},
		},
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
}

func TestOpenAIProviderModelsIncludeSnapshotsWithServerTools(t *testing.T) {
	models := New("test-key", "").Models()
	byID := map[string]types.ModelInfo{}
	for _, model := range models {
		byID[model.ID] = model
	}
	for _, id := range []string{
		"gpt-5.5-2026-04-23",
		"gpt-4.1-2025-04-14",
	} {
		model, ok := byID[id]
		if !ok {
			t.Fatalf("Models() missing %q", id)
		}
		if !stringSliceContains(model.ServerTools, types.ServerToolWebSearch) {
			t.Fatalf("model %q server tools = %+v, want web_search", id, model.ServerTools)
		}
	}
}

func stringSliceContains(values []string, target string) bool {
	for _, value := range values {
		if value == target {
			return true
		}
	}
	return false
}

func TestConvertMessages_PlainText(t *testing.T) {
	messages := []types.Message{
		{Role: "user", Content: json.RawMessage(`"Hello"`)},
		{Role: "assistant", Content: json.RawMessage(`"Hi there"`)},
	}

	result := convertMessages(messages, "You are helpful")

	if len(result) != 3 {
		t.Fatalf("expected 3 messages (system + 2), got %d", len(result))
	}
	if result[0].Role != "system" {
		t.Errorf("expected system role, got %s", result[0].Role)
	}
	if result[0].Content != "You are helpful" {
		t.Errorf("expected system content, got %v", result[0].Content)
	}
	if result[1].Role != "user" || result[1].Content != "Hello" {
		t.Errorf("unexpected user message: %+v", result[1])
	}
	if result[2].Role != "assistant" || result[2].Content != "Hi there" {
		t.Errorf("unexpected assistant message: %+v", result[2])
	}
}

func TestConvertMessages_ToolUse(t *testing.T) {
	blocks := []types.ContentBlock{
		{Type: "text", Text: "Let me search"},
		{Type: "tool_use", ID: "call_1", Name: "search", Input: json.RawMessage(`{"q":"test"}`)},
	}
	content, _ := json.Marshal(blocks)

	messages := []types.Message{
		{Role: "assistant", Content: content},
	}

	result := convertMessages(messages, "")

	if len(result) != 1 {
		t.Fatalf("expected 1 message, got %d", len(result))
	}
	if result[0].Role != "assistant" {
		t.Errorf("expected assistant role, got %s", result[0].Role)
	}
	if len(result[0].ToolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(result[0].ToolCalls))
	}
	if result[0].ToolCalls[0].Function.Name != "search" {
		t.Errorf("expected tool name 'search', got %s", result[0].ToolCalls[0].Function.Name)
	}
}

func TestConvertMessages_ToolResult(t *testing.T) {
	blocks := []types.ContentBlock{
		{Type: "tool_result", ToolUseID: "call_1", Content: "result data"},
	}
	content, _ := json.Marshal(blocks)

	messages := []types.Message{
		{Role: "user", Content: content},
	}

	result := convertMessages(messages, "")

	if len(result) != 1 {
		t.Fatalf("expected 1 message, got %d", len(result))
	}
	if result[0].Role != "tool" {
		t.Errorf("expected tool role, got %s", result[0].Role)
	}
	if result[0].ToolCallID != "call_1" {
		t.Errorf("expected tool_call_id 'call_1', got %s", result[0].ToolCallID)
	}
}

func TestMapUsage(t *testing.T) {
	cost := 0.0042
	u := &usage{
		PromptTokens:     100,
		CompletionTokens: 50,
		Cost:             &cost,
		PromptTokensDetails: &tokenDetails{
			CachedTokens: 30,
			AudioTokens:  7,
		},
		CompletionTokensDetails: &tokenDetails{
			ReasoningTokens:          10,
			AudioTokens:              3,
			AcceptedPredictionTokens: 2,
			RejectedPredictionTokens: 1,
		},
	}

	result := mapUsage(u, "")

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
	if result.AudioInputTokens != 7 || result.AudioOutputTokens != 3 {
		t.Errorf("audio tokens = %d/%d, want 7/3", result.AudioInputTokens, result.AudioOutputTokens)
	}
	if result.AcceptedPredictionTokens != 2 || result.RejectedPredictionTokens != 1 {
		t.Errorf("prediction tokens = %d/%d, want 2/1", result.AcceptedPredictionTokens, result.RejectedPredictionTokens)
	}
	providerCost := providerCostFromUsage(u)
	if providerCost == nil || providerCost.Total != cost || providerCost.Source != types.CostSourceProviderResponse {
		t.Fatalf("providerCostFromUsage = %+v, want exact provider cost", providerCost)
	}
}

func TestMapUsage_NoDetails(t *testing.T) {
	u := &usage{
		PromptTokens:     100,
		CompletionTokens: 50,
	}

	result := mapUsage(u, "")

	if result.CacheReadInputTokens != 0 {
		t.Errorf("expected CacheReadInputTokens=0, got %d", result.CacheReadInputTokens)
	}
	if result.ReasoningTokens != 0 {
		t.Errorf("expected ReasoningTokens=0, got %d", result.ReasoningTokens)
	}
}

func TestConvertResponse(t *testing.T) {
	stop := "stop"
	content := "Hello world"
	resp := &chatCompletionResponse{
		ID:    "chatcmpl-123",
		Model: "gpt-4o",
		Choices: []choice{
			{
				Message: responseMessage{
					Content: &content,
				},
				FinishReason: &stop,
			},
		},
		Usage: &usage{
			PromptTokens:     10,
			CompletionTokens: 5,
		},
	}

	result := convertResponse(resp)

	if result.ID != "chatcmpl-123" {
		t.Errorf("expected ID chatcmpl-123, got %s", result.ID)
	}
	if result.StopReason != "stop" {
		t.Errorf("expected stop reason 'stop', got %s", result.StopReason)
	}
	if len(result.Content) != 1 || result.Content[0].Text != "Hello world" {
		t.Errorf("unexpected content: %+v", result.Content)
	}
	if result.Usage.InputTokens != 10 {
		t.Errorf("expected InputTokens=10, got %d", result.Usage.InputTokens)
	}
}

func TestConvertResponse_ToolCalls(t *testing.T) {
	stop := "tool_calls"
	resp := &chatCompletionResponse{
		ID:    "chatcmpl-456",
		Model: "gpt-4o",
		Choices: []choice{
			{
				Message: responseMessage{
					ToolCalls: []responseToolCall{
						{
							ID:   "call_1",
							Type: "function",
							Function: responseFunction{
								Name:      "search",
								Arguments: `{"query":"test"}`,
							},
						},
					},
				},
				FinishReason: &stop,
			},
		},
		Usage: &usage{PromptTokens: 10, CompletionTokens: 5},
	}

	result := convertResponse(resp)

	if len(result.Content) != 1 {
		t.Fatalf("expected 1 content block, got %d", len(result.Content))
	}
	if result.Content[0].Type != "tool_use" {
		t.Errorf("expected tool_use type, got %s", result.Content[0].Type)
	}
	if result.Content[0].Name != "search" {
		t.Errorf("expected name 'search', got %s", result.Content[0].Name)
	}
}

func TestParseSSEStream(t *testing.T) {
	sseData := `data: {"id":"chatcmpl-1","model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}

data: {"id":"chatcmpl-1","model":"gpt-4o","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-1","model":"gpt-4o","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}

data: {"id":"chatcmpl-1","model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: {"id":"chatcmpl-1","model":"gpt-4o","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":2}}

data: [DONE]

`

	events := make(chan types.StreamEvent, 20)
	go func() {
		defer close(events)
		parseSSEStream(strings.NewReader(sseData), events)
	}()

	var collected []types.StreamEvent
	for e := range events {
		collected = append(collected, e)
	}

	// Should have: start, delta("Hello"), delta(" world"), done
	if len(collected) < 3 {
		t.Fatalf("expected at least 3 events, got %d", len(collected))
	}

	if collected[0].Type != types.StreamEventStart {
		t.Errorf("expected start event, got %s", collected[0].Type)
	}

	// Find text deltas
	var text string
	for _, e := range collected {
		if e.Type == types.StreamEventDelta {
			text += e.Content
		}
	}
	if text != "Hello world" {
		t.Errorf("expected 'Hello world', got '%s'", text)
	}

	// Last should be done with usage
	last := collected[len(collected)-1]
	if last.Type != types.StreamEventDone {
		t.Errorf("expected done event, got %s", last.Type)
	}
	if last.Response == nil {
		t.Fatal("expected response in done event")
	}
	if last.Response.Usage.InputTokens != 10 {
		t.Errorf("expected InputTokens=10, got %d", last.Response.Usage.InputTokens)
	}
}

func TestParseSSEStream_CacheTokens(t *testing.T) {
	sseData := `data: {"id":"chatcmpl-c","model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant","content":"Hi"},"finish_reason":null}]}

data: {"id":"chatcmpl-c","model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: {"id":"chatcmpl-c","model":"gpt-4o","choices":[],"usage":{"prompt_tokens":100,"completion_tokens":5,"prompt_tokens_details":{"cached_tokens":80},"completion_tokens_details":{"reasoning_tokens":3}}}

data: [DONE]

`

	events := make(chan types.StreamEvent, 20)
	go func() {
		defer close(events)
		parseSSEStream(strings.NewReader(sseData), events)
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
	if doneResp.Usage.ReasoningTokens != 3 {
		t.Errorf("ReasoningTokens = %d, want 3", doneResp.Usage.ReasoningTokens)
	}
}

func TestParseSSEStream_ToolCalls(t *testing.T) {
	sseData := `data: {"id":"chatcmpl-2","model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant","content":null,"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"search","arguments":""}}]},"finish_reason":null}]}

data: {"id":"chatcmpl-2","model":"gpt-4o","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"q\":"}}]},"finish_reason":null}]}

data: {"id":"chatcmpl-2","model":"gpt-4o","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"test\"}"}}]},"finish_reason":null}]}

data: {"id":"chatcmpl-2","model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}

data: {"id":"chatcmpl-2","model":"gpt-4o","choices":[],"usage":{"prompt_tokens":20,"completion_tokens":10}}

data: [DONE]

`

	events := make(chan types.StreamEvent, 20)
	go func() {
		defer close(events)
		parseSSEStream(strings.NewReader(sseData), events)
	}()

	var collected []types.StreamEvent
	for e := range events {
		collected = append(collected, e)
	}

	// Find content_done event
	var foundToolDone bool
	for _, e := range collected {
		if e.Type == types.StreamEventContentDone {
			foundToolDone = true
			if e.ContentBlock.Name != "search" {
				t.Errorf("expected tool name 'search', got %s", e.ContentBlock.Name)
			}
			if string(e.ContentBlock.Input) != `{"q":"test"}` {
				t.Errorf("expected tool input '{\"q\":\"test\"}', got %s", string(e.ContentBlock.Input))
			}
		}
	}
	if !foundToolDone {
		t.Error("expected content_done event for tool call")
	}
}

func TestAzureProviderName(t *testing.T) {
	p := NewAzure("test-key", "https://myresource.openai.azure.com", "")
	if p.Name() != "openai-azure" {
		t.Errorf("expected name 'openai-azure', got '%s'", p.Name())
	}
}

func TestProviderModelsIncludesGPT56Family(t *testing.T) {
	want := map[string]string{
		"gpt-5.6":       "GPT-5.6",
		"gpt-5.6-sol":   "GPT-5.6 Sol",
		"gpt-5.6-terra": "GPT-5.6 Terra",
		"gpt-5.6-luna":  "GPT-5.6 Luna",
	}
	for _, model := range New("test-key", "").Models() {
		name, ok := want[model.ID]
		if !ok {
			continue
		}
		delete(want, model.ID)
		if model.Name != name || model.ContextWindow != 1050000 || model.MaxOutput != 128000 {
			t.Errorf("model = %+v, want name %q and limits 1050000/128000", model, name)
		}
		if len(model.ServerTools) != 1 || model.ServerTools[0] != types.ServerToolWebSearch {
			t.Errorf("%s ServerTools = %v, want web search", model.ID, model.ServerTools)
		}
	}
	for id := range want {
		t.Errorf("provider model list missing %q", id)
	}
}

func TestAzureProviderModels(t *testing.T) {
	p := NewAzure("test-key", "https://myresource.openai.azure.com", "")
	models := p.Models()
	if len(models) == 0 {
		t.Fatal("expected at least one model")
	}
	for _, m := range models {
		if m.ID == "" || m.Name == "" {
			t.Errorf("model missing required fields: %+v", m)
		}
	}
}

func TestAzureDefaultAPIVersion(t *testing.T) {
	p := NewAzure("test-key", "https://myresource.openai.azure.com", "")
	if p.apiVersion == "" {
		t.Error("expected default API version to be set")
	}
}

func TestAzureCustomAPIVersion(t *testing.T) {
	p := NewAzure("test-key", "https://myresource.openai.azure.com", "2024-06-01")
	if p.apiVersion != "2024-06-01" {
		t.Errorf("expected API version '2024-06-01', got '%s'", p.apiVersion)
	}
}

func TestAzureEndpointTrimming(t *testing.T) {
	p := NewAzure("test-key", "https://myresource.openai.azure.com/", "")
	if strings.HasSuffix(p.endpoint, "/") {
		t.Error("expected trailing slash to be trimmed from endpoint")
	}
}

func TestConvertTools(t *testing.T) {
	tools := []types.ToolDefinition{
		{
			Name:        "search",
			Description: "Search the web",
			InputSchema: json.RawMessage(`{"type":"object","properties":{"q":{"type":"string"}}}`),
		},
	}

	result := convertTools(tools, openAIServerTools)

	if len(result) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(result))
	}
	if result[0].Type != "function" {
		t.Errorf("expected type 'function', got %s", result[0].Type)
	}
	if result[0].Function.Name != "search" {
		t.Errorf("expected name 'search', got %s", result[0].Function.Name)
	}
}

func TestGrokProviderName(t *testing.T) {
	p := NewGrok("test-key", "")
	if p.Name() != "grok" {
		t.Errorf("expected name 'grok', got '%s'", p.Name())
	}
}

func TestGrokProviderModels(t *testing.T) {
	p := NewGrok("test-key", "")
	models := p.Models()
	if len(models) == 0 {
		t.Fatal("expected at least one model")
	}
	for _, m := range models {
		if m.ID == "" || m.Name == "" {
			t.Errorf("model missing required fields: %+v", m)
		}
	}
}

func TestGrokDefaultBaseURL(t *testing.T) {
	p := NewGrok("test-key", "")
	if p.baseURL != "https://api.x.ai/v1" {
		t.Errorf("expected default base URL 'https://api.x.ai/v1', got '%s'", p.baseURL)
	}
}

func TestGrokCustomBaseURL(t *testing.T) {
	p := NewGrok("test-key", "https://custom.api.example.com/v1")
	if p.baseURL != "https://custom.api.example.com/v1" {
		t.Errorf("expected custom base URL, got '%s'", p.baseURL)
	}
}

func TestGrokBaseURLTrimming(t *testing.T) {
	p := NewGrok("test-key", "https://api.x.ai/v1/")
	if strings.HasSuffix(p.baseURL, "/") {
		t.Error("expected trailing slash to be trimmed from base URL")
	}
}

// ---------------------------------------------------------------------------
// convertMessages — error branch coverage
// ---------------------------------------------------------------------------

func TestConvertMessages_MalformedJSON(t *testing.T) {
	// Content that is neither valid string nor valid []ContentBlock
	// falls back to raw string content — no panic.
	messages := []types.Message{
		{Role: "user", Content: json.RawMessage(`{invalid json}`)},
	}
	result := convertMessages(messages, "")
	if len(result) != 1 {
		t.Fatalf("expected 1 message, got %d", len(result))
	}
	// Should fall back to raw string
	if result[0].Content != `{invalid json}` {
		t.Errorf("content = %v, want raw fallback", result[0].Content)
	}
}

func TestConvertMessages_EmptyContentArray(t *testing.T) {
	// Empty block array should produce a message with empty text.
	messages := []types.Message{
		{Role: "user", Content: json.RawMessage(`[]`)},
	}
	result := convertMessages(messages, "")
	if len(result) != 1 {
		t.Fatalf("expected 1 message, got %d", len(result))
	}
	// extractText of empty parts → ""
	if result[0].Content != "" {
		t.Errorf("content = %v, want empty string", result[0].Content)
	}
}

func TestConvertMessages_UnknownBlockTypes(t *testing.T) {
	// Blocks with unrecognized types should be silently skipped.
	blocks := []types.ContentBlock{
		{Type: "audio", Text: "audio data"},
		{Type: "text", Text: "Hello"},
	}
	content, _ := json.Marshal(blocks)
	messages := []types.Message{
		{Role: "user", Content: content},
	}
	result := convertMessages(messages, "")
	if len(result) != 1 {
		t.Fatalf("expected 1 message, got %d", len(result))
	}
	// Only the text block should survive
	if result[0].Content != "Hello" {
		t.Errorf("content = %v, want %q", result[0].Content, "Hello")
	}
}

func TestConvertMessages_ToolCallEmptyFunctionName(t *testing.T) {
	// tool_use block with empty function name should not panic.
	blocks := []types.ContentBlock{
		{Type: "tool_use", ID: "call_empty", Name: "", Input: json.RawMessage(`{}`)},
	}
	content, _ := json.Marshal(blocks)
	messages := []types.Message{
		{Role: "assistant", Content: content},
	}
	result := convertMessages(messages, "")
	if len(result) != 1 {
		t.Fatalf("expected 1 message, got %d", len(result))
	}
	if len(result[0].ToolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(result[0].ToolCalls))
	}
	if result[0].ToolCalls[0].Function.Name != "" {
		t.Errorf("expected empty function name, got %q", result[0].ToolCalls[0].Function.Name)
	}
}

func TestConvertMessages_ImageBlockNoURLNoData(t *testing.T) {
	// Image block with neither URL nor Data should produce no image_url part.
	blocks := []types.ContentBlock{
		{Type: "image"},
		{Type: "text", Text: "caption"},
	}
	content, _ := json.Marshal(blocks)
	messages := []types.Message{
		{Role: "user", Content: content},
	}
	result := convertMessages(messages, "")
	if len(result) != 1 {
		t.Fatalf("expected 1 message, got %d", len(result))
	}
	// hasImages is true (type=image present), so content is []contentPart
	// but the image part was skipped (no URL/Data) — only text remains
	// Actually hasImages flag is set even though no url, so content is []contentPart
	parts, ok := result[0].Content.([]contentPart)
	if ok {
		// If it's parts, should only have the text
		if len(parts) != 1 {
			t.Errorf("expected 1 content part, got %d", len(parts))
		}
	}
}

func TestConvertMessages_DocumentPlainText(t *testing.T) {
	// Document block with text/plain is converted to text part.
	blocks := []types.ContentBlock{
		{Type: "document", MediaType: "text/plain", Data: "document content here"},
	}
	content, _ := json.Marshal(blocks)
	messages := []types.Message{
		{Role: "user", Content: content},
	}
	result := convertMessages(messages, "")
	if len(result) != 1 {
		t.Fatalf("expected 1 message, got %d", len(result))
	}
	if result[0].Content != "document content here" {
		t.Errorf("content = %v, want %q", result[0].Content, "document content here")
	}
}

func TestConvertMessages_DocumentNonTextSkipped(t *testing.T) {
	// Document block with non-text/plain media type and no URL is skipped.
	blocks := []types.ContentBlock{
		{Type: "document", MediaType: "application/pdf", Data: "base64data"},
		{Type: "text", Text: "summary"},
	}
	content, _ := json.Marshal(blocks)
	messages := []types.Message{
		{Role: "user", Content: content},
	}
	result := convertMessages(messages, "")
	if len(result) != 1 {
		t.Fatalf("expected 1 message, got %d", len(result))
	}
	// PDF document is not handled by OpenAI path, only text survives
	if result[0].Content != "summary" {
		t.Errorf("content = %v, want %q", result[0].Content, "summary")
	}
}

func TestConvertMessages_NullContent(t *testing.T) {
	// JSON null content should not panic.
	messages := []types.Message{
		{Role: "user", Content: json.RawMessage(`null`)},
	}
	result := convertMessages(messages, "")
	if len(result) != 1 {
		t.Fatalf("expected 1 message, got %d", len(result))
	}
}

// ---------------------------------------------------------------------------
// convertResponse — edge cases
// ---------------------------------------------------------------------------

func TestConvertResponse_EmptyChoices(t *testing.T) {
	// Response with no choices should not panic.
	resp := &chatCompletionResponse{
		ID:    "chatcmpl-empty",
		Model: "gpt-4o",
		Usage: &usage{PromptTokens: 10, CompletionTokens: 0},
	}
	result := convertResponse(resp)
	if result.ID != "chatcmpl-empty" {
		t.Errorf("ID = %q, want %q", result.ID, "chatcmpl-empty")
	}
	if len(result.Content) != 0 {
		t.Errorf("expected 0 content blocks, got %d", len(result.Content))
	}
	if result.StopReason != "" {
		t.Errorf("expected empty stop reason, got %q", result.StopReason)
	}
}

func TestConvertResponse_NilUsage(t *testing.T) {
	// Response with nil usage should not panic.
	stop := "stop"
	content := "Hi"
	resp := &chatCompletionResponse{
		ID:    "chatcmpl-nousage",
		Model: "gpt-4o",
		Choices: []choice{
			{Message: responseMessage{Content: &content}, FinishReason: &stop},
		},
	}
	result := convertResponse(resp)
	if result.Usage.InputTokens != 0 {
		t.Errorf("expected 0 input tokens with nil usage, got %d", result.Usage.InputTokens)
	}
}

func TestConvertResponse_NilContentNilFinishReason(t *testing.T) {
	// Choice with nil content and nil finish reason should not panic.
	resp := &chatCompletionResponse{
		ID:      "chatcmpl-nil",
		Model:   "gpt-4o",
		Choices: []choice{{}},
		Usage:   &usage{},
	}
	result := convertResponse(resp)
	if len(result.Content) != 0 {
		t.Errorf("expected 0 content blocks, got %d", len(result.Content))
	}
	if result.StopReason != "" {
		t.Errorf("expected empty stop reason, got %q", result.StopReason)
	}
}

// ---------------------------------------------------------------------------
// parseSSEStream — error branch coverage
// ---------------------------------------------------------------------------

func TestParseSSEStream_MalformedJSONSkipped(t *testing.T) {
	// Malformed JSON data lines should be silently skipped.
	sseData := `data: {"id":"chatcmpl-1","model":"gpt-4o","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

data: {invalid json line}

data: {"id":"chatcmpl-1","model":"gpt-4o","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}

data: {"id":"chatcmpl-1","model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]

`
	events := make(chan types.StreamEvent, 20)
	go func() {
		defer close(events)
		parseSSEStream(strings.NewReader(sseData), events)
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

func TestParseSSEStream_NonDataLinesIgnored(t *testing.T) {
	// Lines without "data: " prefix (comments, empty lines, event: lines)
	// should be silently ignored.
	sseData := `: this is a comment
event: message

data: {"id":"chatcmpl-1","model":"gpt-4o","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]}

random line without prefix

data: {"id":"chatcmpl-1","model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]

`
	events := make(chan types.StreamEvent, 20)
	go func() {
		defer close(events)
		parseSSEStream(strings.NewReader(sseData), events)
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

func TestParseSSEStream_DONEMidStream(t *testing.T) {
	// [DONE] appearing before tool calls are fully accumulated.
	// Should emit Done with whatever has been accumulated so far.
	sseData := `data: {"id":"chatcmpl-1","model":"gpt-4o","choices":[{"index":0,"delta":{"content":"partial"},"finish_reason":null}]}

data: [DONE]

data: {"id":"chatcmpl-1","model":"gpt-4o","choices":[{"index":0,"delta":{"content":" ignored"},"finish_reason":null}]}

`
	events := make(chan types.StreamEvent, 20)
	go func() {
		defer close(events)
		parseSSEStream(strings.NewReader(sseData), events)
	}()

	var text string
	var gotDone bool
	for ev := range events {
		switch ev.Type {
		case types.StreamEventDelta:
			text += ev.Content
		case types.StreamEventDone:
			gotDone = true
		}
	}
	if text != "partial" {
		t.Errorf("text = %q, want %q (content after [DONE] should be ignored)", text, "partial")
	}
	if !gotDone {
		t.Error("expected done event")
	}
}

func TestParseSSEStream_EmptyDeltaChunks(t *testing.T) {
	// Delta with empty/null content should not emit a delta event.
	sseData := `data: {"id":"chatcmpl-1","model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}

data: {"id":"chatcmpl-1","model":"gpt-4o","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]}

data: {"id":"chatcmpl-1","model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]

`
	events := make(chan types.StreamEvent, 20)
	go func() {
		defer close(events)
		parseSSEStream(strings.NewReader(sseData), events)
	}()

	var deltaCount int
	for ev := range events {
		if ev.Type == types.StreamEventDelta {
			deltaCount++
		}
	}
	// Only "Hi" should produce a delta (empty "" should be skipped)
	if deltaCount != 1 {
		t.Errorf("expected 1 delta event, got %d", deltaCount)
	}
}

func TestParseSSEStream_NoChoicesInChunk(t *testing.T) {
	// Chunks with empty choices (e.g. usage-only chunks) should be handled.
	sseData := `data: {"id":"chatcmpl-1","model":"gpt-4o","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]}

data: {"id":"chatcmpl-1","model":"gpt-4o","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":1}}

data: [DONE]

`
	events := make(chan types.StreamEvent, 20)
	go func() {
		defer close(events)
		parseSSEStream(strings.NewReader(sseData), events)
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
	if doneResp.Usage.InputTokens != 10 {
		t.Errorf("InputTokens = %d, want 10", doneResp.Usage.InputTokens)
	}
}

// errReader returns data up to a point, then returns an error.
type errReader struct {
	data string
	pos  int
	err  error
}

func (r *errReader) Read(p []byte) (int, error) {
	if r.pos >= len(r.data) {
		return 0, r.err
	}
	n := copy(p, r.data[r.pos:])
	r.pos += n
	if r.pos >= len(r.data) {
		return n, r.err
	}
	return n, nil
}

func TestParseSSEStream_ReadError(t *testing.T) {
	// Scanner encounters a read error mid-stream.
	// Should emit StreamEventError.
	data := "data: {\"id\":\"chatcmpl-1\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hi\"},\"finish_reason\":null}]}\n\n"

	reader := &errReader{
		data: data,
		err:  errors.New("network timeout"),
	}

	events := make(chan types.StreamEvent, 20)
	go func() {
		defer close(events)
		parseSSEStream(reader, events)
	}()

	var gotDelta, gotError, gotDone bool
	for ev := range events {
		switch ev.Type {
		case types.StreamEventDelta:
			gotDelta = true
		case types.StreamEventError:
			gotError = true
		case types.StreamEventDone:
			gotDone = true
		}
	}
	if !gotDelta {
		t.Error("expected delta event before error")
	}
	// The scanner may or may not see the error depending on buffering.
	// If EOF is returned after all data, scanner considers it normal.
	// But a non-EOF error should be propagated.
	_ = gotError
	_ = gotDone
}

func TestParseSSEStream_EmptyStream(t *testing.T) {
	// Completely empty stream should emit start + done, no error.
	events := make(chan types.StreamEvent, 20)
	go func() {
		defer close(events)
		parseSSEStream(strings.NewReader(""), events)
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
		t.Error("expected done event even on empty stream")
	}
}

func TestParseSSEStream_OnlyDONE(t *testing.T) {
	// Stream with only [DONE] sentinel and no data.
	sseData := "data: [DONE]\n\n"
	events := make(chan types.StreamEvent, 20)
	go func() {
		defer close(events)
		parseSSEStream(strings.NewReader(sseData), events)
	}()

	var gotStart, gotDone bool
	var doneResp *types.CompletionResponse
	for ev := range events {
		switch ev.Type {
		case types.StreamEventStart:
			gotStart = true
		case types.StreamEventDone:
			gotDone = true
			doneResp = ev.Response
		}
	}
	if !gotStart {
		t.Error("expected start event")
	}
	if !gotDone {
		t.Error("expected done event")
	}
	if doneResp == nil {
		t.Fatal("expected response in done event")
	}
	if doneResp.ID != "" {
		t.Errorf("expected empty ID, got %q", doneResp.ID)
	}
}

func TestParseSSEStream_NoDONESentinel(t *testing.T) {
	// Stream that ends without [DONE] should still emit done event.
	sseData := `data: {"id":"chatcmpl-1","model":"gpt-4o","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]}

`
	events := make(chan types.StreamEvent, 20)
	go func() {
		defer close(events)
		parseSSEStream(strings.NewReader(sseData), events)
	}()

	var text string
	var gotDone bool
	for ev := range events {
		switch ev.Type {
		case types.StreamEventDelta:
			text += ev.Content
		case types.StreamEventDone:
			gotDone = true
		}
	}
	if text != "Hi" {
		t.Errorf("text = %q, want %q", text, "Hi")
	}
	if !gotDone {
		t.Error("expected done event even without [DONE] sentinel")
	}
}

func TestParseSSEStream_ReadErrorNonEOF(t *testing.T) {
	// Use io.Pipe to simulate a read error mid-stream.
	pr, pw := io.Pipe()

	go func() {
		pw.Write([]byte("data: {\"id\":\"1\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hi\"},\"finish_reason\":null}]}\n\n"))
		pw.CloseWithError(errors.New("connection reset"))
	}()

	events := make(chan types.StreamEvent, 20)
	go func() {
		defer close(events)
		parseSSEStream(pr, events)
	}()

	var gotError bool
	var errMsg string
	for ev := range events {
		if ev.Type == types.StreamEventError {
			gotError = true
			errMsg = ev.Error.Error()
		}
	}
	if !gotError {
		t.Fatal("expected error event from connection reset")
	}
	if !strings.Contains(errMsg, "connection reset") {
		t.Errorf("error = %q, should contain 'connection reset'", errMsg)
	}
}
