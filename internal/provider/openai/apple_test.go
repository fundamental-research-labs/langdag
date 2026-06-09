package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"langdag.com/langdag/types"
)

func TestAppleModelsFiltersExpectedAvailableModels(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/health" {
			t.Fatalf("path = %q, want /health", r.URL.Path)
		}
		_, _ = w.Write([]byte(`{"status":"fm serve is running","models":[{"name":"system","available":true},{"name":"pcc","available":false},{"name":"other","available":true}]}`))
	}))
	defer srv.Close()

	models := NewApple(srv.URL).Models()
	if len(models) != 1 || models[0].ID != "system" {
		t.Fatalf("models = %+v, want only system", models)
	}
}

func TestAppleModelsSupportsLegacyIDField(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"status":"fm serve is running","models":[{"id":"pcc","available":true}]}`))
	}))
	defer srv.Close()

	models := NewApple(srv.URL).Models()
	if len(models) != 1 || models[0].ID != "pcc" {
		t.Fatalf("models = %+v, want pcc from legacy id field", models)
	}
}

func TestAppleCompleteRequestShapeAndConversion(t *testing.T) {
	var got map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Fatalf("path = %q, want /v1/chat/completions", r.URL.Path)
		}
		if auth := r.Header.Get("Authorization"); auth != "" {
			t.Fatalf("Authorization header = %q, want empty", auth)
		}
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			t.Fatal(err)
		}
		_, _ = w.Write([]byte(`{"id":"1","model":"system","choices":[{"message":{"role":"assistant","content":"hello"},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":3}}`))
	}))
	defer srv.Close()

	resp, err := NewApple(srv.URL).Complete(context.Background(), &types.CompletionRequest{
		Model:       "system",
		System:      "be brief",
		Messages:    []types.Message{{Role: "user", Content: json.RawMessage(`"hi"`)}},
		MaxTokens:   17,
		Temperature: 0.2,
		Think:       boolPtr(true),
	})
	if err != nil {
		t.Fatal(err)
	}
	if got["stream"] != false {
		t.Fatalf("stream = %#v, want false", got["stream"])
	}
	if _, ok := got["max_tokens"]; ok {
		t.Fatalf("max_tokens should be omitted: %#v", got)
	}
	if got["max_completion_tokens"].(float64) != 17 {
		t.Fatalf("max_completion_tokens = %#v, want 17", got["max_completion_tokens"])
	}
	if _, ok := got["think"]; ok {
		t.Fatalf("think should be omitted: %#v", got)
	}
	if _, ok := got["reasoning_effort"]; ok {
		t.Fatalf("reasoning_effort should be omitted: %#v", got)
	}
	if len(resp.Content) != 1 || resp.Content[0].Text != "hello" || resp.Usage.InputTokens != 2 || resp.Usage.OutputTokens != 3 {
		t.Fatalf("response = %+v", resp)
	}
}

func TestAppleCompleteNormalizesMissingRequiredInToolSchema(t *testing.T) {
	var got map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			t.Fatal(err)
		}
		_, _ = w.Write([]byte(`{"id":"1","model":"pcc","choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`))
	}))
	defer srv.Close()

	req := &types.CompletionRequest{
		Model:    "pcc",
		Messages: []types.Message{{Role: "user", Content: json.RawMessage(`"hi"`)}},
		Tools: []types.ToolDefinition{{
			Name:        "outline",
			Description: "Outline files",
			InputSchema: json.RawMessage(`{
				"type": "object",
				"properties": {
					"file_path": {"type": "string"},
					"nested": {
						"type": "object",
						"properties": {"name": {"type": "string"}}
					}
				}
			}`),
		}},
	}

	if _, err := NewApple(srv.URL).Complete(context.Background(), req); err != nil {
		t.Fatal(err)
	}

	tools, ok := got["tools"].([]any)
	if !ok || len(tools) != 1 {
		t.Fatalf("tools = %#v, want one tool", got["tools"])
	}
	tool := tools[0].(map[string]any)
	fn := tool["function"].(map[string]any)
	params := fn["parameters"].(map[string]any)
	required, ok := params["required"].([]any)
	if !ok || len(required) != 0 {
		t.Fatalf("parameters.required = %#v, want empty array", params["required"])
	}
	nested := params["properties"].(map[string]any)["nested"].(map[string]any)
	nestedRequired, ok := nested["required"].([]any)
	if !ok || len(nestedRequired) != 0 {
		t.Fatalf("nested.required = %#v, want empty array", nested["required"])
	}
	if bytes.Contains(req.Tools[0].InputSchema, []byte(`"required"`)) {
		t.Fatalf("original request schema was mutated: %s", req.Tools[0].InputSchema)
	}
}

func TestAppleStreamParsesSSE(t *testing.T) {
	var got map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			t.Fatal(err)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n"))
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer srv.Close()

	ch, err := NewApple(srv.URL).Stream(context.Background(), &types.CompletionRequest{Model: "pcc", MaxTokens: 9})
	if err != nil {
		t.Fatal(err)
	}
	var text string
	for event := range ch {
		if event.Type == types.StreamEventDelta {
			text += event.Content
		}
	}
	if got["stream"] != true || got["max_completion_tokens"].(float64) != 9 {
		t.Fatalf("request = %#v", got)
	}
	if text != "hi" {
		t.Fatalf("stream text = %q, want hi", text)
	}
}

func boolPtr(v bool) *bool {
	return &v
}
