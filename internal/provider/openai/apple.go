package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"langdag.com/langdag/types"
)

const appleFMDefaultBaseURL = "http://127.0.0.1:1976"

var appleExpectedModels = map[string]string{
	"system": "Apple system",
	"pcc":    "Apple Private Cloud Compute",
}

// AppleProvider implements the provider interface for Apple Foundation Models.
// The local fm serve bridge exposes an OpenAI-compatible chat completions API.
type AppleProvider struct {
	baseURL string
	client  *http.Client
}

// NewApple creates a new no-auth Apple Foundation Models provider.
func NewApple(baseURL string) *AppleProvider {
	if baseURL == "" {
		baseURL = appleFMDefaultBaseURL
	}
	return &AppleProvider{
		baseURL: strings.TrimRight(baseURL, "/"),
		client:  &http.Client{},
	}
}

// Name returns the provider name.
func (p *AppleProvider) Name() string {
	return "apple"
}

type appleHealthResponse struct {
	Status string `json:"status"`
	Models []struct {
		ID        string `json:"id"`
		Name      string `json:"name"`
		Available bool   `json:"available"`
	} `json:"models"`
}

// Models returns expected Apple FM models currently reported available by /health.
func (p *AppleProvider) Models() []types.ModelInfo {
	req, err := http.NewRequestWithContext(context.Background(), "GET", p.baseURL+"/health", nil)
	if err != nil {
		return nil
	}
	resp, err := p.client.Do(req)
	if err != nil {
		return nil
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil
	}
	var health appleHealthResponse
	if err := json.NewDecoder(resp.Body).Decode(&health); err != nil {
		return nil
	}
	if health.Status != "fm serve is running" {
		return nil
	}
	var out []types.ModelInfo
	for _, model := range health.Models {
		modelID := appleHealthModelID(model.ID, model.Name)
		name, expected := appleExpectedModels[modelID]
		if !expected || !model.Available {
			continue
		}
		out = append(out, types.ModelInfo{ID: modelID, Name: name})
	}
	return out
}

func appleHealthModelID(id, name string) string {
	if name != "" {
		return name
	}
	return id
}

// Complete performs a synchronous completion request.
func (p *AppleProvider) Complete(ctx context.Context, req *types.CompletionRequest) (*types.CompletionResponse, error) {
	body := buildChatCompletionRequestWithOptions(appleRequestWithNormalizedToolSchemas(req), false, nil, chatCompletionRequestOptions{UseMaxCompletionTokens: true})
	respBody, err := p.doRequest(ctx, body)
	if err != nil {
		return nil, err
	}
	defer respBody.Close()
	var resp chatCompletionResponse
	if err := json.NewDecoder(respBody).Decode(&resp); err != nil {
		return nil, fmt.Errorf("apple: failed to decode response: %w", err)
	}
	return convertResponse(&resp), nil
}

// Stream performs a streaming completion request.
func (p *AppleProvider) Stream(ctx context.Context, req *types.CompletionRequest) (<-chan types.StreamEvent, error) {
	body := buildChatCompletionRequestWithOptions(appleRequestWithNormalizedToolSchemas(req), true, nil, chatCompletionRequestOptions{UseMaxCompletionTokens: true})
	respBody, err := p.doRequest(ctx, body)
	if err != nil {
		return nil, err
	}
	events := make(chan types.StreamEvent, 100)
	go func() {
		defer close(events)
		defer respBody.Close()
		parseSSEStream(respBody, events)
	}()
	return events, nil
}

func appleRequestWithNormalizedToolSchemas(req *types.CompletionRequest) *types.CompletionRequest {
	if req == nil || len(req.Tools) == 0 {
		return req
	}
	var copied *types.CompletionRequest
	for i, tool := range req.Tools {
		if !tool.IsClientTool() {
			continue
		}
		normalized := appleNormalizeToolSchema(tool.InputSchema)
		if bytes.Equal(normalized, tool.InputSchema) {
			continue
		}
		if copied == nil {
			next := *req
			next.Tools = append([]types.ToolDefinition(nil), req.Tools...)
			copied = &next
		}
		copied.Tools[i].InputSchema = normalized
	}
	if copied != nil {
		return copied
	}
	return req
}

func appleNormalizeToolSchema(raw json.RawMessage) json.RawMessage {
	var schema any
	if err := json.Unmarshal(raw, &schema); err != nil {
		return raw
	}
	if !appleNormalizeSchemaValue(schema) {
		return raw
	}
	normalized, err := json.Marshal(schema)
	if err != nil {
		return raw
	}
	return normalized
}

func appleNormalizeSchemaValue(value any) bool {
	switch v := value.(type) {
	case map[string]any:
		changed := false
		if appleSchemaIsObject(v) {
			if _, ok := v["required"]; !ok {
				v["required"] = []any{}
				changed = true
			}
		}
		if props, ok := v["properties"].(map[string]any); ok {
			for _, child := range props {
				changed = appleNormalizeSchemaValue(child) || changed
			}
		}
		for _, key := range []string{"$defs", "definitions"} {
			if defs, ok := v[key].(map[string]any); ok {
				for _, child := range defs {
					changed = appleNormalizeSchemaValue(child) || changed
				}
			}
		}
		if items, ok := v["items"]; ok {
			changed = appleNormalizeSchemaValue(items) || changed
		}
		for _, key := range []string{"allOf", "anyOf", "oneOf"} {
			if variants, ok := v[key].([]any); ok {
				for _, child := range variants {
					changed = appleNormalizeSchemaValue(child) || changed
				}
			}
		}
		return changed
	case []any:
		changed := false
		for _, child := range v {
			changed = appleNormalizeSchemaValue(child) || changed
		}
		return changed
	default:
		return false
	}
}

func appleSchemaIsObject(schema map[string]any) bool {
	if typ, ok := schema["type"].(string); ok && typ == "object" {
		return true
	}
	_, ok := schema["properties"]
	return ok
}

func (p *AppleProvider) doRequest(ctx context.Context, body []byte) (io.ReadCloser, error) {
	httpReq, err := http.NewRequestWithContext(ctx, "POST", p.baseURL+"/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("apple: failed to create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	resp, err := p.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("apple: request failed: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		defer resp.Body.Close()
		bodyBytes, _ := io.ReadAll(io.LimitReader(resp.Body, maxErrorBodySize))
		return nil, fmt.Errorf("apple: API error (status %d): %s", resp.StatusCode, strings.TrimSpace(string(bodyBytes)))
	}
	return resp.Body, nil
}
