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

const (
	openAIProtocolChatCompletions = "openai-chat-completions"
	openAIProtocolResponses       = "openai-responses"
)

// Provider implements the provider interface for OpenAI's public API.
type Provider struct {
	apiKey  string
	baseURL string
	client  *http.Client
}

// New creates a new OpenAI provider.
func New(apiKey, baseURL string) *Provider {
	if baseURL == "" {
		baseURL = "https://api.openai.com/v1"
	}
	baseURL = strings.TrimRight(baseURL, "/")
	return &Provider{
		apiKey:  apiKey,
		baseURL: baseURL,
		client:  &http.Client{},
	}
}

// Name returns the provider name.
func (p *Provider) Name() string {
	return "openai"
}

// Models returns the available models.
func (p *Provider) Models() []types.ModelInfo {
	st := []string{types.ServerToolWebSearch}
	return []types.ModelInfo{
		{ID: "gpt-5.6", Name: "GPT-5.6", ContextWindow: 1050000, MaxOutput: 128000, ServerTools: st},
		{ID: "gpt-5.6-sol", Name: "GPT-5.6 Sol", ContextWindow: 1050000, MaxOutput: 128000, ServerTools: st},
		{ID: "gpt-5.6-terra", Name: "GPT-5.6 Terra", ContextWindow: 1050000, MaxOutput: 128000, ServerTools: st},
		{ID: "gpt-5.6-luna", Name: "GPT-5.6 Luna", ContextWindow: 1050000, MaxOutput: 128000, ServerTools: st},
		{ID: "gpt-5.5", Name: "GPT-5.5", ContextWindow: 1050000, MaxOutput: 128000, ServerTools: st},
		{ID: "gpt-5.5-2026-04-23", Name: "GPT-5.5", ContextWindow: 1050000, MaxOutput: 128000, ServerTools: st},
		{ID: "gpt-5.5-pro", Name: "GPT-5.5 Pro", ContextWindow: 1050000, MaxOutput: 128000, ServerTools: st},
		{ID: "gpt-5.5-pro-2026-04-23", Name: "GPT-5.5 Pro", ContextWindow: 1050000, MaxOutput: 128000, ServerTools: st},
		{ID: "gpt-5.4", Name: "GPT-5.4", ContextWindow: 1050000, MaxOutput: 128000, ServerTools: st},
		{ID: "gpt-5.4-2026-03-05", Name: "GPT-5.4", ContextWindow: 1050000, MaxOutput: 128000, ServerTools: st},
		{ID: "gpt-5.4-mini", Name: "GPT-5.4 Mini", ContextWindow: 400000, MaxOutput: 128000, ServerTools: st},
		{ID: "gpt-5.4-mini-2026-03-17", Name: "GPT-5.4 Mini", ContextWindow: 400000, MaxOutput: 128000, ServerTools: st},
		{ID: "gpt-5.4-nano", Name: "GPT-5.4 Nano", ContextWindow: 400000, MaxOutput: 128000, ServerTools: st},
		{ID: "gpt-5.4-nano-2026-03-17", Name: "GPT-5.4 Nano", ContextWindow: 400000, MaxOutput: 128000, ServerTools: st},
		{ID: "gpt-5", Name: "GPT-5", ContextWindow: 400000, MaxOutput: 128000, ServerTools: st},
		{ID: "gpt-5-2025-08-07", Name: "GPT-5", ContextWindow: 400000, MaxOutput: 128000, ServerTools: st},
		{ID: "gpt-5-mini", Name: "GPT-5 Mini", ContextWindow: 400000, MaxOutput: 128000, ServerTools: st},
		{ID: "gpt-5-mini-2025-08-07", Name: "GPT-5 Mini", ContextWindow: 400000, MaxOutput: 128000, ServerTools: st},
		{ID: "gpt-5-nano", Name: "GPT-5 Nano", ContextWindow: 400000, MaxOutput: 128000, ServerTools: st},
		{ID: "gpt-5-nano-2025-08-07", Name: "GPT-5 Nano", ContextWindow: 400000, MaxOutput: 128000, ServerTools: st},
		{ID: "gpt-4.1", Name: "GPT-4.1", ContextWindow: 1047576, MaxOutput: 32768, ServerTools: st},
		{ID: "gpt-4.1-2025-04-14", Name: "GPT-4.1", ContextWindow: 1047576, MaxOutput: 32768, ServerTools: st},
		{ID: "gpt-4.1-mini", Name: "GPT-4.1 Mini", ContextWindow: 1047576, MaxOutput: 32768, ServerTools: st},
		{ID: "gpt-4.1-mini-2025-04-14", Name: "GPT-4.1 Mini", ContextWindow: 1047576, MaxOutput: 32768, ServerTools: st},
		{ID: "gpt-4.1-nano", Name: "GPT-4.1 Nano", ContextWindow: 1047576, MaxOutput: 32768, ServerTools: st},
		{ID: "gpt-4.1-nano-2025-04-14", Name: "GPT-4.1 Nano", ContextWindow: 1047576, MaxOutput: 32768, ServerTools: st},
		{ID: "gpt-4o", Name: "GPT-4o", ContextWindow: 128000, MaxOutput: 16384, ServerTools: st},
		{ID: "gpt-4o-mini", Name: "GPT-4o Mini", ContextWindow: 128000, MaxOutput: 16384, ServerTools: st},
		{ID: "o3", Name: "o3", ContextWindow: 200000, MaxOutput: 100000, ServerTools: st},
		{ID: "o3-2025-04-16", Name: "o3", ContextWindow: 200000, MaxOutput: 100000, ServerTools: st},
		{ID: "o3-pro", Name: "o3-pro", ContextWindow: 200000, MaxOutput: 100000, ServerTools: st},
		{ID: "o3-pro-2025-06-10", Name: "o3-pro", ContextWindow: 200000, MaxOutput: 100000, ServerTools: st},
		{ID: "o3-mini", Name: "o3-mini", ContextWindow: 200000, MaxOutput: 100000, ServerTools: st},
		{ID: "o3-mini-2025-01-31", Name: "o3-mini", ContextWindow: 200000, MaxOutput: 100000, ServerTools: st},
		{ID: "o4-mini", Name: "o4-mini", ContextWindow: 200000, MaxOutput: 100000, ServerTools: st},
		{ID: "o4-mini-2025-04-16", Name: "o4-mini", ContextWindow: 200000, MaxOutput: 100000, ServerTools: st},
	}
}

// Complete performs a synchronous completion request.
func (p *Provider) Complete(ctx context.Context, req *types.CompletionRequest) (*types.CompletionResponse, error) {
	switch openAIProtocolForRequest(req) {
	case openAIProtocolChatCompletions:
		return p.completeChatCompletions(ctx, req)
	case openAIProtocolResponses:
		return p.completeResponses(ctx, req)
	default:
		return nil, fmt.Errorf("openai: unsupported api_protocol_id %q", req.APIProtocolID)
	}
}

func (p *Provider) completeResponses(ctx context.Context, req *types.CompletionRequest) (*types.CompletionResponse, error) {
	body := buildOpenAIResponsesRequest(req, false)

	respBody, err := p.doRequest(ctx, "/responses", body)
	if err != nil {
		return nil, err
	}
	defer respBody.Close()

	var resp responsesResponse
	if err := json.NewDecoder(respBody).Decode(&resp); err != nil {
		return nil, fmt.Errorf("openai: failed to decode response: %w", err)
	}
	if resp.Status == "failed" {
		return nil, responsesAPIError("openai: responses request failed", resp.Error)
	}

	return convertResponsesResult(&resp), nil
}

func (p *Provider) completeChatCompletions(ctx context.Context, req *types.CompletionRequest) (*types.CompletionResponse, error) {
	body := buildOpenAIChatCompletionRequest(req, false, openAIChatCompletionsServerTools(req))

	respBody, err := p.doRequest(ctx, "/chat/completions", body)
	if err != nil {
		return nil, err
	}
	defer respBody.Close()

	var resp chatCompletionResponse
	if err := json.NewDecoder(respBody).Decode(&resp); err != nil {
		return nil, fmt.Errorf("openai: failed to decode response: %w", err)
	}

	return convertResponse(&resp), nil
}

// Stream performs a streaming completion request.
func (p *Provider) Stream(ctx context.Context, req *types.CompletionRequest) (<-chan types.StreamEvent, error) {
	switch openAIProtocolForRequest(req) {
	case openAIProtocolChatCompletions:
		return p.streamChatCompletions(ctx, req)
	case openAIProtocolResponses:
		return p.streamResponses(ctx, req)
	default:
		return nil, fmt.Errorf("openai: unsupported api_protocol_id %q", req.APIProtocolID)
	}
}

func (p *Provider) streamResponses(ctx context.Context, req *types.CompletionRequest) (<-chan types.StreamEvent, error) {
	body := buildOpenAIResponsesRequest(req, true)

	respBody, err := p.doRequest(ctx, "/responses", body)
	if err != nil {
		return nil, err
	}

	events := make(chan types.StreamEvent, 100)
	go func() {
		defer close(events)
		defer respBody.Close()
		parseResponsesSSEStream(respBody, events)
	}()

	return events, nil
}

func (p *Provider) streamChatCompletions(ctx context.Context, req *types.CompletionRequest) (<-chan types.StreamEvent, error) {
	body := buildOpenAIChatCompletionRequest(req, true, openAIChatCompletionsServerTools(req))

	respBody, err := p.doRequest(ctx, "/chat/completions", body)
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

func openAIProtocolForRequest(req *types.CompletionRequest) string {
	if req != nil && req.APIProtocolID != "" {
		return req.APIProtocolID
	}
	return openAIProtocolResponses
}

func openAIChatCompletionsServerTools(req *types.CompletionRequest) map[string]string {
	if req == nil || !supportsOpenAIChatCompletionsHostedWebSearch(req.Model) {
		return nil
	}
	return openAIServerTools
}

func supportsOpenAIChatCompletionsHostedWebSearch(model string) bool {
	model = strings.ToLower(model)
	return strings.Contains(model, "search-preview") || strings.Contains(model, "search-api")
}

func (p *Provider) doRequest(ctx context.Context, path string, body []byte) (io.ReadCloser, error) {
	httpReq, err := http.NewRequestWithContext(ctx, "POST", p.baseURL+path, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("openai: failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+p.apiKey)

	resp, err := p.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("openai: request failed: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		defer resp.Body.Close()
		bodyBytes, _ := io.ReadAll(io.LimitReader(resp.Body, maxErrorBodySize))
		return nil, fmt.Errorf("openai: API error (status %d): %s", resp.StatusCode, strings.TrimSpace(string(bodyBytes)))
	}

	return resp.Body, nil
}
