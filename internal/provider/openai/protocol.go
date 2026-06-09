// Package openai provides OpenAI-compatible provider implementations.
// This file contains shared request building, response parsing, SSE streaming,
// and message/tool conversion used by all OpenAI-protocol variants (direct, Azure).
package openai

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"strings"

	"langdag.com/langdag/types"
)

// --- Request types ---

type chatCompletionRequest struct {
	Model               string           `json:"model"`
	Messages            []requestMessage `json:"messages"`
	MaxTokens           int              `json:"max_tokens,omitempty"`
	MaxCompletionTokens int              `json:"max_completion_tokens,omitempty"`
	Temperature         *float64         `json:"temperature,omitempty"`
	Stop                []string         `json:"stop,omitempty"`
	Tools               []requestTool    `json:"tools,omitempty"`
	Stream              bool             `json:"stream"`
	StreamOptions       *streamOptions   `json:"stream_options,omitempty"`
	Think               *bool            `json:"think,omitempty"`
	ReasoningEffort     string           `json:"reasoning_effort,omitempty"`
}

type streamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

type chatCompletionRequestOptions struct {
	UseMaxCompletionTokens bool
	IncludeThink           bool
	IncludeReasoningEffort bool
}

type requestMessage struct {
	Role       string            `json:"role"`
	Content    interface{}       `json:"content,omitempty"` // string or []contentPart
	ToolCalls  []requestToolCall `json:"tool_calls,omitempty"`
	ToolCallID string            `json:"tool_call_id,omitempty"`
}

type contentPart struct {
	Type     string    `json:"type"`
	Text     string    `json:"text,omitempty"`
	ImageURL *imageURL `json:"image_url,omitempty"`
}

type imageURL struct {
	URL string `json:"url"`
}

type requestToolCall struct {
	ID       string          `json:"id"`
	Type     string          `json:"type"`
	Function requestFunction `json:"function"`
}

type requestFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type requestTool struct {
	Type     string               `json:"type"`
	Function *requestToolFunction `json:"function,omitempty"`
}

type requestToolFunction struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Parameters  json.RawMessage `json:"parameters"`
}

// --- Response types ---

type chatCompletionResponse struct {
	ID          string   `json:"id"`
	Model       string   `json:"model"`
	Choices     []choice `json:"choices"`
	Usage       *usage   `json:"usage,omitempty"`
	ServiceTier string   `json:"service_tier,omitempty"`
}

type choice struct {
	Index        int             `json:"index"`
	Message      responseMessage `json:"message"`
	Delta        responseMessage `json:"delta"`
	FinishReason *string         `json:"finish_reason,omitempty"`
}

type responseMessage struct {
	Role      string             `json:"role,omitempty"`
	Content   *string            `json:"content,omitempty"`
	ToolCalls []responseToolCall `json:"tool_calls,omitempty"`
}

type responseToolCall struct {
	ID       string           `json:"id"`
	Type     string           `json:"type"`
	Index    int              `json:"index"`
	Function responseFunction `json:"function"`
}

type responseFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type usage struct {
	PromptTokens            int           `json:"prompt_tokens"`
	CompletionTokens        int           `json:"completion_tokens"`
	PromptTokensDetails     *tokenDetails `json:"prompt_tokens_details,omitempty"`
	CompletionTokensDetails *tokenDetails `json:"completion_tokens_details,omitempty"`
	Cost                    *float64      `json:"cost,omitempty"`
}

type tokenDetails struct {
	CachedTokens             int `json:"cached_tokens,omitempty"`
	ReasoningTokens          int `json:"reasoning_tokens,omitempty"`
	AudioTokens              int `json:"audio_tokens,omitempty"`
	AcceptedPredictionTokens int `json:"accepted_prediction_tokens,omitempty"`
	RejectedPredictionTokens int `json:"rejected_prediction_tokens,omitempty"`
}

// Server tool name mappings per OpenAI-protocol variant.
var openAIServerTools = map[string]string{
	types.ServerToolWebSearch: "web_search_preview",
}

// Note: Grok uses the Responses API (responses.go) which supports server-side
// tools natively — no tool name mapping is needed there.

// --- Request building ---

func buildRequest(req *types.CompletionRequest, stream bool, toolMapping map[string]string) []byte {
	return buildChatCompletionRequestWithOptions(req, stream, toolMapping, chatCompletionRequestOptions{IncludeThink: true})
}

func buildOpenAIChatCompletionRequest(req *types.CompletionRequest, stream bool, toolMapping map[string]string) []byte {
	return buildChatCompletionRequestWithOptions(req, stream, toolMapping, chatCompletionRequestOptions{
		UseMaxCompletionTokens: true,
		IncludeReasoningEffort: true,
	})
}

func buildChatCompletionRequestWithOptions(req *types.CompletionRequest, stream bool, toolMapping map[string]string, opts chatCompletionRequestOptions) []byte {
	messages := convertMessages(req.Messages, req.System)

	cr := chatCompletionRequest{
		Model:    req.Model,
		Messages: messages,
		Stream:   stream,
	}

	if req.MaxTokens > 0 {
		if opts.UseMaxCompletionTokens {
			cr.MaxCompletionTokens = req.MaxTokens
		} else {
			cr.MaxTokens = req.MaxTokens
		}
	}
	if req.Temperature > 0 {
		cr.Temperature = &req.Temperature
	}
	if len(req.StopSeqs) > 0 {
		cr.Stop = req.StopSeqs
	}
	if len(req.Tools) > 0 {
		cr.Tools = convertTools(req.Tools, toolMapping)
	}
	if opts.IncludeThink && req.Think != nil {
		cr.Think = req.Think
	}
	if opts.IncludeReasoningEffort {
		cr.ReasoningEffort = openAIResponsesReasoningEffort(req)
	}
	if stream {
		cr.StreamOptions = &streamOptions{IncludeUsage: true}
	}

	body, _ := json.Marshal(cr)
	return body
}

func convertMessages(messages []types.Message, system string) []requestMessage {
	var result []requestMessage

	if system != "" {
		result = append(result, requestMessage{
			Role:    "system",
			Content: system,
		})
	}

	for _, msg := range messages {
		rm := requestMessage{Role: msg.Role}

		var text string
		if err := json.Unmarshal(msg.Content, &text); err == nil {
			rm.Content = text
			result = append(result, rm)
			continue
		}

		var blocks []types.ContentBlock
		if err := json.Unmarshal(msg.Content, &blocks); err != nil {
			rm.Content = string(msg.Content)
			result = append(result, rm)
			continue
		}

		var toolCalls []requestToolCall
		var contentParts []contentPart
		var toolResults []requestMessage
		var hasImages bool

		for _, block := range blocks {
			switch block.Type {
			case "text":
				contentParts = append(contentParts, contentPart{Type: "text", Text: block.Text})
			case "image":
				hasImages = true
				var url string
				if block.URL != "" {
					url = block.URL
				} else if block.Data != "" {
					url = "data:" + block.MediaType + ";base64," + block.Data
				}
				if url != "" {
					contentParts = append(contentParts, contentPart{
						Type:     "image_url",
						ImageURL: &imageURL{URL: url},
					})
				}
			case "document":
				if block.Data != "" && block.MediaType == "text/plain" {
					contentParts = append(contentParts, contentPart{Type: "text", Text: block.Data})
				}
			case "tool_use":
				toolCalls = append(toolCalls, requestToolCall{
					ID:   block.ID,
					Type: "function",
					Function: requestFunction{
						Name:      block.Name,
						Arguments: string(block.Input),
					},
				})
			case "tool_result":
				toolResults = append(toolResults, requestMessage{
					Role:       "tool",
					Content:    block.Content,
					ToolCallID: block.ToolUseID,
				})
			}
		}

		if len(toolCalls) > 0 {
			rm.Role = "assistant"
			if len(contentParts) > 0 {
				rm.Content = extractText(contentParts)
			}
			rm.ToolCalls = toolCalls
			result = append(result, rm)
			result = append(result, toolResults...)
		} else if len(toolResults) > 0 {
			result = append(result, toolResults...)
		} else if hasImages {
			rm.Content = contentParts
			result = append(result, rm)
		} else {
			rm.Content = extractText(contentParts)
			result = append(result, rm)
		}
	}

	return result
}

func extractText(parts []contentPart) string {
	var texts []string
	for _, p := range parts {
		if p.Type == "text" {
			texts = append(texts, p.Text)
		}
	}
	return strings.Join(texts, "\n")
}

func convertTools(tools []types.ToolDefinition, mapping map[string]string) []requestTool {
	result := make([]requestTool, 0, len(tools))
	for _, tool := range tools {
		if tool.IsClientTool() {
			result = append(result, requestTool{
				Type: "function",
				Function: &requestToolFunction{
					Name:        tool.Name,
					Description: tool.Description,
					Parameters:  tool.InputSchema,
				},
			})
			continue
		}

		// Server-side tool: map name if known, otherwise pass through as-is
		if mapping == nil {
			continue
		}
		typeName := tool.Name
		if mapped, ok := mapping[tool.Name]; ok {
			typeName = mapped
		}
		result = append(result, requestTool{Type: typeName})
	}
	return result
}

// --- Response conversion ---

func convertResponse(resp *chatCompletionResponse) *types.CompletionResponse {
	cr := &types.CompletionResponse{
		ID:    resp.ID,
		Model: resp.Model,
	}

	if len(resp.Choices) > 0 {
		c := resp.Choices[0]
		if c.FinishReason != nil {
			cr.StopReason = chatFinishReasonToStopReason(*c.FinishReason)
		}

		if c.Message.Content != nil {
			cr.Content = append(cr.Content, types.ContentBlock{
				Type: "text",
				Text: *c.Message.Content,
			})
		}

		for _, tc := range c.Message.ToolCalls {
			cr.Content = append(cr.Content, types.ContentBlock{
				Type:  "tool_use",
				ID:    tc.ID,
				Name:  tc.Function.Name,
				Input: json.RawMessage(tc.Function.Arguments),
			})
		}
	}

	if resp.Usage != nil {
		cr.Usage = mapUsage(resp.Usage, resp.ServiceTier)
		cr.NormalizedUsage = normalizedUsagePtr(cr.Usage)
		cr.ProviderCost = providerCostFromUsage(resp.Usage)
	}

	return cr
}

func mapUsage(u *usage, serviceTier string) types.Usage {
	cachedTokens := 0
	if u.PromptTokensDetails != nil {
		cachedTokens = u.PromptTokensDetails.CachedTokens
	}
	result := types.Usage{
		InputTokens:  max(0, u.PromptTokens-cachedTokens),
		OutputTokens: u.CompletionTokens,
		ServiceTier:  serviceTier,
	}
	if u.PromptTokensDetails != nil {
		result.CacheReadInputTokens = cachedTokens
		result.AudioInputTokens = u.PromptTokensDetails.AudioTokens
	}
	if u.CompletionTokensDetails != nil {
		result.ReasoningTokens = u.CompletionTokensDetails.ReasoningTokens
		result.AudioOutputTokens = u.CompletionTokensDetails.AudioTokens
		result.AcceptedPredictionTokens = u.CompletionTokensDetails.AcceptedPredictionTokens
		result.RejectedPredictionTokens = u.CompletionTokensDetails.RejectedPredictionTokens
	}
	return result
}

func providerCostFromUsage(u *usage) *types.ProviderCost {
	if u == nil || u.Cost == nil {
		return nil
	}
	raw, _ := json.Marshal(u)
	return &types.ProviderCost{
		Total:    *u.Cost,
		Currency: "USD",
		Source:   types.CostSourceProviderResponse,
		Raw:      raw,
	}
}

func normalizedUsagePtr(usage types.Usage) *types.NormalizedUsage {
	normalized := types.NormalizedUsageFromUsage(usage)
	return &normalized
}

// --- SSE streaming ---

func parseSSEStream(body io.Reader, events chan<- types.StreamEvent) {
	events <- types.StreamEvent{Type: types.StreamEventStart}

	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	var currentToolCalls = map[int]*types.ContentBlock{}
	var finalUsage *types.Usage
	var finalProviderCost *types.ProviderCost
	var responseID, responseModel, serviceTier string
	var finalStopReason string

	for scanner.Scan() {
		line := scanner.Text()

		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")

		if data == "[DONE]" {
			break
		}

		var chunk chatCompletionResponse
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			continue
		}

		if chunk.ID != "" {
			responseID = chunk.ID
		}
		if chunk.Model != "" {
			responseModel = chunk.Model
		}
		if chunk.ServiceTier != "" {
			serviceTier = chunk.ServiceTier
		}

		if chunk.Usage != nil {
			u := mapUsage(chunk.Usage, serviceTier)
			finalUsage = &u
			finalProviderCost = providerCostFromUsage(chunk.Usage)
		}

		if len(chunk.Choices) == 0 {
			continue
		}

		delta := chunk.Choices[0].Delta

		if delta.Content != nil && *delta.Content != "" {
			events <- types.StreamEvent{
				Type:    types.StreamEventDelta,
				Content: *delta.Content,
			}
		}

		for _, tc := range delta.ToolCalls {
			existing, ok := currentToolCalls[tc.Index]
			if !ok {
				existing = &types.ContentBlock{
					Type: "tool_use",
					ID:   tc.ID,
					Name: tc.Function.Name,
				}
				currentToolCalls[tc.Index] = existing
			} else {
				if tc.ID != "" {
					existing.ID = tc.ID
				}
				if tc.Function.Name != "" {
					existing.Name += tc.Function.Name
				}
			}
			if tc.Function.Arguments != "" {
				var prev string
				if existing.Input != nil {
					prev = string(existing.Input)
				}
				existing.Input = json.RawMessage(prev + tc.Function.Arguments)
			}
		}

		if chunk.Choices[0].FinishReason != nil {
			fr := *chunk.Choices[0].FinishReason
			finalStopReason = chatFinishReasonToStopReason(fr)
			if fr == "tool_calls" || fr == "stop" {
				for _, tc := range currentToolCalls {
					events <- types.StreamEvent{
						Type:         types.StreamEventContentDone,
						ContentBlock: tc,
					}
				}
			}
		}
	}

	if err := scanner.Err(); err != nil {
		events <- types.StreamEvent{
			Type:  types.StreamEventError,
			Error: fmt.Errorf("openai: stream read error: %w", err),
		}
		return
	}

	var content []types.ContentBlock
	for _, tc := range currentToolCalls {
		content = append(content, *tc)
	}

	resp := &types.CompletionResponse{
		ID:         responseID,
		Model:      responseModel,
		Content:    content,
		StopReason: finalStopReason,
	}
	if finalUsage != nil {
		resp.Usage = *finalUsage
		resp.NormalizedUsage = normalizedUsagePtr(*finalUsage)
		resp.ProviderCost = finalProviderCost
	}

	events <- types.StreamEvent{
		Type:     types.StreamEventDone,
		Response: resp,
	}
}

func chatFinishReasonToStopReason(reason string) string {
	if reason == "length" {
		return "max_tokens"
	}
	return reason
}
