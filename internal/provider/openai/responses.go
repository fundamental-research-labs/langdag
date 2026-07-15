// Package openai — Responses API support for OpenAI-compatible providers.
//
// This file contains request building, response parsing, and SSE streaming for
// the Responses API (/v1/responses). The Responses API is the modern OpenAI
// endpoint and is also used by xAI/Grok for server-side tools such as
// web_search, x_search, and code_interpreter alongside function calling.
package openai

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"strings"

	"langdag.com/langdag/types"
)

// --- Responses API request types ---

type responsesRequest struct {
	Model           string              `json:"model"`
	Input           []interface{}       `json:"input"`
	Instructions    string              `json:"instructions,omitempty"`
	Tools           []interface{}       `json:"tools,omitempty"`
	MaxOutputTokens int                 `json:"max_output_tokens,omitempty"`
	Temperature     *float64            `json:"temperature,omitempty"`
	Reasoning       *responsesReasoning `json:"reasoning,omitempty"`
	Stream          bool                `json:"stream"`
	Store           bool                `json:"store"`
}

type responsesReasoning struct {
	Effort string `json:"effort,omitempty"`
}

type responsesRequestOptions struct {
	includeReasoning bool
}

// Input item types for the Responses API.

type responsesInputMessage struct {
	Type    string      `json:"type"`
	Role    string      `json:"role"`
	Content interface{} `json:"content"` // string or []responsesContentPart
}

type responsesInputText struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type responsesInputImage struct {
	Type     string `json:"type"`
	ImageURL string `json:"image_url"`
}

type responsesOutputText struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type responsesFunctionCallInput struct {
	Type      string `json:"type"`
	CallID    string `json:"call_id"`
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
	Status    string `json:"status,omitempty"`
}

type responsesFunctionCallOutput struct {
	Type   string `json:"type"`
	CallID string `json:"call_id"`
	Output string `json:"output"`
}

// Tool types for the Responses API.

type responsesFunctionTool struct {
	Type        string          `json:"type"`
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters"`
}

type responsesServerTool struct {
	Type string `json:"type"`
}

// --- Responses API response types ---

type responsesResponse struct {
	ID                string                      `json:"id"`
	Model             string                      `json:"model"`
	Output            []responsesOutput           `json:"output"`
	Usage             *responsesUsage             `json:"usage,omitempty"`
	Status            string                      `json:"status"`
	IncompleteDetails *responsesIncompleteDetails `json:"incomplete_details,omitempty"`
	Error             *responsesError             `json:"error,omitempty"`
	ServiceTier       string                      `json:"service_tier,omitempty"`
}

type responsesIncompleteDetails struct {
	Reason string `json:"reason,omitempty"`
}

type responsesError struct {
	Message string `json:"message,omitempty"`
	Type    string `json:"type,omitempty"`
	Code    string `json:"code,omitempty"`
}

type responsesOutput struct {
	Type string `json:"type"`

	// For "message" type
	ID      string                  `json:"id,omitempty"`
	Role    string                  `json:"role,omitempty"`
	Content []responsesContentBlock `json:"content,omitempty"`
	Status  string                  `json:"status,omitempty"`

	// For "function_call" type
	CallID    string `json:"call_id,omitempty"`
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
	Input     string `json:"input,omitempty"`
}

type responsesContentBlock struct {
	Type string `json:"type"`
	Text string `json:"text,omitempty"`
}

type responsesUsage struct {
	InputTokens         int                           `json:"input_tokens"`
	OutputTokens        int                           `json:"output_tokens"`
	InputTokensDetails  *responsesInputTokensDetails  `json:"input_tokens_details,omitempty"`
	OutputTokensDetails *responsesOutputTokensDetails `json:"output_tokens_details,omitempty"`
	Cost                *float64                      `json:"cost,omitempty"`
}

type responsesInputTokensDetails struct {
	CachedTokens int `json:"cached_tokens,omitempty"`
	AudioTokens  int `json:"audio_tokens,omitempty"`
	ImageTokens  int `json:"image_tokens,omitempty"`
}

type responsesOutputTokensDetails struct {
	ReasoningTokens          int `json:"reasoning_tokens,omitempty"`
	AudioTokens              int `json:"audio_tokens,omitempty"`
	ImageTokens              int `json:"image_tokens,omitempty"`
	AcceptedPredictionTokens int `json:"accepted_prediction_tokens,omitempty"`
	RejectedPredictionTokens int `json:"rejected_prediction_tokens,omitempty"`
}

// --- Responses API streaming event ---

type responsesStreamEvent struct {
	Type string `json:"type"`

	// For response.output_text.delta
	Delta string `json:"delta,omitempty"`

	// For response.function_call_arguments.done and custom tool events
	Arguments string `json:"arguments,omitempty"`
	Input     string `json:"input,omitempty"`
	Name      string `json:"name,omitempty"`

	// For response.output_item.added
	Item *responsesOutput `json:"item,omitempty"`

	// For response.completed
	Response *responsesResponse `json:"response,omitempty"`

	// For error and response.failed
	Error *responsesError `json:"error,omitempty"`

	// Common fields
	OutputIndex  int    `json:"output_index"`
	ContentIndex int    `json:"content_index,omitempty"`
	ItemID       string `json:"item_id,omitempty"`
}

// --- Request building ---

func buildResponsesRequest(req *types.CompletionRequest, stream bool) []byte {
	return buildResponsesRequestWithOptions(req, stream, responsesRequestOptions{})
}

func buildOpenAIResponsesRequest(req *types.CompletionRequest, stream bool) []byte {
	return buildResponsesRequestWithOptions(req, stream, responsesRequestOptions{includeReasoning: true})
}

func buildResponsesRequestWithOptions(req *types.CompletionRequest, stream bool, opts responsesRequestOptions) []byte {
	instructions, input := convertResponsesMessages(req.Messages, req.System)

	rr := responsesRequest{
		Model:        req.Model,
		Input:        input,
		Instructions: instructions,
		Stream:       stream,
		Store:        false,
	}

	if req.MaxTokens > 0 {
		rr.MaxOutputTokens = req.MaxTokens
	}
	if req.Temperature > 0 {
		rr.Temperature = &req.Temperature
	}
	if len(req.Tools) > 0 {
		rr.Tools = convertResponsesTools(req.Tools)
	}
	if opts.includeReasoning {
		if effort := openAIResponsesReasoningEffort(req); effort != "" {
			rr.Reasoning = &responsesReasoning{Effort: effort}
		}
	}

	body, _ := json.Marshal(rr)
	return body
}

func openAIResponsesReasoningEffort(req *types.CompletionRequest) string {
	if req == nil || req.Think == nil {
		return ""
	}
	model := strings.ToLower(req.Model)
	if *req.Think {
		if isOpenAIResponsesReasoningModel(model) {
			return "medium"
		}
		return ""
	}
	if supportsOpenAIReasoningNone(model) {
		return "none"
	}
	return ""
}

func isOpenAIResponsesReasoningModel(model string) bool {
	return strings.HasPrefix(model, "gpt-5") || strings.HasPrefix(model, "o")
}

func supportsOpenAIReasoningNone(model string) bool {
	return strings.HasPrefix(model, "gpt-5.1") ||
		strings.HasPrefix(model, "gpt-5.2") ||
		strings.HasPrefix(model, "gpt-5.3") ||
		strings.HasPrefix(model, "gpt-5.4") ||
		strings.HasPrefix(model, "gpt-5.5") ||
		strings.HasPrefix(model, "gpt-5.6")
}

// convertResponsesMessages converts langdag messages to Responses API input items.
// The system prompt is returned separately as the "instructions" field.
func convertResponsesMessages(messages []types.Message, system string) (string, []interface{}) {
	var input []interface{}

	for _, msg := range messages {
		// Try plain text first
		var text string
		if err := json.Unmarshal(msg.Content, &text); err == nil {
			input = append(input, responsesInputMessage{
				Type:    "message",
				Role:    msg.Role,
				Content: responsesMessageTextContent(msg.Role, text),
			})
			continue
		}

		// Parse as content blocks
		var blocks []types.ContentBlock
		if err := json.Unmarshal(msg.Content, &blocks); err != nil {
			// Fallback: treat raw content as text
			input = append(input, responsesInputMessage{
				Type:    "message",
				Role:    msg.Role,
				Content: responsesMessageTextContent(msg.Role, string(msg.Content)),
			})
			continue
		}

		// Separate blocks by type
		var textParts []interface{}
		var toolCalls []responsesFunctionCallInput
		var toolResults []responsesFunctionCallOutput
		var hasImages bool

		for _, block := range blocks {
			switch block.Type {
			case "text":
				textParts = append(textParts, responsesTextPart(msg.Role, block.Text))
			case "image":
				hasImages = true
				var url string
				if block.URL != "" {
					url = block.URL
				} else if block.Data != "" {
					url = "data:" + block.MediaType + ";base64," + block.Data
				}
				if url != "" {
					textParts = append(textParts, responsesInputImage{
						Type:     "input_image",
						ImageURL: url,
					})
				}
			case "document":
				if block.Data != "" && block.MediaType == "text/plain" {
					textParts = append(textParts, responsesTextPart(msg.Role, block.Data))
				}
			case "tool_use":
				toolCalls = append(toolCalls, responsesFunctionCallInput{
					Type:      "function_call",
					CallID:    block.ID,
					Name:      block.Name,
					Arguments: string(block.Input),
					Status:    "completed",
				})
			case "tool_result":
				content := block.Content
				if len(block.ContentJSON) > 0 {
					content = string(block.ContentJSON)
				}
				toolResults = append(toolResults, responsesFunctionCallOutput{
					Type:   "function_call_output",
					CallID: block.ToolUseID,
					Output: content,
				})
			}
		}

		// Emit items in the order expected by the Responses API:
		// 1. Assistant message text (if any)
		// 2. Function calls
		// 3. Function call outputs

		if len(toolCalls) > 0 {
			// Assistant message with tool calls
			if len(textParts) > 0 {
				input = append(input, responsesInputMessage{
					Type:    "message",
					Role:    "assistant",
					Content: textParts,
				})
			}
			for _, tc := range toolCalls {
				input = append(input, tc)
			}
			for _, tr := range toolResults {
				input = append(input, tr)
			}
		} else if len(toolResults) > 0 {
			// Tool results without associated tool calls in this message
			for _, tr := range toolResults {
				input = append(input, tr)
			}
		} else if hasImages {
			input = append(input, responsesInputMessage{
				Type:    "message",
				Role:    msg.Role,
				Content: textParts,
			})
		} else {
			// Plain text from content blocks
			var texts []string
			for _, block := range blocks {
				if block.Type == "text" {
					texts = append(texts, block.Text)
				}
			}
			content := responsesMessageTextContent(msg.Role, strings.Join(texts, "\n"))
			input = append(input, responsesInputMessage{
				Type:    "message",
				Role:    msg.Role,
				Content: content,
			})
		}
	}

	return system, input
}

func responsesMessageTextContent(role, text string) interface{} {
	if role == "assistant" {
		return []interface{}{responsesTextPart(role, text)}
	}
	return text
}

func responsesTextPart(role, text string) interface{} {
	if role == "assistant" {
		return responsesOutputText{
			Type: "output_text",
			Text: text,
		}
	}
	return responsesInputText{
		Type: "input_text",
		Text: text,
	}
}

// convertResponsesTools converts langdag tool definitions to Responses API tool params.
func convertResponsesTools(tools []types.ToolDefinition) []interface{} {
	result := make([]interface{}, 0, len(tools))
	for _, tool := range tools {
		if tool.IsClientTool() {
			result = append(result, responsesFunctionTool{
				Type:        "function",
				Name:        tool.Name,
				Description: tool.Description,
				Parameters:  tool.InputSchema,
			})
			continue
		}

		// Server-side tool: the standardized name maps directly to the
		// Responses API type (e.g. "web_search" → {"type": "web_search"}).
		result = append(result, responsesServerTool{Type: tool.Name})
	}
	return result
}

// --- Response conversion ---

func convertResponsesResult(resp *responsesResponse) *types.CompletionResponse {
	cr := &types.CompletionResponse{
		ID:    resp.ID,
		Model: resp.Model,
	}

	hasFunctionCalls := false
	for _, out := range resp.Output {
		switch out.Type {
		case "message":
			for _, part := range out.Content {
				if part.Type == "output_text" && part.Text != "" {
					cr.Content = append(cr.Content, types.ContentBlock{
						Type: "text",
						Text: part.Text,
					})
				}
			}
		case "function_call":
			hasFunctionCalls = true
			cr.Content = append(cr.Content, types.ContentBlock{
				Type:  "tool_use",
				ID:    out.CallID,
				Name:  out.Name,
				Input: json.RawMessage(out.Arguments),
			})
		case "custom_tool_call":
			hasFunctionCalls = true
			input, _ := json.Marshal(out.Input)
			cr.Content = append(cr.Content, types.ContentBlock{
				Type:  "tool_use",
				ID:    out.CallID,
				Name:  out.Name,
				Input: input,
			})
		}
	}

	if hasFunctionCalls {
		cr.StopReason = "tool_calls"
	} else if resp.Status == "incomplete" {
		cr.StopReason = "incomplete"
		if resp.IncompleteDetails != nil && resp.IncompleteDetails.Reason != "" {
			if resp.IncompleteDetails.Reason == "max_output_tokens" {
				cr.StopReason = "max_tokens"
			} else {
				cr.StopReason = resp.IncompleteDetails.Reason
			}
		}
	} else if resp.Status == "failed" && resp.Error != nil {
		cr.StopReason = "error"
	} else {
		cr.StopReason = "stop"
	}

	if resp.Usage != nil {
		cr.Usage = mapResponsesUsage(resp.Usage)
		cr.Usage.ServiceTier = resp.ServiceTier
		cr.NormalizedUsage = normalizedUsagePtr(cr.Usage)
		cr.ProviderCost = providerCostFromResponsesUsage(resp.Usage)
	}

	return cr
}

func mapResponsesUsage(u *responsesUsage) types.Usage {
	cachedTokens := 0
	if u.InputTokensDetails != nil {
		cachedTokens = u.InputTokensDetails.CachedTokens
	}
	result := types.Usage{
		InputTokens:  max(0, u.InputTokens-cachedTokens),
		OutputTokens: u.OutputTokens,
	}
	if u.InputTokensDetails != nil {
		result.CacheReadInputTokens = cachedTokens
		result.AudioInputTokens = u.InputTokensDetails.AudioTokens
		result.ImageInputTokens = u.InputTokensDetails.ImageTokens
	}
	if u.OutputTokensDetails != nil {
		result.ReasoningTokens = u.OutputTokensDetails.ReasoningTokens
		result.AudioOutputTokens = u.OutputTokensDetails.AudioTokens
		result.ImageOutputTokens = u.OutputTokensDetails.ImageTokens
		result.AcceptedPredictionTokens = u.OutputTokensDetails.AcceptedPredictionTokens
		result.RejectedPredictionTokens = u.OutputTokensDetails.RejectedPredictionTokens
	}
	return result
}

func providerCostFromResponsesUsage(u *responsesUsage) *types.ProviderCost {
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

func contentBlockFromResponsesToolCall(out *responsesOutput) *types.ContentBlock {
	if out == nil {
		return &types.ContentBlock{Type: "tool_use"}
	}
	id := out.CallID
	if id == "" {
		id = out.ID
	}
	cb := &types.ContentBlock{
		Type: "tool_use",
		ID:   id,
		Name: out.Name,
	}
	switch out.Type {
	case "custom_tool_call":
		cb.Input = rawJSONString(out.Input)
	default:
		cb.Input = json.RawMessage(out.Arguments)
	}
	return cb
}

func emitResponsesToolCallDone(events chan<- types.StreamEvent, emitted map[int]bool, outputIndex int, cb *types.ContentBlock) {
	if cb == nil || emitted[outputIndex] {
		return
	}
	emitted[outputIndex] = true
	events <- types.StreamEvent{
		Type:         types.StreamEventContentDone,
		ContentBlock: cb,
	}
}

func rawJSONString(value string) json.RawMessage {
	data, _ := json.Marshal(value)
	return data
}

func responsesAPIError(prefix string, errResp *responsesError) error {
	return responsesStreamError(prefix, errResp)
}

func responsesStreamError(prefix string, errResp *responsesError) error {
	if errResp == nil {
		return fmt.Errorf("%s", prefix)
	}
	if errResp.Message != "" {
		if errResp.Code != "" || errResp.Type != "" {
			return fmt.Errorf("%s: %s (type=%s code=%s)", prefix, errResp.Message, errResp.Type, errResp.Code)
		}
		return fmt.Errorf("%s: %s", prefix, errResp.Message)
	}
	if errResp.Code != "" || errResp.Type != "" {
		return fmt.Errorf("%s: type=%s code=%s", prefix, errResp.Type, errResp.Code)
	}
	return fmt.Errorf("%s", prefix)
}

// --- SSE streaming ---

func parseResponsesSSEStream(body io.Reader, events chan<- types.StreamEvent) {
	events <- types.StreamEvent{Type: types.StreamEventStart}

	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	currentFunctionCalls := map[int]*types.ContentBlock{}
	customToolInputs := map[int]string{}
	emittedFunctionCalls := map[int]bool{}

	for scanner.Scan() {
		line := scanner.Text()

		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")

		if data == "[DONE]" {
			break
		}

		var event responsesStreamEvent
		if err := json.Unmarshal([]byte(data), &event); err != nil {
			continue
		}

		switch event.Type {
		case "error":
			events <- types.StreamEvent{
				Type:  types.StreamEventError,
				Error: responsesStreamError("responses: stream error", event.Error),
			}
			return

		case "response.failed":
			streamErr := event.Error
			if streamErr == nil && event.Response != nil {
				streamErr = event.Response.Error
			}
			events <- types.StreamEvent{
				Type:  types.StreamEventError,
				Error: responsesStreamError("responses: response failed", streamErr),
			}
			return

		case "response.output_text.delta":
			events <- types.StreamEvent{
				Type:    types.StreamEventDelta,
				Content: event.Delta,
			}

		case "response.output_item.added":
			// Track new function call items
			if event.Item != nil && (event.Item.Type == "function_call" || event.Item.Type == "custom_tool_call") {
				currentFunctionCalls[event.OutputIndex] = contentBlockFromResponsesToolCall(event.Item)
			}

		case "response.function_call_arguments.delta":
			existing, ok := currentFunctionCalls[event.OutputIndex]
			if !ok {
				existing = &types.ContentBlock{Type: "tool_use"}
				currentFunctionCalls[event.OutputIndex] = existing
			}
			if event.Delta != "" {
				var prev string
				if existing.Input != nil {
					prev = string(existing.Input)
				}
				existing.Input = json.RawMessage(prev + event.Delta)
			}

		case "response.function_call_arguments.done":
			if cb, ok := currentFunctionCalls[event.OutputIndex]; ok {
				if event.Name != "" {
					cb.Name = event.Name
				}
				if event.Arguments != "" {
					cb.Input = json.RawMessage(event.Arguments)
				}
				emitResponsesToolCallDone(events, emittedFunctionCalls, event.OutputIndex, cb)
			}

		case "response.custom_tool_call_input.delta":
			existing, ok := currentFunctionCalls[event.OutputIndex]
			if !ok {
				existing = &types.ContentBlock{Type: "tool_use"}
				currentFunctionCalls[event.OutputIndex] = existing
			}
			if event.Delta != "" {
				customToolInputs[event.OutputIndex] += event.Delta
				existing.Input = rawJSONString(customToolInputs[event.OutputIndex])
			}

		case "response.custom_tool_call_input.done":
			if cb, ok := currentFunctionCalls[event.OutputIndex]; ok {
				if event.Name != "" {
					cb.Name = event.Name
				}
				if event.Input != "" {
					customToolInputs[event.OutputIndex] = event.Input
					cb.Input = rawJSONString(event.Input)
				}
				emitResponsesToolCallDone(events, emittedFunctionCalls, event.OutputIndex, cb)
			}

		case "response.output_item.done":
			if event.Item != nil && (event.Item.Type == "function_call" || event.Item.Type == "custom_tool_call") {
				cb := contentBlockFromResponsesToolCall(event.Item)
				if existing, ok := currentFunctionCalls[event.OutputIndex]; ok {
					if existing.ID != "" {
						cb.ID = existing.ID
					}
					if existing.Name != "" {
						cb.Name = existing.Name
					}
					if len(existing.Input) > 0 && (len(cb.Input) == 0 || string(cb.Input) == `""`) {
						cb.Input = existing.Input
					}
				}
				currentFunctionCalls[event.OutputIndex] = cb
				emitResponsesToolCallDone(events, emittedFunctionCalls, event.OutputIndex, cb)
			}

		case "response.completed":
			if event.Response != nil {
				resp := convertResponsesResult(event.Response)
				events <- types.StreamEvent{
					Type:     types.StreamEventDone,
					Response: resp,
				}
				return
			}
		}
	}

	if err := scanner.Err(); err != nil {
		events <- types.StreamEvent{
			Type:  types.StreamEventError,
			Error: fmt.Errorf("responses: stream read error: %w", err),
		}
		return
	}

	// Fallback: build a minimal response from accumulated function calls.
	var content []types.ContentBlock
	for _, cb := range currentFunctionCalls {
		content = append(content, *cb)
	}
	stopReason := "stop"
	if len(content) > 0 {
		stopReason = "tool_calls"
	}

	events <- types.StreamEvent{
		Type: types.StreamEventDone,
		Response: &types.CompletionResponse{
			Content:    content,
			StopReason: stopReason,
		},
	}
}
