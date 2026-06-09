package provider

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"langdag.com/langdag/internal/models"
	"langdag.com/langdag/types"
)

type captureProvider struct {
	name         string
	models       []types.ModelInfo
	failCount    int
	streamEvents []types.StreamEvent
	calls        int
	lastReq      *types.CompletionRequest
}

func (p *captureProvider) Name() string { return p.name }

func (p *captureProvider) Models() []types.ModelInfo {
	return append([]types.ModelInfo(nil), p.models...)
}

func (p *captureProvider) Complete(_ context.Context, req *types.CompletionRequest) (*types.CompletionResponse, error) {
	p.calls++
	copied := *req
	p.lastReq = &copied
	if p.failCount > 0 {
		p.failCount--
		return nil, fmt.Errorf("%s failed", p.name)
	}
	return &types.CompletionResponse{
		ID:       "resp-" + p.name,
		Model:    req.Model,
		Provider: p.name,
		Content:  []types.ContentBlock{{Type: "text", Text: "ok"}},
		Usage:    types.Usage{InputTokens: 10, OutputTokens: 5},
	}, nil
}

func (p *captureProvider) Stream(_ context.Context, req *types.CompletionRequest) (<-chan types.StreamEvent, error) {
	p.calls++
	copied := *req
	p.lastReq = &copied
	ch := make(chan types.StreamEvent, len(p.streamEvents)+1)
	go func() {
		defer close(ch)
		if len(p.streamEvents) > 0 {
			for _, event := range p.streamEvents {
				ch <- event
			}
			return
		}
		ch <- types.StreamEvent{
			Type: types.StreamEventDone,
			Response: &types.CompletionResponse{
				ID:    "stream-" + p.name,
				Model: req.Model,
				Usage: types.Usage{InputTokens: 1, OutputTokens: 1},
			},
		}
	}()
	return ch, nil
}

func newTestDeploymentRouter(t *testing.T, deployments map[string]DeploymentAdapter, routing RoutingPolicy) *DeploymentRouter {
	t.Helper()
	catalog := models.ReferenceCatalogV1()
	compiled, err := models.CompileCatalogV1(catalog)
	if err != nil {
		t.Fatal(err)
	}
	router, err := NewDeploymentRouter(DeploymentRouterOptions{
		Catalog:     compiled,
		Deployments: deployments,
		Routing:     routing,
	})
	if err != nil {
		t.Fatal(err)
	}
	return router
}

func deploymentAdapter(id string, p *captureProvider) DeploymentAdapter {
	return DeploymentAdapter{DeploymentID: id, Provider: p}
}

func TestDeploymentRouterResolvesAnthropicDeploymentNativeIDs(t *testing.T) {
	cases := []struct {
		deploymentID string
		wantNative   string
	}{
		{"anthropic-direct", "claude-sonnet-4-20250514"},
		{"anthropic-bedrock", "anthropic.claude-sonnet-4-20250514-v1:0"},
		{"anthropic-vertex", "claude-sonnet-4@20250514"},
	}
	for _, tc := range cases {
		t.Run(tc.deploymentID, func(t *testing.T) {
			inner := &captureProvider{name: tc.deploymentID}
			router := newTestDeploymentRouter(t, map[string]DeploymentAdapter{
				tc.deploymentID: deploymentAdapter(tc.deploymentID, inner),
			}, RoutingPolicy{})

			resp, err := router.Complete(context.Background(), &types.CompletionRequest{
				Model:    "anthropic/claude-sonnet-4-20250514",
				Messages: []types.Message{{Role: "user", Content: json.RawMessage(`"hello"`)}},
			})
			if err != nil {
				t.Fatal(err)
			}
			if inner.lastReq.Model != tc.wantNative {
				t.Fatalf("native model = %q, want %q", inner.lastReq.Model, tc.wantNative)
			}
			if resp.Provider != tc.deploymentID {
				t.Fatalf("response provider = %q, want deployment id %q", resp.Provider, tc.deploymentID)
			}
			if resp.ModelResolution == nil || resp.ModelResolution.ProviderID != "anthropic" || resp.ModelResolution.DeploymentID != tc.deploymentID {
				t.Fatalf("bad model resolution: %+v", resp.ModelResolution)
			}
		})
	}
}

func TestDeploymentRouterMaterializesAzureModelMapping(t *testing.T) {
	inner := &captureProvider{name: "openai-azure"}
	router := newTestDeploymentRouter(t, map[string]DeploymentAdapter{
		"openai-azure": {
			DeploymentID:  "openai-azure",
			Provider:      inner,
			ModelMappings: map[string]string{"openai/gpt-4.1-2025-04-14": "my-gpt-41-prod"},
		},
	}, RoutingPolicy{})

	resp, err := router.Complete(context.Background(), &types.CompletionRequest{
		Model:    "openai/gpt-4.1-2025-04-14",
		Messages: []types.Message{{Role: "user", Content: json.RawMessage(`"hello"`)}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if inner.lastReq.Model != "my-gpt-41-prod" {
		t.Fatalf("Azure native model = %q, want mapped deployment name", inner.lastReq.Model)
	}
	if resp.ModelResolution == nil {
		t.Fatal("missing model resolution")
	}
	if resp.ModelResolution.ProviderID != "openai" || resp.ModelResolution.DeploymentID != "openai-azure" || resp.ModelResolution.NativeModelID != "my-gpt-41-prod" {
		t.Fatalf("bad model resolution: %+v", resp.ModelResolution)
	}
}

func TestDeploymentRouterPassesOfferingAPIProtocol(t *testing.T) {
	catalog := models.ReferenceCatalogV1()
	for i := range catalog.Offerings {
		if catalog.Offerings[i].ID == "openai-direct:gpt-4.1-2025-04-14" {
			catalog.Offerings[i].APIProtocolID = "openai-chat-completions"
		}
	}
	compiled, err := models.CompileCatalogV1(catalog)
	if err != nil {
		t.Fatal(err)
	}
	inner := &captureProvider{name: "openai-direct"}
	router, err := NewDeploymentRouter(DeploymentRouterOptions{
		Catalog: compiled,
		Deployments: map[string]DeploymentAdapter{
			"openai-direct": deploymentAdapter("openai-direct", inner),
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	resp, err := router.Complete(context.Background(), &types.CompletionRequest{
		Model:    "openai/gpt-4.1-2025-04-14",
		Messages: []types.Message{{Role: "user", Content: json.RawMessage(`"hello"`)}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if inner.lastReq.APIProtocolID != "openai-chat-completions" {
		t.Fatalf("request api protocol = %q, want openai-chat-completions", inner.lastReq.APIProtocolID)
	}
	if resp.ModelResolution == nil || resp.ModelResolution.APIProtocolID != "openai-chat-completions" {
		t.Fatalf("response model resolution = %+v, want chat completions protocol", resp.ModelResolution)
	}
}

func TestDeploymentRouterDefaultsOpenAIDirectToResponsesDespiteStaleDeploymentProtocol(t *testing.T) {
	catalog := models.ReferenceCatalogV1()
	catalog.Deployments["openai-direct"].APIProtocolID = "openai-chat-completions"
	catalog.Deployments["openai-direct"].APIProtocolIDs = nil
	compiled, err := models.CompileCatalogV1(catalog)
	if err != nil {
		t.Fatal(err)
	}
	inner := &captureProvider{name: "openai-direct"}
	router, err := NewDeploymentRouter(DeploymentRouterOptions{
		Catalog: compiled,
		Deployments: map[string]DeploymentAdapter{
			"openai-direct": deploymentAdapter("openai-direct", inner),
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	resp, err := router.Complete(context.Background(), &types.CompletionRequest{
		Model:    "openai/gpt-4.1-2025-04-14",
		Messages: []types.Message{{Role: "user", Content: json.RawMessage(`"hello"`)}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if inner.lastReq.APIProtocolID != "openai-responses" {
		t.Fatalf("request api protocol = %q, want openai-responses", inner.lastReq.APIProtocolID)
	}
	if resp.ModelResolution == nil || resp.ModelResolution.APIProtocolID != "openai-responses" {
		t.Fatalf("response model resolution = %+v, want responses protocol", resp.ModelResolution)
	}
}

func TestDeploymentRouterSkipsIneligibleDeployments(t *testing.T) {
	openAI := &captureProvider{name: "openai-direct"}
	openRouter := &captureProvider{name: "openrouter"}
	router := newTestDeploymentRouter(t, map[string]DeploymentAdapter{
		"openai-direct": deploymentAdapter("openai-direct", openAI),
		"openrouter":    deploymentAdapter("openrouter", openRouter),
	}, RoutingPolicy{
		Default: []RoutingStage{
			{Deployments: []DeploymentChoice{{DeploymentID: "openai-direct", Weight: 100}}},
			{Deployments: []DeploymentChoice{{DeploymentID: "openrouter", Weight: 100}}},
		},
	})

	resp, err := router.Complete(context.Background(), &types.CompletionRequest{Model: "z-ai/glm-4.5-air:free"})
	if err != nil {
		t.Fatal(err)
	}
	if openAI.calls != 0 {
		t.Fatalf("ineligible openai-direct was called %d times", openAI.calls)
	}
	if openRouter.calls != 1 || resp.Provider != "openrouter" {
		t.Fatalf("openrouter calls/provider = %d/%q, want 1/openrouter", openRouter.calls, resp.Provider)
	}
	if resp.ModelResolution == nil || resp.ModelResolution.ProviderID != "z-ai" {
		t.Fatalf("openrouter-hosted model owner was not preserved: %+v", resp.ModelResolution)
	}
}

func TestDeploymentRouterSingleDeploymentUnknownNativeModelPassesThrough(t *testing.T) {
	inner := &captureProvider{name: "anthropic-direct"}
	router := newTestDeploymentRouter(t, map[string]DeploymentAdapter{
		"anthropic-direct": deploymentAdapter("anthropic-direct", inner),
	}, RoutingPolicy{})

	resp, err := router.Complete(context.Background(), &types.CompletionRequest{Model: "claude-brand-new-20990101"})
	if err != nil {
		t.Fatal(err)
	}
	if inner.lastReq.Model != "claude-brand-new-20990101" {
		t.Fatalf("native model = %q, want passthrough native id", inner.lastReq.Model)
	}
	if resp.ModelResolution == nil || resp.ModelResolution.CanonicalModelID != "anthropic/claude-brand-new-20990101" {
		t.Fatalf("bad synthetic resolution: %+v", resp.ModelResolution)
	}
}

func TestDeploymentRouterOpenRouterDiscoveryAndUnqualifiedNativeID(t *testing.T) {
	openRouter := &captureProvider{
		name: "openrouter",
		models: []types.ModelInfo{
			{ID: "gpt-oss-20b:free", Name: "GPT OSS Free"},
			{ID: "anthropic/claude-3.5-sonnet", Name: "Claude via OpenRouter"},
		},
	}
	router := newTestDeploymentRouter(t, map[string]DeploymentAdapter{
		"openrouter": deploymentAdapter("openrouter", openRouter),
	}, RoutingPolicy{})

	models := router.Models()
	seen := map[string]bool{}
	for _, model := range models {
		seen[model.ID] = true
	}
	if !seen["openrouter/gpt-oss-20b:free"] || !seen["anthropic/claude-3.5-sonnet"] {
		t.Fatalf("discovered OpenRouter models missing from Models(): %+v", models)
	}

	resp, err := router.Complete(context.Background(), &types.CompletionRequest{Model: "gpt-oss-20b:free"})
	if err != nil {
		t.Fatal(err)
	}
	if openRouter.lastReq.Model != "gpt-oss-20b:free" {
		t.Fatalf("native model = %q, want unqualified OpenRouter native id", openRouter.lastReq.Model)
	}
	if resp.ModelResolution == nil || resp.ModelResolution.CanonicalModelID != "openrouter/gpt-oss-20b:free" || resp.ModelResolution.NativeModelID != "gpt-oss-20b:free" {
		t.Fatalf("bad OpenRouter resolution: %+v", resp.ModelResolution)
	}
}

func TestDeploymentRouterOpenRouterDiscoveredCanonicalInMultiDeploymentRoute(t *testing.T) {
	openAI := &captureProvider{name: "openai-direct"}
	openRouter := &captureProvider{
		name:   "openrouter",
		models: []types.ModelInfo{{ID: "gpt-oss-20b:free", Name: "GPT OSS Free"}},
	}
	router := newTestDeploymentRouter(t, map[string]DeploymentAdapter{
		"openai-direct": deploymentAdapter("openai-direct", openAI),
		"openrouter":    deploymentAdapter("openrouter", openRouter),
	}, RoutingPolicy{
		Default: []RoutingStage{
			{Deployments: []DeploymentChoice{{DeploymentID: "openai-direct", Weight: 100}}},
			{Deployments: []DeploymentChoice{{DeploymentID: "openrouter", Weight: 100}}},
		},
	})

	resp, err := router.Complete(context.Background(), &types.CompletionRequest{Model: "openrouter/gpt-oss-20b:free"})
	if err != nil {
		t.Fatal(err)
	}
	if openAI.calls != 0 {
		t.Fatalf("ineligible openai-direct was called")
	}
	if openRouter.lastReq.Model != "gpt-oss-20b:free" {
		t.Fatalf("native model = %q, want stripped OpenRouter native id", openRouter.lastReq.Model)
	}
	if resp.ModelResolution == nil || resp.ModelResolution.CanonicalModelID != "openrouter/gpt-oss-20b:free" || resp.ModelResolution.NativeModelID != "gpt-oss-20b:free" {
		t.Fatalf("bad OpenRouter canonical resolution: %+v", resp.ModelResolution)
	}
}

func TestDeploymentRouterUnknownFutureOpenAIModelWorksWithMultipleDeployments(t *testing.T) {
	openAI := &captureProvider{name: "openai-direct"}
	openRouter := &captureProvider{name: "openrouter"}
	router := newTestDeploymentRouter(t, map[string]DeploymentAdapter{
		"openai-direct": deploymentAdapter("openai-direct", openAI),
		"openrouter":    deploymentAdapter("openrouter", openRouter),
	}, RoutingPolicy{
		Default: []RoutingStage{{Deployments: []DeploymentChoice{{DeploymentID: "openrouter", Weight: 100}}}},
		Providers: map[string][]RoutingStage{
			"openai": {{Deployments: []DeploymentChoice{{DeploymentID: "openai-direct", Weight: 100}}}},
		},
	})

	resp, err := router.Complete(context.Background(), &types.CompletionRequest{
		Model:    "openai/gpt-next-2099-01-01",
		Messages: []types.Message{{Role: "user", Content: json.RawMessage(`"hello"`)}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if openAI.calls != 1 {
		t.Fatalf("openai-direct calls = %d, want 1", openAI.calls)
	}
	if openRouter.calls != 0 {
		t.Fatalf("openrouter calls = %d, want 0", openRouter.calls)
	}
	if openAI.lastReq.Model != "gpt-next-2099-01-01" {
		t.Fatalf("native model = %q, want gpt-next-2099-01-01", openAI.lastReq.Model)
	}
	if resp.ModelResolution == nil || resp.ModelResolution.APIProtocolID != "openai-responses" {
		t.Fatalf("model resolution = %+v, want openai responses", resp.ModelResolution)
	}
}

func TestDeploymentRouterRouteOverridePrecedence(t *testing.T) {
	openAI := &captureProvider{name: "openai-direct"}
	azure := &captureProvider{name: "openai-azure"}
	router := newTestDeploymentRouter(t, map[string]DeploymentAdapter{
		"openai-direct": deploymentAdapter("openai-direct", openAI),
		"openai-azure": {
			DeploymentID:  "openai-azure",
			Provider:      azure,
			ModelMappings: map[string]string{"openai/gpt-4.1-2025-04-14": "azure-gpt"},
		},
	}, RoutingPolicy{
		Default: []RoutingStage{{Deployments: []DeploymentChoice{{DeploymentID: "openai-direct", Weight: 100}}}},
		Providers: map[string][]RoutingStage{
			"openai": {{Deployments: []DeploymentChoice{{DeploymentID: "openai-direct", Weight: 100}}}},
		},
		Models: map[string][]RoutingStage{
			"openai/gpt-4.1-2025-04-14": {{Deployments: []DeploymentChoice{{DeploymentID: "openai-azure", Weight: 100}}}},
		},
	})

	resp, err := router.Complete(context.Background(), &types.CompletionRequest{Model: "openai/gpt-4.1-2025-04-14"})
	if err != nil {
		t.Fatal(err)
	}
	if openAI.calls != 0 {
		t.Fatalf("provider/default route was used despite exact model route")
	}
	if azure.calls != 1 || resp.Provider != "openai-azure" {
		t.Fatalf("azure calls/provider = %d/%q, want 1/openai-azure", azure.calls, resp.Provider)
	}
}

func TestDeploymentRouterProviderRouteOverridesDefault(t *testing.T) {
	openAI := &captureProvider{name: "openai-direct"}
	azure := &captureProvider{name: "openai-azure"}
	router := newTestDeploymentRouter(t, map[string]DeploymentAdapter{
		"openai-direct": deploymentAdapter("openai-direct", openAI),
		"openai-azure": {
			DeploymentID:  "openai-azure",
			Provider:      azure,
			ModelMappings: map[string]string{"openai/gpt-4.1-2025-04-14": "azure-gpt"},
		},
	}, RoutingPolicy{
		Default: []RoutingStage{{Deployments: []DeploymentChoice{{DeploymentID: "openai-direct", Weight: 100}}}},
		Providers: map[string][]RoutingStage{
			"openai": {{Deployments: []DeploymentChoice{{DeploymentID: "openai-azure", Weight: 100}}}},
		},
	})

	resp, err := router.Complete(context.Background(), &types.CompletionRequest{Model: "openai/gpt-4.1-2025-04-14"})
	if err != nil {
		t.Fatal(err)
	}
	if openAI.calls != 0 || azure.calls != 1 || resp.Provider != "openai-azure" {
		t.Fatalf("openAI/azure/provider = %d/%d/%q, want 0/1/openai-azure", openAI.calls, azure.calls, resp.Provider)
	}
}

func TestDeploymentRouterProviderOverrideDoesNotCascadeToDefault(t *testing.T) {
	openAI := &captureProvider{name: "openai-direct"}
	azure := &captureProvider{name: "openai-azure"}
	router := newTestDeploymentRouter(t, map[string]DeploymentAdapter{
		"openai-direct": deploymentAdapter("openai-direct", openAI),
		"openai-azure":  deploymentAdapter("openai-azure", azure),
	}, RoutingPolicy{
		Default: []RoutingStage{{Deployments: []DeploymentChoice{{DeploymentID: "openai-direct", Weight: 100}}}},
		Providers: map[string][]RoutingStage{
			"openai": {{Deployments: []DeploymentChoice{{DeploymentID: "openai-azure", Weight: 100}}}},
		},
	})

	_, err := router.Complete(context.Background(), &types.CompletionRequest{Model: "openai/gpt-4.1-2025-04-14"})
	if err == nil {
		t.Fatal("expected provider override to fail without cascading to default")
	}
	if openAI.calls != 0 || azure.calls != 0 {
		t.Fatalf("ineligible route should not call adapters, got openAI=%d azure=%d", openAI.calls, azure.calls)
	}
}

func TestDeploymentRouterScopedProviderRouteDoesNotBlockUnmatchedModel(t *testing.T) {
	openAI := &captureProvider{name: "openai-direct"}
	openRouter := &captureProvider{name: "openrouter"}
	router := newTestDeploymentRouter(t, map[string]DeploymentAdapter{
		"openai-direct": deploymentAdapter("openai-direct", openAI),
		"openrouter":    deploymentAdapter("openrouter", openRouter),
	}, RoutingPolicy{
		Providers: map[string][]RoutingStage{
			"openai": {{Deployments: []DeploymentChoice{{DeploymentID: "openai-direct", Weight: 100}}}},
		},
	})

	resp, err := router.Complete(context.Background(), &types.CompletionRequest{Model: "z-ai/glm-4.5-air:free"})
	if err != nil {
		t.Fatal(err)
	}
	if openAI.calls != 0 {
		t.Fatalf("unmatched model should not use openai provider rule, calls=%d", openAI.calls)
	}
	if openRouter.calls != 1 || resp.Provider != "openrouter" {
		t.Fatalf("openrouter calls/provider = %d/%q, want 1/openrouter", openRouter.calls, resp.Provider)
	}
}

func TestDeploymentRouterScopedModelRouteDoesNotBlockUnmatchedModel(t *testing.T) {
	openAI := &captureProvider{name: "openai-direct"}
	openRouter := &captureProvider{name: "openrouter"}
	router := newTestDeploymentRouter(t, map[string]DeploymentAdapter{
		"openai-direct": deploymentAdapter("openai-direct", openAI),
		"openrouter":    deploymentAdapter("openrouter", openRouter),
	}, RoutingPolicy{
		Models: map[string][]RoutingStage{
			"openai/gpt-4.1-2025-04-14": {{Deployments: []DeploymentChoice{{DeploymentID: "openai-direct", Weight: 100}}}},
		},
	})

	resp, err := router.Complete(context.Background(), &types.CompletionRequest{Model: "z-ai/glm-4.5-air:free"})
	if err != nil {
		t.Fatal(err)
	}
	if openAI.calls != 0 {
		t.Fatalf("unmatched model should not use exact model rule, calls=%d", openAI.calls)
	}
	if openRouter.calls != 1 || resp.Provider != "openrouter" {
		t.Fatalf("openrouter calls/provider = %d/%q, want 1/openrouter", openRouter.calls, resp.Provider)
	}
}

func TestDeploymentRouterExplicitDefaultRouteOverridesAutomaticBaseline(t *testing.T) {
	openAI := &captureProvider{name: "openai-direct"}
	openRouter := &captureProvider{name: "openrouter"}
	router := newTestDeploymentRouter(t, map[string]DeploymentAdapter{
		"openai-direct": deploymentAdapter("openai-direct", openAI),
		"openrouter":    deploymentAdapter("openrouter", openRouter),
	}, RoutingPolicy{
		Default: []RoutingStage{{Deployments: []DeploymentChoice{{DeploymentID: "openrouter", Weight: 100}}}},
	})

	resp, err := router.Complete(context.Background(), &types.CompletionRequest{Model: "openai/gpt-4.1-2025-04-14"})
	if err != nil {
		t.Fatal(err)
	}
	if openAI.calls != 0 {
		t.Fatalf("explicit default route should replace automatic openai baseline, calls=%d", openAI.calls)
	}
	if openRouter.calls != 1 || resp.Provider != "openrouter" {
		t.Fatalf("openrouter calls/provider = %d/%q, want 1/openrouter", openRouter.calls, resp.Provider)
	}
}

func TestDeploymentRouterExplicitEmptyDefaultRouteDisablesAutomaticBaseline(t *testing.T) {
	openAI := &captureProvider{name: "openai-direct"}
	router := newTestDeploymentRouter(t, map[string]DeploymentAdapter{
		"openai-direct": deploymentAdapter("openai-direct", openAI),
	}, RoutingPolicy{
		Default: []RoutingStage{},
	})

	_, err := router.Complete(context.Background(), &types.CompletionRequest{Model: "openai/gpt-4.1-2025-04-14"})
	if err == nil {
		t.Fatal("expected explicit empty default route to fail")
	}
	if openAI.calls != 0 {
		t.Fatalf("explicit empty default should not use automatic deployment baseline, calls=%d", openAI.calls)
	}
}

func TestDeploymentRouterWeightedSelectionUsesPositiveEligibleChoices(t *testing.T) {
	openAI := &captureProvider{name: "openai-direct"}
	azure := &captureProvider{name: "openai-azure"}
	router := newTestDeploymentRouter(t, map[string]DeploymentAdapter{
		"openai-direct": deploymentAdapter("openai-direct", openAI),
		"openai-azure": {
			DeploymentID:  "openai-azure",
			Provider:      azure,
			ModelMappings: map[string]string{"openai/gpt-4.1-2025-04-14": "azure-gpt"},
		},
	}, RoutingPolicy{
		Default: []RoutingStage{{
			Deployments: []DeploymentChoice{
				{DeploymentID: "openai-direct", Weight: 0},
				{DeploymentID: "openai-azure", Weight: 100},
			},
		}},
	})

	resp, err := router.Complete(context.Background(), &types.CompletionRequest{Model: "openai/gpt-4.1-2025-04-14"})
	if err != nil {
		t.Fatal(err)
	}
	if openAI.calls != 0 {
		t.Fatalf("zero-weight deployment was called")
	}
	if azure.calls != 1 || resp.Provider != "openai-azure" {
		t.Fatalf("azure calls/provider = %d/%q, want 1/openai-azure", azure.calls, resp.Provider)
	}
}

func TestDeploymentRouterStageRetries(t *testing.T) {
	primary := &captureProvider{name: "openai-direct", failCount: 1}
	router := newTestDeploymentRouter(t, map[string]DeploymentAdapter{
		"openai-direct": deploymentAdapter("openai-direct", primary),
	}, RoutingPolicy{
		Default: []RoutingStage{{Deployments: []DeploymentChoice{{DeploymentID: "openai-direct", Weight: 100}}, Retries: 1}},
	})

	if _, err := router.Complete(context.Background(), &types.CompletionRequest{Model: "openai/gpt-4.1-2025-04-14"}); err != nil {
		t.Fatal(err)
	}
	if primary.calls != 2 {
		t.Fatalf("calls = %d, want one retry after first failure", primary.calls)
	}
}

func TestDeploymentRouterDefaultStagesUseMaxRetriesWithMultipleDeployments(t *testing.T) {
	stages := defaultStagesForDeployments(map[string]DeploymentAdapter{
		"openai-direct": {DeploymentID: "openai-direct", Provider: &captureProvider{name: "openai"}, DefaultRetries: 2},
		"openrouter":    {DeploymentID: "openrouter", Provider: &captureProvider{name: "openrouter"}, DefaultRetries: 1},
	})
	if len(stages) != 1 || stages[0].Retries != 2 {
		t.Fatalf("default stages = %+v, want one stage with max retries 2", stages)
	}
}

func TestDeploymentRouterMissingAzureMappingIsIneligible(t *testing.T) {
	azure := &captureProvider{name: "openai-azure"}
	router := newTestDeploymentRouter(t, map[string]DeploymentAdapter{
		"openai-azure": deploymentAdapter("openai-azure", azure),
	}, RoutingPolicy{})

	_, err := router.Complete(context.Background(), &types.CompletionRequest{Model: "openai/gpt-4.1-2025-04-14"})
	if err == nil {
		t.Fatal("expected missing model mapping error")
	}
	if azure.calls != 0 {
		t.Fatalf("azure adapter should not be called without mapping")
	}
	if !strings.Contains(err.Error(), "no eligible deployments") {
		t.Fatalf("error = %v, want no eligible deployments diagnostic", err)
	}
}

func TestDeploymentRouterFiltersUnknownServerToolCapability(t *testing.T) {
	inner := &captureProvider{name: "openai-direct"}
	router := newTestDeploymentRouter(t, map[string]DeploymentAdapter{
		"openai-direct": deploymentAdapter("openai-direct", inner),
	}, RoutingPolicy{})

	_, err := router.Complete(context.Background(), &types.CompletionRequest{
		Model: "openai/gpt-4.1-2025-04-14",
		Tools: []types.ToolDefinition{
			{Name: types.ServerToolWebSearch, Description: "search"},
			{Name: "client_func", Description: "client", InputSchema: json.RawMessage(`{"type":"object"}`)},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(inner.lastReq.Tools) != 1 || inner.lastReq.Tools[0].Name != "client_func" {
		t.Fatalf("filtered tools = %+v, want client_func only", inner.lastReq.Tools)
	}
}

func TestDeploymentRouterPrefersDeploymentWithRequiredServerTool(t *testing.T) {
	catalog := models.ReferenceCatalogV1()
	for i := range catalog.OfferingTemplates {
		if catalog.OfferingTemplates[i].DeploymentID == "openai-azure" &&
			catalog.OfferingTemplates[i].CanonicalModelID == "openai/gpt-4.1-2025-04-14" {
			catalog.OfferingTemplates[i].Capabilities.ServerTools[types.ServerToolWebSearch] = models.CapabilitySupported
		}
	}
	compiled, err := models.CompileCatalogV1(catalog)
	if err != nil {
		t.Fatal(err)
	}
	direct := &captureProvider{name: "openai-direct"}
	azure := &captureProvider{name: "openai-azure"}
	router, err := NewDeploymentRouter(DeploymentRouterOptions{
		Catalog: compiled,
		Deployments: map[string]DeploymentAdapter{
			"openai-direct": deploymentAdapter("openai-direct", direct),
			"openai-azure": {
				DeploymentID:  "openai-azure",
				Provider:      azure,
				ModelMappings: map[string]string{"openai/gpt-4.1-2025-04-14": "azure-gpt"},
			},
		},
		Routing: RoutingPolicy{Default: []RoutingStage{{Deployments: []DeploymentChoice{
			{DeploymentID: "openai-direct", Weight: 100},
			{DeploymentID: "openai-azure", Weight: 100},
		}}}},
	})
	if err != nil {
		t.Fatal(err)
	}

	_, err = router.Complete(context.Background(), &types.CompletionRequest{
		Model: "openai/gpt-4.1-2025-04-14",
		Tools: []types.ToolDefinition{{Name: types.ServerToolWebSearch, Description: "search"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if direct.calls != 0 || azure.calls != 1 {
		t.Fatalf("direct/azure calls = %d/%d, want 0/1", direct.calls, azure.calls)
	}
	if len(azure.lastReq.Tools) != 1 || azure.lastReq.Tools[0].Name != types.ServerToolWebSearch {
		t.Fatalf("azure tools = %+v, want web_search preserved", azure.lastReq.Tools)
	}
}

func TestDeploymentRouterOllamaDiscoveryFallback(t *testing.T) {
	ollama := &captureProvider{
		name:   "ollama",
		models: []types.ModelInfo{{ID: "llama3.2:3b", Name: "llama3.2:3b"}},
	}
	router := newTestDeploymentRouter(t, map[string]DeploymentAdapter{
		"ollama-local": deploymentAdapter("ollama-local", ollama),
	}, RoutingPolicy{})

	resp, err := router.Complete(context.Background(), &types.CompletionRequest{Model: "ollama/llama3.2:3b"})
	if err != nil {
		t.Fatal(err)
	}
	if ollama.lastReq.Model != "llama3.2:3b" {
		t.Fatalf("native model = %q, want discovered Ollama tag", ollama.lastReq.Model)
	}
	if resp.PricingSnapshot == nil || resp.PricingSnapshot.Status != types.CostStatusFree {
		t.Fatalf("Ollama pricing snapshot = %+v, want free", resp.PricingSnapshot)
	}
}

func TestDeploymentRouterAppleDiscoveryFallback(t *testing.T) {
	apple := &captureProvider{
		name:   "apple",
		models: []types.ModelInfo{{ID: "system", Name: "Apple system"}},
	}
	router := newTestDeploymentRouter(t, map[string]DeploymentAdapter{
		"apple-local": deploymentAdapter("apple-local", apple),
	}, RoutingPolicy{})

	resp, err := router.Complete(context.Background(), &types.CompletionRequest{Model: "apple/system"})
	if err != nil {
		t.Fatal(err)
	}
	if apple.lastReq.Model != "system" {
		t.Fatalf("native model = %q, want system", apple.lastReq.Model)
	}
	if resp.ModelResolution == nil || resp.ModelResolution.CanonicalModelID != "apple/system" || resp.ModelResolution.NativeModelID != "system" {
		t.Fatalf("bad Apple resolution: %+v", resp.ModelResolution)
	}
	if resp.PricingSnapshot == nil || resp.PricingSnapshot.Status != types.CostStatusFree {
		t.Fatalf("Apple pricing snapshot = %+v, want free", resp.PricingSnapshot)
	}
}

func TestDeploymentRouterOllamaSlashNativeID(t *testing.T) {
	ollama := &captureProvider{
		name:   "ollama",
		models: []types.ModelInfo{{ID: "library/custom:latest", Name: "library/custom:latest"}},
	}
	router := newTestDeploymentRouter(t, map[string]DeploymentAdapter{
		"ollama-local": deploymentAdapter("ollama-local", ollama),
	}, RoutingPolicy{})

	resp, err := router.Complete(context.Background(), &types.CompletionRequest{Model: "library/custom:latest"})
	if err != nil {
		t.Fatal(err)
	}
	if ollama.lastReq.Model != "library/custom:latest" {
		t.Fatalf("native model = %q, want slash-containing Ollama tag", ollama.lastReq.Model)
	}
	if resp.ModelResolution == nil || resp.ModelResolution.CanonicalModelID != "ollama/library/custom:latest" {
		t.Fatalf("bad Ollama slash resolution: %+v", resp.ModelResolution)
	}
}

func TestDeploymentRouterOllamaSlashNativeIDWithMultipleDeployments(t *testing.T) {
	openAI := &captureProvider{name: "openai-direct"}
	ollama := &captureProvider{
		name:   "ollama",
		models: []types.ModelInfo{{ID: "library/custom:latest", Name: "library/custom:latest"}},
	}
	router := newTestDeploymentRouter(t, map[string]DeploymentAdapter{
		"openai-direct": deploymentAdapter("openai-direct", openAI),
		"ollama-local":  deploymentAdapter("ollama-local", ollama),
	}, RoutingPolicy{
		Default: []RoutingStage{{Deployments: []DeploymentChoice{
			{DeploymentID: "openai-direct", Weight: 100},
			{DeploymentID: "ollama-local", Weight: 100},
		}}},
	})

	resp, err := router.Complete(context.Background(), &types.CompletionRequest{Model: "library/custom:latest"})
	if err != nil {
		t.Fatal(err)
	}
	if openAI.calls != 0 {
		t.Fatalf("ineligible OpenAI deployment was called")
	}
	if ollama.lastReq.Model != "library/custom:latest" {
		t.Fatalf("native model = %q, want slash-containing Ollama tag", ollama.lastReq.Model)
	}
	if resp.ModelResolution == nil || resp.ModelResolution.CanonicalModelID != "ollama/library/custom:latest" {
		t.Fatalf("bad multi-deployment Ollama slash resolution: %+v", resp.ModelResolution)
	}
}

func TestDeploymentRouterDoesNotFallbackAfterStreamOutput(t *testing.T) {
	primary := &captureProvider{
		name: "openai-direct",
		streamEvents: []types.StreamEvent{
			{Type: types.StreamEventDelta, Content: "partial"},
			{Type: types.StreamEventError, Error: fmt.Errorf("boom after output")},
		},
	}
	fallback := &captureProvider{name: "openrouter"}
	router := newTestDeploymentRouter(t, map[string]DeploymentAdapter{
		"openai-direct": deploymentAdapter("openai-direct", primary),
		"openrouter":    deploymentAdapter("openrouter", fallback),
	}, RoutingPolicy{
		Default: []RoutingStage{
			{Deployments: []DeploymentChoice{{DeploymentID: "openai-direct", Weight: 100}}},
			{Deployments: []DeploymentChoice{{DeploymentID: "openrouter", Weight: 100}}},
		},
	})

	ch, err := router.Stream(context.Background(), &types.CompletionRequest{Model: "openai/gpt-4.1-2025-04-14"})
	if err != nil {
		t.Fatal(err)
	}
	var sawDelta, sawError bool
	for event := range ch {
		if event.Type == types.StreamEventDelta && event.Content == "partial" {
			sawDelta = true
		}
		if event.Type == types.StreamEventError {
			sawError = true
		}
	}
	if !sawDelta || !sawError {
		t.Fatalf("sawDelta/sawError = %v/%v, want true/true", sawDelta, sawError)
	}
	if fallback.calls != 0 {
		t.Fatalf("fallback was called after partial output")
	}
}

func TestDeploymentRouterDiscardsPreOutputEventsBeforeFallback(t *testing.T) {
	primary := &captureProvider{
		name: "openai-direct",
		streamEvents: []types.StreamEvent{
			{Type: types.StreamEventStart},
			{Type: types.StreamEventError, Error: fmt.Errorf("boom before output")},
		},
	}
	fallback := &captureProvider{
		name: "openrouter",
		streamEvents: []types.StreamEvent{
			{Type: types.StreamEventStart},
			{Type: types.StreamEventDone, Response: &types.CompletionResponse{ID: "fallback", Model: "z-ai/glm-4.5-air:free"}},
		},
	}
	router := newTestDeploymentRouter(t, map[string]DeploymentAdapter{
		"openai-direct": deploymentAdapter("openai-direct", primary),
		"openrouter":    deploymentAdapter("openrouter", fallback),
	}, RoutingPolicy{
		Default: []RoutingStage{
			{Deployments: []DeploymentChoice{{DeploymentID: "openai-direct", Weight: 100}}},
			{Deployments: []DeploymentChoice{{DeploymentID: "openrouter", Weight: 100}}},
		},
	})

	ch, err := router.Stream(context.Background(), &types.CompletionRequest{Model: "openai/gpt-4.1-2025-04-14"})
	if err != nil {
		t.Fatal(err)
	}
	startEvents := 0
	doneEvents := 0
	for event := range ch {
		if event.Type == types.StreamEventStart {
			startEvents++
		}
		if event.Type == types.StreamEventDone {
			doneEvents++
		}
	}
	if primary.calls != 1 {
		t.Fatalf("primary calls = %d, want 1 pre-output failure", primary.calls)
	}
	if fallback.calls != 1 || startEvents != 1 || doneEvents != 1 {
		t.Fatalf("fallback calls/start/done = %d/%d/%d, want 1/1/1", fallback.calls, startEvents, doneEvents)
	}
}
