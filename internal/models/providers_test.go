package models

import (
	"context"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"
)

func TestParseOpenAIText(t *testing.T) {
	text := `# GPT-4o

**Current Snapshot:** gpt-4o-2024-08-06

## Snapshots

### gpt-4o-2024-08-06

- Context window size: 128000
- Knowledge cutoff date: 2023-10-01
- Maximum output tokens: 16384

### gpt-4o-mini-2024-07-18

- Context window size: 128000
- Maximum output tokens: 16384

## Text tokens

| Name | Input | Cached input | Output | Unit |
| --- | --- | --- | --- | --- |
| gpt-4o | 2.5 | 1.25 | 10 | 1M tokens |
| gpt-4o (batch) | 1.25 | | 5 | 1M tokens |
| gpt-4o-2024-08-06 | 2.5 | 1.25 | 10 | 1M tokens |
| gpt-4o-mini | 0.15 | 0.075 | 0.6 | 1M tokens |
| o1-pro | 100 | | 400 | 1M tokens |

## Audio tokens
`

	models, err := parseOpenAIText(text)
	if err != nil {
		t.Fatalf("parseOpenAIText error: %v", err)
	}

	byID := make(map[string]ModelPricing)
	for _, m := range models {
		byID[m.ID] = m
	}

	// gpt-4o: pricing only (no metadata section for family name)
	if m, ok := byID["gpt-4o"]; !ok {
		t.Error("gpt-4o not found")
	} else {
		if m.InputPricePer1M != 2.5 {
			t.Errorf("gpt-4o input = %f, want 2.5", m.InputPricePer1M)
		}
		if m.OutputPricePer1M != 10 {
			t.Errorf("gpt-4o output = %f, want 10", m.OutputPricePer1M)
		}
	}

	// gpt-4o-2024-08-06: pricing + metadata
	if m, ok := byID["gpt-4o-2024-08-06"]; !ok {
		t.Error("gpt-4o-2024-08-06 not found")
	} else {
		if m.InputPricePer1M != 2.5 {
			t.Errorf("gpt-4o-2024-08-06 input = %f, want 2.5", m.InputPricePer1M)
		}
		if m.ContextWindow != 128000 {
			t.Errorf("gpt-4o-2024-08-06 context = %d, want 128000", m.ContextWindow)
		}
		if m.MaxOutput != 16384 {
			t.Errorf("gpt-4o-2024-08-06 maxOutput = %d, want 16384", m.MaxOutput)
		}
	}

	// gpt-4o-mini-2024-07-18: metadata only (no pricing row for this snapshot)
	if m, ok := byID["gpt-4o-mini-2024-07-18"]; !ok {
		t.Error("gpt-4o-mini-2024-07-18 not found")
	} else {
		if m.ContextWindow != 128000 {
			t.Errorf("context = %d, want 128000", m.ContextWindow)
		}
		if m.InputPricePer1M != 0 {
			t.Errorf("input = %f, want 0 (no pricing row)", m.InputPricePer1M)
		}
	}

	// Batch entries should be filtered
	if _, ok := byID["gpt-4o (batch)"]; ok {
		t.Error("batch entry should be filtered out")
	}

	// o1-pro: high pricing
	if m, ok := byID["o1-pro"]; !ok {
		t.Error("o1-pro not found")
	} else {
		if m.InputPricePer1M != 100 {
			t.Errorf("o1-pro input = %f, want 100", m.InputPricePer1M)
		}
		if m.OutputPricePer1M != 400 {
			t.Errorf("o1-pro output = %f, want 400", m.OutputPricePer1M)
		}
	}
}

func TestFetchOpenAIModelsFromDocsPages(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/api/docs/models/all", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`<html><body>
<a href="/api/docs/models/gpt-5.5">GPT-5.5</a>
<a href="/api/docs/models/gpt-5.5-pro">GPT-5.5 pro</a>
<a href="/api/docs/models/gpt-image-2">GPT Image 2</a>
</body></html>`))
	})
	mux.HandleFunc("/api/docs/models/gpt-5.5", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`<html><body>
<h1>GPT-5.5</h1>
<p>1,050,000 context window</p>
<p>128,000 max output tokens</p>
<h2>Pricing</h2>
<p>Text tokens</p><p>Per 1M tokens</p>
<p>Input</p><p>$5.00</p><p>Cached input</p><p>$0.50</p><p>Output</p><p>$30.00</p>
<h2>Snapshots</h2><p>gpt-5.5</p><p>gpt-5.5-2026-04-23</p><h2>Rate limits</h2>
</body></html>`))
	})
	mux.HandleFunc("/api/docs/models/gpt-5.5-pro", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`<html><body>
<h1>GPT-5.5 pro</h1>
<p>1,050,000 context window</p>
<p>128,000 max output tokens</p>
<h2>Pricing</h2>
<p>Text tokens</p><p>Per 1M tokens</p>
<p>Input</p><p>$30.00</p><p>Output</p><p>$180.00</p>
<h2>Snapshots</h2><p>Snapshots for GPT-5.5 pro.</p><p>gpt-5.5-pro</p><p>gpt-5.5-pro-2026-04-23</p><h2>Rate limits</h2>
</body></html>`))
	})
	mux.HandleFunc("/api/docs/models/gpt-image-2", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`<html><body><h1>GPT Image 2</h1><p>Text tokens</p></body></html>`))
	})
	server := httptest.NewServer(mux)
	defer server.Close()

	origSource, origDetails := openAISourceURL, openAIModelDetailsURL
	defer func() {
		openAISourceURL = origSource
		openAIModelDetailsURL = origDetails
	}()
	openAISourceURL = server.URL + "/api/docs/models/all"
	openAIModelDetailsURL = server.URL + "/api/docs/models"

	models, err := fetchOpenAIModels(context.Background())
	if err != nil {
		t.Fatalf("fetchOpenAIModels error: %v", err)
	}

	byID := make(map[string]ModelPricing)
	for _, m := range models {
		byID[m.ID] = m
	}

	if m, ok := byID["gpt-5.5"]; !ok {
		t.Fatal("gpt-5.5 not found")
	} else {
		if m.InputPricePer1M != 5 || m.OutputPricePer1M != 30 {
			t.Errorf("gpt-5.5 pricing = %f/%f, want 5/30", m.InputPricePer1M, m.OutputPricePer1M)
		}
		if m.ContextWindow != 1050000 || m.MaxOutput != 128000 {
			t.Errorf("gpt-5.5 limits = %d/%d, want 1050000/128000", m.ContextWindow, m.MaxOutput)
		}
	}
	if _, ok := byID["gpt-5.5-2026-04-23"]; !ok {
		t.Error("gpt-5.5 snapshot not found")
	}
	if m, ok := byID["gpt-5.5-pro"]; !ok {
		t.Fatal("gpt-5.5-pro not found")
	} else if m.InputPricePer1M != 30 || m.OutputPricePer1M != 180 {
		t.Errorf("gpt-5.5-pro pricing = %f/%f, want 30/180", m.InputPricePer1M, m.OutputPricePer1M)
	}
	if _, ok := byID["gpt-5.5-pro-2026-04-23"]; !ok {
		t.Error("gpt-5.5-pro snapshot not found")
	}
	if _, ok := byID["gpt-image-2"]; ok {
		t.Error("gpt-image-2 should be skipped without text pricing and context")
	}
}

func TestAnthropicBedrockNativeModelIDV1(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want string
	}{
		{name: "dated", in: "claude-sonnet-4-5-20250929", want: "anthropic.claude-sonnet-4-5-20250929-v1:0"},
		{name: "already provider qualified", in: "anthropic.claude-sonnet-4-6", want: "anthropic.claude-sonnet-4-6"},
		{name: "opus 4.6 last v1", in: "claude-opus-4-6", want: "anthropic.claude-opus-4-6-v1"},
		{name: "sonnet 4.6 unversioned", in: "claude-sonnet-4-6", want: "anthropic.claude-sonnet-4-6"},
		{name: "future semantic unversioned", in: "claude-opus-4-7", want: "anthropic.claude-opus-4-7"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := anthropicBedrockNativeModelIDV1(tt.in); got != tt.want {
				t.Fatalf("anthropicBedrockNativeModelIDV1(%q) = %q, want %q", tt.in, got, tt.want)
			}
		})
	}
}

func TestParseAnthropicHTML(t *testing.T) {
	html := `<html><body>
<table>
<tr><th>Feature</th><th>Claude Opus 4.6</th><th>Claude Sonnet 4.6</th></tr>
<tr><td><strong>Claude API ID</strong></td><td>claude-opus-4-6</td><td>claude-sonnet-4-6</td></tr>
<tr><td><strong>Claude API alias</strong></td><td>claude-opus-4-6</td><td>claude-sonnet-4-6</td></tr>
<tr><td><strong>Pricing</strong></td><td>$5 / input MTok<br/>$25 / output MTok</td><td>$3 / input MTok<br/>$15 / output MTok</td></tr>
<tr><td><strong>Context window</strong></td><td>200K tokens</td><td>200K tokens</td></tr>
<tr><td><strong>Max output</strong></td><td>128K tokens</td><td>64K tokens</td></tr>
</table>
<table>
<tr><th>Feature</th><th>Claude Sonnet 4.5</th></tr>
<tr><td><strong>Claude API ID</strong></td><td>claude-sonnet-4-5-20250929</td></tr>
<tr><td><strong>Claude API alias</strong></td><td>claude-sonnet-4-5</td></tr>
<tr><td><strong>Pricing</strong></td><td>$3 / input MTok<br/>$15 / output MTok</td></tr>
<tr><td><strong>Context window</strong></td><td>200K tokens</td></tr>
<tr><td><strong>Max output</strong></td><td>64K tokens</td></tr>
</table>
</body></html>`

	models, err := parseAnthropicHTML(html)
	if err != nil {
		t.Fatalf("parseAnthropicHTML error: %v", err)
	}

	byID := make(map[string]ModelPricing)
	for _, m := range models {
		byID[m.ID] = m
	}

	// Opus 4.6
	if m, ok := byID["claude-opus-4-6"]; !ok {
		t.Error("claude-opus-4-6 not found")
	} else {
		if m.InputPricePer1M != 5 {
			t.Errorf("opus input = %f, want 5", m.InputPricePer1M)
		}
		if m.OutputPricePer1M != 25 {
			t.Errorf("opus output = %f, want 25", m.OutputPricePer1M)
		}
		if m.ContextWindow != 200000 {
			t.Errorf("opus context = %d, want 200000", m.ContextWindow)
		}
		if m.MaxOutput != 128000 {
			t.Errorf("opus maxOutput = %d, want 128000", m.MaxOutput)
		}
	}

	// Sonnet 4.6
	if m, ok := byID["claude-sonnet-4-6"]; !ok {
		t.Error("claude-sonnet-4-6 not found")
	} else {
		if m.OutputPricePer1M != 15 {
			t.Errorf("sonnet output = %f, want 15", m.OutputPricePer1M)
		}
		if m.MaxOutput != 64000 {
			t.Errorf("sonnet maxOutput = %d, want 64000", m.MaxOutput)
		}
	}

	// Sonnet 4.5 (legacy) — both ID and alias should be present
	if _, ok := byID["claude-sonnet-4-5-20250929"]; !ok {
		t.Error("claude-sonnet-4-5-20250929 not found")
	}
	if m, ok := byID["claude-sonnet-4-5"]; !ok {
		t.Error("claude-sonnet-4-5 alias not found")
	} else {
		if m.InputPricePer1M != 3 {
			t.Errorf("sonnet 4.5 alias input = %f, want 3", m.InputPricePer1M)
		}
	}
}

func TestFetchAnthropicModelsAPIMergesHTMLPricingAndOverrides(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/models", func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("x-api-key"); got != "test-key" {
			http.Error(w, "missing api key", http.StatusBadRequest)
			return
		}
		if got := r.Header.Get("anthropic-version"); got != "2023-06-01" {
			http.Error(w, "missing version", http.StatusBadRequest)
			return
		}
		if got := r.Header.Get("Accept"); got != "application/json" {
			http.Error(w, "missing accept", http.StatusBadRequest)
			return
		}
		if got := r.Header.Get("User-Agent"); got != providerUserAgent {
			http.Error(w, "missing user agent", http.StatusBadRequest)
			return
		}
		if got := r.URL.Query().Get("limit"); got != "1000" {
			http.Error(w, "missing limit", http.StatusBadRequest)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{
			"data": [
				{"id": "claude-fable-5", "type": "model", "display_name": "Claude Fable 5", "max_input_tokens": 1000000, "max_tokens": 128000, "capabilities": {"vision": true}},
				{"id": "claude-sonnet-4-6", "type": "model", "display_name": "Claude Sonnet 4.6", "max_input_tokens": 1000000, "max_tokens": 128000}
			],
			"has_more": false,
			"last_id": "claude-sonnet-4-6"
		}`))
	})
	mux.HandleFunc("/docs", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`<html><body>
<table>
<tr><th>Feature</th><th>Claude Sonnet 4.6</th><th>Claude Sonnet 4.5</th></tr>
<tr><td><strong>Claude API ID</strong></td><td>claude-sonnet-4-6</td><td>claude-sonnet-4-5-20250929</td></tr>
<tr><td><strong>Claude API alias</strong></td><td>claude-sonnet-4-6</td><td>claude-sonnet-4-5</td></tr>
<tr><td><strong>Pricing</strong></td><td>$3 / input MTok<br/>$15 / output MTok</td><td>$3 / input MTok<br/>$15 / output MTok</td></tr>
<tr><td><strong>Context window</strong></td><td>200K tokens</td><td>200K tokens</td></tr>
<tr><td><strong>Max output</strong></td><td>64K tokens</td><td>64K tokens</td></tr>
</table>
</body></html>`))
	})
	server := httptest.NewServer(mux)
	defer server.Close()

	origSource, origAPI := anthropicSourceURL, anthropicModelsAPIURL
	defer func() {
		anthropicSourceURL = origSource
		anthropicModelsAPIURL = origAPI
	}()
	anthropicSourceURL = server.URL + "/docs"
	anthropicModelsAPIURL = server.URL + "/v1/models"
	t.Setenv("ANTHROPIC_API_KEY", "test-key")
	t.Setenv("LANGDAG_REQUIRE_ANTHROPIC_MODELS_API", "")

	models, err := fetchAnthropicModels(context.Background())
	if err != nil {
		t.Fatalf("fetchAnthropicModels error: %v", err)
	}

	byID := make(map[string]ModelPricing)
	for _, m := range models {
		byID[m.ID] = m
	}

	if m, ok := byID["claude-fable-5"]; !ok {
		t.Fatal("claude-fable-5 not found")
	} else {
		if m.ContextWindow != 1000000 || m.MaxOutput != 128000 {
			t.Errorf("fable limits = %d/%d, want 1000000/128000", m.ContextWindow, m.MaxOutput)
		}
		if m.InputPricePer1M != 10 || m.OutputPricePer1M != 50 {
			t.Errorf("fable pricing = %f/%f, want 10/50", m.InputPricePer1M, m.OutputPricePer1M)
		}
	}
	if m, ok := byID["claude-sonnet-4-6"]; !ok {
		t.Fatal("claude-sonnet-4-6 not found")
	} else {
		if m.ContextWindow != 1000000 || m.MaxOutput != 128000 {
			t.Errorf("sonnet API limits = %d/%d, want 1000000/128000", m.ContextWindow, m.MaxOutput)
		}
		if m.InputPricePer1M != 3 || m.OutputPricePer1M != 15 {
			t.Errorf("sonnet pricing = %f/%f, want 3/15", m.InputPricePer1M, m.OutputPricePer1M)
		}
	}
	if _, ok := byID["claude-sonnet-4-5-20250929"]; !ok {
		t.Error("scraped-only sonnet 4.5 ID not preserved")
	}
	if _, ok := byID["claude-sonnet-4-5"]; !ok {
		t.Error("scraped-only sonnet 4.5 alias not preserved")
	}
}

func TestFetchAnthropicModelsAPIBackfillsZeroLimitsFromHTML(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/models", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{
			"data": [
				{"id": "claude-fable-5", "max_input_tokens": 0, "max_tokens": 0}
			],
			"has_more": false,
			"last_id": "claude-fable-5"
		}`))
	})
	mux.HandleFunc("/docs", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`<html><body>
<table>
<tr><th>Feature</th><th>Claude Fable 5</th></tr>
<tr><td><strong>Claude API ID</strong></td><td>claude-fable-5</td></tr>
<tr><td><strong>Claude API alias</strong></td><td>claude-fable-5</td></tr>
<tr><td><strong>Pricing</strong></td><td>$10 / $50 per MTok (input / output)</td></tr>
<tr><td><strong>Context window</strong></td><td>1M tokens</td></tr>
<tr><td><strong>Max output</strong></td><td>128K tokens</td></tr>
</table>
</body></html>`))
	})
	server := httptest.NewServer(mux)
	defer server.Close()

	origSource, origAPI := anthropicSourceURL, anthropicModelsAPIURL
	defer func() {
		anthropicSourceURL = origSource
		anthropicModelsAPIURL = origAPI
	}()
	anthropicSourceURL = server.URL + "/docs"
	anthropicModelsAPIURL = server.URL + "/v1/models"
	t.Setenv("ANTHROPIC_API_KEY", "test-key")
	t.Setenv("LANGDAG_REQUIRE_ANTHROPIC_MODELS_API", "1")

	models, err := fetchAnthropicModels(context.Background())
	if err != nil {
		t.Fatalf("fetchAnthropicModels error: %v", err)
	}
	if len(models) != 1 {
		t.Fatalf("got %d models, want 1", len(models))
	}
	model := models[0]
	if model.ID != "claude-fable-5" {
		t.Fatalf("model ID = %q, want claude-fable-5", model.ID)
	}
	if model.ContextWindow != 1000000 || model.MaxOutput != 128000 {
		t.Fatalf("backfilled limits = %d/%d, want 1000000/128000", model.ContextWindow, model.MaxOutput)
	}
	if model.InputPricePer1M != 10 || model.OutputPricePer1M != 50 {
		t.Fatalf("override pricing = %f/%f, want 10/50", model.InputPricePer1M, model.OutputPricePer1M)
	}
}

func TestFetchAnthropicModelsNoKeyUsesHTMLOnly(t *testing.T) {
	var apiCalls atomic.Int32
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/models", func(w http.ResponseWriter, r *http.Request) {
		apiCalls.Add(1)
		http.Error(w, "should not be called", http.StatusInternalServerError)
	})
	mux.HandleFunc("/docs", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`<html><body>
<table>
<tr><th>Feature</th><th>Claude Fable 5</th></tr>
<tr><td><strong>Claude API ID</strong></td><td>claude-fable-5</td></tr>
<tr><td><strong>Claude API alias</strong></td><td>claude-fable-5</td></tr>
<tr><td><strong>Pricing</strong></td><td>$10 / $50 per MTok (input / output)</td></tr>
<tr><td><strong>Context window</strong></td><td>1M tokens</td></tr>
<tr><td><strong>Max output</strong></td><td>128K tokens</td></tr>
</table>
</body></html>`))
	})
	server := httptest.NewServer(mux)
	defer server.Close()

	origSource, origAPI := anthropicSourceURL, anthropicModelsAPIURL
	defer func() {
		anthropicSourceURL = origSource
		anthropicModelsAPIURL = origAPI
	}()
	anthropicSourceURL = server.URL + "/docs"
	anthropicModelsAPIURL = server.URL + "/v1/models"
	t.Setenv("ANTHROPIC_API_KEY", "")
	t.Setenv("LANGDAG_REQUIRE_ANTHROPIC_MODELS_API", "")

	models, err := fetchAnthropicModels(context.Background())
	if err != nil {
		t.Fatalf("fetchAnthropicModels error: %v", err)
	}
	if apiCalls.Load() != 0 {
		t.Fatalf("API was called without an API key")
	}
	if len(models) != 1 || models[0].ID != "claude-fable-5" {
		t.Fatalf("models = %#v, want only claude-fable-5", models)
	}
	if models[0].InputPricePer1M != 10 || models[0].OutputPricePer1M != 50 {
		t.Errorf("pricing = %f/%f, want 10/50", models[0].InputPricePer1M, models[0].OutputPricePer1M)
	}
}

func TestFetchAnthropicModelsRequiredAPIRejectsMissingKey(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/docs", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`<html><body>
<table>
<tr><th>Feature</th><th>Claude Haiku 4.5</th></tr>
<tr><td><strong>Claude API ID</strong></td><td>claude-haiku-4-5</td></tr>
<tr><td><strong>Pricing</strong></td><td>$1 / input MTok<br/>$5 / output MTok</td></tr>
<tr><td><strong>Context window</strong></td><td>200K tokens</td></tr>
<tr><td><strong>Max output</strong></td><td>64K tokens</td></tr>
</table>
</body></html>`))
	})
	server := httptest.NewServer(mux)
	defer server.Close()

	origSource, origAPI := anthropicSourceURL, anthropicModelsAPIURL
	defer func() {
		anthropicSourceURL = origSource
		anthropicModelsAPIURL = origAPI
	}()
	anthropicSourceURL = server.URL + "/docs"
	anthropicModelsAPIURL = server.URL + "/v1/models"
	t.Setenv("ANTHROPIC_API_KEY", "")
	t.Setenv("LANGDAG_REQUIRE_ANTHROPIC_MODELS_API", "1")

	_, err := fetchAnthropicModels(context.Background())
	if err == nil {
		t.Fatal("fetchAnthropicModels error = nil, want missing API key error")
	}
}

func TestFetchAnthropicModelsAPIFailureFallsBackToHTML(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/models", func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "api unavailable", http.StatusInternalServerError)
	})
	mux.HandleFunc("/docs", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`<html><body>
<table>
<tr><th>Feature</th><th>Claude Opus 4.6</th></tr>
<tr><td><strong>Claude API ID</strong></td><td>claude-opus-4-6</td></tr>
<tr><td><strong>Claude API alias</strong></td><td>claude-opus-4-6</td></tr>
<tr><td><strong>Pricing</strong></td><td>$5 / input MTok<br/>$25 / output MTok</td></tr>
<tr><td><strong>Context window</strong></td><td>1M tokens</td></tr>
<tr><td><strong>Max output</strong></td><td>128K tokens</td></tr>
</table>
</body></html>`))
	})
	server := httptest.NewServer(mux)
	defer server.Close()

	origSource, origAPI := anthropicSourceURL, anthropicModelsAPIURL
	defer func() {
		anthropicSourceURL = origSource
		anthropicModelsAPIURL = origAPI
	}()
	anthropicSourceURL = server.URL + "/docs"
	anthropicModelsAPIURL = server.URL + "/v1/models"
	t.Setenv("ANTHROPIC_API_KEY", "test-key")
	t.Setenv("LANGDAG_REQUIRE_ANTHROPIC_MODELS_API", "")

	models, err := fetchAnthropicModels(context.Background())
	if err != nil {
		t.Fatalf("fetchAnthropicModels error: %v", err)
	}
	if len(models) != 1 || models[0].ID != "claude-opus-4-6" {
		t.Fatalf("models = %#v, want only claude-opus-4-6", models)
	}
	if models[0].ContextWindow != 1000000 || models[0].MaxOutput != 128000 {
		t.Errorf("limits = %d/%d, want 1000000/128000", models[0].ContextWindow, models[0].MaxOutput)
	}
}

func TestFetchAnthropicModelsRequiredAPIFailureDoesNotFallback(t *testing.T) {
	var docsCalls atomic.Int32
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/models", func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "api unavailable", http.StatusInternalServerError)
	})
	mux.HandleFunc("/docs", func(w http.ResponseWriter, r *http.Request) {
		docsCalls.Add(1)
		w.Write([]byte(`<html><body>
<table>
<tr><th>Feature</th><th>Claude Opus 4.6</th></tr>
<tr><td><strong>Claude API ID</strong></td><td>claude-opus-4-6</td></tr>
<tr><td><strong>Pricing</strong></td><td>$5 / input MTok<br/>$25 / output MTok</td></tr>
<tr><td><strong>Context window</strong></td><td>1M tokens</td></tr>
<tr><td><strong>Max output</strong></td><td>128K tokens</td></tr>
</table>
</body></html>`))
	})
	server := httptest.NewServer(mux)
	defer server.Close()

	origSource, origAPI := anthropicSourceURL, anthropicModelsAPIURL
	defer func() {
		anthropicSourceURL = origSource
		anthropicModelsAPIURL = origAPI
	}()
	anthropicSourceURL = server.URL + "/docs"
	anthropicModelsAPIURL = server.URL + "/v1/models"
	t.Setenv("ANTHROPIC_API_KEY", "test-key")
	t.Setenv("LANGDAG_REQUIRE_ANTHROPIC_MODELS_API", "1")

	_, err := fetchAnthropicModels(context.Background())
	if err == nil {
		t.Fatal("fetchAnthropicModels error = nil, want API failure")
	}
	if docsCalls.Load() != 0 {
		t.Fatalf("docs fallback calls = %d, want 0 when API is required", docsCalls.Load())
	}
}

func TestFetchAnthropicModelsAPIPagination(t *testing.T) {
	var seenSecondPage atomic.Bool
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/models", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch afterID := r.URL.Query().Get("after_id"); afterID {
		case "":
			if got := r.URL.Query().Get("limit"); got != "1000" {
				http.Error(w, "missing limit", http.StatusBadRequest)
				return
			}
			w.Write([]byte(`{
				"data": [
					{"id": "claude-opus-4-8", "max_input_tokens": 1000000, "max_tokens": 128000}
				],
				"has_more": true,
				"last_id": "claude-opus-4-8"
			}`))
		case "claude-opus-4-8":
			seenSecondPage.Store(true)
			w.Write([]byte(`{
				"data": [
					{"id": "claude-haiku-4-5", "max_input_tokens": 200000, "max_tokens": 64000}
				],
				"has_more": false,
				"last_id": "claude-haiku-4-5"
			}`))
		default:
			http.Error(w, "unexpected cursor", http.StatusBadRequest)
		}
	})
	server := httptest.NewServer(mux)
	defer server.Close()

	origSource, origAPI := anthropicSourceURL, anthropicModelsAPIURL
	defer func() {
		anthropicSourceURL = origSource
		anthropicModelsAPIURL = origAPI
	}()
	anthropicSourceURL = server.URL + "/docs"
	anthropicModelsAPIURL = server.URL + "/v1/models"
	t.Setenv("ANTHROPIC_API_KEY", "test-key")
	t.Setenv("LANGDAG_REQUIRE_ANTHROPIC_MODELS_API", "")

	models, err := fetchAnthropicModels(context.Background())
	if err != nil {
		t.Fatalf("fetchAnthropicModels error: %v", err)
	}
	if !seenSecondPage.Load() {
		t.Fatal("second page was not fetched")
	}
	byID := make(map[string]ModelPricing)
	for _, m := range models {
		byID[m.ID] = m
	}
	if m, ok := byID["claude-opus-4-8"]; !ok {
		t.Error("claude-opus-4-8 not found")
	} else if m.InputPricePer1M != 5 || m.OutputPricePer1M != 25 {
		t.Errorf("opus override pricing = %f/%f, want 5/25", m.InputPricePer1M, m.OutputPricePer1M)
	}
	if m, ok := byID["claude-haiku-4-5"]; !ok {
		t.Error("claude-haiku-4-5 not found")
	} else if m.InputPricePer1M != 1 || m.OutputPricePer1M != 5 {
		t.Errorf("haiku override pricing = %f/%f, want 1/5", m.InputPricePer1M, m.OutputPricePer1M)
	}
}

func TestFetchAnthropicModelsAPIEmptyLastIDGuard(t *testing.T) {
	var apiCalls atomic.Int32
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/models", func(w http.ResponseWriter, r *http.Request) {
		apiCalls.Add(1)
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{
			"data": [
				{"id": "claude-unknown-test-model", "max_input_tokens": 123000, "max_tokens": 456}
			],
			"has_more": true,
			"last_id": ""
		}`))
	})
	mux.HandleFunc("/docs", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`<html><body>
<table>
<tr><th>Feature</th><th>Claude Haiku 4.5</th></tr>
<tr><td><strong>Claude API ID</strong></td><td>claude-haiku-4-5</td></tr>
<tr><td><strong>Claude API alias</strong></td><td>claude-haiku-4-5</td></tr>
<tr><td><strong>Pricing</strong></td><td>$1 / input MTok<br/>$5 / output MTok</td></tr>
<tr><td><strong>Context window</strong></td><td>200K tokens</td></tr>
<tr><td><strong>Max output</strong></td><td>64K tokens</td></tr>
</table>
</body></html>`))
	})
	server := httptest.NewServer(mux)
	defer server.Close()

	origSource, origAPI := anthropicSourceURL, anthropicModelsAPIURL
	defer func() {
		anthropicSourceURL = origSource
		anthropicModelsAPIURL = origAPI
	}()
	anthropicSourceURL = server.URL + "/docs"
	anthropicModelsAPIURL = server.URL + "/v1/models"
	t.Setenv("ANTHROPIC_API_KEY", "test-key")
	t.Setenv("LANGDAG_REQUIRE_ANTHROPIC_MODELS_API", "")

	models, err := fetchAnthropicModels(context.Background())
	if err != nil {
		t.Fatalf("fetchAnthropicModels error: %v", err)
	}
	if apiCalls.Load() != 1 {
		t.Fatalf("API calls = %d, want 1", apiCalls.Load())
	}
	if len(models) != 1 || models[0].ID != "claude-haiku-4-5" {
		t.Fatalf("models = %#v, want HTML fallback model claude-haiku-4-5", models)
	}
	if models[0].InputPricePer1M != 1 || models[0].OutputPricePer1M != 5 {
		t.Errorf("fallback pricing = %f/%f, want 1/5", models[0].InputPricePer1M, models[0].OutputPricePer1M)
	}
}

func TestParseGrokPage(t *testing.T) {
	// Simulates the RSC payload format found in the xAI docs page
	html := `<html><script>
some stuff \"name\":\"grok-3\",\"version\":\"1.0\",\"inputModalities\":[1],\"outputModalities\":[1],\"promptTextTokenPrice\":\"$n30000\",\"completionTextTokenPrice\":\"$n150000\",\"maxPromptLength\":131072
more stuff \"name\":\"grok-3-mini\",\"version\":\"1.0\",\"promptTextTokenPrice\":\"$n3000\",\"completionTextTokenPrice\":\"$n5000\",\"maxPromptLength\":131072
\"name\":\"grok-4-fast-reasoning\",\"version\":\"1.0\",\"promptTextTokenPrice\":\"$n2000\",\"completionTextTokenPrice\":\"$n5000\",\"maxPromptLength\":2000000
\"name\":\"grok-imagine-image\",\"version\":\"1.0\",\"promptTextTokenPrice\":\"$n0\",\"completionTextTokenPrice\":\"$n0\",\"maxPromptLength\":0
duplicate \"name\":\"grok-3\",\"version\":\"1.0\",\"promptTextTokenPrice\":\"$n30000\",\"completionTextTokenPrice\":\"$n150000\",\"maxPromptLength\":131072
</script></html>`

	models, err := parseGrokPage(html)
	if err != nil {
		t.Fatalf("parseGrokPage error: %v", err)
	}

	byID := make(map[string]ModelPricing)
	for _, m := range models {
		byID[m.ID] = m
	}

	// grok-3: $n30000/10000 = $3.00, $n150000/10000 = $15.00
	if m, ok := byID["grok-3"]; !ok {
		t.Error("grok-3 not found")
	} else {
		if m.InputPricePer1M != 3.0 {
			t.Errorf("grok-3 input = %f, want 3.0", m.InputPricePer1M)
		}
		if m.OutputPricePer1M != 15.0 {
			t.Errorf("grok-3 output = %f, want 15.0", m.OutputPricePer1M)
		}
		if m.ContextWindow != 131072 {
			t.Errorf("grok-3 context = %d, want 131072", m.ContextWindow)
		}
	}

	// grok-4-fast-reasoning: $n2000/10000 = $0.20
	if m, ok := byID["grok-4-fast-reasoning"]; !ok {
		t.Error("grok-4-fast-reasoning not found")
	} else {
		if m.InputPricePer1M != 0.2 {
			t.Errorf("input = %f, want 0.2", m.InputPricePer1M)
		}
		if m.ContextWindow != 2000000 {
			t.Errorf("context = %d, want 2000000", m.ContextWindow)
		}
	}

	// grok-imagine-image should be filtered out
	if _, ok := byID["grok-imagine-image"]; ok {
		t.Error("image model should be filtered out")
	}

	// Duplicates should be deduplicated
	if len(models) != 3 {
		t.Errorf("got %d models, want 3 (deduplicated)", len(models))
	}
}

func TestParseGeminiHTML(t *testing.T) {
	html := `<html><body>
<h3>Gemini 3.1 Pro Preview</h3>
<p>Input price $2.00</p>
<p>Output price $12.00</p>

<h3>Gemini 3 Flash Preview</h3>
<p>Input price $0.50</p>
<p>Output price $3.00</p>

<h3>Gemini 3.1 Flash-Lite Preview</h3>
<p>Input price $0.25</p>
<p>Output price $1.50</p>

<h3>Gemini 3 Flash Image</h3>
<p>Output price $30.00</p>
</body></html>`

	models, err := parseGeminiHTML(html)
	if err != nil {
		t.Fatalf("parseGeminiHTML error: %v", err)
	}

	byID := make(map[string]ModelPricing)
	for _, m := range models {
		byID[m.ID] = m
	}

	if m, ok := byID["gemini-3.1-pro-preview"]; !ok {
		t.Error("gemini-3.1-pro-preview not found")
	} else {
		if m.InputPricePer1M != 2.0 {
			t.Errorf("input = %f, want 2.0", m.InputPricePer1M)
		}
		if m.OutputPricePer1M != 12.0 {
			t.Errorf("output = %f, want 12.0", m.OutputPricePer1M)
		}
	}

	if m, ok := byID["gemini-3-flash-preview"]; !ok {
		t.Error("gemini-3-flash-preview not found")
	} else {
		if m.InputPricePer1M != 0.50 {
			t.Errorf("input = %f, want 0.50", m.InputPricePer1M)
		}
		if m.OutputPricePer1M != 3.0 {
			t.Errorf("output = %f, want 3.0", m.OutputPricePer1M)
		}
	}

	if m, ok := byID["gemini-3.1-flash-lite-preview"]; !ok {
		t.Error("gemini-3.1-flash-lite-preview not found")
	} else {
		if m.InputPricePer1M != 0.25 {
			t.Errorf("input = %f, want 0.25", m.InputPricePer1M)
		}
	}

	// Image model should be skipped
	if _, ok := byID["gemini-3-flash-image"]; ok {
		t.Error("image model should be filtered")
	}
}

func TestGemmaHardcodedModelsAreFreeWithoutServerTools(t *testing.T) {
	models := gemmaHardcodedModels()
	if len(models) == 0 {
		t.Fatal("gemmaHardcodedModels returned no models")
	}
	for _, model := range models {
		if !model.Free {
			t.Errorf("%s Free = false, want true", model.ID)
		}
		if model.InputPricePer1M != 0 || model.OutputPricePer1M != 0 {
			t.Errorf("%s pricing = %f/%f, want zero free pricing", model.ID, model.InputPricePer1M, model.OutputPricePer1M)
		}
		if len(model.ServerTools) != 0 {
			t.Errorf("%s ServerTools = %+v, want none", model.ID, model.ServerTools)
		}
		if model.ContextWindow == 0 || model.MaxOutput == 0 {
			t.Errorf("%s context/max_output = %d/%d, want populated metadata", model.ID, model.ContextWindow, model.MaxOutput)
		}
	}
}

func TestParseGeminiSpecPage(t *testing.T) {
	html := `<html><body>
<dt>Model code</dt><dd>gemini-3-flash-preview</dd>
<dt>Input token limit</dt><dd>1,048,576</dd>
<dt>Output token limit</dt><dd>65,536</dd>
</body></html>`

	ctx, maxOut := parseGeminiSpecPage(html)
	if ctx != 1048576 {
		t.Errorf("contextWindow = %d, want 1048576", ctx)
	}
	if maxOut != 65536 {
		t.Errorf("maxOutput = %d, want 65536", maxOut)
	}
}

func TestParseGeminiDeprecations(t *testing.T) {
	html := `<html><body>
<table>
<tr><th>Model</th><th>Release date</th><th>Shutdown date</th><th>Replacement</th></tr>
<tr><td>gemini-3-flash-preview</td><td>December 17, 2025</td><td>No shutdown date announced</td><td>-</td></tr>
<tr><td>gemini-3.1-pro-preview</td><td>February 19, 2026</td><td>No shutdown date announced</td><td>-</td></tr>
<tr><td>gemini-2.5-pro</td><td>June 17, 2025</td><td>June 17, 2026</td><td>gemini-3.1-pro-preview</td></tr>
<tr><td>gemini-2.5-flash</td><td>June 17, 2025</td><td>June 17, 2026</td><td>gemini-3-flash-preview</td></tr>
<tr><td>gemini-2.0-flash</td><td>February 5, 2025</td><td>June 1, 2026</td><td>gemini-3-flash-preview</td></tr>
</table>
</body></html>`

	shutdown := parseGeminiDeprecations(html)

	wantDates := map[string]string{
		"gemini-2.5-pro":   "2026-06-17",
		"gemini-2.5-flash": "2026-06-17",
		"gemini-2.0-flash": "2026-06-01",
	}
	for id, want := range wantDates {
		if got, ok := shutdown[id]; !ok {
			t.Errorf("expected %s to have shutdown date", id)
		} else if got.Format("2006-01-02") != want {
			t.Errorf("%s shutdown = %s, want %s", id, got.Format("2006-01-02"), want)
		}
	}

	// Models with "No shutdown date announced" should NOT be in the set
	for _, id := range []string{"gemini-3-flash-preview", "gemini-3.1-pro-preview"} {
		if _, ok := shutdown[id]; ok {
			t.Errorf("expected %s to NOT have shutdown date", id)
		}
	}
}

func TestFilterGeminiModelsByShutdownKeepsFutureDates(t *testing.T) {
	models := []ModelPricing{
		{ID: "gemini-active"},
		{ID: "gemini-shuts-down-tomorrow"},
		{ID: "gemini-shut-down-yesterday"},
	}
	shutdown := map[string]time.Time{
		"gemini-shuts-down-tomorrow": time.Date(2026, 5, 20, 0, 0, 0, 0, time.UTC),
		"gemini-shut-down-yesterday": time.Date(2026, 5, 18, 0, 0, 0, 0, time.UTC),
	}

	filtered := filterGeminiModelsByShutdown(models, shutdown, time.Date(2026, 5, 19, 0, 0, 0, 0, time.UTC))
	byID := map[string]bool{}
	for _, model := range filtered {
		byID[model.ID] = true
	}

	if !byID["gemini-active"] {
		t.Fatal("undated model was filtered out")
	}
	if !byID["gemini-shuts-down-tomorrow"] {
		t.Fatal("future shutdown model was filtered out")
	}
	if byID["gemini-shut-down-yesterday"] {
		t.Fatal("past shutdown model was retained")
	}
}

func TestParseGeminiDeprecations_Empty(t *testing.T) {
	shutdown := parseGeminiDeprecations("<html><body>No tables here</body></html>")
	if len(shutdown) != 0 {
		t.Errorf("expected empty set, got %d entries", len(shutdown))
	}
}

func TestParseGeminiSpecPage_NoData(t *testing.T) {
	ctx, maxOut := parseGeminiSpecPage("<html><body>No specs here</body></html>")
	if ctx != 0 || maxOut != 0 {
		t.Errorf("expected zeros, got ctx=%d, maxOut=%d", ctx, maxOut)
	}
}

func TestParseCommaNumber(t *testing.T) {
	tests := []struct {
		input string
		want  int
	}{
		{"1,048,576", 1048576},
		{"65,536", 65536},
		{"128000", 128000},
		{"0", 0},
	}
	for _, tt := range tests {
		if got := parseCommaNumber(tt.input); got != tt.want {
			t.Errorf("parseCommaNumber(%q) = %d, want %d", tt.input, got, tt.want)
		}
	}
}

func TestParseHelpers(t *testing.T) {
	t.Run("parseTokenCount", func(t *testing.T) {
		tests := []struct {
			input string
			want  int
		}{
			{"200K tokens", 200000},
			{"1M tokens", 1000000},
			{"128K tokens", 128000},
			{"64K tokens", 64000},
			{"no match", 0},
		}
		for _, tt := range tests {
			if got := parseTokenCount(tt.input); got != tt.want {
				t.Errorf("parseTokenCount(%q) = %d, want %d", tt.input, got, tt.want)
			}
		}
	})

	t.Run("parseSizeStr", func(t *testing.T) {
		tests := []struct {
			input string
			want  int
		}{
			{"2M", 2000000},
			{"256K", 256000},
			{"131K", 131000},
			{"", 0},
		}
		for _, tt := range tests {
			if got := parseSizeStr(tt.input); got != tt.want {
				t.Errorf("parseSizeStr(%q) = %d, want %d", tt.input, got, tt.want)
			}
		}
	})

	t.Run("parseDollarAmount", func(t *testing.T) {
		tests := []struct {
			input string
			want  float64
		}{
			{"$3.00", 3.0},
			{"$0.20", 0.2},
			{"$15.00", 15.0},
			{"no price", 0},
		}
		for _, tt := range tests {
			if got := parseDollarAmount(tt.input); got != tt.want {
				t.Errorf("parseDollarAmount(%q) = %f, want %f", tt.input, got, tt.want)
			}
		}
	})

	t.Run("stripHTMLTags", func(t *testing.T) {
		input := `<strong>Bold</strong> and <a href="x">link</a>`
		got := stripHTMLTags(input)
		if got != "Bold and link" {
			t.Errorf("stripHTMLTags = %q, want %q", got, "Bold and link")
		}
	})
}
