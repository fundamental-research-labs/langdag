package models

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestDefaultCatalog(t *testing.T) {
	catalog, err := DefaultCatalog()
	if err != nil {
		t.Fatalf("DefaultCatalog() error: %v", err)
	}

	if catalog.SchemaVersion != CatalogV1SchemaVersion {
		t.Errorf("SchemaVersion = %q, want %q", catalog.SchemaVersion, CatalogV1SchemaVersion)
	}
	if catalog.GeneratedAt.IsZero() {
		t.Error("GeneratedAt is zero")
	}

	expectedProviders := []string{"anthropic", "openai", "gemini", "grok"}
	for _, p := range expectedProviders {
		models := catalog.ForProvider(p)
		if len(models) == 0 {
			t.Errorf("provider %q has no models", p)
		}
	}
}

func TestDefaultCatalog_KnownModels(t *testing.T) {
	catalog, err := DefaultCatalog()
	if err != nil {
		t.Fatalf("DefaultCatalog() error: %v", err)
	}

	tests := []struct {
		modelID      string
		wantProvider string
		wantInputGt0 bool
		wantCtxGt0   bool
	}{
		{"gpt-4o-2024-08-06", "openai", true, true},
		{"grok-4.3", "grok", true, true},
	}

	for _, tt := range tests {
		t.Run(tt.modelID, func(t *testing.T) {
			m, provider, ok := catalog.LookupModel(tt.modelID)
			if !ok {
				t.Fatalf("model %q not found in catalog", tt.modelID)
			}
			if provider != tt.wantProvider {
				t.Errorf("provider = %q, want %q", provider, tt.wantProvider)
			}
			if tt.wantInputGt0 && m.InputPricePer1M <= 0 {
				t.Errorf("InputPricePer1M = %f, want > 0", m.InputPricePer1M)
			}
			if tt.wantCtxGt0 && m.ContextWindow <= 0 {
				t.Errorf("ContextWindow = %d, want > 0", m.ContextWindow)
			}
		})
	}
}

func TestLookupModel_NotFound(t *testing.T) {
	catalog, err := DefaultCatalog()
	if err != nil {
		t.Fatalf("DefaultCatalog() error: %v", err)
	}

	_, _, ok := catalog.LookupModel("nonexistent-model")
	if ok {
		t.Error("expected LookupModel to return false for nonexistent model")
	}
}

func TestForProvider_Unknown(t *testing.T) {
	catalog, err := DefaultCatalog()
	if err != nil {
		t.Fatalf("DefaultCatalog() error: %v", err)
	}

	models := catalog.ForProvider("unknown-provider")
	if models != nil {
		t.Errorf("expected nil for unknown provider, got %d models", len(models))
	}
}

func TestForProvider_NilCatalog(t *testing.T) {
	var catalog *Catalog
	models := catalog.ForProvider("anthropic")
	if models != nil {
		t.Error("expected nil from nil catalog")
	}
}

func TestLoadCatalog_FallsBackToDefault(t *testing.T) {
	catalog, err := LoadCatalog("/nonexistent/path/catalog.json")
	if err != nil {
		t.Fatalf("LoadCatalog() error: %v", err)
	}

	if catalog.SchemaVersion != CatalogV1SchemaVersion {
		t.Errorf("SchemaVersion = %q, want %q", catalog.SchemaVersion, CatalogV1SchemaVersion)
	}
}

func TestLoadCatalog_EmptyPath(t *testing.T) {
	catalog, err := LoadCatalog("")
	if err != nil {
		t.Fatalf("LoadCatalog(\"\") error: %v", err)
	}

	if len(catalog.Offerings) == 0 {
		t.Error("expected non-empty offerings from default catalog")
	}
}

func TestSaveAndLoadCatalog(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "catalog.json")

	original, err := DefaultCatalog()
	if err != nil {
		t.Fatalf("DefaultCatalog() error: %v", err)
	}

	if err := SaveCatalog(original, path); err != nil {
		t.Fatalf("SaveCatalog() error: %v", err)
	}

	loaded, err := LoadCatalog(path)
	if err != nil {
		t.Fatalf("LoadCatalog() error: %v", err)
	}

	if !loaded.GeneratedAt.Equal(original.GeneratedAt) {
		t.Errorf("GeneratedAt = %s, want %s", loaded.GeneratedAt, original.GeneratedAt)
	}

	if len(loaded.Offerings) != len(original.Offerings) {
		t.Errorf("offerings: got %d, want %d", len(loaded.Offerings), len(original.Offerings))
	}
}

func TestLoadCatalog_PrefersCache(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "catalog.json")

	custom := ReferenceCatalogV1()
	custom.GeneratedAt = custom.GeneratedAt.Add(24 * time.Hour)
	custom.StaleAfter = custom.GeneratedAt.Add(30 * 24 * time.Hour)
	custom.Provenance = &ProvenanceV1{Source: "custom-cache", ObservedAt: custom.GeneratedAt}

	if err := SaveCatalog(custom, path); err != nil {
		t.Fatalf("SaveCatalog() error: %v", err)
	}

	loaded, err := LoadCatalog(path)
	if err != nil {
		t.Fatalf("LoadCatalog() error: %v", err)
	}

	if loaded.Provenance == nil || loaded.Provenance.Source != "custom-cache" {
		t.Errorf("Provenance = %+v, want custom-cache (cache should take precedence)", loaded.Provenance)
	}
}

func TestLoadRuntimeCatalogUsesEmbeddedByDefault(t *testing.T) {
	result, err := LoadRuntimeCatalog(CatalogLoadOptions{})
	if err != nil {
		t.Fatalf("LoadRuntimeCatalog() error: %v", err)
	}
	if result.Source != CatalogSourceEmbedded {
		t.Fatalf("Source = %q, want embedded", result.Source)
	}
	if result.CachePath != "" {
		t.Fatalf("CachePath = %q, want empty", result.CachePath)
	}
	if result.Catalog == nil {
		t.Fatal("Catalog is nil")
	}
}

func TestLoadRuntimeCatalogIgnoresImplicitUserCache(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	cachePath := filepath.Join(home, ".config", "langdag", "model_catalog.json")

	custom := ReferenceCatalogV1()
	custom.GeneratedAt = custom.GeneratedAt.Add(24 * time.Hour)
	custom.StaleAfter = custom.GeneratedAt.Add(30 * 24 * time.Hour)
	custom.Provenance = &ProvenanceV1{Source: "implicit-user-cache", ObservedAt: custom.GeneratedAt}
	if err := SaveCatalog(custom, cachePath); err != nil {
		t.Fatalf("SaveCatalog user cache: %v", err)
	}

	result, err := LoadRuntimeCatalog(CatalogLoadOptions{})
	if err != nil {
		t.Fatalf("LoadRuntimeCatalog() error: %v", err)
	}
	if result.Source != CatalogSourceEmbedded {
		t.Fatalf("Source = %q, want embedded", result.Source)
	}
	if result.Catalog == nil {
		t.Fatal("Catalog is nil")
	}
	if result.Catalog.Provenance != nil && result.Catalog.Provenance.Source == "implicit-user-cache" {
		t.Fatal("runtime catalog loaded implicit ~/.config/langdag/model_catalog.json cache")
	}
}

func TestLoadCatalog_InvalidCacheFallsBack(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "catalog.json")

	if err := os.WriteFile(path, []byte("invalid json"), 0644); err != nil {
		t.Fatal(err)
	}

	catalog, err := LoadCatalog(path)
	if err != nil {
		t.Fatalf("LoadCatalog() error: %v", err)
	}

	if catalog.SchemaVersion != CatalogV1SchemaVersion {
		t.Errorf("SchemaVersion = %q, want %q (should fall back to default)", catalog.SchemaVersion, CatalogV1SchemaVersion)
	}
}

func TestLoadCatalogWithOptionsReportsStaleCache(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "catalog.json")
	catalog := ReferenceCatalogV1()
	catalog.GeneratedAt = mustParseTime(t, "2026-01-01T00:00:00Z")
	catalog.StaleAfter = mustParseTime(t, "2026-01-02T00:00:00Z")
	if err := SaveCatalog(catalog, path); err != nil {
		t.Fatalf("SaveCatalog: %v", err)
	}

	result, err := LoadCatalogWithOptions(CatalogLoadOptions{
		CachePath: path,
		Now:       func() time.Time { return mustParseTime(t, "2026-01-03T00:00:00Z") },
	})
	if err != nil {
		t.Fatalf("LoadCatalogWithOptions: %v", err)
	}
	if result.Source != CatalogSourceCache {
		t.Fatalf("Source = %q, want cache", result.Source)
	}
	if !hasDiagnostic(result.Diagnostics, "stale_catalog") {
		t.Fatalf("diagnostics = %+v, want stale_catalog", result.Diagnostics)
	}
}

func TestLoadCatalogWithOptionsFallsBackToEmbeddedWithDiagnostics(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "catalog.json")
	if err := os.WriteFile(path, []byte(`{"schema_version":"model-catalog/v1","unknown":true}`), 0644); err != nil {
		t.Fatal(err)
	}

	result, err := LoadCatalogWithOptions(CatalogLoadOptions{CachePath: path})
	if err != nil {
		t.Fatalf("LoadCatalogWithOptions: %v", err)
	}
	if result.Source != CatalogSourceEmbedded {
		t.Fatalf("Source = %q, want embedded", result.Source)
	}
	if result.Catalog == nil {
		t.Fatal("Catalog is nil")
	}
	if !hasDiagnostic(result.Diagnostics, "cache_invalid") {
		t.Fatalf("diagnostics = %+v, want cache_invalid", result.Diagnostics)
	}
}

func TestRefreshCatalogCacheRemoteSuccessReplacesCache(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "catalog.json")
	oldCatalog := ReferenceCatalogV1()
	oldCatalog.Provenance = &ProvenanceV1{Source: "old-cache", ObservedAt: oldCatalog.GeneratedAt}
	if err := SaveCatalog(oldCatalog, path); err != nil {
		t.Fatalf("SaveCatalog old: %v", err)
	}

	remoteCatalog := ReferenceCatalogV1()
	remoteCatalog.GeneratedAt = mustParseTime(t, "2026-05-20T00:00:00Z")
	remoteCatalog.StaleAfter = mustParseTime(t, "2026-06-20T00:00:00Z")
	remoteCatalog.Provenance = &ProvenanceV1{Source: "remote-test", ObservedAt: remoteCatalog.GeneratedAt}
	server := serveCatalog(t, remoteCatalog)
	defer server.Close()

	result, err := RefreshCatalogCache(context.Background(), CatalogRefreshOptions{
		CachePath: path,
		Endpoint:  server.URL,
		Timeout:   time.Second,
		Now:       func() time.Time { return mustParseTime(t, "2026-05-21T00:00:00Z") },
	})
	if err != nil {
		t.Fatalf("RefreshCatalogCache: %v", err)
	}
	if result.Source != CatalogSourceRemote || !result.ReplacedCache || result.Catalog == nil {
		t.Fatalf("refresh result = %+v, want remote replacement", result)
	}
	loaded, err := LoadCatalog(path)
	if err != nil {
		t.Fatalf("LoadCatalog refreshed cache: %v", err)
	}
	if loaded.Provenance == nil || loaded.Provenance.Source != "remote-test" {
		t.Fatalf("cache provenance = %+v, want remote-test", loaded.Provenance)
	}
}

func TestRefreshCatalogCacheRemoteSuccessWithoutCachePath(t *testing.T) {
	remoteCatalog := ReferenceCatalogV1()
	remoteCatalog.GeneratedAt = mustParseTime(t, "2026-05-20T00:00:00Z")
	remoteCatalog.StaleAfter = mustParseTime(t, "2026-06-20T00:00:00Z")
	remoteCatalog.Provenance = &ProvenanceV1{Source: "remote-test", ObservedAt: remoteCatalog.GeneratedAt}
	server := serveCatalog(t, remoteCatalog)
	defer server.Close()

	result, err := RefreshCatalogCache(context.Background(), CatalogRefreshOptions{
		Endpoint: server.URL,
		Timeout:  time.Second,
		Now:      func() time.Time { return mustParseTime(t, "2026-05-21T00:00:00Z") },
	})
	if err != nil {
		t.Fatalf("RefreshCatalogCache: %v", err)
	}
	if result.Source != CatalogSourceRemote || result.ReplacedCache {
		t.Fatalf("refresh result = %+v, want remote catalog without cache replacement", result)
	}
	if result.CachePath != "" {
		t.Fatalf("CachePath = %q, want empty", result.CachePath)
	}
	if result.Catalog == nil || result.Catalog.Provenance == nil || result.Catalog.Provenance.Source != "remote-test" {
		t.Fatalf("remote catalog provenance = %+v, want remote-test", result.Catalog)
	}
}

func TestLoadRemoteCatalogDoesNotWriteCache(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "catalog.json")
	remoteCatalog := ReferenceCatalogV1()
	remoteCatalog.GeneratedAt = mustParseTime(t, "2026-05-20T00:00:00Z")
	remoteCatalog.StaleAfter = mustParseTime(t, "2026-06-20T00:00:00Z")
	server := serveCatalog(t, remoteCatalog)
	defer server.Close()

	result, err := LoadRemoteCatalog(context.Background(), CatalogRefreshOptions{
		CachePath: path,
		Endpoint:  server.URL,
		Timeout:   time.Second,
		Now:       func() time.Time { return mustParseTime(t, "2026-05-21T00:00:00Z") },
	})
	if err != nil {
		t.Fatalf("LoadRemoteCatalog: %v", err)
	}
	if result.Source != CatalogSourceRemote || result.ReplacedCache || result.Catalog == nil {
		t.Fatalf("remote result = %+v, want remote without cache replacement", result)
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatalf("cache file exists after LoadRemoteCatalog: %v", err)
	}
}

func TestRefreshCatalogCacheOptOutDoesNotFetch(t *testing.T) {
	result, err := RefreshCatalogCache(context.Background(), CatalogRefreshOptions{
		Endpoint: "http://127.0.0.1:1/catalog.json",
		Disabled: true,
	})
	if err != nil {
		t.Fatalf("RefreshCatalogCache disabled: %v", err)
	}
	if result.Catalog != nil || result.ReplacedCache {
		t.Fatalf("disabled result = %+v, want no catalog replacement", result)
	}
	if !hasDiagnostic(result.Diagnostics, "refresh_disabled") {
		t.Fatalf("diagnostics = %+v, want refresh_disabled", result.Diagnostics)
	}
}

func TestRefreshCatalogCacheTimeoutKeepsExistingCache(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "catalog.json")
	if err := SaveCatalog(ReferenceCatalogV1(), path); err != nil {
		t.Fatalf("SaveCatalog: %v", err)
	}
	before, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(100 * time.Millisecond)
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	_, err = RefreshCatalogCache(context.Background(), CatalogRefreshOptions{
		CachePath: path,
		Endpoint:  server.URL,
		Timeout:   10 * time.Millisecond,
	})
	if err == nil {
		t.Fatal("RefreshCatalogCache succeeded, want timeout error")
	}
	after, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(after) != string(before) {
		t.Fatal("cache changed after timeout")
	}
}

func TestRefreshCatalogCacheInvalidRemoteKeepsExistingCache(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "catalog.json")
	if err := SaveCatalog(ReferenceCatalogV1(), path); err != nil {
		t.Fatalf("SaveCatalog: %v", err)
	}
	before, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`{"schema_version":"model-catalog/v1","unknown":true}`))
	}))
	defer server.Close()

	_, err = RefreshCatalogCache(context.Background(), CatalogRefreshOptions{
		CachePath: path,
		Endpoint:  server.URL,
		Timeout:   time.Second,
	})
	if err == nil || !strings.Contains(err.Error(), "invalid remote catalog") {
		t.Fatalf("err = %v, want invalid remote catalog", err)
	}
	after, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(after) != string(before) {
		t.Fatal("cache changed after invalid remote catalog")
	}
}

func TestRefreshCatalogCacheRejectsLegacyRemoteShape(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "catalog.json")
	if err := SaveCatalog(ReferenceCatalogV1(), path); err != nil {
		t.Fatalf("SaveCatalog: %v", err)
	}
	before, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`{"updated_at":"2026-05-01T00:00:00Z","source":"legacy","providers":{}}`))
	}))
	defer server.Close()

	_, err = RefreshCatalogCache(context.Background(), CatalogRefreshOptions{
		CachePath: path,
		Endpoint:  server.URL,
		Timeout:   time.Second,
	})
	if err == nil || !strings.Contains(err.Error(), "schema_version") {
		t.Fatalf("err = %v, want schema_version rejection", err)
	}
	after, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(after) != string(before) {
		t.Fatal("cache changed after legacy remote catalog")
	}
}

func TestRefreshCatalogCacheRejectsEmptyRemoteObject(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "catalog.json")
	if err := SaveCatalog(ReferenceCatalogV1(), path); err != nil {
		t.Fatalf("SaveCatalog: %v", err)
	}
	before, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`{}`))
	}))
	defer server.Close()

	_, err = RefreshCatalogCache(context.Background(), CatalogRefreshOptions{
		CachePath: path,
		Endpoint:  server.URL,
		Timeout:   time.Second,
	})
	if err == nil || !strings.Contains(err.Error(), "schema_version") {
		t.Fatalf("err = %v, want schema_version rejection", err)
	}
	after, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(after) != string(before) {
		t.Fatal("cache changed after empty remote catalog")
	}
}

func TestRefreshCatalogCacheStaleRemoteKeepsExistingCache(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "catalog.json")
	if err := SaveCatalog(ReferenceCatalogV1(), path); err != nil {
		t.Fatalf("SaveCatalog: %v", err)
	}
	before, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	remoteCatalog := ReferenceCatalogV1()
	remoteCatalog.GeneratedAt = mustParseTime(t, "2026-01-01T00:00:00Z")
	remoteCatalog.StaleAfter = mustParseTime(t, "2026-01-02T00:00:00Z")
	server := serveCatalog(t, remoteCatalog)
	defer server.Close()

	result, err := RefreshCatalogCache(context.Background(), CatalogRefreshOptions{
		CachePath: path,
		Endpoint:  server.URL,
		Timeout:   time.Second,
		Now:       func() time.Time { return mustParseTime(t, "2026-01-03T00:00:00Z") },
	})
	if err == nil {
		t.Fatal("RefreshCatalogCache succeeded, want stale remote error")
	}
	if !hasDiagnostic(result.Diagnostics, "stale_catalog") {
		t.Fatalf("diagnostics = %+v, want stale_catalog", result.Diagnostics)
	}
	after, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(after) != string(before) {
		t.Fatal("cache changed after stale remote catalog")
	}
}

func TestRefreshCatalogCachePartialRemoteKeepsExistingCache(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "catalog.json")
	if err := SaveCatalog(ReferenceCatalogV1(), path); err != nil {
		t.Fatalf("SaveCatalog: %v", err)
	}
	before, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	partial := ReferenceCatalogV1()
	partial.Offerings = append(partial.Offerings, ModelOfferingV1{
		ID:               "openai-direct:not-in-model-map",
		CanonicalModelID: "openai/not-in-model-map",
		DeploymentID:     "openai-direct",
		NativeModelID:    "not-in-model-map",
		Pricing: PricingV1{
			Status:      PricingKnown,
			Currency:    "USD",
			RatesPer1M:  map[string]float64{"input_tokens": 1, "output_tokens": 1},
			EffectiveAt: partial.GeneratedAt,
		},
	})
	server := serveCatalogWithoutValidation(t, partial)
	defer server.Close()

	_, err = RefreshCatalogCache(context.Background(), CatalogRefreshOptions{
		CachePath: path,
		Endpoint:  server.URL,
		Timeout:   time.Second,
	})
	if err == nil {
		t.Fatal("RefreshCatalogCache succeeded, want invalid partial remote error")
	}
	after, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(after) != string(before) {
		t.Fatal("cache changed after partial remote catalog")
	}
}

func TestCatalogRefreshOptionsFromEnv(t *testing.T) {
	t.Setenv(envCatalogRefresh, "false")
	t.Setenv(envCatalogURL, "http://example.test/catalog.json")
	t.Setenv(envCatalogTimeout, "750ms")

	opts := CatalogRefreshOptionsFromEnv("/tmp/catalog.json")
	if !opts.Disabled {
		t.Fatal("Disabled = false, want true")
	}
	if opts.Endpoint != "http://example.test/catalog.json" {
		t.Fatalf("Endpoint = %q", opts.Endpoint)
	}
	if opts.Timeout != 750*time.Millisecond {
		t.Fatalf("Timeout = %s", opts.Timeout)
	}
	if opts.CachePath != "/tmp/catalog.json" {
		t.Fatalf("CachePath = %q", opts.CachePath)
	}
}

func hasDiagnostic(diagnostics []CatalogDiagnosticV1, code string) bool {
	for _, diagnostic := range diagnostics {
		if diagnostic.Code == code {
			return true
		}
	}
	return false
}

func serveCatalog(t *testing.T, catalog *Catalog) *httptest.Server {
	t.Helper()
	if err := ValidateCatalogV1(catalog); err != nil {
		t.Fatalf("test catalog is invalid: %v", err)
	}
	return serveCatalogWithoutValidation(t, catalog)
}

func serveCatalogWithoutValidation(t *testing.T, catalog *Catalog) *httptest.Server {
	t.Helper()
	data, err := json.Marshal(catalog)
	if err != nil {
		t.Fatalf("marshal catalog: %v", err)
	}
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write(data)
	}))
}

func TestFetchLatest(t *testing.T) {
	t.Setenv("ANTHROPIC_API_KEY", "")
	t.Setenv("LANGDAG_REQUIRE_ANTHROPIC_MODELS_API", "")

	openAIServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`# Models

### gpt-4o-2024-08-06

- Context window size: 128000
- Maximum output tokens: 16384

## Text tokens

| Name | Input | Cached input | Output | Unit |
| --- | --- | --- | --- | --- |
| gpt-4o | 2.5 | 1.25 | 10 | 1M tokens |
| gpt-4o-2024-08-06 | 2.5 | 1.25 | 10 | 1M tokens |
`))
	}))
	defer openAIServer.Close()

	anthropicServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`<table>
<tr><th>Feature</th><th>Model</th></tr>
<tr><td>Claude API ID</td><td>claude-sonnet-4-6</td></tr>
<tr><td>Pricing</td><td>$3 / input MTok $15 / output MTok</td></tr>
<tr><td>Context window</td><td>200K tokens</td></tr>
<tr><td>Max output</td><td>64K tokens</td></tr>
</table>`))
	}))
	defer anthropicServer.Close()

	// Gemini: pricing page + spec page
	geminiMux := http.NewServeMux()
	geminiMux.HandleFunc("/pricing", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`<h3>Gemini 3 Flash Preview</h3><p>Input price $0.50</p><p>Output price $3.00</p>`))
	})
	geminiMux.HandleFunc("/models/gemini-3-flash-preview", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`<html>Input token limit 1,048,576 Output token limit 65,536</html>`))
	})
	geminiServer := httptest.NewServer(geminiMux)
	defer geminiServer.Close()

	grokServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`\"name\":\"grok-3\",\"promptTextTokenPrice\":\"$n30000\",\"completionTextTokenPrice\":\"$n150000\",\"maxPromptLength\":131072`))
	}))
	defer grokServer.Close()

	origOpenAI, origAnthropic, origGemini, origGeminiSpec, origGrok := openAISourceURL, anthropicSourceURL, geminiSourceURL, geminiSpecBaseURL, grokSourceURL
	defer func() {
		openAISourceURL = origOpenAI
		anthropicSourceURL = origAnthropic
		geminiSourceURL = origGemini
		geminiSpecBaseURL = origGeminiSpec
		grokSourceURL = origGrok
	}()
	openAISourceURL = openAIServer.URL
	anthropicSourceURL = anthropicServer.URL
	geminiSourceURL = geminiServer.URL + "/pricing"
	geminiSpecBaseURL = geminiServer.URL + "/models"
	grokSourceURL = grokServer.URL

	catalog, err := FetchLatest(context.Background())
	if err != nil {
		t.Fatalf("FetchLatest() error: %v", err)
	}

	if catalog.SchemaVersion != CatalogV1SchemaVersion {
		t.Errorf("SchemaVersion = %q, want %q", catalog.SchemaVersion, CatalogV1SchemaVersion)
	}

	// Check at least one model per provider
	for _, p := range []string{"openai", "anthropic", "gemini", "grok"} {
		if len(catalog.ForProvider(p)) == 0 {
			t.Errorf("no models for provider %q", p)
		}
	}

	// gpt-4o-2024-08-06 should have full data (pricing + context from same file)
	m, _, ok := catalog.LookupModel("gpt-4o-2024-08-06")
	if !ok {
		t.Fatal("gpt-4o-2024-08-06 not found")
	}
	if m.InputPricePer1M != 2.5 {
		t.Errorf("gpt-4o input = %f, want 2.5", m.InputPricePer1M)
	}
	if m.ContextWindow != 128000 {
		t.Errorf("gpt-4o context = %d, want 128000", m.ContextWindow)
	}

	// gpt-4o has pricing but no context window → should be filtered out
	_, _, ok = catalog.LookupModel("gpt-4o")
	if ok {
		t.Error("gpt-4o should be filtered out (no context window)")
	}

	// Gemini model should have spec data from spec page
	gm, _, ok := catalog.LookupModel("gemini-3-flash-preview")
	if !ok {
		t.Fatal("gemini-3-flash-preview not found")
	}
	if gm.ContextWindow != 1048576 {
		t.Errorf("gemini context = %d, want 1048576", gm.ContextWindow)
	}
	if gm.MaxOutput != 65536 {
		t.Errorf("gemini maxOutput = %d, want 65536", gm.MaxOutput)
	}

	compiled, err := CompileCatalogV1(catalog)
	if err != nil {
		t.Fatalf("CompileCatalogV1: %v", err)
	}
	bedrockOffering, ok := compiled.OfferingForDeployment("anthropic/claude-sonnet-4-6", "anthropic-bedrock")
	if !ok {
		t.Fatal("FetchLatest should generate an Anthropic Bedrock offering")
	}
	if bedrockOffering.Pricing.Status != PricingUnknown {
		t.Fatalf("generated Bedrock pricing status = %q, want unknown placeholder", bedrockOffering.Pricing.Status)
	}
	if bedrockOffering.Capabilities.ServerTools["web_search"] != CapabilityUnknown {
		t.Fatalf("generated Bedrock web_search capability = %q, want unknown", bedrockOffering.Capabilities.ServerTools["web_search"])
	}
	if _, ok := compiled.OfferingForDeployment("anthropic/claude-sonnet-4-6", "anthropic-vertex"); !ok {
		t.Fatal("FetchLatest should generate an Anthropic Vertex offering")
	}
	if _, ok := compiled.OfferingForDeployment("google/gemini-3-flash-preview", "gemini-vertex"); !ok {
		t.Fatal("FetchLatest should generate a Gemini Vertex offering")
	}
	if len(compiled.OfferingTemplatesByDeployment["openai-azure"]) == 0 {
		t.Fatal("FetchLatest should generate Azure mapping-required templates")
	}
	if len(compiled.OfferingsByDeployment["openrouter"]) == 0 {
		t.Fatal("FetchLatest should include an OpenRouter placeholder offering")
	}
	if len(compiled.OfferingsByDeployment["ollama-local"]) == 0 {
		t.Fatal("FetchLatest should include an Ollama local placeholder offering")
	}
}

func TestFetchLatest_KeepsAnthropicAPIModelWithUnknownPricing(t *testing.T) {
	t.Setenv("ANTHROPIC_API_KEY", "test-key")

	openAIServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`### gpt-4o-2024-08-06

- Context window size: 128000
- Maximum output tokens: 16384

| Name | Input | Cached input | Output | Unit |
| --- | --- | --- | --- | --- |
| gpt-4o-2024-08-06 | 2.5 | 1.25 | 10 | 1M tokens |
`))
	}))
	defer openAIServer.Close()

	anthropicMux := http.NewServeMux()
	anthropicMux.HandleFunc("/v1/models", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{
			"data": [
				{"id": "claude-unknown-pricing-5", "max_input_tokens": 123000, "max_tokens": 456}
			],
			"has_more": false,
			"last_id": "claude-unknown-pricing-5"
		}`))
	})
	anthropicMux.HandleFunc("/docs", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`<table>
<tr><th>Feature</th><th>Claude Sonnet 4.6</th></tr>
<tr><td>Claude API ID</td><td>claude-sonnet-4-6</td></tr>
<tr><td>Pricing</td><td>$3 / input MTok $15 / output MTok</td></tr>
<tr><td>Context window</td><td>200K tokens</td></tr>
<tr><td>Max output</td><td>64K tokens</td></tr>
</table>`))
	})
	anthropicServer := httptest.NewServer(anthropicMux)
	defer anthropicServer.Close()

	geminiMux := http.NewServeMux()
	geminiMux.HandleFunc("/pricing", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`<h3>Gemini 3 Flash Preview</h3><p>Input price $0.50</p><p>Output price $3.00</p>`))
	})
	geminiMux.HandleFunc("/models/gemini-3-flash-preview", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`<html>Input token limit 1,048,576 Output token limit 65,536</html>`))
	})
	geminiServer := httptest.NewServer(geminiMux)
	defer geminiServer.Close()

	grokServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`\"name\":\"grok-3\",\"promptTextTokenPrice\":\"$n30000\",\"completionTextTokenPrice\":\"$n150000\",\"maxPromptLength\":131072`))
	}))
	defer grokServer.Close()

	origOpenAI, origAnthropic, origAnthropicAPI, origGemini, origGeminiSpec, origGrok := openAISourceURL, anthropicSourceURL, anthropicModelsAPIURL, geminiSourceURL, geminiSpecBaseURL, grokSourceURL
	defer func() {
		openAISourceURL = origOpenAI
		anthropicSourceURL = origAnthropic
		anthropicModelsAPIURL = origAnthropicAPI
		geminiSourceURL = origGemini
		geminiSpecBaseURL = origGeminiSpec
		grokSourceURL = origGrok
	}()
	openAISourceURL = openAIServer.URL
	anthropicSourceURL = anthropicServer.URL + "/docs"
	anthropicModelsAPIURL = anthropicServer.URL + "/v1/models"
	geminiSourceURL = geminiServer.URL + "/pricing"
	geminiSpecBaseURL = geminiServer.URL + "/models"
	grokSourceURL = grokServer.URL

	catalog, err := FetchLatest(context.Background())
	if err != nil {
		t.Fatalf("FetchLatest() error: %v", err)
	}
	compiled, err := CompileCatalogV1(catalog)
	if err != nil {
		t.Fatalf("CompileCatalogV1: %v", err)
	}

	model := compiled.ModelsByID["anthropic/claude-unknown-pricing-5"]
	if model == nil {
		t.Fatal("API-only Anthropic model with unknown pricing was filtered out")
	}
	if model.ContextWindow != 123000 || model.MaxOutput != 456 {
		t.Fatalf("model limits = %d/%d, want 123000/456", model.ContextWindow, model.MaxOutput)
	}
	offering, ok := compiled.OfferingForDeployment("anthropic/claude-unknown-pricing-5", "anthropic-direct")
	if !ok {
		t.Fatal("direct offering for unknown-pricing Anthropic model not found")
	}
	if offering.Pricing.Status != PricingUnknown {
		t.Fatalf("pricing status = %q, want %q", offering.Pricing.Status, PricingUnknown)
	}
	if len(offering.Pricing.MissingDimensions) == 0 {
		t.Fatal("unknown pricing should include missing dimensions")
	}
}

func TestFetchLatest_FiltersIncomplete(t *testing.T) {
	t.Setenv("ANTHROPIC_API_KEY", "")
	t.Setenv("LANGDAG_REQUIRE_ANTHROPIC_MODELS_API", "")

	// Models without context window or pricing should be filtered
	openAIServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`### gpt-4o-2024-08-06

- Context window size: 128000
- Maximum output tokens: 16384

## Text tokens

| Name | Input | Cached input | Output | Unit |
| --- | --- | --- | --- | --- |
| gpt-4o-2024-08-06 | 2.5 | 1.25 | 10 | 1M tokens |
| gpt-4o | 2.5 | 1.25 | 10 | 1M tokens |
`))
	}))
	defer openAIServer.Close()

	anthropicServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`<table>
<tr><th>Feature</th><th>Model</th></tr>
<tr><td>Claude API ID</td><td>claude-sonnet-4-6</td></tr>
<tr><td>Pricing</td><td>$3 / input MTok $15 / output MTok</td></tr>
<tr><td>Context window</td><td>200K tokens</td></tr>
<tr><td>Max output</td><td>64K tokens</td></tr>
</table>`))
	}))
	defer anthropicServer.Close()

	geminiMux := http.NewServeMux()
	geminiMux.HandleFunc("/pricing", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`<h3>Gemini 3 Flash Preview</h3><p>Input price $0.50</p><p>Output price $3.00</p>`))
	})
	geminiMux.HandleFunc("/models/gemini-3-flash-preview", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`<html>Input token limit 1,048,576 Output token limit 65,536</html>`))
	})
	geminiServer := httptest.NewServer(geminiMux)
	defer geminiServer.Close()

	grokServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`\"name\":\"grok-3\",\"promptTextTokenPrice\":\"$n30000\",\"completionTextTokenPrice\":\"$n150000\",\"maxPromptLength\":131072`))
	}))
	defer grokServer.Close()

	origOpenAI, origAnthropic, origGemini, origGeminiSpec, origGrok := openAISourceURL, anthropicSourceURL, geminiSourceURL, geminiSpecBaseURL, grokSourceURL
	defer func() {
		openAISourceURL = origOpenAI
		anthropicSourceURL = origAnthropic
		geminiSourceURL = origGemini
		geminiSpecBaseURL = origGeminiSpec
		grokSourceURL = origGrok
	}()
	openAISourceURL = openAIServer.URL
	anthropicSourceURL = anthropicServer.URL
	geminiSourceURL = geminiServer.URL + "/pricing"
	geminiSpecBaseURL = geminiServer.URL + "/models"
	grokSourceURL = grokServer.URL

	catalog, err := FetchLatest(context.Background())
	if err != nil {
		t.Fatalf("FetchLatest() error: %v", err)
	}

	// All models should have both pricing and context window
	for _, provider := range []string{"openai", "anthropic", "gemini", "grok"} {
		for _, m := range catalog.ForProvider(provider) {
			if !m.Free && m.InputPricePer1M <= 0 && m.OutputPricePer1M <= 0 {
				t.Errorf("%s/%s: missing pricing", provider, m.ID)
			}
			if m.ContextWindow <= 0 {
				t.Errorf("%s/%s: missing context window", provider, m.ID)
			}
		}
	}
}

func TestFetchLatest_ServerError(t *testing.T) {
	t.Setenv("ANTHROPIC_API_KEY", "")
	t.Setenv("LANGDAG_REQUIRE_ANTHROPIC_MODELS_API", "")

	errorServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer errorServer.Close()

	origOpenAI, origAnthropic, origGemini, origGrok := openAISourceURL, anthropicSourceURL, geminiSourceURL, grokSourceURL
	defer func() {
		openAISourceURL = origOpenAI
		anthropicSourceURL = origAnthropic
		geminiSourceURL = origGemini
		grokSourceURL = origGrok
	}()
	openAISourceURL = errorServer.URL
	anthropicSourceURL = errorServer.URL
	geminiSourceURL = errorServer.URL
	grokSourceURL = errorServer.URL

	_, err := FetchLatest(context.Background())
	if err == nil {
		t.Error("expected error when providers fail")
	}
}
