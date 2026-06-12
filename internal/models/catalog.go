// Package models provides a catalog of LLM model pricing and capabilities.
package models

import (
	"context"
	_ "embed"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"langdag.com/langdag/types"
)

//go:embed catalog.json
var defaultCatalogJSON []byte

const (
	// DefaultRemoteCatalogURL is the static, data-only catalog artifact served
	// from the langdag documentation site.
	DefaultRemoteCatalogURL = "https://langdag.com/model-catalog/v1/catalog.json"

	envCatalogRefresh = "LANGDAG_MODEL_CATALOG_REFRESH"
	envCatalogURL     = "LANGDAG_MODEL_CATALOG_URL"
	envCatalogTimeout = "LANGDAG_MODEL_CATALOG_TIMEOUT"

	defaultRemoteCatalogTimeout = 5 * time.Second
	maxRemoteCatalogBytes       = 10 << 20
)

// CatalogSource records where a catalog was loaded from.
type CatalogSource string

const (
	CatalogSourceEmbedded CatalogSource = "embedded"
	CatalogSourceCache    CatalogSource = "cache"
	CatalogSourceRemote   CatalogSource = "remote"
)

// CatalogLoadOptions configures cache-or-embedded catalog loading.
type CatalogLoadOptions struct {
	CachePath string
	Now       func() time.Time
}

// CatalogLoadResult returns the loaded catalog plus non-fatal diagnostics.
type CatalogLoadResult struct {
	Catalog     *Catalog
	Source      CatalogSource
	CachePath   string
	Diagnostics []CatalogDiagnosticV1
}

// CatalogRefreshOptions configures remote catalog refresh.
type CatalogRefreshOptions struct {
	CachePath  string
	Endpoint   string
	Disabled   bool
	Timeout    time.Duration
	HTTPClient *http.Client
	Now        func() time.Time
}

// CatalogRefreshResult returns the refreshed remote catalog plus cache status.
type CatalogRefreshResult struct {
	Catalog       *Catalog
	Source        CatalogSource
	Endpoint      string
	CachePath     string
	ReplacedCache bool
	Diagnostics   []CatalogDiagnosticV1
}

// ModelPricing contains pricing and capability information for a model.
type ModelPricing struct {
	ID                  string   `json:"id"`
	InputPricePer1M     float64  `json:"input_price_per_1m"`
	OutputPricePer1M    float64  `json:"output_price_per_1m"`
	Free                bool     `json:"free,omitempty"`
	ContextWindow       int      `json:"context_window"`
	MaxOutput           int      `json:"max_output"`
	ServerTools         []string `json:"server_tools,omitempty"`
	AllowUnknownPricing bool     `json:"-"`
}

// LegacyCatalog is the pre-v1 provider-keyed catalog shape. It is kept only so
// old embedded catalogs and cache files can be migrated into CatalogV1.
type LegacyCatalog struct {
	UpdatedAt time.Time                 `json:"updated_at"`
	Source    string                    `json:"source"`
	Providers map[string][]ModelPricing `json:"providers"`
}

// Catalog is the deployment-aware model catalog v1 shape.
type Catalog = CatalogV1

// DefaultCatalog returns the embedded model catalog bundled with the library.
func DefaultCatalog() (*Catalog, error) {
	catalog, err := ParseCatalogV1(defaultCatalogJSON)
	if err != nil {
		return nil, fmt.Errorf("models: failed to parse embedded catalog: %w", err)
	}
	NormalizeCatalogV1(catalog)
	return catalog, nil
}

// LoadRuntimeCatalog loads the catalog used by prompt/runtime routing. By
// default, runtime uses the embedded catalog generated from the published
// model-catalog branch. A cache path is only used when explicitly provided.
func LoadRuntimeCatalog(opts CatalogLoadOptions) (*CatalogLoadResult, error) {
	return LoadCatalogWithOptions(opts)
}

// LoadCatalog loads the catalog from a cache file, falling back to the
// embedded default if the file does not exist or is invalid.
func LoadCatalog(cachePath string) (*Catalog, error) {
	result, err := LoadCatalogWithOptions(CatalogLoadOptions{CachePath: cachePath})
	if err != nil {
		return nil, err
	}
	return result.Catalog, nil
}

// LoadCatalogWithOptions loads a usable catalog immediately. It uses an
// explicit valid cache path when provided and falls back to the embedded catalog
// with diagnostics when the cache is missing or invalid. Stale cached data
// remains usable when the caller explicitly opts into that cache path.
func LoadCatalogWithOptions(opts CatalogLoadOptions) (*CatalogLoadResult, error) {
	now := catalogNow(opts.Now)
	var diagnostics []CatalogDiagnosticV1
	if opts.CachePath != "" {
		data, err := os.ReadFile(opts.CachePath)
		if err == nil {
			catalog, err := ParseCatalogV1(data)
			if err == nil {
				NormalizeCatalogV1(catalog)
				diagnostics = append(diagnostics, catalogFreshnessDiagnostics(catalog, now)...)
				return &CatalogLoadResult{
					Catalog:     catalog,
					Source:      CatalogSourceCache,
					CachePath:   opts.CachePath,
					Diagnostics: diagnostics,
				}, nil
			}
			diagnostics = append(diagnostics, CatalogDiagnosticV1{
				Code:    "cache_invalid",
				Message: fmt.Sprintf("model catalog cache %s is invalid: %v", opts.CachePath, err),
			})
		} else if os.IsNotExist(err) {
			diagnostics = append(diagnostics, CatalogDiagnosticV1{
				Code:    "cache_missing",
				Message: fmt.Sprintf("model catalog cache %s does not exist", opts.CachePath),
			})
		} else {
			diagnostics = append(diagnostics, CatalogDiagnosticV1{
				Code:    "cache_unavailable",
				Message: fmt.Sprintf("model catalog cache %s could not be read: %v", opts.CachePath, err),
			})
		}
	}
	catalog, err := DefaultCatalog()
	if err != nil {
		return nil, err
	}
	diagnostics = append(diagnostics, catalogFreshnessDiagnostics(catalog, now)...)
	return &CatalogLoadResult{
		Catalog:     catalog,
		Source:      CatalogSourceEmbedded,
		CachePath:   opts.CachePath,
		Diagnostics: diagnostics,
	}, nil
}

// SaveCatalog writes the catalog to a JSON file.
func SaveCatalog(catalog *Catalog, path string) error {
	if catalog == nil {
		return fmt.Errorf("models: cannot save nil catalog")
	}
	if path == "" {
		return fmt.Errorf("models: catalog path is required")
	}
	NormalizeCatalogV1(catalog)
	if err := ValidateCatalogV1(catalog); err != nil {
		return fmt.Errorf("models: invalid catalog: %w", err)
	}
	data, err := json.MarshalIndent(catalog, "", "  ")
	if err != nil {
		return fmt.Errorf("models: failed to marshal catalog: %w", err)
	}
	data = append(data, '\n')
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("models: failed to create catalog directory: %w", err)
	}
	tmp, err := os.CreateTemp(dir, ".catalog-*.tmp")
	if err != nil {
		return fmt.Errorf("models: failed to create temporary catalog file: %w", err)
	}
	tmpPath := tmp.Name()
	defer os.Remove(tmpPath)
	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		return fmt.Errorf("models: failed to write temporary catalog file: %w", err)
	}
	if err := tmp.Close(); err != nil {
		return fmt.Errorf("models: failed to close temporary catalog file: %w", err)
	}
	if err := os.Chmod(tmpPath, 0644); err != nil {
		return fmt.Errorf("models: failed to chmod temporary catalog file: %w", err)
	}
	if err := os.Rename(tmpPath, path); err != nil {
		return fmt.Errorf("models: failed to replace catalog cache: %w", err)
	}
	return nil
}

// CatalogRefreshOptionsFromEnv builds default remote refresh options from
// LANGDAG_MODEL_CATALOG_* environment variables.
func CatalogRefreshOptionsFromEnv(cachePath string) CatalogRefreshOptions {
	opts := CatalogRefreshOptions{CachePath: cachePath}
	if disabledCatalogRefresh(os.Getenv(envCatalogRefresh)) {
		opts.Disabled = true
	}
	if endpoint := strings.TrimSpace(os.Getenv(envCatalogURL)); endpoint != "" {
		opts.Endpoint = endpoint
	}
	if timeout := strings.TrimSpace(os.Getenv(envCatalogTimeout)); timeout != "" {
		if parsed, err := time.ParseDuration(timeout); err == nil {
			opts.Timeout = parsed
		}
	}
	return opts
}

// RefreshCatalogCache fetches the published remote catalog and validates it
// strictly. If opts.CachePath is non-empty, it atomically replaces that cache
// file. Invalid, stale, or partial remote data never overwrites an existing
// cache.
func RefreshCatalogCache(ctx context.Context, opts CatalogRefreshOptions) (*CatalogRefreshResult, error) {
	endpoint := strings.TrimSpace(opts.Endpoint)
	if endpoint == "" {
		endpoint = DefaultRemoteCatalogURL
	}
	result := &CatalogRefreshResult{
		Source:    CatalogSourceRemote,
		Endpoint:  endpoint,
		CachePath: opts.CachePath,
	}
	if opts.Disabled {
		result.Diagnostics = append(result.Diagnostics, CatalogDiagnosticV1{
			Code:    "refresh_disabled",
			Message: "remote model catalog refresh is disabled",
		})
		return result, nil
	}

	timeout := opts.Timeout
	if timeout <= 0 {
		timeout = defaultRemoteCatalogTimeout
	}
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	catalog, diagnostics, err := fetchRemoteCatalog(ctx, opts.HTTPClient, endpoint, catalogNow(opts.Now))
	result.Diagnostics = append(result.Diagnostics, diagnostics...)
	if err != nil {
		return result, err
	}
	if opts.CachePath != "" {
		if err := SaveCatalog(catalog, opts.CachePath); err != nil {
			return result, err
		}
		result.ReplacedCache = true
	}
	result.Catalog = catalog
	return result, nil
}

// LoadRemoteCatalog fetches the published remote catalog and validates it
// without writing any local cache file.
func LoadRemoteCatalog(ctx context.Context, opts CatalogRefreshOptions) (*CatalogRefreshResult, error) {
	opts.CachePath = ""
	return RefreshCatalogCache(ctx, opts)
}

func fetchRemoteCatalog(ctx context.Context, client *http.Client, endpoint string, now time.Time) (*Catalog, []CatalogDiagnosticV1, error) {
	if client == nil {
		client = http.DefaultClient
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return nil, nil, err
	}
	req.Header.Set("Accept", "application/json")
	req.Header.Set("User-Agent", "langdag-model-catalog/1.0")
	resp, err := client.Do(req)
	if err != nil {
		return nil, nil, fmt.Errorf("models: fetch remote catalog: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, nil, fmt.Errorf("models: fetch remote catalog: HTTP %d from %s", resp.StatusCode, endpoint)
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, maxRemoteCatalogBytes+1))
	if err != nil {
		return nil, nil, fmt.Errorf("models: read remote catalog: %w", err)
	}
	if len(body) > maxRemoteCatalogBytes {
		return nil, nil, fmt.Errorf("models: remote catalog exceeds %d bytes", maxRemoteCatalogBytes)
	}
	catalog, err := ParseRemoteCatalogV1(body)
	if err != nil {
		return nil, nil, fmt.Errorf("models: invalid remote catalog: %w", err)
	}
	compiled, err := CompileCatalogV1(catalog)
	if err != nil {
		return nil, nil, fmt.Errorf("models: invalid remote catalog: %w", err)
	}
	diagnostics := append([]CatalogDiagnosticV1{}, compiled.Diagnostics...)
	diagnostics = append(diagnostics, catalogFreshnessDiagnostics(catalog, now)...)
	if catalogIsStale(catalog, now) {
		return nil, diagnostics, fmt.Errorf("models: remote catalog is stale after %s", catalog.StaleAfter.Format(time.RFC3339))
	}
	for _, diagnostic := range compiled.Diagnostics {
		switch diagnostic.Code {
		case "dropped_offering", "dropped_offering_template":
			return nil, diagnostics, fmt.Errorf("models: remote catalog is partially generated: %s", diagnostic.Message)
		}
	}
	return catalog, diagnostics, nil
}

func catalogNow(now func() time.Time) time.Time {
	if now != nil {
		return now().UTC()
	}
	return time.Now().UTC()
}

func catalogFreshnessDiagnostics(catalog *Catalog, now time.Time) []CatalogDiagnosticV1 {
	if !catalogIsStale(catalog, now) {
		return nil
	}
	return []CatalogDiagnosticV1{{
		Code:    "stale_catalog",
		Message: fmt.Sprintf("model catalog generated at %s is stale after %s", catalog.GeneratedAt.Format(time.RFC3339), catalog.StaleAfter.Format(time.RFC3339)),
	}}
}

func catalogIsStale(catalog *Catalog, now time.Time) bool {
	return catalog != nil && !catalog.StaleAfter.IsZero() && now.After(catalog.StaleAfter)
}

func disabledCatalogRefresh(value string) bool {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "0", "false", "no", "off", "disabled":
		return true
	default:
		return false
	}
}

// ForProvider returns a legacy provider view over routeable deployment
// offerings. Herm still uses this until it switches its picker to canonical
// model rows in a later phase.
func (c *CatalogV1) ForProvider(provider string) []ModelPricing {
	if c == nil {
		return nil
	}
	deployments := legacyProviderDeploymentsV1(provider)
	if len(deployments) == 0 {
		return nil
	}
	compiled, err := CompileCatalogV1(c)
	if err != nil {
		return nil
	}
	var out []ModelPricing
	for _, deploymentID := range deployments {
		for _, offering := range compiled.OfferingsByDeployment[deploymentID] {
			out = append(out, modelPricingFromOfferingV1(compiled, offering))
		}
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	return out
}

// LookupModel finds a model by ID across all providers.
// Returns the model pricing and provider name, or false if not found.
func (c *CatalogV1) LookupModel(modelID string) (ModelPricing, string, bool) {
	if c == nil {
		return ModelPricing{}, "", false
	}
	compiled, err := CompileCatalogV1(c)
	if err != nil {
		return ModelPricing{}, "", false
	}
	for _, provider := range []string{"anthropic", "anthropic-bedrock", "anthropic-vertex", "openai", "openai-azure", "gemini", "gemini-vertex", "grok", "openrouter", "ollama", "apple"} {
		for _, deploymentID := range legacyProviderDeploymentsV1(provider) {
			for _, offering := range compiled.OfferingsByDeployment[deploymentID] {
				if offering.NativeModelID == modelID || offering.CanonicalModelID == modelID {
					return modelPricingFromOfferingV1(compiled, offering), provider, true
				}
			}
		}
	}
	return ModelPricing{}, "", false
}

func legacyProviderDeploymentsV1(provider string) []string {
	switch provider {
	case "anthropic":
		return []string{"anthropic-direct"}
	case "anthropic-bedrock":
		return []string{"anthropic-bedrock"}
	case "anthropic-vertex":
		return []string{"anthropic-vertex"}
	case "openai":
		return []string{"openai-direct"}
	case "openai-azure":
		return []string{"openai-azure"}
	case "gemini", "gemma":
		return []string{"gemini-direct"}
	case "gemini-vertex":
		return []string{"gemini-vertex"}
	case "grok":
		return []string{"grok-direct"}
	case "openrouter":
		return []string{"openrouter"}
	case "ollama":
		return []string{"ollama-local"}
	case "apple":
		return []string{"apple-local"}
	default:
		return nil
	}
}

func modelPricingFromOfferingV1(compiled *CompiledCatalogV1, offering *ModelOfferingV1) ModelPricing {
	model := compiled.ModelsByID[offering.CanonicalModelID]
	pricing := offering.Pricing.RatesPer1M
	out := ModelPricing{
		ID:               offering.NativeModelID,
		InputPricePer1M:  pricing["input_tokens"],
		OutputPricePer1M: pricing["output_tokens"],
		Free:             offering.Pricing.Status == PricingFree,
	}
	if model != nil {
		out.ContextWindow = model.ContextWindow
		out.MaxOutput = model.MaxOutput
	}
	for tool, state := range offering.Capabilities.ServerTools {
		if state == CapabilitySupported {
			out.ServerTools = append(out.ServerTools, tool)
		}
	}
	sort.Strings(out.ServerTools)
	return out
}

// MetadataForLegacyProviderModel returns best-effort deployment-aware metadata
// for legacy callers that still pass provider/native model IDs directly. The
// routing phase replaces this with explicit offering resolution before adapter
// execution, but this keeps Phase 3 persistence typed and deterministic.
func (c *CatalogV1) MetadataForLegacyProviderModel(providerName, requestedModel, responseModel string) (*types.ModelResolutionMetadata, *types.PricingSnapshot, bool) {
	if c == nil {
		return legacyResolutionOnly(providerName, requestedModel, responseModel), nil, false
	}
	compiled, err := CompileCatalogV1(c)
	if err != nil {
		return legacyResolutionOnly(providerName, requestedModel, responseModel), nil, false
	}
	for _, modelID := range compactStrings(responseModel, requestedModel) {
		if offering, ok := compiled.OfferingForLegacyProviderModel(providerName, modelID); ok {
			resolution := ModelResolutionMetadataFromOffering(offering)
			snapshot := PricingSnapshotFromPricingV1(offering.Pricing)
			return &resolution, &snapshot, true
		}
	}
	return legacyResolutionOnly(providerName, requestedModel, responseModel), nil, false
}

func (c *CompiledCatalogV1) OfferingForLegacyProviderModel(providerName, modelID string) (*ModelOfferingV1, bool) {
	if c == nil || modelID == "" {
		return nil, false
	}
	deployments := legacyProviderDeploymentsV1(providerName)
	if len(deployments) == 0 {
		if c.Catalog == nil {
			return nil, false
		}
		for i := range c.Catalog.Offerings {
			offering := &c.Catalog.Offerings[i]
			if offering.NativeModelID == modelID || offering.CanonicalModelID == modelID {
				return offering, true
			}
		}
		return nil, false
	}
	for _, deploymentID := range deployments {
		for _, offering := range c.OfferingsByDeployment[deploymentID] {
			if offering.NativeModelID == modelID || offering.CanonicalModelID == modelID {
				return offering, true
			}
		}
	}
	return nil, false
}

func ModelResolutionMetadataFromOffering(offering *ModelOfferingV1) types.ModelResolutionMetadata {
	if offering == nil {
		return types.ModelResolutionMetadata{}
	}
	resolution := types.ModelResolutionMetadata{
		CanonicalModelID: offering.CanonicalModelID,
		OfferingID:       offering.ID,
		DeploymentID:     offering.DeploymentID,
		NativeModelID:    offering.NativeModelID,
	}
	if offering.Model != nil {
		resolution.ProviderID = offering.Model.ProviderID
	}
	if resolution.ProviderID == "" && offering.Provider != nil {
		resolution.ProviderID = offering.Provider.ID
	}
	if resolution.ProviderID == "" {
		if owner, _, ok := strings.Cut(offering.CanonicalModelID, "/"); ok {
			resolution.ProviderID = owner
		}
	}
	if offering.APIProtocolID != "" {
		resolution.APIProtocolID = offering.APIProtocolID
	}
	if resolution.APIProtocolID == "" && offering.APIProtocol != nil {
		resolution.APIProtocolID = offering.APIProtocol.ID
	}
	if resolution.APIProtocolID == "" && offering.Deployment != nil {
		resolution.APIProtocolID = offering.Deployment.APIProtocolID
	}
	return resolution
}

func PricingSnapshotFromPricingV1(pricing PricingV1) types.PricingSnapshot {
	source := types.CostSourceCatalog
	if pricing.Source != "" && pricing.Source != "catalog" {
		source = types.CostSource(pricing.Source)
	}
	return types.PricingSnapshot{
		Status:            types.CostStatus(pricing.Status),
		Currency:          pricing.Currency,
		EffectiveAt:       pricing.EffectiveAt,
		Source:            source,
		RatesPer1M:        cloneFloat64Map(pricing.RatesPer1M),
		MissingDimensions: append([]string(nil), pricing.MissingDimensions...),
	}
}

func legacyResolutionOnly(providerName, requestedModel, responseModel string) *types.ModelResolutionMetadata {
	nativeID := responseModel
	if nativeID == "" {
		nativeID = requestedModel
	}
	if nativeID == "" && providerName == "" {
		return nil
	}
	resolution := &types.ModelResolutionMetadata{NativeModelID: nativeID}
	if strings.Contains(nativeID, "/") {
		resolution.CanonicalModelID = nativeID
	}
	if deployments := legacyProviderDeploymentsV1(providerName); len(deployments) > 0 {
		resolution.DeploymentID = deployments[0]
	}
	return resolution
}

func compactStrings(values ...string) []string {
	seen := map[string]bool{}
	var out []string
	for _, value := range values {
		if value == "" || seen[value] {
			continue
		}
		seen[value] = true
		out = append(out, value)
	}
	return out
}

func cloneFloat64Map(in map[string]float64) map[string]float64 {
	if len(in) == 0 {
		return nil
	}
	out := make(map[string]float64, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}
