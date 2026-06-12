package models

import (
	"context"
	"fmt"
	"math"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"
)

// providerServerTools maps provider names to the server tools their models support.
// Used to annotate fetched catalog entries with capability metadata.
var providerServerTools = map[string][]string{
	"anthropic": {"web_search"},
	"openai":    {"web_search"},
	"gemini":    {"web_search"},
	"grok":      {"web_search"},
}

// FetchLatest fetches the latest model catalog from official provider
// documentation pages and compiles it into the deployment-aware v1 shape.
// All provider fetches must succeed; partial failures return an error.
func FetchLatest(ctx context.Context) (*Catalog, error) {
	return FetchLatestV1(ctx)
}

// FetchLatestV1 fetches the latest provider catalog and compiles it into the
// deployment-aware v1 shape used by the published catalog artifact.
func FetchLatestV1(ctx context.Context) (*CatalogV1, error) {
	legacy, err := fetchLatestLegacy(ctx)
	if err != nil {
		return nil, err
	}
	catalog := CatalogV1FromLegacyCatalog(legacy)
	NormalizeCatalogV1(catalog)
	if err := ValidateCatalogV1(catalog); err != nil {
		return nil, err
	}
	return catalog, nil
}

func fetchLatestLegacy(ctx context.Context) (*LegacyCatalog, error) {
	ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()

	type result struct {
		provider string
		models   []ModelPricing
		err      error
	}

	type fetcher struct {
		provider string
		fn       func(context.Context) ([]ModelPricing, error)
	}

	fetchers := []fetcher{
		{"openai", fetchOpenAIModels},
		{"anthropic", fetchAnthropicModels},
		{"gemini", fetchGeminiAndGemmaModels},
		{"grok", fetchGrokModels},
	}

	results := make([]result, len(fetchers))
	var wg sync.WaitGroup
	for i, f := range fetchers {
		wg.Add(1)
		go func(i int, f fetcher) {
			defer wg.Done()
			models, err := f.fn(ctx)
			results[i] = result{provider: f.provider, models: models, err: err}
		}(i, f)
	}
	wg.Wait()

	// All providers must succeed
	var errs []string
	for _, r := range results {
		if r.err != nil {
			errs = append(errs, fmt.Sprintf("%s: %v", r.provider, r.err))
		} else if len(r.models) == 0 {
			errs = append(errs, fmt.Sprintf("%s: no models found", r.provider))
		}
	}
	if len(errs) > 0 {
		return nil, fmt.Errorf("models: provider fetch failed: %s", strings.Join(errs, "; "))
	}

	legacy := &LegacyCatalog{
		UpdatedAt: time.Now().UTC(),
		Source:    "providers",
		Providers: make(map[string][]ModelPricing),
	}

	for _, r := range results {
		// Filter: require usable pricing information and context_window > 0.
		var filtered []ModelPricing
		for _, m := range r.models {
			hasPrice := m.Free || m.InputPricePer1M > 0 || m.OutputPricePer1M > 0 || m.AllowUnknownPricing
			if hasPrice && m.ContextWindow > 0 {
				filtered = append(filtered, m)
			}
		}
		// Annotate with known server tool capabilities.
		if st := providerServerTools[r.provider]; len(st) > 0 {
			for i := range filtered {
				if r.provider == "gemini" && strings.HasPrefix(filtered[i].ID, "gemma-") {
					continue
				}
				filtered[i].ServerTools = st
			}
		}
		slices.SortFunc(filtered, func(a, b ModelPricing) int {
			return strings.Compare(a.ID, b.ID)
		})
		legacy.Providers[r.provider] = filtered
	}

	return legacy, nil
}

func addDerivedDeploymentOfferingsV1(catalog *CatalogV1, observedAt time.Time) {
	if catalog == nil {
		return
	}
	provenance := &ProvenanceV1{Source: "generated-from-provider-catalog", ObservedAt: observedAt}
	var derived []ModelOfferingV1
	for _, offering := range catalog.Offerings {
		switch offering.DeploymentID {
		case "anthropic-direct":
			if nativeID := anthropicBedrockNativeModelIDV1(offering.NativeModelID); nativeID != "" {
				derived = append(derived, derivedOfferingV1(offering, "anthropic-bedrock", nativeID, provenance))
			}
			if nativeID := anthropicVertexNativeModelIDV1(offering.NativeModelID); nativeID != "" {
				derived = append(derived, derivedOfferingV1(offering, "anthropic-vertex", nativeID, provenance))
			}
		case "gemini-direct":
			if strings.HasPrefix(offering.NativeModelID, "gemini-") {
				derived = append(derived, derivedOfferingV1(offering, "gemini-vertex", offering.NativeModelID, provenance))
			}
		}
	}
	for _, offering := range derived {
		if !catalogV1HasOffering(catalog.Offerings, offering.ID) {
			catalog.Offerings = append(catalog.Offerings, offering)
		}
	}
	for modelID, model := range catalog.Models {
		if model.ProviderID != "openai" {
			continue
		}
		template := offeringTemplateV1("openai-azure", modelID, PricingV1{
			Status:            PricingUnknown,
			Currency:          "USD",
			EffectiveAt:       observedAt,
			MissingDimensions: []string{"input_tokens", "output_tokens"},
			Source:            "user-configured-azure-deployment",
		}, provenance)
		if !catalogV1HasOfferingTemplate(catalog.OfferingTemplates, template.ID) {
			catalog.OfferingTemplates = append(catalog.OfferingTemplates, template)
		}
	}
}

func derivedOfferingV1(source ModelOfferingV1, deploymentID, nativeModelID string, provenance *ProvenanceV1) ModelOfferingV1 {
	return ModelOfferingV1{
		ID:               deploymentID + ":" + nativeModelID,
		CanonicalModelID: source.CanonicalModelID,
		DeploymentID:     deploymentID,
		NativeModelID:    nativeModelID,
		Capabilities: CapabilitySetV1{
			ServerTools: map[string]CapabilityState{"web_search": CapabilityUnknown},
		},
		Pricing: PricingV1{
			Status:            PricingUnknown,
			Currency:          "USD",
			EffectiveAt:       provenance.ObservedAt,
			MissingDimensions: []string{"input_tokens", "output_tokens"},
			Source:            "generated-deployment-placeholder",
			Notes:             []string{"Deployment-specific pricing is not inferred from direct provider pricing."},
		},
		Provenance: provenance,
	}
}

func anthropicBedrockNativeModelIDV1(nativeID string) string {
	if nativeID == "" {
		return ""
	}
	if strings.HasPrefix(nativeID, "anthropic.") {
		return nativeID
	}
	if anthropicUsesUnversionedBedrockID(nativeID) {
		return "anthropic." + nativeID
	}
	if nativeID == "claude-opus-4-6" {
		return "anthropic." + nativeID + "-v1"
	}
	if strings.HasSuffix(nativeID, "-v1:0") {
		return "anthropic." + nativeID
	}
	return "anthropic." + nativeID + "-v1:0"
}

func anthropicUsesUnversionedBedrockID(nativeID string) bool {
	parts := strings.Split(nativeID, "-")
	if len(parts) < 4 || parts[0] != "claude" {
		return false
	}
	if len(parts[len(parts)-1]) > 2 || len(parts[len(parts)-2]) > 2 {
		return false
	}
	major, err := strconv.Atoi(parts[len(parts)-2])
	if err != nil {
		return false
	}
	minor, err := strconv.Atoi(parts[len(parts)-1])
	if err != nil {
		return false
	}
	if major > 4 {
		return true
	}
	return major == 4 && minor >= 6 && nativeID != "claude-opus-4-6"
}

func anthropicVertexNativeModelIDV1(nativeID string) string {
	if nativeID == "" {
		return ""
	}
	parts := strings.Split(nativeID, "-")
	if len(parts) < 2 {
		return nativeID
	}
	last := parts[len(parts)-1]
	if len(last) == 8 {
		for _, r := range last {
			if r < '0' || r > '9' {
				return nativeID
			}
		}
		return strings.Join(parts[:len(parts)-1], "-") + "@" + last
	}
	return nativeID
}

func catalogV1HasOfferingTemplate(templates []ModelOfferingTemplateV1, id string) bool {
	for _, template := range templates {
		if template.ID == id {
			return true
		}
	}
	return false
}

// roundPrice rounds a price to 4 decimal places to avoid floating point artifacts.
func roundPrice(p float64) float64 {
	return math.Round(p*10000) / 10000
}
