package models

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestCatalogV1ReferenceValidatesAndCompiles(t *testing.T) {
	if !json.Valid([]byte(CatalogV1JSONSchema)) {
		t.Fatal("CatalogV1JSONSchema is not valid JSON")
	}

	catalog := ReferenceCatalogV1()
	compiled, err := CompileCatalogV1(catalog)
	if err != nil {
		t.Fatalf("CompileCatalogV1: %v", err)
	}

	wantDeployments := []string{
		"anthropic-direct",
		"anthropic-bedrock",
		"anthropic-vertex",
		"openai-direct",
		"openai-azure",
		"gemini-direct",
		"gemini-vertex",
		"grok-direct",
		"openrouter",
		"ollama-local",
		"apple-local",
	}
	for _, deploymentID := range wantDeployments {
		if compiled.DeploymentsByID[deploymentID] == nil {
			t.Errorf("reference catalog missing deployment %q", deploymentID)
		}
	}

	bindings := DeploymentBindingsV1()
	if len(bindings) != len(wantDeployments) {
		t.Fatalf("DeploymentBindingsV1 returned %d bindings, want %d", len(bindings), len(wantDeployments))
	}
	for _, binding := range bindings {
		deployment := compiled.DeploymentsByID[binding.DeploymentID]
		if deployment == nil {
			t.Fatalf("binding %q missing matching deployment", binding.DeploymentID)
		}
		if deployment.ProviderID != binding.ProviderID {
			t.Errorf("%s ProviderID = %q, want %q", binding.DeploymentID, deployment.ProviderID, binding.ProviderID)
		}
		if deployment.APIProtocolID != binding.APIProtocolID {
			t.Errorf("%s APIProtocolID = %q, want %q", binding.DeploymentID, deployment.APIProtocolID, binding.APIProtocolID)
		}
		if deployment.NativeModelIDSource != binding.NativeModelIDSource {
			t.Errorf("%s NativeModelIDSource = %q, want %q", binding.DeploymentID, deployment.NativeModelIDSource, binding.NativeModelIDSource)
		}
	}

	offering, ok := compiled.OfferingForDeployment("anthropic/claude-sonnet-4-20250514", "anthropic-bedrock")
	if !ok {
		t.Fatal("missing anthropic bedrock offering")
	}
	deploymentID, nativeID, ok := SplitOfferingIDV1(offering.ID)
	if !ok {
		t.Fatalf("SplitOfferingIDV1(%q) failed", offering.ID)
	}
	if deploymentID != "anthropic-bedrock" {
		t.Errorf("deployment segment = %q", deploymentID)
	}
	if nativeID != "anthropic.claude-sonnet-4-20250514-v1:0" {
		t.Errorf("native segment = %q, want full native id with colon preserved", nativeID)
	}

	if _, ok := compiled.OfferingForDeployment("openai/gpt-4.1-2025-04-14", "openai-azure"); ok {
		t.Fatal("Azure should not have a static routeable offering before model_mappings supplies a native model ID")
	}
	templates := compiled.OfferingTemplatesByDeployment["openai-azure"]
	if len(templates) != 1 {
		t.Fatalf("openai-azure templates = %d, want 1", len(templates))
	}
	if !templates[0].MappingRequired || templates[0].NativeModelIDSource != NativeModelIDUserConfigured {
		t.Fatalf("azure template should require user-configured native IDs: %+v", templates[0])
	}
	materialized, err := templates[0].Materialize("my-gpt-4-1-prod")
	if err != nil {
		t.Fatalf("Materialize: %v", err)
	}
	if materialized.ID != "openai-azure:my-gpt-4-1-prod" || materialized.NativeModelID != "my-gpt-4-1-prod" {
		t.Fatalf("materialized azure offering = %+v", materialized)
	}
	if materialized.Model == nil || materialized.Deployment == nil || materialized.Provider == nil || materialized.APIProtocol == nil {
		t.Fatalf("materialized azure offering should preserve compiled pointer links: %+v", materialized)
	}

	if compiled.ModelsByID["z-ai/glm-4.5-air:free"] == nil {
		t.Fatal("OpenRouter-hosted model should keep the model owner, not openrouter, in its canonical ID")
	}
	if canonicalID, ok := compiled.CanonicalModelForAliasOrID("gemini/gemini-2.5-pro"); !ok || canonicalID != "google/gemini-2.5-pro" {
		t.Fatalf("model-level alias resolved to %q/%v, want google/gemini-2.5-pro", canonicalID, ok)
	}
}

func TestReferenceCatalogHasNoStaticAppleOfferings(t *testing.T) {
	catalog := ReferenceCatalogV1()
	for _, offering := range catalog.Offerings {
		if offering.DeploymentID == "apple-local" {
			t.Fatalf("reference catalog must not include static Apple offering: %+v", offering)
		}
	}
}

func TestCatalogV1NormalizesOpenAIDirectToResponses(t *testing.T) {
	catalog := ReferenceCatalogV1()
	deployment := catalog.Deployments["openai-direct"]
	deployment.APIProtocolID = "openai-chat-completions"
	deployment.APIProtocolIDs = nil

	data := mustMarshalCatalogV1(t, catalog)
	parsed, err := ParseRemoteCatalogV1(data)
	if err != nil {
		t.Fatalf("ParseRemoteCatalogV1: %v", err)
	}
	normalized := parsed.Deployments["openai-direct"]
	if normalized == nil {
		t.Fatal("missing openai-direct deployment")
	}
	if normalized.APIProtocolID != "openai-responses" {
		t.Fatalf("openai-direct api_protocol_id = %q, want openai-responses", normalized.APIProtocolID)
	}
	if !stringSliceContainsV1(normalized.APIProtocolIDs, "openai-responses") || !stringSliceContainsV1(normalized.APIProtocolIDs, "openai-chat-completions") {
		t.Fatalf("openai-direct api_protocol_ids = %+v, want responses and chat completions", normalized.APIProtocolIDs)
	}
}

func TestCatalogV1SchemaDefinesNestedStrictContract(t *testing.T) {
	var schema map[string]any
	if err := json.Unmarshal([]byte(CatalogV1JSONSchema), &schema); err != nil {
		t.Fatalf("schema unmarshal: %v", err)
	}
	defs, ok := schema["$defs"].(map[string]any)
	if !ok {
		t.Fatal("schema missing $defs")
	}
	for _, name := range []string{"provider", "api_protocol", "deployment", "model", "offering", "offering_template", "pricing", "capabilities"} {
		def, ok := defs[name].(map[string]any)
		if !ok {
			t.Fatalf("schema missing definition %q", name)
		}
		if def["additionalProperties"] != false {
			t.Fatalf("schema definition %q must set additionalProperties false", name)
		}
	}
	pricing := defs["pricing"].(map[string]any)
	properties := pricing["properties"].(map[string]any)
	status := properties["status"].(map[string]any)
	enum := status["enum"].([]any)
	if len(enum) != 4 {
		t.Fatalf("pricing status enum = %+v, want known/partial/unknown/free", enum)
	}
}

func TestPublishedCatalogArtifactValidates(t *testing.T) {
	path := filepath.Join("..", "..", "docs", "model-catalog", "v1", "catalog.json")
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			t.Skip("published catalog artifact is stored on the model-catalog branch")
		}
		t.Fatalf("read published catalog artifact: %v", err)
	}
	catalog, err := ParseRemoteCatalogV1(data)
	if err != nil {
		t.Fatalf("ParseCatalogV1(%s): %v", path, err)
	}
	if catalog.SchemaVersion != CatalogV1SchemaVersion {
		t.Fatalf("schema_version = %q", catalog.SchemaVersion)
	}
	if _, err := CompileCatalogV1(catalog); err != nil {
		t.Fatalf("CompileCatalogV1(%s): %v", path, err)
	}
}

func TestCatalogV1SemanticJSONIgnoresVolatileTimes(t *testing.T) {
	left := ReferenceCatalogV1()
	right := ReferenceCatalogV1()
	shiftCatalogV1Times(right, 48*time.Hour)

	leftData := mustMarshalCatalogV1(t, left)
	rightData := mustMarshalCatalogV1(t, right)

	leftSemantic, err := CatalogV1SemanticJSON(leftData)
	if err != nil {
		t.Fatalf("CatalogV1SemanticJSON(left): %v", err)
	}
	rightSemantic, err := CatalogV1SemanticJSON(rightData)
	if err != nil {
		t.Fatalf("CatalogV1SemanticJSON(right): %v", err)
	}
	if string(leftSemantic) != string(rightSemantic) {
		t.Fatalf("semantic JSON differs for timestamp-only changes")
	}
}

func TestCatalogV1SemanticJSONDetectsContentChanges(t *testing.T) {
	left := ReferenceCatalogV1()
	right := ReferenceCatalogV1()
	right.Offerings[0].Pricing.RatesPer1M["input_tokens"] += 0.25

	leftSemantic, err := CatalogV1SemanticJSON(mustMarshalCatalogV1(t, left))
	if err != nil {
		t.Fatalf("CatalogV1SemanticJSON(left): %v", err)
	}
	rightSemantic, err := CatalogV1SemanticJSON(mustMarshalCatalogV1(t, right))
	if err != nil {
		t.Fatalf("CatalogV1SemanticJSON(right): %v", err)
	}
	if string(leftSemantic) == string(rightSemantic) {
		t.Fatalf("semantic JSON did not change for pricing update")
	}
}

func TestDeploymentBindingsRecordCostSourceAudit(t *testing.T) {
	bindings := DeploymentBindingsV1()
	if len(bindings) == 0 {
		t.Fatal("DeploymentBindingsV1 returned no bindings")
	}

	seen := map[string]bool{}
	for _, binding := range bindings {
		if binding.DeploymentID == "" {
			t.Fatal("binding missing DeploymentID")
		}
		if seen[binding.DeploymentID] {
			t.Fatalf("duplicate binding for %q", binding.DeploymentID)
		}
		seen[binding.DeploymentID] = true
		if binding.AdapterConstructor == "" || binding.APIProtocolID == "" || binding.ProviderID == "" {
			t.Fatalf("binding %q is incomplete: %+v", binding.DeploymentID, binding)
		}
		if binding.ResponseCostSource == "" {
			t.Fatalf("binding %q missing ResponseCostSource audit", binding.DeploymentID)
		}
	}
	for _, deploymentID := range []string{"ollama-local", "apple-local"} {
		found := false
		for _, binding := range bindings {
			if binding.DeploymentID == deploymentID {
				found = true
				if binding.ResponseCostSource != ResponseCostLocalFree {
					t.Fatalf("%s should be audited as local/free, got %+v", deploymentID, binding)
				}
			}
		}
		if !found {
			t.Fatalf("%s binding missing", deploymentID)
		}
	}
}

func mustMarshalCatalogV1(t *testing.T, catalog *CatalogV1) []byte {
	t.Helper()
	data, err := json.Marshal(catalog)
	if err != nil {
		t.Fatalf("marshal catalog: %v", err)
	}
	return data
}

func shiftCatalogV1Times(catalog *CatalogV1, delta time.Duration) {
	catalog.GeneratedAt = catalog.GeneratedAt.Add(delta)
	catalog.StaleAfter = catalog.StaleAfter.Add(delta)
	if catalog.Provenance != nil {
		catalog.Provenance.ObservedAt = catalog.Provenance.ObservedAt.Add(delta)
	}
	for _, provider := range catalog.Providers {
		if provider.Provenance != nil {
			provider.Provenance.ObservedAt = provider.Provenance.ObservedAt.Add(delta)
		}
	}
	for _, protocol := range catalog.APIProtocols {
		if protocol.Provenance != nil {
			protocol.Provenance.ObservedAt = protocol.Provenance.ObservedAt.Add(delta)
		}
	}
	for _, deployment := range catalog.Deployments {
		if deployment.Provenance != nil {
			deployment.Provenance.ObservedAt = deployment.Provenance.ObservedAt.Add(delta)
		}
	}
	for _, model := range catalog.Models {
		if model.Provenance != nil {
			model.Provenance.ObservedAt = model.Provenance.ObservedAt.Add(delta)
		}
	}
	for i := range catalog.Offerings {
		catalog.Offerings[i].Pricing.EffectiveAt = catalog.Offerings[i].Pricing.EffectiveAt.Add(delta)
		if catalog.Offerings[i].Provenance != nil {
			catalog.Offerings[i].Provenance.ObservedAt = catalog.Offerings[i].Provenance.ObservedAt.Add(delta)
		}
	}
	for i := range catalog.OfferingTemplates {
		catalog.OfferingTemplates[i].Pricing.EffectiveAt = catalog.OfferingTemplates[i].Pricing.EffectiveAt.Add(delta)
		if catalog.OfferingTemplates[i].Provenance != nil {
			catalog.OfferingTemplates[i].Provenance.ObservedAt = catalog.OfferingTemplates[i].Provenance.ObservedAt.Add(delta)
		}
	}
}

func TestCatalogV1RejectsInvalidContracts(t *testing.T) {
	catalog := ReferenceCatalogV1()
	catalog.Offerings[0].Pricing = PricingV1{Status: PricingKnown}

	err := ValidateCatalogV1(catalog)
	if err == nil {
		t.Fatal("ValidateCatalogV1 succeeded for known pricing without currency/rates")
	}
	if !strings.Contains(err.Error(), "known pricing missing currency") {
		t.Fatalf("validation error = %v, want missing currency detail", err)
	}
	if !strings.Contains(err.Error(), "known pricing missing rates_per_1m") {
		t.Fatalf("validation error = %v, want missing rates detail", err)
	}
}

func TestCatalogV1RejectsAmbiguousDeploymentRoutes(t *testing.T) {
	catalog := ReferenceCatalogV1()
	duplicateOffering := catalog.Offerings[0]
	duplicateOffering.NativeModelID += "-duplicate"
	duplicateOffering.ID = duplicateOffering.DeploymentID + ":" + duplicateOffering.NativeModelID
	catalog.Offerings = append(catalog.Offerings, duplicateOffering)

	if len(catalog.OfferingTemplates) == 0 {
		t.Fatal("reference catalog missing offering template fixture")
	}
	duplicateTemplate := catalog.OfferingTemplates[0]
	duplicateTemplate.ID += "-duplicate"
	catalog.OfferingTemplates = append(catalog.OfferingTemplates, duplicateTemplate)

	err := ValidateCatalogV1(catalog)
	if err == nil {
		t.Fatal("ValidateCatalogV1 succeeded with ambiguous deployment routes")
	}
	if !strings.Contains(err.Error(), "offering ") || !strings.Contains(err.Error(), "canonical_model_id") || !strings.Contains(err.Error(), "deployment_id") {
		t.Fatalf("validation error = %v, want ambiguous offering route detail", err)
	}
	if !strings.Contains(err.Error(), "offering_template ") {
		t.Fatalf("validation error = %v, want ambiguous offering_template route detail", err)
	}
}

func TestCatalogV1ParsesLegacyProviderKeyedCache(t *testing.T) {
	path := filepath.Join("testdata", "deployment_aware_compatibility", "old_provider_keyed_catalog_cache.json")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}

	catalog, err := ParseCatalogV1(data)
	if err != nil {
		t.Fatalf("ParseCatalogV1: %v", err)
	}
	compiled, err := CompileCatalogV1(catalog)
	if err != nil {
		t.Fatalf("CompileCatalogV1: %v", err)
	}

	canonicalID, ok := compiled.CanonicalModelForAliasOrID("gpt-4.1-2025-04-14")
	if !ok || canonicalID != "openai/gpt-4.1-2025-04-14" {
		t.Fatalf("CanonicalModelForAliasOrID = %q/%v, want openai canonical", canonicalID, ok)
	}
	offering, ok := compiled.OfferingForDeployment(canonicalID, "openai-direct")
	if !ok {
		t.Fatal("legacy openai model did not produce an openai-direct offering")
	}
	if offering.NativeModelID != "gpt-4.1-2025-04-14" {
		t.Errorf("NativeModelID = %q", offering.NativeModelID)
	}
	if offering.Pricing.RatesPer1M["input_tokens"] != 2 || offering.Pricing.RatesPer1M["output_tokens"] != 8 {
		t.Errorf("legacy pricing = %+v, want 2/8", offering.Pricing.RatesPer1M)
	}

	geminiDirect, ok := compiled.OfferingForDeployment("google/gemini-2.5-pro", "gemini-direct")
	if !ok {
		t.Fatal("legacy gemini model did not produce a gemini-direct offering")
	}
	geminiVertex, ok := compiled.OfferingForDeployment("google/gemini-2.5-pro", "gemini-vertex")
	if !ok {
		t.Fatal("legacy gemini-vertex model did not produce a gemini-vertex offering")
	}
	if geminiDirect.Pricing.RatesPer1M["input_tokens"] == geminiVertex.Pricing.RatesPer1M["input_tokens"] {
		t.Fatalf("deployment-specific pricing collapsed: direct=%+v vertex=%+v", geminiDirect.Pricing.RatesPer1M, geminiVertex.Pricing.RatesPer1M)
	}
}

func TestCatalogV1FromLegacyCatalogPreservesGemmaFreeDirectOnly(t *testing.T) {
	generatedAt := time.Date(2026, 5, 19, 0, 0, 0, 0, time.UTC)
	legacy := &LegacyCatalog{
		UpdatedAt: generatedAt,
		Source:    "providers",
		Providers: map[string][]ModelPricing{
			"gemini": {
				{
					ID:            "gemma-3-1b-it",
					Free:          true,
					ContextWindow: 32768,
					MaxOutput:     8192,
				},
			},
		},
	}

	catalog := CatalogV1FromLegacyCatalog(legacy)
	compiled, err := CompileCatalogV1(catalog)
	if err != nil {
		t.Fatalf("CompileCatalogV1: %v", err)
	}

	direct, ok := compiled.OfferingForDeployment("google/gemma-3-1b-it", "gemini-direct")
	if !ok {
		t.Fatal("gemma model did not produce a gemini-direct offering")
	}
	if direct.Pricing.Status != PricingFree {
		t.Fatalf("gemma pricing status = %q, want %q", direct.Pricing.Status, PricingFree)
	}
	if direct.Pricing.RatesPer1M["input_tokens"] != 0 || direct.Pricing.RatesPer1M["output_tokens"] != 0 {
		t.Fatalf("gemma pricing rates = %+v, want zero rates", direct.Pricing.RatesPer1M)
	}
	if direct.Capabilities.ServerTools["web_search"] == CapabilitySupported {
		t.Fatalf("gemma web_search = supported, want unknown or absent")
	}
	if _, ok := compiled.OfferingForDeployment("google/gemma-3-1b-it", "gemini-vertex"); ok {
		t.Fatal("gemma model produced a gemini-vertex offering")
	}
}

func TestParseCatalogV1RejectsEmptyObjectAsLegacy(t *testing.T) {
	_, err := ParseCatalogV1([]byte(`{}`))
	if err == nil {
		t.Fatal("ParseCatalogV1 succeeded with empty object")
	}
	if !strings.Contains(err.Error(), "schema_version") || !strings.Contains(err.Error(), "legacy provider-keyed catalog") {
		t.Fatalf("ParseCatalogV1 error = %v, want missing schema_version legacy rejection", err)
	}
}

func TestParseCatalogV1RejectsUnknownV1Fields(t *testing.T) {
	data, err := json.Marshal(ReferenceCatalogV1())
	if err != nil {
		t.Fatalf("marshal reference: %v", err)
	}
	var raw map[string]any
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatalf("unmarshal reference: %v", err)
	}
	raw["unexpected"] = true
	data, err = json.Marshal(raw)
	if err != nil {
		t.Fatalf("marshal raw: %v", err)
	}

	if _, err := ParseCatalogV1(data); err == nil {
		t.Fatal("ParseCatalogV1 succeeded with an unknown top-level field")
	}
}

func TestCompileCatalogV1ReportsStaleAndDroppedOfferings(t *testing.T) {
	catalog := ReferenceCatalogV1()
	catalog.GeneratedAt = time.Date(2000, 1, 1, 0, 0, 0, 0, time.UTC)
	catalog.StaleAfter = catalog.GeneratedAt.Add(24 * time.Hour)
	catalog.Offerings = append(catalog.Offerings, ModelOfferingV1{
		ID:               "missing-deployment:native",
		CanonicalModelID: "openai/gpt-4.1-2025-04-14",
		DeploymentID:     "missing-deployment",
		NativeModelID:    "native",
		Pricing: PricingV1{
			Status:      PricingKnown,
			Currency:    "USD",
			RatesPer1M:  map[string]float64{"input_tokens": 1, "output_tokens": 2},
			EffectiveAt: catalog.GeneratedAt,
		},
	})

	compiled, err := CompileCatalogV1(catalog)
	if err != nil {
		t.Fatalf("CompileCatalogV1: %v", err)
	}
	if compiled.OfferingsByID["missing-deployment:native"] != nil {
		t.Fatal("invalid offering should be dropped from indexes")
	}
	var sawStale, sawDropped bool
	for _, diagnostic := range compiled.Diagnostics {
		if diagnostic.Code == "stale_catalog" {
			sawStale = true
		}
		if diagnostic.Code == "dropped_offering" && strings.Contains(diagnostic.Message, "missing-deployment") {
			sawDropped = true
		}
	}
	if !sawStale || !sawDropped {
		t.Fatalf("diagnostics = %+v, want stale_catalog and dropped_offering", compiled.Diagnostics)
	}
}
