package models

import (
	"encoding/json"
	"fmt"
	"regexp"
	"sort"
	"strings"
	"time"
)

const CatalogV1SchemaVersion = "model-catalog/v1"

type CapabilityState string

const (
	CapabilitySupported   CapabilityState = "supported"
	CapabilityUnsupported CapabilityState = "unsupported"
	CapabilityUnknown     CapabilityState = "unknown"
)

type PricingStatus string

const (
	PricingKnown   PricingStatus = "known"
	PricingPartial PricingStatus = "partial"
	PricingUnknown PricingStatus = "unknown"
	PricingFree    PricingStatus = "free"
)

type NativeModelIDSource string

const (
	NativeModelIDCatalogKnown   NativeModelIDSource = "catalog_known"
	NativeModelIDDiscovered     NativeModelIDSource = "discovered"
	NativeModelIDUserConfigured NativeModelIDSource = "user_configured"
	NativeModelIDCatalogOrUser  NativeModelIDSource = "catalog_or_user_configured"
)

type ResponseCostSource string

const (
	ResponseCostUsageCountersOnly ResponseCostSource = "usage_counters_only"
	ResponseCostProviderExact     ResponseCostSource = "provider_exact"
	ResponseCostProviderAsync     ResponseCostSource = "provider_async_not_in_adapter"
	ResponseCostLocalFree         ResponseCostSource = "local_free"
)

// CatalogV1 is the normalized deployment-aware model catalog contract. It is
// intentionally data-only: it can select among known deployments and API
// protocols, but it cannot define arbitrary request templates or auth behavior.
type CatalogV1 struct {
	SchemaVersion     string                    `json:"schema_version"`
	GeneratedAt       time.Time                 `json:"generated_at"`
	StaleAfter        time.Time                 `json:"stale_after"`
	Providers         map[string]*ProviderV1    `json:"providers"`
	APIProtocols      map[string]*APIProtocolV1 `json:"api_protocols"`
	Deployments       map[string]*DeploymentV1  `json:"deployments"`
	Models            map[string]*ModelV1       `json:"models"`
	Offerings         []ModelOfferingV1         `json:"offerings"`
	OfferingTemplates []ModelOfferingTemplateV1 `json:"offering_templates,omitempty"`
	Aliases           map[string]string         `json:"aliases,omitempty"`
	Provenance        *ProvenanceV1             `json:"provenance,omitempty"`
}

type ProviderV1 struct {
	ID          string        `json:"id"`
	Name        string        `json:"name"`
	Description string        `json:"description,omitempty"`
	HomepageURL string        `json:"homepage_url,omitempty"`
	Aliases     []string      `json:"aliases,omitempty"`
	Provenance  *ProvenanceV1 `json:"provenance,omitempty"`
}

type APIProtocolV1 struct {
	ID          string        `json:"id"`
	Name        string        `json:"name"`
	Description string        `json:"description,omitempty"`
	Provenance  *ProvenanceV1 `json:"provenance,omitempty"`
}

type DeploymentV1 struct {
	ID                     string              `json:"id"`
	Name                   string              `json:"name"`
	ProviderID             string              `json:"provider_id"`
	APIProtocolID          string              `json:"api_protocol_id"`
	APIProtocolIDs         []string            `json:"api_protocol_ids,omitempty"`
	AdapterConstructor     string              `json:"adapter_constructor"`
	CredentialRequirements []CredentialV1      `json:"credential_requirements,omitempty"`
	EnvFallbacks           []EnvFallbackV1     `json:"env_fallbacks,omitempty"`
	NativeModelIDSource    NativeModelIDSource `json:"native_model_id_source"`
	ModelMappingsRequired  bool                `json:"model_mappings_required,omitempty"`
	Local                  bool                `json:"local,omitempty"`
	Provenance             *ProvenanceV1       `json:"provenance,omitempty"`
	Provider               *ProviderV1         `json:"-"`
	APIProtocol            *APIProtocolV1      `json:"-"`
}

type CredentialV1 struct {
	Field    string `json:"field"`
	Secret   bool   `json:"secret,omitempty"`
	Required bool   `json:"required,omitempty"`
}

type EnvFallbackV1 struct {
	Field string   `json:"field"`
	Env   []string `json:"env"`
}

type ModelV1 struct {
	ID            string        `json:"id"`
	ProviderID    string        `json:"provider_id"`
	Name          string        `json:"name"`
	Family        string        `json:"family,omitempty"`
	ContextWindow int           `json:"context_window,omitempty"`
	MaxOutput     int           `json:"max_output,omitempty"`
	Aliases       []string      `json:"aliases,omitempty"`
	Provenance    *ProvenanceV1 `json:"provenance,omitempty"`
	Provider      *ProviderV1   `json:"-"`
}

type ModelOfferingV1 struct {
	ID               string          `json:"id"`
	CanonicalModelID string          `json:"canonical_model_id"`
	DeploymentID     string          `json:"deployment_id"`
	NativeModelID    string          `json:"native_model_id"`
	APIProtocolID    string          `json:"api_protocol_id,omitempty"`
	Capabilities     CapabilitySetV1 `json:"capabilities,omitempty"`
	Pricing          PricingV1       `json:"pricing"`
	Provenance       *ProvenanceV1   `json:"provenance,omitempty"`
	Model            *ModelV1        `json:"-"`
	Deployment       *DeploymentV1   `json:"-"`
	Provider         *ProviderV1     `json:"-"`
	APIProtocol      *APIProtocolV1  `json:"-"`
}

// ModelOfferingTemplateV1 records that a deployment can serve a canonical
// model only after user config supplies the exact native model ID. It is not a
// routeable offering until resolution materializes a ModelOfferingV1 with the
// mapped native_model_id.
type ModelOfferingTemplateV1 struct {
	ID                  string              `json:"id"`
	CanonicalModelID    string              `json:"canonical_model_id"`
	DeploymentID        string              `json:"deployment_id"`
	APIProtocolID       string              `json:"api_protocol_id,omitempty"`
	NativeModelIDSource NativeModelIDSource `json:"native_model_id_source"`
	MappingRequired     bool                `json:"mapping_required"`
	Capabilities        CapabilitySetV1     `json:"capabilities,omitempty"`
	Pricing             PricingV1           `json:"pricing"`
	Provenance          *ProvenanceV1       `json:"provenance,omitempty"`
	Model               *ModelV1            `json:"-"`
	Deployment          *DeploymentV1       `json:"-"`
	Provider            *ProviderV1         `json:"-"`
	APIProtocol         *APIProtocolV1      `json:"-"`
}

type CapabilitySetV1 struct {
	ServerTools            map[string]CapabilityState `json:"server_tools,omitempty"`
	FunctionCalling        CapabilityState            `json:"function_calling,omitempty"`
	ExplicitThinkingBudget CapabilityState            `json:"explicit_thinking_budget,omitempty"`
}

type PricingV1 struct {
	Status            PricingStatus      `json:"status"`
	Currency          string             `json:"currency,omitempty"`
	EffectiveAt       time.Time          `json:"effective_at,omitempty"`
	RatesPer1M        map[string]float64 `json:"rates_per_1m,omitempty"`
	MissingDimensions []string           `json:"missing_dimensions,omitempty"`
	Notes             []string           `json:"notes,omitempty"`
	Source            string             `json:"source,omitempty"`
}

type ProvenanceV1 struct {
	Source     string    `json:"source"`
	SourceURL  string    `json:"source_url,omitempty"`
	ObservedAt time.Time `json:"observed_at,omitempty"`
}

type CatalogDiagnosticV1 struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

type CompiledCatalogV1 struct {
	Catalog                           *CatalogV1
	ProvidersByID                     map[string]*ProviderV1
	ProtocolsByID                     map[string]*APIProtocolV1
	DeploymentsByID                   map[string]*DeploymentV1
	ModelsByID                        map[string]*ModelV1
	OfferingsByID                     map[string]*ModelOfferingV1
	OfferingsByCanonicalModel         map[string][]*ModelOfferingV1
	OfferingsByDeployment             map[string][]*ModelOfferingV1
	OfferingTemplatesByCanonicalModel map[string][]*ModelOfferingTemplateV1
	OfferingTemplatesByDeployment     map[string][]*ModelOfferingTemplateV1
	Diagnostics                       []CatalogDiagnosticV1
}

type DeploymentBindingV1 struct {
	DeploymentID        string              `json:"deployment_id"`
	ProviderID          string              `json:"provider_id"`
	APIProtocolID       string              `json:"api_protocol_id"`
	APIProtocolIDs      []string            `json:"api_protocol_ids,omitempty"`
	AdapterConstructor  string              `json:"adapter_constructor"`
	CredentialFields    []string            `json:"credential_fields"`
	NativeModelIDSource NativeModelIDSource `json:"native_model_id_source"`
	ResponseCostSource  ResponseCostSource  `json:"response_cost_source"`
	Notes               string              `json:"notes,omitempty"`
}

var catalogTokenIDRe = regexp.MustCompile(`^[a-z0-9][a-z0-9._-]*$`)
var catalogProtocolIDRe = regexp.MustCompile(`^[a-z0-9][a-z0-9._-]*$`)

// CatalogV1JSONSchema is the public schema contract for remote catalog data.
// Runtime validation uses ValidateCatalogV1 so the repo does not need a JSON
// schema dependency just to enforce the catalog contract.
const CatalogV1JSONSchema = `{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://langdag.com/schemas/model-catalog-v1.schema.json",
  "type": "object",
  "additionalProperties": false,
  "required": ["schema_version", "generated_at", "stale_after", "providers", "api_protocols", "deployments", "models", "offerings"],
  "properties": {
    "schema_version": { "const": "model-catalog/v1" },
    "generated_at": { "type": "string", "format": "date-time" },
    "stale_after": { "type": "string", "format": "date-time" },
    "providers": { "type": "object", "patternProperties": { "^[a-z0-9][a-z0-9._-]*$": { "$ref": "#/$defs/provider" } }, "additionalProperties": false },
    "api_protocols": { "type": "object", "patternProperties": { "^[a-z0-9][a-z0-9._-]*$": { "$ref": "#/$defs/api_protocol" } }, "additionalProperties": false },
    "deployments": { "type": "object", "patternProperties": { "^[a-z0-9][a-z0-9._-]*$": { "$ref": "#/$defs/deployment" } }, "additionalProperties": false },
    "models": { "type": "object", "patternProperties": { "^[a-z0-9][^\\s/]+/.+$": { "$ref": "#/$defs/model" } }, "additionalProperties": false },
    "offerings": { "type": "array", "items": { "$ref": "#/$defs/offering" } },
    "offering_templates": { "type": "array", "items": { "$ref": "#/$defs/offering_template" } },
    "aliases": { "type": "object", "additionalProperties": { "type": "string" } },
    "provenance": { "$ref": "#/$defs/provenance" }
  },
  "$defs": {
    "provider": {
      "type": "object",
      "additionalProperties": false,
      "required": ["id", "name"],
      "properties": {
        "id": { "$ref": "#/$defs/token_id" },
        "name": { "type": "string", "minLength": 1 },
        "description": { "type": "string" },
        "homepage_url": { "type": "string" },
        "aliases": { "$ref": "#/$defs/string_list" },
        "provenance": { "$ref": "#/$defs/provenance" }
      }
    },
    "api_protocol": {
      "type": "object",
      "additionalProperties": false,
      "required": ["id", "name"],
      "properties": {
        "id": { "$ref": "#/$defs/token_id" },
        "name": { "type": "string", "minLength": 1 },
        "description": { "type": "string" },
        "provenance": { "$ref": "#/$defs/provenance" }
      }
    },
    "deployment": {
      "type": "object",
      "additionalProperties": false,
      "required": ["id", "name", "provider_id", "api_protocol_id", "adapter_constructor", "native_model_id_source"],
      "properties": {
        "id": { "$ref": "#/$defs/token_id" },
        "name": { "type": "string", "minLength": 1 },
        "provider_id": { "$ref": "#/$defs/token_id" },
        "api_protocol_id": { "$ref": "#/$defs/token_id" },
        "api_protocol_ids": { "$ref": "#/$defs/string_list" },
        "adapter_constructor": { "type": "string", "minLength": 1 },
        "credential_requirements": { "type": "array", "items": { "$ref": "#/$defs/credential" } },
        "env_fallbacks": { "type": "array", "items": { "$ref": "#/$defs/env_fallback" } },
        "native_model_id_source": { "$ref": "#/$defs/native_model_id_source" },
        "model_mappings_required": { "type": "boolean" },
        "local": { "type": "boolean" },
        "provenance": { "$ref": "#/$defs/provenance" }
      }
    },
    "model": {
      "type": "object",
      "additionalProperties": false,
      "required": ["id", "provider_id", "name"],
      "properties": {
        "id": { "type": "string", "pattern": "^[a-z0-9][^\\s/]+/.+$" },
        "provider_id": { "$ref": "#/$defs/token_id" },
        "name": { "type": "string", "minLength": 1 },
        "family": { "type": "string" },
        "context_window": { "type": "integer", "minimum": 0 },
        "max_output": { "type": "integer", "minimum": 0 },
        "aliases": { "$ref": "#/$defs/string_list" },
        "provenance": { "$ref": "#/$defs/provenance" }
      }
    },
    "offering": {
      "type": "object",
      "additionalProperties": false,
      "required": ["id", "canonical_model_id", "deployment_id", "native_model_id", "pricing"],
      "properties": {
        "id": { "type": "string", "pattern": "^[a-z0-9][a-z0-9._-]*:.+$" },
        "canonical_model_id": { "type": "string", "pattern": "^[a-z0-9][^\\s/]+/.+$" },
        "deployment_id": { "$ref": "#/$defs/token_id" },
        "native_model_id": { "type": "string", "minLength": 1 },
        "api_protocol_id": { "$ref": "#/$defs/token_id" },
        "capabilities": { "$ref": "#/$defs/capabilities" },
        "pricing": { "$ref": "#/$defs/pricing" },
        "provenance": { "$ref": "#/$defs/provenance" }
      }
    },
    "offering_template": {
      "type": "object",
      "additionalProperties": false,
      "required": ["id", "canonical_model_id", "deployment_id", "native_model_id_source", "mapping_required", "pricing"],
      "properties": {
        "id": { "type": "string", "minLength": 1 },
        "canonical_model_id": { "type": "string", "pattern": "^[a-z0-9][^\\s/]+/.+$" },
        "deployment_id": { "$ref": "#/$defs/token_id" },
        "api_protocol_id": { "$ref": "#/$defs/token_id" },
        "native_model_id_source": { "$ref": "#/$defs/native_model_id_source" },
        "mapping_required": { "type": "boolean", "const": true },
        "capabilities": { "$ref": "#/$defs/capabilities" },
        "pricing": { "$ref": "#/$defs/pricing" },
        "provenance": { "$ref": "#/$defs/provenance" }
      }
    },
    "capabilities": {
      "type": "object",
      "additionalProperties": false,
      "properties": {
        "server_tools": { "type": "object", "additionalProperties": { "$ref": "#/$defs/capability_state" } },
        "function_calling": { "$ref": "#/$defs/capability_state" },
        "explicit_thinking_budget": { "$ref": "#/$defs/capability_state" }
      }
    },
    "pricing": {
      "type": "object",
      "additionalProperties": false,
      "required": ["status"],
      "properties": {
        "status": { "enum": ["known", "partial", "unknown", "free"] },
        "currency": { "type": "string" },
        "effective_at": { "type": "string", "format": "date-time" },
        "rates_per_1m": { "type": "object", "additionalProperties": { "type": "number", "minimum": 0 } },
        "missing_dimensions": { "$ref": "#/$defs/string_list" },
        "notes": { "$ref": "#/$defs/string_list" },
        "source": { "type": "string" }
      }
    },
    "credential": {
      "type": "object",
      "additionalProperties": false,
      "required": ["field"],
      "properties": {
        "field": { "type": "string", "minLength": 1 },
        "secret": { "type": "boolean" },
        "required": { "type": "boolean" }
      }
    },
    "env_fallback": {
      "type": "object",
      "additionalProperties": false,
      "required": ["field", "env"],
      "properties": {
        "field": { "type": "string", "minLength": 1 },
        "env": { "type": "array", "items": { "type": "string", "minLength": 1 }, "minItems": 1 }
      }
    },
    "provenance": {
      "type": "object",
      "additionalProperties": false,
      "required": ["source"],
      "properties": {
        "source": { "type": "string", "minLength": 1 },
        "source_url": { "type": "string" },
        "observed_at": { "type": "string", "format": "date-time" }
      }
    },
    "token_id": { "type": "string", "pattern": "^[a-z0-9][a-z0-9._-]*$" },
    "native_model_id_source": { "enum": ["catalog_known", "discovered", "user_configured", "catalog_or_user_configured"] },
    "capability_state": { "enum": ["supported", "unsupported", "unknown"] },
    "string_list": { "type": "array", "items": { "type": "string" } }
  }
}`

func DeploymentBindingsV1() []DeploymentBindingV1 {
	return []DeploymentBindingV1{
		{
			DeploymentID:        "anthropic-direct",
			ProviderID:          "anthropic",
			APIProtocolID:       "anthropic-messages",
			AdapterConstructor:  "anthropic.New",
			CredentialFields:    []string{"api_key", "base_url"},
			NativeModelIDSource: NativeModelIDCatalogKnown,
			ResponseCostSource:  ResponseCostUsageCountersOnly,
		},
		{
			DeploymentID:        "anthropic-bedrock",
			ProviderID:          "anthropic",
			APIProtocolID:       "anthropic-messages",
			AdapterConstructor:  "anthropic.NewBedrock",
			CredentialFields:    []string{"region"},
			NativeModelIDSource: NativeModelIDCatalogKnown,
			ResponseCostSource:  ResponseCostUsageCountersOnly,
		},
		{
			DeploymentID:        "anthropic-vertex",
			ProviderID:          "anthropic",
			APIProtocolID:       "anthropic-messages",
			AdapterConstructor:  "anthropic.NewVertex",
			CredentialFields:    []string{"project_id", "region"},
			NativeModelIDSource: NativeModelIDCatalogKnown,
			ResponseCostSource:  ResponseCostUsageCountersOnly,
		},
		{
			DeploymentID:        "openai-direct",
			ProviderID:          "openai",
			APIProtocolID:       "openai-responses",
			APIProtocolIDs:      []string{"openai-responses", "openai-chat-completions"},
			AdapterConstructor:  "openai.New",
			CredentialFields:    []string{"api_key", "base_url"},
			NativeModelIDSource: NativeModelIDCatalogKnown,
			ResponseCostSource:  ResponseCostUsageCountersOnly,
		},
		{
			DeploymentID:        "openai-azure",
			ProviderID:          "openai",
			APIProtocolID:       "openai-chat-completions",
			AdapterConstructor:  "openai.NewAzure",
			CredentialFields:    []string{"api_key", "endpoint", "api_version", "model_mappings"},
			NativeModelIDSource: NativeModelIDUserConfigured,
			ResponseCostSource:  ResponseCostUsageCountersOnly,
			Notes:               "Azure deployment names are user-configured native_model_id values.",
		},
		{
			DeploymentID:        "gemini-direct",
			ProviderID:          "google",
			APIProtocolID:       "gemini-generate-content",
			AdapterConstructor:  "gemini.New",
			CredentialFields:    []string{"api_key", "base_url"},
			NativeModelIDSource: NativeModelIDCatalogKnown,
			ResponseCostSource:  ResponseCostUsageCountersOnly,
		},
		{
			DeploymentID:        "gemini-vertex",
			ProviderID:          "google",
			APIProtocolID:       "gemini-generate-content",
			AdapterConstructor:  "gemini.NewVertex",
			CredentialFields:    []string{"project_id", "region"},
			NativeModelIDSource: NativeModelIDCatalogKnown,
			ResponseCostSource:  ResponseCostUsageCountersOnly,
		},
		{
			DeploymentID:        "grok-direct",
			ProviderID:          "xai",
			APIProtocolID:       "openai-responses",
			AdapterConstructor:  "openai.NewGrok",
			CredentialFields:    []string{"api_key", "base_url"},
			NativeModelIDSource: NativeModelIDCatalogKnown,
			ResponseCostSource:  ResponseCostUsageCountersOnly,
		},
		{
			DeploymentID:        "openrouter",
			ProviderID:          "openrouter",
			APIProtocolID:       "openai-chat-completions",
			AdapterConstructor:  "openai.NewOpenRouter",
			CredentialFields:    []string{"api_key", "base_url"},
			NativeModelIDSource: NativeModelIDDiscovered,
			ResponseCostSource:  ResponseCostProviderAsync,
			Notes:               "Current adapter gets usage counters synchronously; exact credit accounting may require OpenRouter generation lookup.",
		},
		{
			DeploymentID:        "ollama-local",
			ProviderID:          "ollama",
			APIProtocolID:       "openai-chat-completions",
			AdapterConstructor:  "openai.NewOllama",
			CredentialFields:    []string{"base_url"},
			NativeModelIDSource: NativeModelIDDiscovered,
			ResponseCostSource:  ResponseCostLocalFree,
		},
		{
			DeploymentID:        "apple-local",
			ProviderID:          "apple",
			APIProtocolID:       "openai-chat-completions",
			AdapterConstructor:  "openai.NewApple",
			CredentialFields:    []string{"base_url"},
			NativeModelIDSource: NativeModelIDDiscovered,
			ResponseCostSource:  ResponseCostLocalFree,
		},
	}
}

func ReferenceCatalogV1() *CatalogV1 {
	generatedAt := time.Date(2026, 5, 15, 0, 0, 0, 0, time.UTC)
	staleAfter := generatedAt.Add(30 * 24 * time.Hour)
	observed := &ProvenanceV1{Source: "reference-catalog", ObservedAt: generatedAt}

	catalog := &CatalogV1{
		SchemaVersion: CatalogV1SchemaVersion,
		GeneratedAt:   generatedAt,
		StaleAfter:    staleAfter,
		Providers: map[string]*ProviderV1{
			"anthropic":  {ID: "anthropic", Name: "Anthropic", HomepageURL: "https://www.anthropic.com", Provenance: observed},
			"openai":     {ID: "openai", Name: "OpenAI", HomepageURL: "https://openai.com", Provenance: observed},
			"google":     {ID: "google", Name: "Google", HomepageURL: "https://ai.google.dev", Aliases: []string{"gemini"}, Provenance: observed},
			"xai":        {ID: "xai", Name: "xAI", HomepageURL: "https://x.ai", Aliases: []string{"grok"}, Provenance: observed},
			"z-ai":       {ID: "z-ai", Name: "Z.AI", Provenance: observed},
			"openrouter": {ID: "openrouter", Name: "OpenRouter", HomepageURL: "https://openrouter.ai", Provenance: observed},
			"ollama":     {ID: "ollama", Name: "Ollama", HomepageURL: "https://ollama.com", Provenance: observed},
			"apple":      {ID: "apple", Name: "Apple", HomepageURL: "https://developer.apple.com", Provenance: observed},
		},
		APIProtocols: map[string]*APIProtocolV1{
			"anthropic-messages":      {ID: "anthropic-messages", Name: "Anthropic Messages", Provenance: observed},
			"openai-chat-completions": {ID: "openai-chat-completions", Name: "OpenAI Chat Completions", Provenance: observed},
			"openai-responses":        {ID: "openai-responses", Name: "OpenAI Responses", Provenance: observed},
			"gemini-generate-content": {ID: "gemini-generate-content", Name: "Gemini generateContent", Provenance: observed},
		},
		Deployments: map[string]*DeploymentV1{},
		Models: map[string]*ModelV1{
			"anthropic/claude-sonnet-4-20250514": {
				ID: "anthropic/claude-sonnet-4-20250514", ProviderID: "anthropic", Name: "Claude Sonnet 4", Family: "claude-sonnet", ContextWindow: 200000, MaxOutput: 64000, Provenance: observed,
			},
			"openai/gpt-4.1-2025-04-14": {
				ID: "openai/gpt-4.1-2025-04-14", ProviderID: "openai", Name: "GPT-4.1", Family: "gpt-4.1", ContextWindow: 1047576, MaxOutput: 32768, Provenance: observed,
			},
			"google/gemini-2.5-pro": {
				ID: "google/gemini-2.5-pro", ProviderID: "google", Name: "Gemini 2.5 Pro", Family: "gemini-2.5", ContextWindow: 1048576, MaxOutput: 65536, Aliases: []string{"gemini/gemini-2.5-pro"}, Provenance: observed,
			},
			"xai/grok-4-1-fast-reasoning": {
				ID: "xai/grok-4-1-fast-reasoning", ProviderID: "xai", Name: "Grok 4.1 Fast Reasoning", Family: "grok-4.1", ContextWindow: 131072, MaxOutput: 16384, Provenance: observed,
			},
			"z-ai/glm-4.5-air:free": {
				ID: "z-ai/glm-4.5-air:free", ProviderID: "z-ai", Name: "Z.AI GLM 4.5 Air Free", Provenance: observed,
			},
			"ollama/llama3.1:8b": {
				ID: "ollama/llama3.1:8b", ProviderID: "ollama", Name: "llama3.1:8b", Provenance: observed,
			},
			"apple/system": {
				ID: "apple/system", ProviderID: "apple", Name: "Apple system", Provenance: observed,
			},
			"apple/pcc": {
				ID: "apple/pcc", ProviderID: "apple", Name: "Apple Private Cloud Compute", Provenance: observed,
			},
		},
		Offerings: []ModelOfferingV1{
			offeringV1("anthropic-direct", "anthropic/claude-sonnet-4-20250514", "claude-sonnet-4-20250514", PricingV1{Status: PricingKnown, Currency: "USD", EffectiveAt: generatedAt, RatesPer1M: map[string]float64{"input_tokens": 3, "output_tokens": 15, "cache_read_input_tokens": 0.30}, Source: "catalog"}, observed),
			offeringV1("anthropic-bedrock", "anthropic/claude-sonnet-4-20250514", "anthropic.claude-sonnet-4-20250514-v1:0", PricingV1{Status: PricingKnown, Currency: "USD", EffectiveAt: generatedAt, RatesPer1M: map[string]float64{"input_tokens": 3, "output_tokens": 15}, Source: "catalog"}, observed),
			offeringV1("anthropic-vertex", "anthropic/claude-sonnet-4-20250514", "claude-sonnet-4@20250514", PricingV1{Status: PricingPartial, Currency: "USD", EffectiveAt: generatedAt, RatesPer1M: map[string]float64{"input_tokens": 3}, MissingDimensions: []string{"output_tokens"}, Source: "catalog"}, observed),
			offeringV1("openai-direct", "openai/gpt-4.1-2025-04-14", "gpt-4.1-2025-04-14", PricingV1{Status: PricingKnown, Currency: "USD", EffectiveAt: generatedAt, RatesPer1M: map[string]float64{"input_tokens": 2, "output_tokens": 8, "cache_read_input_tokens": 0.50}, Source: "catalog"}, observed),
			offeringV1("gemini-direct", "google/gemini-2.5-pro", "gemini-2.5-pro", PricingV1{Status: PricingKnown, Currency: "USD", EffectiveAt: generatedAt, RatesPer1M: map[string]float64{"input_tokens": 1.25, "output_tokens": 10}, Source: "catalog"}, observed),
			offeringV1("gemini-vertex", "google/gemini-2.5-pro", "gemini-2.5-pro", PricingV1{Status: PricingKnown, Currency: "USD", EffectiveAt: generatedAt, RatesPer1M: map[string]float64{"input_tokens": 1.25, "output_tokens": 10}, Source: "catalog"}, observed),
			offeringV1("grok-direct", "xai/grok-4-1-fast-reasoning", "grok-4-1-fast-reasoning", PricingV1{Status: PricingKnown, Currency: "USD", EffectiveAt: generatedAt, RatesPer1M: map[string]float64{"input_tokens": 3, "output_tokens": 15}, Source: "catalog"}, observed),
			offeringV1("openrouter", "z-ai/glm-4.5-air:free", "z-ai/glm-4.5-air:free", PricingV1{Status: PricingFree, Currency: "USD", EffectiveAt: generatedAt, RatesPer1M: map[string]float64{"input_tokens": 0, "output_tokens": 0}, Source: "openrouter"}, observed),
			offeringV1("ollama-local", "ollama/llama3.1:8b", "llama3.1:8b", PricingV1{Status: PricingFree, Currency: "USD", EffectiveAt: generatedAt, RatesPer1M: map[string]float64{"input_tokens": 0, "output_tokens": 0}, Source: "local"}, observed),
		},
		OfferingTemplates: []ModelOfferingTemplateV1{
			offeringTemplateV1("openai-azure", "openai/gpt-4.1-2025-04-14", PricingV1{Status: PricingUnknown, Currency: "USD", EffectiveAt: generatedAt, MissingDimensions: []string{"input_tokens", "output_tokens"}, Source: "user-configured-azure-deployment"}, observed),
		},
		Aliases: map[string]string{
			"claude-sonnet-4-20250514":         "anthropic/claude-sonnet-4-20250514",
			"gpt-4.1-2025-04-14":               "openai/gpt-4.1-2025-04-14",
			"gemini-2.5-pro":                   "google/gemini-2.5-pro",
			"grok-4-1-fast-reasoning":          "xai/grok-4-1-fast-reasoning",
			"openrouter/z-ai/glm-4.5-air:free": "z-ai/glm-4.5-air:free",
		},
		Provenance: observed,
	}

	for _, binding := range DeploymentBindingsV1() {
		catalog.Deployments[binding.DeploymentID] = deploymentFromBindingV1(binding, observed)
	}

	return catalog
}

func offeringV1(deploymentID, canonicalID, nativeID string, pricing PricingV1, provenance *ProvenanceV1) ModelOfferingV1 {
	return ModelOfferingV1{
		ID:               deploymentID + ":" + nativeID,
		CanonicalModelID: canonicalID,
		DeploymentID:     deploymentID,
		NativeModelID:    nativeID,
		Capabilities: CapabilitySetV1{
			ServerTools: map[string]CapabilityState{"web_search": CapabilityUnknown},
		},
		Pricing:    pricing,
		Provenance: provenance,
	}
}

func offeringTemplateV1(deploymentID, canonicalID string, pricing PricingV1, provenance *ProvenanceV1) ModelOfferingTemplateV1 {
	return ModelOfferingTemplateV1{
		ID:                  deploymentID + ":" + canonicalID,
		CanonicalModelID:    canonicalID,
		DeploymentID:        deploymentID,
		NativeModelIDSource: NativeModelIDUserConfigured,
		MappingRequired:     true,
		Capabilities: CapabilitySetV1{
			ServerTools: map[string]CapabilityState{"web_search": CapabilityUnknown},
		},
		Pricing:    pricing,
		Provenance: provenance,
	}
}

func (t ModelOfferingTemplateV1) Materialize(nativeModelID string) (ModelOfferingV1, error) {
	if strings.TrimSpace(nativeModelID) == "" {
		return ModelOfferingV1{}, fmt.Errorf("catalog v1: native_model_id is required to materialize offering template %q", t.ID)
	}
	return ModelOfferingV1{
		ID:               t.DeploymentID + ":" + nativeModelID,
		CanonicalModelID: t.CanonicalModelID,
		DeploymentID:     t.DeploymentID,
		NativeModelID:    nativeModelID,
		APIProtocolID:    t.APIProtocolID,
		Capabilities:     t.Capabilities,
		Pricing:          t.Pricing,
		Provenance:       t.Provenance,
		Model:            t.Model,
		Deployment:       t.Deployment,
		Provider:         t.Provider,
		APIProtocol:      t.APIProtocol,
	}, nil
}

func deploymentFromBindingV1(binding DeploymentBindingV1, provenance *ProvenanceV1) *DeploymentV1 {
	requirements := make([]CredentialV1, 0, len(binding.CredentialFields))
	env := make([]EnvFallbackV1, 0, len(binding.CredentialFields))
	apiProtocolIDs := append([]string(nil), binding.APIProtocolIDs...)
	for _, field := range binding.CredentialFields {
		if field == "model_mappings" {
			requirements = append(requirements, CredentialV1{Field: field, Required: true})
			continue
		}
		secret := strings.Contains(field, "key")
		requirements = append(requirements, CredentialV1{Field: field, Secret: secret, Required: field != "base_url"})
	}
	switch binding.DeploymentID {
	case "anthropic-direct":
		env = append(env, EnvFallbackV1{Field: "api_key", Env: []string{"ANTHROPIC_API_KEY"}})
	case "openai-direct":
		env = append(env, EnvFallbackV1{Field: "api_key", Env: []string{"OPENAI_API_KEY"}}, EnvFallbackV1{Field: "base_url", Env: []string{"OPENAI_BASE_URL"}})
	case "openai-azure":
		env = append(env, EnvFallbackV1{Field: "api_key", Env: []string{"AZURE_OPENAI_API_KEY"}}, EnvFallbackV1{Field: "endpoint", Env: []string{"AZURE_OPENAI_ENDPOINT"}}, EnvFallbackV1{Field: "api_version", Env: []string{"AZURE_OPENAI_API_VERSION"}})
	case "gemini-direct":
		env = append(env, EnvFallbackV1{Field: "api_key", Env: []string{"GEMINI_API_KEY"}})
	case "grok-direct":
		env = append(env, EnvFallbackV1{Field: "api_key", Env: []string{"XAI_API_KEY"}}, EnvFallbackV1{Field: "base_url", Env: []string{"XAI_BASE_URL"}})
	case "openrouter":
		env = append(env, EnvFallbackV1{Field: "api_key", Env: []string{"OPENROUTER_API_KEY"}}, EnvFallbackV1{Field: "base_url", Env: []string{"OPENROUTER_BASE_URL"}})
	case "ollama-local":
		env = append(env, EnvFallbackV1{Field: "base_url", Env: []string{"OLLAMA_BASE_URL"}})
	case "apple-local":
		env = append(env, EnvFallbackV1{Field: "base_url", Env: []string{"APPLE_FM_BASE_URL"}})
	case "anthropic-bedrock":
		env = append(env, EnvFallbackV1{Field: "region", Env: []string{"AWS_REGION"}})
	case "anthropic-vertex", "gemini-vertex":
		env = append(env, EnvFallbackV1{Field: "project_id", Env: []string{"VERTEX_PROJECT_ID"}}, EnvFallbackV1{Field: "region", Env: []string{"VERTEX_REGION"}})
	}
	return &DeploymentV1{
		ID:                     binding.DeploymentID,
		Name:                   deploymentDisplayNameV1(binding.DeploymentID),
		ProviderID:             binding.ProviderID,
		APIProtocolID:          binding.APIProtocolID,
		APIProtocolIDs:         apiProtocolIDs,
		AdapterConstructor:     binding.AdapterConstructor,
		CredentialRequirements: requirements,
		EnvFallbacks:           env,
		NativeModelIDSource:    binding.NativeModelIDSource,
		ModelMappingsRequired:  binding.NativeModelIDSource == NativeModelIDUserConfigured,
		Local:                  binding.ResponseCostSource == ResponseCostLocalFree,
		Provenance:             provenance,
	}
}

func deploymentDisplayNameV1(id string) string {
	switch id {
	case "anthropic-direct":
		return "Anthropic direct"
	case "anthropic-bedrock":
		return "Anthropic on Bedrock"
	case "anthropic-vertex":
		return "Anthropic on Vertex"
	case "openai-direct":
		return "OpenAI direct"
	case "openai-azure":
		return "Azure OpenAI"
	case "gemini-direct":
		return "Gemini direct"
	case "gemini-vertex":
		return "Gemini on Vertex"
	case "grok-direct":
		return "Grok direct"
	case "openrouter":
		return "OpenRouter"
	case "ollama-local":
		return "Ollama local"
	case "apple-local":
		return "Apple Foundation Models local"
	default:
		return id
	}
}

func ValidateCatalogV1(catalog *CatalogV1) error {
	if catalog == nil {
		return fmt.Errorf("catalog v1: nil catalog")
	}
	var problems []string
	add := func(format string, args ...any) {
		problems = append(problems, fmt.Sprintf(format, args...))
	}
	if catalog.SchemaVersion != CatalogV1SchemaVersion {
		add("schema_version = %q, want %q", catalog.SchemaVersion, CatalogV1SchemaVersion)
	}
	if catalog.GeneratedAt.IsZero() {
		add("generated_at is required")
	}
	if catalog.StaleAfter.IsZero() {
		add("stale_after is required")
	} else if !catalog.GeneratedAt.IsZero() && !catalog.StaleAfter.After(catalog.GeneratedAt) {
		add("stale_after must be after generated_at")
	}
	if len(catalog.Providers) == 0 {
		add("providers is required")
	}
	if len(catalog.APIProtocols) == 0 {
		add("api_protocols is required")
	}
	if len(catalog.Deployments) == 0 {
		add("deployments is required")
	}
	if len(catalog.Models) == 0 {
		add("models is required")
	}
	if len(catalog.Offerings) == 0 {
		add("offerings is required")
	}

	for id, provider := range catalog.Providers {
		if provider == nil {
			add("provider %q is nil", id)
			continue
		}
		if id != provider.ID {
			add("provider map key %q does not match id %q", id, provider.ID)
		}
		if !catalogTokenIDRe.MatchString(provider.ID) {
			add("provider %q has invalid id", provider.ID)
		}
		if provider.Name == "" {
			add("provider %q missing name", provider.ID)
		}
	}
	for id, protocol := range catalog.APIProtocols {
		if protocol == nil {
			add("api_protocol %q is nil", id)
			continue
		}
		if id != protocol.ID {
			add("api_protocol map key %q does not match id %q", id, protocol.ID)
		}
		if !catalogProtocolIDRe.MatchString(protocol.ID) {
			add("api_protocol %q has invalid id", protocol.ID)
		}
		if protocol.Name == "" {
			add("api_protocol %q missing name", protocol.ID)
		}
	}
	for id, deployment := range catalog.Deployments {
		if deployment == nil {
			add("deployment %q is nil", id)
			continue
		}
		if id != deployment.ID {
			add("deployment map key %q does not match id %q", id, deployment.ID)
		}
		if !catalogTokenIDRe.MatchString(deployment.ID) {
			add("deployment %q has invalid id", deployment.ID)
		}
		if catalog.Providers[deployment.ProviderID] == nil {
			add("deployment %q references unknown provider %q", deployment.ID, deployment.ProviderID)
		}
		if catalog.APIProtocols[deployment.APIProtocolID] == nil {
			add("deployment %q references unknown api_protocol %q", deployment.ID, deployment.APIProtocolID)
		}
		if len(deployment.APIProtocolIDs) > 0 && !stringSliceContainsV1(deployment.APIProtocolIDs, deployment.APIProtocolID) {
			add("deployment %q api_protocol_ids must include api_protocol_id %q", deployment.ID, deployment.APIProtocolID)
		}
		for _, protocolID := range deployment.APIProtocolIDs {
			if !catalogProtocolIDRe.MatchString(protocolID) {
				add("deployment %q has invalid api_protocol_ids entry %q", deployment.ID, protocolID)
			}
			if catalog.APIProtocols[protocolID] == nil {
				add("deployment %q references unknown api_protocol %q in api_protocol_ids", deployment.ID, protocolID)
			}
		}
		if deployment.AdapterConstructor == "" {
			add("deployment %q missing adapter_constructor", deployment.ID)
		}
		if !validNativeModelSourceV1(deployment.NativeModelIDSource) {
			add("deployment %q has invalid native_model_id_source %q", deployment.ID, deployment.NativeModelIDSource)
		}
	}
	for id, model := range catalog.Models {
		if model == nil {
			add("model %q is nil", id)
			continue
		}
		if id != model.ID {
			add("model map key %q does not match id %q", id, model.ID)
		}
		if !validCanonicalModelIDV1(model.ID) {
			add("model %q has invalid canonical id", model.ID)
		}
		if catalog.Providers[model.ProviderID] == nil {
			add("model %q references unknown provider %q", model.ID, model.ProviderID)
		}
		if !strings.HasPrefix(model.ID, model.ProviderID+"/") {
			add("model %q must be owner-qualified with provider_id %q", model.ID, model.ProviderID)
		}
		if model.Name == "" {
			add("model %q missing name", model.ID)
		}
	}
	seenOfferings := map[string]bool{}
	seenOfferingRoutes := map[catalogV1DeploymentRouteKey]string{}
	for i := range catalog.Offerings {
		offering := &catalog.Offerings[i]
		if offering.ID == "" {
			add("offering[%d] missing id", i)
			continue
		}
		if seenOfferings[offering.ID] {
			add("offering %q is duplicated", offering.ID)
		}
		seenOfferings[offering.ID] = true
		if offering.CanonicalModelID != "" && offering.DeploymentID != "" {
			key := catalogV1DeploymentRouteKey{
				CanonicalModelID: offering.CanonicalModelID,
				DeploymentID:     offering.DeploymentID,
			}
			if previousID := seenOfferingRoutes[key]; previousID != "" && previousID != offering.ID {
				add("offering %q conflicts with offering %q for canonical_model_id %q deployment_id %q", offering.ID, previousID, offering.CanonicalModelID, offering.DeploymentID)
			} else if previousID == "" {
				seenOfferingRoutes[key] = offering.ID
			}
		}
		deploymentID, nativeID, ok := SplitOfferingIDV1(offering.ID)
		if !ok {
			add("offering %q must be deployment_id:native_model_id", offering.ID)
		}
		if ok && deploymentID != offering.DeploymentID {
			add("offering %q deployment segment %q does not match deployment_id %q", offering.ID, deploymentID, offering.DeploymentID)
		}
		if ok && nativeID != offering.NativeModelID {
			add("offering %q native segment %q does not match native_model_id %q", offering.ID, nativeID, offering.NativeModelID)
		}
		if catalog.Models[offering.CanonicalModelID] == nil {
			add("offering %q references unknown model %q", offering.ID, offering.CanonicalModelID)
		}
		if catalog.Deployments[offering.DeploymentID] == nil {
			add("offering %q references unknown deployment %q", offering.ID, offering.DeploymentID)
		}
		if offering.APIProtocolID != "" {
			if catalog.APIProtocols[offering.APIProtocolID] == nil {
				add("offering %q references unknown api_protocol %q", offering.ID, offering.APIProtocolID)
			}
			if deployment := catalog.Deployments[offering.DeploymentID]; deployment != nil && !deploymentSupportsProtocolV1(deployment, offering.APIProtocolID) {
				add("offering %q api_protocol_id %q is not supported by deployment %q", offering.ID, offering.APIProtocolID, offering.DeploymentID)
			}
		}
		validateCapabilitiesV1(add, offering.ID, offering.Capabilities)
		validatePricingV1(add, offering.ID, offering.Pricing)
	}
	seenTemplates := map[string]bool{}
	seenTemplateRoutes := map[catalogV1DeploymentRouteKey]string{}
	for i := range catalog.OfferingTemplates {
		template := &catalog.OfferingTemplates[i]
		if template.ID == "" {
			add("offering_template[%d] missing id", i)
			continue
		}
		if seenTemplates[template.ID] {
			add("offering_template %q is duplicated", template.ID)
		}
		seenTemplates[template.ID] = true
		if template.CanonicalModelID != "" && template.DeploymentID != "" {
			key := catalogV1DeploymentRouteKey{
				CanonicalModelID: template.CanonicalModelID,
				DeploymentID:     template.DeploymentID,
			}
			if previousID := seenTemplateRoutes[key]; previousID != "" && previousID != template.ID {
				add("offering_template %q conflicts with offering_template %q for canonical_model_id %q deployment_id %q", template.ID, previousID, template.CanonicalModelID, template.DeploymentID)
			} else if previousID == "" {
				seenTemplateRoutes[key] = template.ID
			}
		}
		if catalog.Models[template.CanonicalModelID] == nil {
			add("offering_template %q references unknown model %q", template.ID, template.CanonicalModelID)
		}
		deployment := catalog.Deployments[template.DeploymentID]
		if deployment == nil {
			add("offering_template %q references unknown deployment %q", template.ID, template.DeploymentID)
		} else if !deployment.ModelMappingsRequired {
			add("offering_template %q deployment %q does not require model mappings", template.ID, template.DeploymentID)
		}
		if template.NativeModelIDSource != NativeModelIDUserConfigured {
			add("offering_template %q must use native_model_id_source %q", template.ID, NativeModelIDUserConfigured)
		}
		if template.APIProtocolID != "" {
			if catalog.APIProtocols[template.APIProtocolID] == nil {
				add("offering_template %q references unknown api_protocol %q", template.ID, template.APIProtocolID)
			}
			if deployment != nil && !deploymentSupportsProtocolV1(deployment, template.APIProtocolID) {
				add("offering_template %q api_protocol_id %q is not supported by deployment %q", template.ID, template.APIProtocolID, template.DeploymentID)
			}
		}
		if !template.MappingRequired {
			add("offering_template %q must set mapping_required", template.ID)
		}
		validateCapabilitiesV1(add, template.ID, template.Capabilities)
		validatePricingV1(add, template.ID, template.Pricing)
	}
	for alias, target := range catalog.Aliases {
		if alias == "" || target == "" {
			add("aliases must not contain empty keys or values")
			continue
		}
		if catalog.Models[target] == nil {
			add("alias %q references unknown model %q", alias, target)
		}
	}
	if len(problems) > 0 {
		sort.Strings(problems)
		return fmt.Errorf("catalog v1 validation failed: %s", strings.Join(problems, "; "))
	}
	return nil
}

type catalogV1DeploymentRouteKey struct {
	CanonicalModelID string
	DeploymentID     string
}

func CompileCatalogV1(catalog *CatalogV1) (*CompiledCatalogV1, error) {
	if err := validateCatalogV1CompileEnvelope(catalog); err != nil {
		return nil, err
	}
	compiled := &CompiledCatalogV1{
		Catalog:                           catalog,
		ProvidersByID:                     map[string]*ProviderV1{},
		ProtocolsByID:                     map[string]*APIProtocolV1{},
		DeploymentsByID:                   map[string]*DeploymentV1{},
		ModelsByID:                        map[string]*ModelV1{},
		OfferingsByID:                     map[string]*ModelOfferingV1{},
		OfferingsByCanonicalModel:         map[string][]*ModelOfferingV1{},
		OfferingsByDeployment:             map[string][]*ModelOfferingV1{},
		OfferingTemplatesByCanonicalModel: map[string][]*ModelOfferingTemplateV1{},
		OfferingTemplatesByDeployment:     map[string][]*ModelOfferingTemplateV1{},
	}
	for id, provider := range catalog.Providers {
		if provider == nil || id != provider.ID || !catalogTokenIDRe.MatchString(provider.ID) || provider.Name == "" {
			return nil, fmt.Errorf("catalog v1: invalid provider %q", id)
		}
		compiled.ProvidersByID[id] = provider
	}
	for id, protocol := range catalog.APIProtocols {
		if protocol == nil || id != protocol.ID || !catalogProtocolIDRe.MatchString(protocol.ID) || protocol.Name == "" {
			return nil, fmt.Errorf("catalog v1: invalid api_protocol %q", id)
		}
		compiled.ProtocolsByID[id] = protocol
	}
	for id, deployment := range catalog.Deployments {
		if deployment == nil || id != deployment.ID || !catalogTokenIDRe.MatchString(deployment.ID) || deployment.AdapterConstructor == "" || !validNativeModelSourceV1(deployment.NativeModelIDSource) {
			return nil, fmt.Errorf("catalog v1: invalid deployment %q", id)
		}
		deployment.Provider = compiled.ProvidersByID[deployment.ProviderID]
		deployment.APIProtocol = compiled.ProtocolsByID[deployment.APIProtocolID]
		if deployment.Provider == nil {
			return nil, fmt.Errorf("catalog v1: deployment %q references unknown provider %q", deployment.ID, deployment.ProviderID)
		}
		if deployment.APIProtocol == nil {
			return nil, fmt.Errorf("catalog v1: deployment %q references unknown api_protocol %q", deployment.ID, deployment.APIProtocolID)
		}
		if len(deployment.APIProtocolIDs) > 0 && !stringSliceContainsV1(deployment.APIProtocolIDs, deployment.APIProtocolID) {
			return nil, fmt.Errorf("catalog v1: deployment %q api_protocol_ids must include api_protocol_id %q", deployment.ID, deployment.APIProtocolID)
		}
		for _, protocolID := range deployment.APIProtocolIDs {
			if compiled.ProtocolsByID[protocolID] == nil {
				return nil, fmt.Errorf("catalog v1: deployment %q references unknown api_protocol %q in api_protocol_ids", deployment.ID, protocolID)
			}
		}
		compiled.DeploymentsByID[id] = deployment
	}
	for id, model := range catalog.Models {
		if model == nil || id != model.ID || !validCanonicalModelIDV1(model.ID) || model.Name == "" {
			return nil, fmt.Errorf("catalog v1: invalid model %q", id)
		}
		model.Provider = compiled.ProvidersByID[model.ProviderID]
		if model.Provider == nil {
			return nil, fmt.Errorf("catalog v1: model %q references unknown provider %q", model.ID, model.ProviderID)
		}
		if !strings.HasPrefix(model.ID, model.ProviderID+"/") {
			return nil, fmt.Errorf("catalog v1: model %q must be owner-qualified with provider_id %q", model.ID, model.ProviderID)
		}
		compiled.ModelsByID[id] = model
	}
	if time.Now().UTC().After(catalog.StaleAfter) {
		compiled.Diagnostics = append(compiled.Diagnostics, CatalogDiagnosticV1{
			Code:    "stale_catalog",
			Message: fmt.Sprintf("catalog generated at %s is stale after %s", catalog.GeneratedAt.Format(time.RFC3339), catalog.StaleAfter.Format(time.RFC3339)),
		})
	}
	seenOfferings := map[string]bool{}
	seenOfferingRoutes := map[catalogV1DeploymentRouteKey]string{}
	for i := range catalog.Offerings {
		offering := &catalog.Offerings[i]
		if seenOfferings[offering.ID] {
			compiled.Diagnostics = append(compiled.Diagnostics, CatalogDiagnosticV1{Code: "dropped_offering", Message: fmt.Sprintf("dropped duplicate offering %q", offering.ID)})
			continue
		}
		seenOfferings[offering.ID] = true
		if problems := compileOfferingProblemsV1(compiled, offering); len(problems) > 0 {
			compiled.Diagnostics = append(compiled.Diagnostics, CatalogDiagnosticV1{
				Code:    "dropped_offering",
				Message: fmt.Sprintf("dropped offering %q: %s", offering.ID, strings.Join(problems, "; ")),
			})
			continue
		}
		key := catalogV1DeploymentRouteKey{CanonicalModelID: offering.CanonicalModelID, DeploymentID: offering.DeploymentID}
		if previousID := seenOfferingRoutes[key]; previousID != "" {
			compiled.Diagnostics = append(compiled.Diagnostics, CatalogDiagnosticV1{
				Code:    "dropped_offering",
				Message: fmt.Sprintf("dropped ambiguous offering %q: canonical_model_id %q deployment_id %q already provided by offering %q", offering.ID, offering.CanonicalModelID, offering.DeploymentID, previousID),
			})
			continue
		}
		seenOfferingRoutes[key] = offering.ID
		linkOfferingV1(compiled, offering)
		compiled.OfferingsByID[offering.ID] = offering
		compiled.OfferingsByCanonicalModel[offering.CanonicalModelID] = append(compiled.OfferingsByCanonicalModel[offering.CanonicalModelID], offering)
		compiled.OfferingsByDeployment[offering.DeploymentID] = append(compiled.OfferingsByDeployment[offering.DeploymentID], offering)
	}
	seenTemplates := map[string]bool{}
	seenTemplateRoutes := map[catalogV1DeploymentRouteKey]string{}
	for i := range catalog.OfferingTemplates {
		template := &catalog.OfferingTemplates[i]
		if seenTemplates[template.ID] {
			compiled.Diagnostics = append(compiled.Diagnostics, CatalogDiagnosticV1{Code: "dropped_offering_template", Message: fmt.Sprintf("dropped duplicate offering_template %q", template.ID)})
			continue
		}
		seenTemplates[template.ID] = true
		if problems := compileOfferingTemplateProblemsV1(compiled, template); len(problems) > 0 {
			compiled.Diagnostics = append(compiled.Diagnostics, CatalogDiagnosticV1{
				Code:    "dropped_offering_template",
				Message: fmt.Sprintf("dropped offering_template %q: %s", template.ID, strings.Join(problems, "; ")),
			})
			continue
		}
		key := catalogV1DeploymentRouteKey{CanonicalModelID: template.CanonicalModelID, DeploymentID: template.DeploymentID}
		if previousID := seenTemplateRoutes[key]; previousID != "" {
			compiled.Diagnostics = append(compiled.Diagnostics, CatalogDiagnosticV1{
				Code:    "dropped_offering_template",
				Message: fmt.Sprintf("dropped ambiguous offering_template %q: canonical_model_id %q deployment_id %q already provided by offering_template %q", template.ID, template.CanonicalModelID, template.DeploymentID, previousID),
			})
			continue
		}
		seenTemplateRoutes[key] = template.ID
		linkOfferingTemplateV1(compiled, template)
		compiled.OfferingTemplatesByCanonicalModel[template.CanonicalModelID] = append(compiled.OfferingTemplatesByCanonicalModel[template.CanonicalModelID], template)
		compiled.OfferingTemplatesByDeployment[template.DeploymentID] = append(compiled.OfferingTemplatesByDeployment[template.DeploymentID], template)
	}
	return compiled, nil
}

func validateCatalogV1CompileEnvelope(catalog *CatalogV1) error {
	if catalog == nil {
		return fmt.Errorf("catalog v1: nil catalog")
	}
	var problems []string
	if catalog.SchemaVersion != CatalogV1SchemaVersion {
		problems = append(problems, fmt.Sprintf("schema_version = %q, want %q", catalog.SchemaVersion, CatalogV1SchemaVersion))
	}
	if catalog.GeneratedAt.IsZero() {
		problems = append(problems, "generated_at is required")
	}
	if catalog.StaleAfter.IsZero() {
		problems = append(problems, "stale_after is required")
	} else if !catalog.GeneratedAt.IsZero() && !catalog.StaleAfter.After(catalog.GeneratedAt) {
		problems = append(problems, "stale_after must be after generated_at")
	}
	if len(catalog.Providers) == 0 {
		problems = append(problems, "providers is required")
	}
	if len(catalog.APIProtocols) == 0 {
		problems = append(problems, "api_protocols is required")
	}
	if len(catalog.Deployments) == 0 {
		problems = append(problems, "deployments is required")
	}
	if len(catalog.Models) == 0 {
		problems = append(problems, "models is required")
	}
	if len(catalog.Offerings) == 0 {
		problems = append(problems, "offerings is required")
	}
	if len(problems) > 0 {
		sort.Strings(problems)
		return fmt.Errorf("catalog v1 validation failed: %s", strings.Join(problems, "; "))
	}
	return nil
}

func compileOfferingProblemsV1(compiled *CompiledCatalogV1, offering *ModelOfferingV1) []string {
	var problems []string
	if offering.ID == "" {
		problems = append(problems, "missing id")
		return problems
	}
	deploymentID, nativeID, ok := SplitOfferingIDV1(offering.ID)
	if !ok {
		problems = append(problems, "id must be deployment_id:native_model_id")
	}
	if ok && deploymentID != offering.DeploymentID {
		problems = append(problems, fmt.Sprintf("deployment segment %q does not match deployment_id %q", deploymentID, offering.DeploymentID))
	}
	if ok && nativeID != offering.NativeModelID {
		problems = append(problems, fmt.Sprintf("native segment %q does not match native_model_id %q", nativeID, offering.NativeModelID))
	}
	if compiled.ModelsByID[offering.CanonicalModelID] == nil {
		problems = append(problems, fmt.Sprintf("unknown model %q", offering.CanonicalModelID))
	}
	if compiled.DeploymentsByID[offering.DeploymentID] == nil {
		problems = append(problems, fmt.Sprintf("unknown deployment %q", offering.DeploymentID))
	}
	if offering.APIProtocolID != "" {
		if compiled.ProtocolsByID[offering.APIProtocolID] == nil {
			problems = append(problems, fmt.Sprintf("unknown api_protocol %q", offering.APIProtocolID))
		} else if deployment := compiled.DeploymentsByID[offering.DeploymentID]; deployment != nil && !deploymentSupportsProtocolV1(deployment, offering.APIProtocolID) {
			problems = append(problems, fmt.Sprintf("api_protocol %q is not supported by deployment %q", offering.APIProtocolID, offering.DeploymentID))
		}
	}
	collectCapabilityProblemsV1(&problems, offering.ID, offering.Capabilities)
	collectPricingProblemsV1(&problems, offering.ID, offering.Pricing)
	return problems
}

func compileOfferingTemplateProblemsV1(compiled *CompiledCatalogV1, template *ModelOfferingTemplateV1) []string {
	var problems []string
	if template.ID == "" {
		problems = append(problems, "missing id")
		return problems
	}
	if compiled.ModelsByID[template.CanonicalModelID] == nil {
		problems = append(problems, fmt.Sprintf("unknown model %q", template.CanonicalModelID))
	}
	deployment := compiled.DeploymentsByID[template.DeploymentID]
	if deployment == nil {
		problems = append(problems, fmt.Sprintf("unknown deployment %q", template.DeploymentID))
	} else if !deployment.ModelMappingsRequired {
		problems = append(problems, fmt.Sprintf("deployment %q does not require model mappings", template.DeploymentID))
	}
	if template.NativeModelIDSource != NativeModelIDUserConfigured {
		problems = append(problems, fmt.Sprintf("native_model_id_source must be %q", NativeModelIDUserConfigured))
	}
	if template.APIProtocolID != "" {
		if compiled.ProtocolsByID[template.APIProtocolID] == nil {
			problems = append(problems, fmt.Sprintf("unknown api_protocol %q", template.APIProtocolID))
		} else if deployment != nil && !deploymentSupportsProtocolV1(deployment, template.APIProtocolID) {
			problems = append(problems, fmt.Sprintf("api_protocol %q is not supported by deployment %q", template.APIProtocolID, template.DeploymentID))
		}
	}
	if !template.MappingRequired {
		problems = append(problems, "mapping_required must be set")
	}
	collectCapabilityProblemsV1(&problems, template.ID, template.Capabilities)
	collectPricingProblemsV1(&problems, template.ID, template.Pricing)
	return problems
}

func linkOfferingV1(compiled *CompiledCatalogV1, offering *ModelOfferingV1) {
	offering.Model = compiled.ModelsByID[offering.CanonicalModelID]
	offering.Deployment = compiled.DeploymentsByID[offering.DeploymentID]
	if offering.Model != nil {
		offering.Provider = offering.Model.Provider
	}
	protocolID := offering.APIProtocolID
	if protocolID == "" && offering.Deployment != nil {
		protocolID = offering.Deployment.APIProtocolID
	}
	offering.APIProtocol = compiled.ProtocolsByID[protocolID]
}

func linkOfferingTemplateV1(compiled *CompiledCatalogV1, template *ModelOfferingTemplateV1) {
	template.Model = compiled.ModelsByID[template.CanonicalModelID]
	template.Deployment = compiled.DeploymentsByID[template.DeploymentID]
	if template.Model != nil {
		template.Provider = template.Model.Provider
	}
	protocolID := template.APIProtocolID
	if protocolID == "" && template.Deployment != nil {
		protocolID = template.Deployment.APIProtocolID
	}
	template.APIProtocol = compiled.ProtocolsByID[protocolID]
}

func ParseCatalogV1(data []byte) (*CatalogV1, error) {
	var envelope struct {
		SchemaVersion string `json:"schema_version"`
	}
	if err := json.Unmarshal(data, &envelope); err != nil {
		return nil, fmt.Errorf("catalog v1: unmarshal envelope: %w", err)
	}
	if envelope.SchemaVersion == "" {
		var legacy LegacyCatalog
		if err := json.Unmarshal(data, &legacy); err != nil {
			return nil, fmt.Errorf("catalog v1: unmarshal legacy catalog: %w", err)
		}
		if !legacyCatalogV1HasMigratableModel(&legacy) {
			return nil, fmt.Errorf("catalog v1: schema_version is required unless data is a legacy provider-keyed catalog with at least one recognized model")
		}
		catalog := CatalogV1FromLegacyCatalog(&legacy)
		NormalizeCatalogV1(catalog)
		if err := ValidateCatalogV1(catalog); err != nil {
			return nil, err
		}
		return catalog, nil
	}
	var catalog CatalogV1
	if err := strictJSONUnmarshal(data, &catalog); err != nil {
		return nil, fmt.Errorf("catalog v1: unmarshal: %w", err)
	}
	NormalizeCatalogV1(&catalog)
	if err := ValidateCatalogV1(&catalog); err != nil {
		return nil, err
	}
	return &catalog, nil
}

func legacyCatalogV1HasMigratableModel(legacy *LegacyCatalog) bool {
	if legacy == nil {
		return false
	}
	for provider, models := range legacy.Providers {
		deploymentID, _ := legacyDeploymentAndOwnerV1(provider)
		if deploymentID == "" {
			continue
		}
		for _, model := range models {
			if strings.TrimSpace(model.ID) != "" {
				return true
			}
		}
	}
	return false
}

// ParseRemoteCatalogV1 parses catalog data from the published remote artifact.
// Unlike local cache loading, remote loading is intentionally v1-only and does
// not accept legacy provider-keyed cache data.
func ParseRemoteCatalogV1(data []byte) (*CatalogV1, error) {
	var envelope struct {
		SchemaVersion string `json:"schema_version"`
	}
	if err := json.Unmarshal(data, &envelope); err != nil {
		return nil, fmt.Errorf("catalog v1: unmarshal envelope: %w", err)
	}
	if envelope.SchemaVersion != CatalogV1SchemaVersion {
		return nil, fmt.Errorf("catalog v1: schema_version = %q, want %q", envelope.SchemaVersion, CatalogV1SchemaVersion)
	}
	var catalog CatalogV1
	if err := strictJSONUnmarshal(data, &catalog); err != nil {
		return nil, fmt.Errorf("catalog v1: unmarshal: %w", err)
	}
	NormalizeCatalogV1(&catalog)
	if err := ValidateCatalogV1(&catalog); err != nil {
		return nil, err
	}
	return &catalog, nil
}

func SplitOfferingIDV1(id string) (deploymentID string, nativeModelID string, ok bool) {
	left, right, found := strings.Cut(id, ":")
	if !found || left == "" || right == "" {
		return "", "", false
	}
	return left, right, true
}

func (c *CompiledCatalogV1) OfferingForDeployment(canonicalModelID, deploymentID string) (*ModelOfferingV1, bool) {
	if c == nil {
		return nil, false
	}
	for _, offering := range c.OfferingsByCanonicalModel[canonicalModelID] {
		if offering.DeploymentID == deploymentID {
			return offering, true
		}
	}
	return nil, false
}

func (c *CompiledCatalogV1) CanonicalModelForAliasOrID(value string) (string, bool) {
	if c == nil || value == "" {
		return "", false
	}
	if c.ModelsByID[value] != nil {
		return value, true
	}
	if target := c.Catalog.Aliases[value]; target != "" && c.ModelsByID[target] != nil {
		return target, true
	}
	for _, model := range c.ModelsByID {
		for _, alias := range model.Aliases {
			if alias == value {
				return model.ID, true
			}
		}
	}
	return "", false
}

func CatalogV1FromLegacyCatalog(legacy *LegacyCatalog) *CatalogV1 {
	base := ReferenceCatalogV1()
	if legacy == nil || len(legacy.Providers) == 0 {
		return base
	}
	base.Models = map[string]*ModelV1{}
	base.Offerings = nil
	base.OfferingTemplates = nil
	base.Aliases = map[string]string{}
	if !legacy.UpdatedAt.IsZero() {
		base.GeneratedAt = legacy.UpdatedAt
		base.StaleAfter = legacy.UpdatedAt.Add(30 * 24 * time.Hour)
	}
	base.SourceFromLegacyCatalog(legacy)
	addDerivedDeploymentOfferingsV1(base, base.GeneratedAt)
	mergeReferenceDynamicPlaceholdersV1(base)
	NormalizeCatalogV1(base)
	return base
}

func (c *CatalogV1) SourceFromLegacyCatalog(legacy *LegacyCatalog) {
	for oldProvider, oldModels := range legacy.Providers {
		deploymentID, ownerProviderID := legacyDeploymentAndOwnerV1(oldProvider)
		if deploymentID == "" {
			continue
		}
		if c.Deployments[deploymentID] == nil {
			for _, binding := range DeploymentBindingsV1() {
				if binding.DeploymentID == deploymentID {
					c.Deployments[deploymentID] = deploymentFromBindingV1(binding, &ProvenanceV1{Source: "legacy-provider-keyed-catalog", ObservedAt: legacy.UpdatedAt})
					break
				}
			}
		}
		for _, oldModel := range oldModels {
			if oldModel.ID == "" {
				continue
			}
			modelOwnerProviderID := ownerProviderID
			if oldProvider == "openrouter" {
				if owner, _, ok := strings.Cut(oldModel.ID, "/"); ok && catalogTokenIDRe.MatchString(owner) {
					modelOwnerProviderID = owner
					if c.Providers[modelOwnerProviderID] == nil {
						c.Providers[modelOwnerProviderID] = &ProviderV1{
							ID:         modelOwnerProviderID,
							Name:       modelOwnerProviderID,
							Provenance: &ProvenanceV1{Source: "legacy-provider-keyed-catalog", ObservedAt: legacy.UpdatedAt},
						}
					}
				}
			}
			canonicalID := modelOwnerProviderID + "/" + oldModel.ID
			if oldProvider == "openrouter" && strings.HasPrefix(oldModel.ID, modelOwnerProviderID+"/") {
				canonicalID = oldModel.ID
			}
			if c.Models[canonicalID] == nil {
				c.Models[canonicalID] = &ModelV1{
					ID:            canonicalID,
					ProviderID:    modelOwnerProviderID,
					Name:          oldModel.ID,
					ContextWindow: oldModel.ContextWindow,
					MaxOutput:     oldModel.MaxOutput,
					Aliases:       []string{oldModel.ID},
					Provenance:    &ProvenanceV1{Source: "legacy-provider-keyed-catalog", ObservedAt: legacy.UpdatedAt},
				}
			}
			pricing := PricingV1{
				Status:      PricingKnown,
				Currency:    "USD",
				EffectiveAt: legacy.UpdatedAt,
				RatesPer1M: map[string]float64{
					"input_tokens":  oldModel.InputPricePer1M,
					"output_tokens": oldModel.OutputPricePer1M,
				},
				Source: legacy.Source,
			}
			if oldModel.Free {
				pricing.Status = PricingFree
				pricing.RatesPer1M = map[string]float64{
					"input_tokens":  0,
					"output_tokens": 0,
				}
			} else if oldModel.InputPricePer1M == 0 && oldModel.OutputPricePer1M == 0 {
				pricing.Status = PricingUnknown
				pricing.RatesPer1M = nil
				pricing.MissingDimensions = []string{"input_tokens", "output_tokens"}
			}
			capabilities := CapabilitySetV1{ServerTools: map[string]CapabilityState{}}
			for _, tool := range oldModel.ServerTools {
				capabilities.ServerTools[tool] = CapabilitySupported
			}
			if len(capabilities.ServerTools) == 0 {
				capabilities.ServerTools["web_search"] = CapabilityUnknown
			}
			offering := ModelOfferingV1{
				ID:               deploymentID + ":" + oldModel.ID,
				CanonicalModelID: canonicalID,
				DeploymentID:     deploymentID,
				NativeModelID:    oldModel.ID,
				Capabilities:     capabilities,
				Pricing:          pricing,
				Provenance:       &ProvenanceV1{Source: "legacy-provider-keyed-catalog", ObservedAt: legacy.UpdatedAt},
			}
			if !catalogV1HasOffering(c.Offerings, offering.ID) {
				c.Offerings = append(c.Offerings, offering)
			}
			if c.Aliases == nil {
				c.Aliases = map[string]string{}
			}
			if _, exists := c.Aliases[oldModel.ID]; !exists {
				c.Aliases[oldModel.ID] = canonicalID
			}
		}
	}
}

func mergeReferenceDynamicPlaceholdersV1(catalog *CatalogV1) {
	if catalog == nil {
		return
	}
	ensureReferencePlaceholderMapsV1(catalog)
	reference := ReferenceCatalogV1()
	for id, provider := range reference.Providers {
		if catalog.Providers[id] == nil {
			copyProvider := *provider
			catalog.Providers[id] = &copyProvider
		}
	}
	for id, protocol := range reference.APIProtocols {
		if catalog.APIProtocols[id] == nil {
			copyProtocol := *protocol
			catalog.APIProtocols[id] = &copyProtocol
		}
	}
	for id, deployment := range reference.Deployments {
		if catalog.Deployments[id] == nil {
			copyDeployment := *deployment
			catalog.Deployments[id] = &copyDeployment
		}
	}
	for id, model := range reference.Models {
		if model.ProviderID != "z-ai" && model.ProviderID != "ollama" && model.ProviderID != "apple" {
			continue
		}
		if catalog.Models[id] == nil {
			copyModel := *model
			catalog.Models[id] = &copyModel
		}
	}
	for _, offering := range reference.Offerings {
		if offering.DeploymentID != "openrouter" && offering.DeploymentID != "ollama-local" {
			continue
		}
		if !catalogV1HasOffering(catalog.Offerings, offering.ID) {
			catalog.Offerings = append(catalog.Offerings, offering)
		}
	}
	for alias, target := range reference.Aliases {
		if catalog.Models[target] == nil {
			continue
		}
		if catalog.Aliases == nil {
			catalog.Aliases = map[string]string{}
		}
		if catalog.Aliases[alias] == "" {
			catalog.Aliases[alias] = target
		}
	}
}

func mergeReferenceRuntimePlaceholdersV1(catalog *CatalogV1) {
	if catalog == nil {
		return
	}
	ensureReferencePlaceholderMapsV1(catalog)
	reference := ReferenceCatalogV1()
	for id, provider := range reference.Providers {
		if catalog.Providers[id] == nil {
			copyProvider := *provider
			catalog.Providers[id] = &copyProvider
		}
	}
	for id, protocol := range reference.APIProtocols {
		if catalog.APIProtocols[id] == nil {
			copyProtocol := *protocol
			catalog.APIProtocols[id] = &copyProtocol
		}
	}
	for id, deployment := range reference.Deployments {
		if catalog.Deployments[id] == nil {
			copyDeployment := *deployment
			catalog.Deployments[id] = &copyDeployment
		}
	}
	for id, model := range reference.Models {
		if model.ProviderID != "apple" {
			continue
		}
		if catalog.Models[id] == nil {
			copyModel := *model
			catalog.Models[id] = &copyModel
		}
	}
}

func ensureReferencePlaceholderMapsV1(catalog *CatalogV1) {
	if catalog.Providers == nil {
		catalog.Providers = map[string]*ProviderV1{}
	}
	if catalog.APIProtocols == nil {
		catalog.APIProtocols = map[string]*APIProtocolV1{}
	}
	if catalog.Deployments == nil {
		catalog.Deployments = map[string]*DeploymentV1{}
	}
	if catalog.Models == nil {
		catalog.Models = map[string]*ModelV1{}
	}
}

func legacyDeploymentAndOwnerV1(provider string) (deploymentID string, ownerProviderID string) {
	switch provider {
	case "anthropic":
		return "anthropic-direct", "anthropic"
	case "anthropic-bedrock":
		return "anthropic-bedrock", "anthropic"
	case "anthropic-vertex":
		return "anthropic-vertex", "anthropic"
	case "openai":
		return "openai-direct", "openai"
	case "openai-azure":
		return "openai-azure", "openai"
	case "gemini", "gemma":
		return "gemini-direct", "google"
	case "gemini-vertex":
		return "gemini-vertex", "google"
	case "grok":
		return "grok-direct", "xai"
	case "openrouter":
		return "openrouter", "openrouter"
	case "ollama":
		return "ollama-local", "ollama"
	case "apple":
		return "apple-local", "apple"
	default:
		return "", ""
	}
}

func catalogV1HasOffering(offerings []ModelOfferingV1, id string) bool {
	for _, offering := range offerings {
		if offering.ID == id {
			return true
		}
	}
	return false
}

func validCanonicalModelIDV1(id string) bool {
	owner, model, ok := strings.Cut(id, "/")
	return ok && catalogTokenIDRe.MatchString(owner) && model != "" && !strings.ContainsAny(model, " \t\r\n")
}

func validNativeModelSourceV1(source NativeModelIDSource) bool {
	switch source {
	case NativeModelIDCatalogKnown, NativeModelIDDiscovered, NativeModelIDUserConfigured, NativeModelIDCatalogOrUser:
		return true
	default:
		return false
	}
}

func deploymentSupportsProtocolV1(deployment *DeploymentV1, protocolID string) bool {
	if deployment == nil || protocolID == "" {
		return false
	}
	if len(deployment.APIProtocolIDs) == 0 {
		return deployment.APIProtocolID == protocolID
	}
	return stringSliceContainsV1(deployment.APIProtocolIDs, protocolID)
}

func stringSliceContainsV1(values []string, target string) bool {
	for _, value := range values {
		if value == target {
			return true
		}
	}
	return false
}

func validateCapabilitiesV1(add func(string, ...any), offeringID string, capabilities CapabilitySetV1) {
	var problems []string
	collectCapabilityProblemsV1(&problems, offeringID, capabilities)
	for _, problem := range problems {
		add("%s", problem)
	}
}

func collectCapabilityProblemsV1(problems *[]string, offeringID string, capabilities CapabilitySetV1) {
	validateState := func(name string, state CapabilityState) {
		if state == "" {
			return
		}
		if state != CapabilitySupported && state != CapabilityUnsupported && state != CapabilityUnknown {
			*problems = append(*problems, fmt.Sprintf("offering %q capability %s has invalid state %q", offeringID, name, state))
		}
	}
	validateState("function_calling", capabilities.FunctionCalling)
	validateState("explicit_thinking_budget", capabilities.ExplicitThinkingBudget)
	for name, state := range capabilities.ServerTools {
		if name == "" {
			*problems = append(*problems, fmt.Sprintf("offering %q has empty server tool capability name", offeringID))
		}
		validateState("server_tools."+name, state)
	}
}

func validatePricingV1(add func(string, ...any), offeringID string, pricing PricingV1) {
	var problems []string
	collectPricingProblemsV1(&problems, offeringID, pricing)
	for _, problem := range problems {
		add("%s", problem)
	}
}

func collectPricingProblemsV1(problems *[]string, offeringID string, pricing PricingV1) {
	switch pricing.Status {
	case PricingKnown:
		if pricing.Currency == "" {
			*problems = append(*problems, fmt.Sprintf("offering %q known pricing missing currency", offeringID))
		}
		if len(pricing.RatesPer1M) == 0 {
			*problems = append(*problems, fmt.Sprintf("offering %q known pricing missing rates_per_1m", offeringID))
		}
	case PricingPartial:
		if pricing.Currency == "" {
			*problems = append(*problems, fmt.Sprintf("offering %q partial pricing missing currency", offeringID))
		}
		if len(pricing.RatesPer1M) == 0 {
			*problems = append(*problems, fmt.Sprintf("offering %q partial pricing missing known rates_per_1m", offeringID))
		}
		if len(pricing.MissingDimensions) == 0 {
			*problems = append(*problems, fmt.Sprintf("offering %q partial pricing missing missing_dimensions", offeringID))
		}
	case PricingUnknown:
		if len(pricing.RatesPer1M) > 0 {
			*problems = append(*problems, fmt.Sprintf("offering %q unknown pricing must not include rates_per_1m", offeringID))
		}
	case PricingFree:
		if pricing.Currency == "" {
			*problems = append(*problems, fmt.Sprintf("offering %q free pricing missing currency", offeringID))
		}
		for dim, rate := range pricing.RatesPer1M {
			if rate != 0 {
				*problems = append(*problems, fmt.Sprintf("offering %q free pricing dimension %q has non-zero rate %f", offeringID, dim, rate))
			}
		}
	default:
		*problems = append(*problems, fmt.Sprintf("offering %q has invalid pricing status %q", offeringID, pricing.Status))
	}
	for dim, rate := range pricing.RatesPer1M {
		if dim == "" {
			*problems = append(*problems, fmt.Sprintf("offering %q has empty pricing dimension", offeringID))
		}
		if rate < 0 {
			*problems = append(*problems, fmt.Sprintf("offering %q pricing dimension %q has negative rate %f", offeringID, dim, rate))
		}
	}
}
