// Package langdag provides a Go library for managing AI agent conversations
// with tree-structured storage and multi-provider LLM routing.
package langdag

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"sync"
	"time"

	"langdag.com/langdag/internal/conversation"
	"langdag.com/langdag/internal/models"
	internalprovider "langdag.com/langdag/internal/provider"
	anthropicprovider "langdag.com/langdag/internal/provider/anthropic"
	geminiprovider "langdag.com/langdag/internal/provider/gemini"
	openaiprovider "langdag.com/langdag/internal/provider/openai"
	internalstorage "langdag.com/langdag/internal/storage"
	"langdag.com/langdag/internal/storage/sqlite"
	"langdag.com/langdag/types"
)

// Storage is the interface for persisting conversation nodes.
// It is re-exported here so that external consumers can use the return value
// of Client.Storage() without importing an internal package.
type Storage = internalstorage.Storage

// Provider is the interface for LLM providers.
// It is re-exported here so that external consumers can use the return value
// of Client.Provider() without importing an internal package.
type Provider = internalprovider.Provider

// ContextWithRetryCallback returns a child context that carries a per-call
// retry callback. This takes priority over the config-level OnRetry.
var ContextWithRetryCallback = internalprovider.ContextWithRetryCallback

// ModelPricing contains pricing and capability information for a model.
type ModelPricing = models.ModelPricing

// ModelCatalog contains deployment-aware model, offering, pricing, and
// capability information.
type ModelCatalog = models.Catalog

// Deployment-aware model catalog v1 contract types.
type CatalogV1 = models.CatalogV1
type ProviderV1 = models.ProviderV1
type APIProtocolV1 = models.APIProtocolV1
type DeploymentV1 = models.DeploymentV1
type ModelV1 = models.ModelV1
type ModelOfferingV1 = models.ModelOfferingV1
type ModelOfferingTemplateV1 = models.ModelOfferingTemplateV1
type PricingV1 = models.PricingV1
type PricingStatus = models.PricingStatus
type CapabilityState = models.CapabilityState
type CapabilitySetV1 = models.CapabilitySetV1
type NativeModelIDSource = models.NativeModelIDSource
type ResponseCostSource = models.ResponseCostSource
type ProvenanceV1 = models.ProvenanceV1
type CredentialV1 = models.CredentialV1
type EnvFallbackV1 = models.EnvFallbackV1
type CatalogDiagnosticV1 = models.CatalogDiagnosticV1
type CompiledCatalogV1 = models.CompiledCatalogV1
type DeploymentBindingV1 = models.DeploymentBindingV1
type CatalogSource = models.CatalogSource
type CatalogLoadOptions = models.CatalogLoadOptions
type CatalogLoadResult = models.CatalogLoadResult
type CatalogRefreshOptions = models.CatalogRefreshOptions
type CatalogRefreshResult = models.CatalogRefreshResult

const CatalogV1SchemaVersion = models.CatalogV1SchemaVersion
const CatalogV1JSONSchema = models.CatalogV1JSONSchema
const DefaultRemoteCatalogURL = models.DefaultRemoteCatalogURL
const PricingKnown = models.PricingKnown
const PricingPartial = models.PricingPartial
const PricingUnknown = models.PricingUnknown
const PricingFree = models.PricingFree
const CatalogSourceEmbedded = models.CatalogSourceEmbedded
const CatalogSourceCache = models.CatalogSourceCache
const CatalogSourceRemote = models.CatalogSourceRemote
const CapabilitySupported = models.CapabilitySupported
const CapabilityUnsupported = models.CapabilityUnsupported
const CapabilityUnknown = models.CapabilityUnknown
const NativeModelIDCatalogKnown = models.NativeModelIDCatalogKnown
const NativeModelIDDiscovered = models.NativeModelIDDiscovered
const NativeModelIDUserConfigured = models.NativeModelIDUserConfigured
const NativeModelIDCatalogOrUser = models.NativeModelIDCatalogOrUser
const ResponseCostUsageCountersOnly = models.ResponseCostUsageCountersOnly
const ResponseCostProviderExact = models.ResponseCostProviderExact
const ResponseCostProviderAsync = models.ResponseCostProviderAsync
const ResponseCostLocalFree = models.ResponseCostLocalFree

var ReferenceCatalogV1 = models.ReferenceCatalogV1
var CompileCatalogV1 = models.CompileCatalogV1
var ValidateCatalogV1 = models.ValidateCatalogV1
var ParseCatalogV1 = models.ParseCatalogV1
var ParseRemoteCatalogV1 = models.ParseRemoteCatalogV1
var SplitOfferingIDV1 = models.SplitOfferingIDV1
var DeploymentBindingsV1 = models.DeploymentBindingsV1
var CatalogRefreshOptionsFromEnv = models.CatalogRefreshOptionsFromEnv

// Config holds all configuration for the langdag client.
type Config struct {
	// StoragePath is the path to the SQLite database file.
	// Defaults to "$HOME/.config/langdag/langdag.db"
	StoragePath string

	// Provider is the default LLM provider to use.
	// Valid values: "anthropic", "openai", "gemini", "grok", "openrouter", "ollama", "apple",
	// "anthropic-vertex", "anthropic-bedrock", "openai-azure", "gemini-vertex"
	// "gemma" is accepted as an alias for "gemini".
	// Defaults to "anthropic"
	Provider string

	// APIKeys maps provider names to their API keys.
	// Keys: "anthropic", "openai", "gemini"
	APIKeys map[string]string

	// AnthropicConfig holds Anthropic-specific config (optional base URL override).
	AnthropicConfig *AnthropicConfig

	// OpenAIConfig holds OpenAI-specific config.
	OpenAIConfig *OpenAIConfig

	// GeminiConfig holds Gemini-specific config.
	GeminiConfig *GeminiConfig

	// GrokConfig holds Grok (xAI)-specific config.
	GrokConfig *GrokConfig

	// OpenRouterConfig holds OpenRouter-specific config.
	OpenRouterConfig *OpenRouterConfig

	// AzureOpenAIConfig holds Azure OpenAI-specific config.
	AzureOpenAIConfig *AzureOpenAIConfig

	// VertexConfig holds Google Vertex AI config.
	VertexConfig *VertexConfig

	// BedrockConfig holds AWS Bedrock config.
	BedrockConfig *BedrockConfig

	// OllamaConfig holds Ollama-specific config (local LLM server).
	OllamaConfig *OllamaConfig

	// AppleConfig holds Apple Foundation Models config (local fm serve bridge).
	AppleConfig *AppleConfig

	// ModelCatalog is the deployment-aware catalog used for canonical model
	// resolution. Defaults to the embedded catalog generated from the published
	// model-catalog branch when nil. Mutually exclusive with RemoteModelCatalog.
	ModelCatalog *ModelCatalog

	// RemoteModelCatalog, when non-nil, makes New fetch the published catalog
	// instead of using the embedded catalog. Fetch failures are returned from
	// New; no local cache file is read or written. Mutually exclusive with
	// ModelCatalog.
	RemoteModelCatalog *RemoteModelCatalogConfig

	// Deployments configures routeable deployment credentials and deployment-
	// scoped model mappings. Legacy provider-specific fields above remain
	// readable and are merged into this shape.
	Deployments map[string]DeploymentConfig

	// RoutingPolicy configures canonical-model deployment routing. Exact model
	// routes override matching provider routes. Non-matching models use
	// automatic eligible deployment resolution unless Default is explicitly set.
	RoutingPolicy *RoutingPolicy

	// Routing configures multi-provider routing (optional).
	// Deprecated: use RoutingPolicy with deployment IDs.
	Routing []RoutingEntry

	// FallbackOrder specifies provider fallback order (optional).
	// Deprecated: use RoutingPolicy with deployment IDs.
	FallbackOrder []string

	// RetryConfig configures retry behavior.
	RetryConfig *RetryConfig
}

// RemoteModelCatalogConfig configures an explicit runtime fetch of the
// published model catalog. Leave Endpoint empty to use DefaultRemoteCatalogURL.
type RemoteModelCatalogConfig struct {
	Endpoint   string
	Timeout    time.Duration
	HTTPClient *http.Client
}

// AnthropicConfig holds Anthropic-specific configuration.
type AnthropicConfig struct {
	BaseURL string
}

// OpenAIConfig holds OpenAI-specific configuration.
type OpenAIConfig struct {
	BaseURL string
}

// GeminiConfig holds Gemini-specific configuration.
type GeminiConfig struct {
	BaseURL string
}

// AzureOpenAIConfig holds Azure OpenAI-specific configuration.
type AzureOpenAIConfig struct {
	Endpoint   string
	APIVersion string
	APIKey     string
}

// VertexConfig holds Google Vertex AI configuration.
type VertexConfig struct {
	ProjectID string
	Region    string
}

// BedrockConfig holds AWS Bedrock configuration.
type BedrockConfig struct {
	Region string
}

// OllamaConfig holds Ollama-specific configuration.
// Ollama is a local LLM server that provides an OpenAI-compatible API.
type OllamaConfig struct {
	// BaseURL is the Ollama server address (e.g., "http://localhost:11434" or "http://100.93.184.1:11434")
	BaseURL string
}

// AppleConfig holds Apple Foundation Models configuration.
type AppleConfig struct {
	BaseURL string
}

// GrokConfig holds Grok (xAI)-specific configuration.
type GrokConfig struct {
	BaseURL string
}

// OpenRouterConfig holds OpenRouter-specific configuration.
type OpenRouterConfig struct {
	BaseURL string
}

// DeploymentConfig holds deployment-scoped credentials and native model
// mappings. Azure OpenAI uses ModelMappings to translate canonical model IDs to
// the caller's Azure deployment names.
type DeploymentConfig struct {
	APIKey        string
	BaseURL       string
	Endpoint      string
	APIVersion    string
	ProjectID     string
	Region        string
	ModelMappings map[string]string
}

// RoutingEntry configures a single provider entry in the routing table.
// Deprecated: use RoutingPolicy with deployment IDs.
type RoutingEntry struct {
	Provider string
	Weight   int
	Retry    *RetryConfig
}

type DeploymentChoice = internalprovider.DeploymentChoice
type RoutingStage = internalprovider.RoutingStage
type RoutingPolicy = internalprovider.RoutingPolicy

// RetryEvent holds information about a retry attempt.
type RetryEvent = internalprovider.RetryEvent

// RetryConfig configures retry behavior for LLM provider calls.
type RetryConfig struct {
	MaxRetries int
	BaseDelay  time.Duration
	MaxDelay   time.Duration
	// OnRetry is called before each retry wait. It may be nil.
	OnRetry func(RetryEvent)
}

// Client is the main langdag client for managing AI conversations.
//
// A Client is safe for concurrent use by multiple goroutines. The underlying
// storage (SQLite with WAL mode and busy_timeout) and providers (stateless HTTP
// clients) are themselves concurrent-safe, and each call to Prompt or PromptFrom
// returns an independent PromptResult with its own streaming channel.
type Client struct {
	store   internalstorage.Storage
	prov    internalprovider.Provider
	convMgr *conversation.Manager
}

// New creates a new langdag client with the given configuration.
func New(cfg Config) (*Client, error) {
	ctx := context.Background()

	// Resolve storage path
	storagePath := cfg.StoragePath
	if storagePath == "" {
		storagePath = defaultStoragePath()
	}

	// Ensure the directory for the storage file exists
	if err := os.MkdirAll(filepath.Dir(storagePath), 0755); err != nil {
		return nil, fmt.Errorf("langdag: failed to create storage directory: %w", err)
	}

	// Initialize SQLite storage
	store, err := sqlite.New(storagePath)
	if err != nil {
		return nil, fmt.Errorf("langdag: failed to open storage: %w", err)
	}

	if err := store.Init(ctx); err != nil {
		store.Close()
		return nil, fmt.Errorf("langdag: failed to initialize storage: %w", err)
	}

	// Build the provider
	prov, err := buildProvider(ctx, cfg)
	if err != nil {
		store.Close()
		return nil, fmt.Errorf("langdag: failed to create provider: %w", err)
	}

	convMgr := conversation.NewManager(store, prov)

	return &Client{
		store:   store,
		prov:    prov,
		convMgr: convMgr,
	}, nil
}

// NewWithDeps creates a Client from pre-built dependencies.
// Useful for testing or custom integrations where the caller has already
// constructed a Storage and Provider.
func NewWithDeps(store Storage, prov Provider) *Client {
	return &Client{
		store:   store,
		prov:    prov,
		convMgr: conversation.NewManager(store, prov),
	}
}

// WithRetry wraps a Provider with exponential backoff retry logic.
// Only transient errors (5xx, rate limits, timeouts) are retried.
func WithRetry(p Provider, cfg RetryConfig) Provider {
	return internalprovider.WithRetry(p, resolveRetryConfig(&cfg))
}

// Close releases all resources held by the client.
func (c *Client) Close() error {
	return c.store.Close()
}

// Storage returns the underlying storage interface for advanced use cases.
func (c *Client) Storage() Storage {
	return c.store
}

// Provider returns the underlying provider for advanced use cases.
func (c *Client) Provider() Provider {
	return c.prov
}

// PromptOption configures a prompt request.
type PromptOption func(*promptOptions)

type promptOptions struct {
	model                string
	apiProtocolID        string
	systemPrompt         string
	maxTokens            int
	maxOutputGroupTokens int
	maxTurns             int
	tools                []types.ToolDefinition
	think                *bool
}

// WithModel sets the model for the prompt.
func WithModel(model string) PromptOption {
	return func(o *promptOptions) {
		o.model = model
	}
}

// WithAPIProtocol selects a provider API surface for providers that expose
// multiple protocols for the same deployment, for example
// "openai-responses" or "openai-chat-completions".
func WithAPIProtocol(apiProtocolID string) PromptOption {
	return func(o *promptOptions) {
		o.apiProtocolID = apiProtocolID
	}
}

// WithSystemPrompt sets the system prompt.
func WithSystemPrompt(prompt string) PromptOption {
	return func(o *promptOptions) {
		o.systemPrompt = prompt
	}
}

// WithMaxTokens sets the max tokens for the response.
func WithMaxTokens(n int) PromptOption {
	return func(o *promptOptions) {
		o.maxTokens = n
	}
}

// WithMaxOutputGroupTokens sets the maximum total output tokens across all
// continuation calls in an output group. When a response hits max_tokens and
// is continued, langdag tracks cumulative output tokens; if they exceed this
// budget the continuation stops. A value of 0 (the default) means 4× the
// per-call max_tokens.
func WithMaxOutputGroupTokens(n int) PromptOption {
	return func(o *promptOptions) {
		o.maxOutputGroupTokens = n
	}
}

// WithMaxTurns sets the maximum number of LLM round-trips for a single
// Prompt/PromptFrom call. Since langdag does not have a built-in tool-use
// loop, the value is exposed on the PromptResult so the caller can
// decrement and check it in their own multi-turn loop.
// A value of 0 (the default) means unlimited.
func WithMaxTurns(n int) PromptOption {
	return func(o *promptOptions) {
		o.maxTurns = n
	}
}

// WithTools sets the tool definitions for the prompt.
// When tools are provided, the LLM may respond with tool_use content blocks.
func WithTools(tools []types.ToolDefinition) PromptOption {
	return func(o *promptOptions) {
		o.tools = tools
	}
}

// WithThink controls whether the model should use extended thinking.
// true = enable thinking, false = disable thinking. Omitting this option
// leaves the decision to the provider/model default.
func WithThink(enabled bool) PromptOption {
	return func(o *promptOptions) {
		o.think = &enabled
	}
}

// PromptResult holds the result of a prompt call.
//
// The NodeID and Content fields are written by a background goroutine as the
// stream is consumed. Reading them directly before the stream is fully drained
// is a data race. Use the GetNodeID and GetContent accessor methods for
// concurrent-safe access at any time, or read the fields only after the Stream
// channel has been fully drained (closed).
type PromptResult struct {
	// NodeID is the ID of the saved assistant node (set when streaming completes,
	// or immediately for non-streaming use when the stream is consumed).
	NodeID string

	// Content is the full response text (empty until streaming completes).
	Content string

	// Stream is the streaming channel. Range over it to receive chunks.
	// It is never nil; even for simple use cases, consumers should drain it.
	Stream <-chan StreamChunk

	// MaxTurns is the maximum number of LLM round-trips the caller should
	// allow for this conversation. 0 means unlimited. Since langdag does not
	// have a built-in tool-use loop, the caller can use this value to enforce
	// a turn budget in their own multi-turn loop.
	MaxTurns int

	// mu protects concurrent writes to NodeID and Content from the background
	// goroutine in buildResult.
	mu sync.Mutex
}

// GetNodeID returns the node ID in a concurrent-safe manner.
// The value is only meaningful after the stream has delivered a Done chunk.
func (r *PromptResult) GetNodeID() string {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.NodeID
}

// GetContent returns the full response content in a concurrent-safe manner.
// The value is only meaningful after the stream has been fully drained.
func (r *PromptResult) GetContent() string {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.Content
}

// StreamChunk is a piece of a streaming response.
type StreamChunk struct {
	// Content is the incremental text content for this chunk.
	Content string

	// ContentBlock is set for content_done events (e.g. tool_use blocks).
	ContentBlock *types.ContentBlock

	// Done indicates the stream has completed.
	Done bool

	// Error holds any error that occurred during streaming.
	Error error

	// NodeID is set when Done=true and the assistant node has been saved to storage.
	NodeID string

	// StopReason is the reason the LLM stopped generating (e.g. "end_turn", "tool_use").
	// Set when Done=true.
	StopReason string

	Usage           *types.Usage
	ModelResolution *types.ModelResolutionMetadata
	NormalizedUsage *types.NormalizedUsage
	PricingSnapshot *types.PricingSnapshot
	ProviderCost    *types.ProviderCost
}

// Prompt starts a new conversation with the given message.
// Returns a PromptResult with the streaming response.
func (c *Client) Prompt(ctx context.Context, message string, opts ...PromptOption) (*PromptResult, error) {
	o := applyOptions(opts)
	events, err := c.convMgr.PromptWithAPIProtocol(ctx, message, o.model, o.apiProtocolID, o.systemPrompt, o.tools, o.think, o.maxTokens, o.maxOutputGroupTokens)
	if err != nil {
		return nil, err
	}
	result := buildResult(events)
	result.MaxTurns = o.maxTurns
	return result, nil
}

// PromptFrom continues a conversation from an existing node.
func (c *Client) PromptFrom(ctx context.Context, nodeID string, message string, opts ...PromptOption) (*PromptResult, error) {
	o := applyOptions(opts)
	events, err := c.convMgr.PromptFromWithAPIProtocol(ctx, nodeID, message, o.model, o.apiProtocolID, o.tools, o.think, o.maxTokens, o.maxOutputGroupTokens)
	if err != nil {
		return nil, err
	}
	result := buildResult(events)
	result.MaxTurns = o.maxTurns
	return result, nil
}

// ListConversations returns all root conversation nodes.
func (c *Client) ListConversations(ctx context.Context) ([]*types.Node, error) {
	return c.convMgr.ListRoots(ctx)
}

// GetNode returns a node by ID or ID prefix.
func (c *Client) GetNode(ctx context.Context, id string) (*types.Node, error) {
	return c.convMgr.ResolveNode(ctx, id)
}

// GetSubtree returns a node and all its descendants.
func (c *Client) GetSubtree(ctx context.Context, id string) ([]*types.Node, error) {
	node, err := c.convMgr.ResolveNode(ctx, id)
	if err != nil {
		return nil, err
	}
	if node == nil {
		return nil, fmt.Errorf("langdag: node not found: %s", id)
	}
	return c.convMgr.GetSubtree(ctx, node.ID)
}

// GetAncestors returns all ancestors of a node (the conversation history leading to it).
func (c *Client) GetAncestors(ctx context.Context, id string) ([]*types.Node, error) {
	node, err := c.convMgr.ResolveNode(ctx, id)
	if err != nil {
		return nil, err
	}
	if node == nil {
		return nil, fmt.Errorf("langdag: node not found: %s", id)
	}
	return c.store.GetAncestors(ctx, node.ID)
}

// DeleteNode deletes a node and all its descendants.
func (c *Client) DeleteNode(ctx context.Context, id string) error {
	node, err := c.convMgr.ResolveNode(ctx, id)
	if err != nil {
		return err
	}
	if node == nil {
		return fmt.Errorf("langdag: node not found: %s", id)
	}
	return c.convMgr.DeleteNode(ctx, node.ID)
}

// applyOptions applies prompt options and returns the resulting promptOptions.
func applyOptions(opts []PromptOption) *promptOptions {
	o := &promptOptions{
		model: "claude-sonnet-4-20250514",
	}
	for _, opt := range opts {
		opt(o)
	}
	return o
}

// buildResult converts a channel of types.StreamEvent into a PromptResult with a StreamChunk channel.
// The returned PromptResult.Content and PromptResult.NodeID are populated once the stream completes
// (i.e., after the Stream channel is drained).
func buildResult(events <-chan types.StreamEvent) *PromptResult {
	ch := make(chan StreamChunk, 100)
	result := &PromptResult{Stream: ch}

	go func() {
		defer close(ch)
		var accumulated string
		var stopReason string
		var doneResponse *types.CompletionResponse
		emittedContentBlocks := map[string]bool{}
		var terminated bool
		for event := range events {
			switch event.Type {
			case types.StreamEventDelta:
				accumulated += event.Content
				ch <- StreamChunk{Content: event.Content}
			case types.StreamEventContentDone:
				markContentBlockEmitted(emittedContentBlocks, event.ContentBlock)
				ch <- StreamChunk{ContentBlock: event.ContentBlock}
			case types.StreamEventDone:
				if event.Response != nil {
					stopReason = event.Response.StopReason
					doneResponse = event.Response
				}
			case types.StreamEventError:
				ch <- StreamChunk{Error: event.Error, Done: true}
				terminated = true
				return
			case types.StreamEventNodeSaved:
				result.mu.Lock()
				result.NodeID = event.NodeID
				result.Content = accumulated
				result.mu.Unlock()
				emitMissingDoneContentBlocks(ch, emittedContentBlocks, doneResponse)
				ch <- streamDoneChunk(event.NodeID, stopReason, doneResponse)
				terminated = true
			}
		}
		// If the stream ended without a NodeSaved or Error event (e.g. provider
		// closed the channel early or returned an empty stream), send a Done
		// chunk so consumers never hang waiting for one.
		if !terminated {
			result.mu.Lock()
			result.Content = accumulated
			result.mu.Unlock()
			ch <- StreamChunk{
				Done:       true,
				Error:      fmt.Errorf("stream ended without completion"),
				StopReason: stopReason,
			}
		}
	}()

	return result
}

func markContentBlockEmitted(emitted map[string]bool, block *types.ContentBlock) {
	if key := contentBlockStreamKey(block); key != "" {
		emitted[key] = true
	}
}

func emitMissingDoneContentBlocks(ch chan<- StreamChunk, emitted map[string]bool, response *types.CompletionResponse) {
	if response == nil {
		return
	}
	for i := range response.Content {
		block := response.Content[i]
		if block.Type != "tool_use" {
			continue
		}
		key := contentBlockStreamKey(&block)
		if key != "" && emitted[key] {
			continue
		}
		if key != "" {
			emitted[key] = true
		}
		ch <- StreamChunk{ContentBlock: &block}
	}
}

func contentBlockStreamKey(block *types.ContentBlock) string {
	if block == nil || block.ID == "" {
		return ""
	}
	return block.Type + ":" + block.ID
}

func streamDoneChunk(nodeID, stopReason string, response *types.CompletionResponse) StreamChunk {
	chunk := StreamChunk{Done: true, NodeID: nodeID, StopReason: stopReason}
	if response == nil {
		return chunk
	}
	chunk.Usage = &response.Usage
	chunk.ModelResolution = response.ModelResolution
	chunk.NormalizedUsage = response.NormalizedUsage
	chunk.PricingSnapshot = response.PricingSnapshot
	chunk.ProviderCost = response.ProviderCost
	return chunk
}

// defaultStoragePath returns the default path for the SQLite database.
func defaultStoragePath() string {
	homeDir, err := os.UserHomeDir()
	if err != nil {
		return "./langdag.db"
	}
	return filepath.Join(homeDir, ".config", "langdag", "langdag.db")
}

// buildProvider creates the appropriate provider based on configuration.
func buildProvider(ctx context.Context, cfg Config) (internalprovider.Provider, error) {
	// Resolve global retry config
	globalRetry := resolveRetryConfig(cfg.RetryConfig)

	if !hasDeploymentAwareRuntimeConfig(cfg) && len(cfg.Routing) > 0 {
		return buildRouter(ctx, cfg, globalRetry)
	}

	catalog, err := resolveModelCatalog(ctx, cfg)
	if err != nil {
		return nil, err
	}
	compiled, err := models.CompileCatalogV1(catalog)
	if err != nil {
		return nil, fmt.Errorf("langdag: invalid model catalog: %w", err)
	}

	return buildDeploymentAwareProvider(ctx, cfg, compiled, globalRetry)
}

func hasDeploymentAwareRuntimeConfig(cfg Config) bool {
	return cfg.ModelCatalog != nil || cfg.RemoteModelCatalog != nil || len(cfg.Deployments) > 0 || cfg.RoutingPolicy != nil
}

func resolveModelCatalog(ctx context.Context, cfg Config) (*models.Catalog, error) {
	if cfg.ModelCatalog != nil && cfg.RemoteModelCatalog != nil {
		return nil, fmt.Errorf("langdag: ModelCatalog and RemoteModelCatalog cannot both be set")
	}
	if cfg.ModelCatalog != nil {
		return cfg.ModelCatalog, nil
	}
	if cfg.RemoteModelCatalog != nil {
		result, err := models.LoadRemoteCatalog(ctx, models.CatalogRefreshOptions{
			Endpoint:   cfg.RemoteModelCatalog.Endpoint,
			Timeout:    cfg.RemoteModelCatalog.Timeout,
			HTTPClient: cfg.RemoteModelCatalog.HTTPClient,
		})
		if err != nil {
			return nil, fmt.Errorf("langdag: failed to fetch remote model catalog: %w", err)
		}
		if result.Catalog == nil {
			return nil, fmt.Errorf("langdag: remote model catalog was not loaded")
		}
		return result.Catalog, nil
	}
	result, err := models.LoadRuntimeCatalog(models.CatalogLoadOptions{})
	if err != nil {
		return nil, err
	}
	return result.Catalog, nil
}

func buildDeploymentAwareProvider(ctx context.Context, cfg Config, catalog *models.CompiledCatalogV1, globalRetry internalprovider.RetryConfig) (internalprovider.Provider, error) {
	deploymentIDs, routingConfigured := deploymentIDsForConfig(cfg)
	adapters := map[string]internalprovider.DeploymentAdapter{}
	for _, deploymentID := range deploymentIDs {
		adapter, err := createDeploymentAdapter(ctx, deploymentID, cfg, globalRetry)
		if err != nil {
			if !routingConfigured && len(deploymentIDs) == 1 {
				return nil, err
			}
			log.Printf("langdag: skipping unavailable deployment %q: %v", deploymentID, err)
			continue
		}
		adapters[deploymentID] = adapter
		log.Printf("langdag: configured deployment: %s", deploymentID)
	}
	if len(adapters) == 0 {
		return nil, fmt.Errorf("langdag: no configured deployments are available")
	}

	policy := internalprovider.RoutingPolicy{}
	if cfg.RoutingPolicy != nil {
		policy = *cfg.RoutingPolicy
	} else if len(cfg.Routing) > 0 || len(cfg.FallbackOrder) > 0 {
		policy = legacyRoutingPolicyFromConfig(cfg, globalRetry)
	}

	return internalprovider.NewDeploymentRouter(internalprovider.DeploymentRouterOptions{
		Catalog:     catalog,
		Deployments: adapters,
		Routing:     policy,
	})
}

// createSingleProvider constructs a single provider by name.
func createSingleProvider(ctx context.Context, name string, cfg Config) (internalprovider.Provider, error) {
	switch name {
	case "anthropic":
		apiKey := cfg.APIKeys["anthropic"]
		if apiKey == "" {
			apiKey = os.Getenv("ANTHROPIC_API_KEY")
		}
		if apiKey == "" {
			return nil, fmt.Errorf("langdag: ANTHROPIC_API_KEY not set")
		}
		return anthropicprovider.New(apiKey), nil

	case "openai":
		apiKey := cfg.APIKeys["openai"]
		if apiKey == "" {
			apiKey = os.Getenv("OPENAI_API_KEY")
		}
		if apiKey == "" {
			return nil, fmt.Errorf("langdag: OPENAI_API_KEY not set")
		}
		baseURL := ""
		if cfg.OpenAIConfig != nil {
			baseURL = cfg.OpenAIConfig.BaseURL
		}
		if baseURL == "" {
			baseURL = os.Getenv("OPENAI_BASE_URL")
		}
		return openaiprovider.New(apiKey, baseURL), nil

	case "gemini", "gemma":
		apiKey := cfg.APIKeys["gemini"]
		if apiKey == "" {
			apiKey = os.Getenv("GEMINI_API_KEY")
		}
		if apiKey == "" {
			return nil, fmt.Errorf("langdag: GEMINI_API_KEY not set")
		}
		return geminiprovider.New(apiKey), nil

	case "anthropic-vertex":
		vc := cfg.VertexConfig
		if vc == nil {
			return nil, fmt.Errorf("langdag: VertexConfig must be set for anthropic-vertex provider")
		}
		if vc.ProjectID == "" || vc.Region == "" {
			return nil, fmt.Errorf("langdag: VertexConfig.ProjectID and VertexConfig.Region must be set for anthropic-vertex")
		}
		return anthropicprovider.NewVertex(ctx, vc.Region, vc.ProjectID)

	case "anthropic-bedrock":
		region := ""
		if cfg.BedrockConfig != nil {
			region = cfg.BedrockConfig.Region
		}
		if region == "" {
			region = os.Getenv("AWS_REGION")
		}
		return anthropicprovider.NewBedrock(ctx, region)

	case "openai-azure":
		ac := cfg.AzureOpenAIConfig
		if ac == nil {
			return nil, fmt.Errorf("langdag: AzureOpenAIConfig must be set for openai-azure provider")
		}
		if ac.APIKey == "" || ac.Endpoint == "" {
			return nil, fmt.Errorf("langdag: AzureOpenAIConfig.APIKey and AzureOpenAIConfig.Endpoint must be set for openai-azure")
		}
		return openaiprovider.NewAzure(ac.APIKey, ac.Endpoint, ac.APIVersion), nil

	case "grok":
		apiKey := cfg.APIKeys["grok"]
		if apiKey == "" {
			apiKey = os.Getenv("XAI_API_KEY")
		}
		if apiKey == "" {
			return nil, fmt.Errorf("langdag: XAI_API_KEY not set")
		}
		baseURL := ""
		if cfg.GrokConfig != nil {
			baseURL = cfg.GrokConfig.BaseURL
		}
		if baseURL == "" {
			baseURL = os.Getenv("XAI_BASE_URL")
		}
		return openaiprovider.NewGrok(apiKey, baseURL), nil

	case "openrouter":
		apiKey := cfg.APIKeys["openrouter"]
		if apiKey == "" {
			apiKey = os.Getenv("OPENROUTER_API_KEY")
		}
		if apiKey == "" {
			return nil, fmt.Errorf("langdag: OPENROUTER_API_KEY not set")
		}
		baseURL := ""
		if cfg.OpenRouterConfig != nil {
			baseURL = cfg.OpenRouterConfig.BaseURL
		}
		if baseURL == "" {
			baseURL = os.Getenv("OPENROUTER_BASE_URL")
		}
		return openaiprovider.NewOpenRouter(apiKey, baseURL), nil

	case "gemini-vertex":
		vc := cfg.VertexConfig
		if vc == nil {
			return nil, fmt.Errorf("langdag: VertexConfig must be set for gemini-vertex provider")
		}
		if vc.ProjectID == "" || vc.Region == "" {
			return nil, fmt.Errorf("langdag: VertexConfig.ProjectID and VertexConfig.Region must be set for gemini-vertex")
		}
		return geminiprovider.NewVertex(ctx, vc.ProjectID, vc.Region)

	case "ollama":
		baseURL := "http://localhost:11434"
		if cfg.OllamaConfig != nil && cfg.OllamaConfig.BaseURL != "" {
			baseURL = cfg.OllamaConfig.BaseURL
		}
		return openaiprovider.NewOllama(baseURL), nil

	case "apple":
		baseURL := ""
		if cfg.AppleConfig != nil {
			baseURL = cfg.AppleConfig.BaseURL
		}
		if baseURL == "" {
			baseURL = os.Getenv("APPLE_FM_BASE_URL")
		}
		return openaiprovider.NewApple(baseURL), nil

	default:
		return nil, fmt.Errorf("langdag: unknown provider: %s", name)
	}
}

func createDeploymentAdapter(ctx context.Context, deploymentID string, cfg Config, globalRetry internalprovider.RetryConfig) (internalprovider.DeploymentAdapter, error) {
	deploymentCfg := deploymentConfigForID(deploymentID, cfg)
	var prov internalprovider.Provider
	var err error
	switch deploymentID {
	case "anthropic-direct":
		if deploymentCfg.APIKey == "" {
			return internalprovider.DeploymentAdapter{}, fmt.Errorf("langdag: ANTHROPIC_API_KEY not set")
		}
		prov = anthropicprovider.New(deploymentCfg.APIKey)
	case "anthropic-bedrock":
		prov, err = anthropicprovider.NewBedrock(ctx, deploymentCfg.Region)
	case "anthropic-vertex":
		if deploymentCfg.ProjectID == "" || deploymentCfg.Region == "" {
			return internalprovider.DeploymentAdapter{}, fmt.Errorf("langdag: project_id and region must be set for anthropic-vertex")
		}
		prov, err = anthropicprovider.NewVertex(ctx, deploymentCfg.Region, deploymentCfg.ProjectID)
	case "openai-direct":
		if deploymentCfg.APIKey == "" {
			return internalprovider.DeploymentAdapter{}, fmt.Errorf("langdag: OPENAI_API_KEY not set")
		}
		prov = openaiprovider.New(deploymentCfg.APIKey, deploymentCfg.BaseURL)
	case "openai-azure":
		if deploymentCfg.APIKey == "" || deploymentCfg.Endpoint == "" {
			return internalprovider.DeploymentAdapter{}, fmt.Errorf("langdag: AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT must be set for openai-azure")
		}
		prov = openaiprovider.NewAzure(deploymentCfg.APIKey, deploymentCfg.Endpoint, deploymentCfg.APIVersion)
	case "gemini-direct":
		if deploymentCfg.APIKey == "" {
			return internalprovider.DeploymentAdapter{}, fmt.Errorf("langdag: GEMINI_API_KEY not set")
		}
		prov = geminiprovider.New(deploymentCfg.APIKey)
	case "gemini-vertex":
		if deploymentCfg.ProjectID == "" || deploymentCfg.Region == "" {
			return internalprovider.DeploymentAdapter{}, fmt.Errorf("langdag: project_id and region must be set for gemini-vertex")
		}
		prov, err = geminiprovider.NewVertex(ctx, deploymentCfg.ProjectID, deploymentCfg.Region)
	case "grok-direct":
		if deploymentCfg.APIKey == "" {
			return internalprovider.DeploymentAdapter{}, fmt.Errorf("langdag: XAI_API_KEY not set")
		}
		prov = openaiprovider.NewGrok(deploymentCfg.APIKey, deploymentCfg.BaseURL)
	case "openrouter":
		if deploymentCfg.APIKey == "" {
			return internalprovider.DeploymentAdapter{}, fmt.Errorf("langdag: OPENROUTER_API_KEY not set")
		}
		prov = openaiprovider.NewOpenRouter(deploymentCfg.APIKey, deploymentCfg.BaseURL)
	case "ollama-local":
		prov = openaiprovider.NewOllama(deploymentCfg.BaseURL)
	case "apple-local":
		prov = openaiprovider.NewApple(deploymentCfg.BaseURL)
	default:
		return internalprovider.DeploymentAdapter{}, fmt.Errorf("langdag: unknown deployment: %s", deploymentID)
	}
	if err != nil {
		return internalprovider.DeploymentAdapter{}, err
	}
	prov = internalprovider.WithRetry(prov, globalRetry)
	return internalprovider.DeploymentAdapter{
		DeploymentID:  deploymentID,
		Provider:      prov,
		ModelMappings: deploymentCfg.ModelMappings,
	}, nil
}

func deploymentIDsForConfig(cfg Config) ([]string, bool) {
	seen := map[string]bool{}
	var ids []string
	add := func(id string) {
		if id == "" || seen[id] {
			return
		}
		seen[id] = true
		ids = append(ids, id)
	}
	routingConfigured := cfg.RoutingPolicy != nil || len(cfg.Routing) > 0
	for deploymentID := range cfg.Deployments {
		add(deploymentID)
	}
	if cfg.RoutingPolicy != nil {
		collectDeploymentChoices := func(stages []internalprovider.RoutingStage) {
			for _, stage := range stages {
				for _, choice := range stage.Deployments {
					add(choice.DeploymentID)
				}
			}
		}
		collectDeploymentChoices(cfg.RoutingPolicy.Default)
		for _, stages := range cfg.RoutingPolicy.Providers {
			collectDeploymentChoices(stages)
		}
		for _, stages := range cfg.RoutingPolicy.Models {
			collectDeploymentChoices(stages)
		}
	}
	for _, entry := range cfg.Routing {
		add(deploymentIDForProviderName(entry.Provider))
	}
	if len(cfg.Routing) > 0 || len(cfg.Deployments) > 0 {
		for _, providerName := range cfg.FallbackOrder {
			add(deploymentIDForProviderName(providerName))
		}
	}
	if len(ids) == 0 {
		providerName := cfg.Provider
		if providerName == "" {
			providerName = "anthropic"
		}
		add(deploymentIDForProviderName(providerName))
	}
	return ids, routingConfigured
}

func legacyRoutingPolicyFromConfig(cfg Config, globalRetry internalprovider.RetryConfig) internalprovider.RoutingPolicy {
	var defaultStage internalprovider.RoutingStage
	for _, entry := range cfg.Routing {
		deploymentID := deploymentIDForProviderName(entry.Provider)
		if deploymentID == "" {
			continue
		}
		defaultStage.Deployments = append(defaultStage.Deployments, internalprovider.DeploymentChoice{
			DeploymentID: deploymentID,
			Weight:       entry.Weight,
		})
		if entry.Retry != nil && entry.Retry.MaxRetries > defaultStage.Retries {
			defaultStage.Retries = entry.Retry.MaxRetries
		}
	}
	if defaultStage.Retries == 0 {
		defaultStage.Retries = globalRetry.MaxRetries
	}
	var stages []internalprovider.RoutingStage
	if len(defaultStage.Deployments) > 0 {
		stages = append(stages, defaultStage)
	}
	for _, providerName := range cfg.FallbackOrder {
		deploymentID := deploymentIDForProviderName(providerName)
		if deploymentID == "" {
			continue
		}
		stages = append(stages, internalprovider.RoutingStage{
			Deployments: []internalprovider.DeploymentChoice{{DeploymentID: deploymentID, Weight: 100}},
			Retries:     globalRetry.MaxRetries,
		})
	}
	return internalprovider.RoutingPolicy{Default: stages}
}

func deploymentIDForProviderName(providerName string) string {
	switch providerName {
	case "", "anthropic":
		return "anthropic-direct"
	case "openai":
		return "openai-direct"
	case "gemini", "gemma":
		return "gemini-direct"
	case "grok":
		return "grok-direct"
	case "ollama":
		return "ollama-local"
	case "apple":
		return "apple-local"
	case "anthropic-direct", "anthropic-bedrock", "anthropic-vertex", "openai-direct", "openai-azure", "gemini-direct", "gemini-vertex", "grok-direct", "openrouter", "ollama-local", "apple-local":
		return providerName
	default:
		return ""
	}
}

func deploymentConfigForID(deploymentID string, cfg Config) DeploymentConfig {
	out := cfg.Deployments[deploymentID]
	if out.ModelMappings != nil {
		out.ModelMappings = cloneLangdagStringMap(out.ModelMappings)
	}
	applyEnv := func(field *string, names ...string) {
		if *field != "" {
			return
		}
		for _, name := range names {
			if value := os.Getenv(name); value != "" {
				*field = value
				return
			}
		}
	}
	switch deploymentID {
	case "anthropic-direct":
		if out.APIKey == "" {
			out.APIKey = cfg.APIKeys["anthropic"]
		}
		if out.BaseURL == "" && cfg.AnthropicConfig != nil {
			out.BaseURL = cfg.AnthropicConfig.BaseURL
		}
		applyEnv(&out.APIKey, "ANTHROPIC_API_KEY")
	case "anthropic-bedrock":
		if cfg.BedrockConfig != nil && out.Region == "" {
			out.Region = cfg.BedrockConfig.Region
		}
		applyEnv(&out.Region, "AWS_REGION")
	case "anthropic-vertex":
		if cfg.VertexConfig != nil {
			if out.ProjectID == "" {
				out.ProjectID = cfg.VertexConfig.ProjectID
			}
			if out.Region == "" {
				out.Region = cfg.VertexConfig.Region
			}
		}
		applyEnv(&out.ProjectID, "VERTEX_PROJECT_ID")
		applyEnv(&out.Region, "VERTEX_REGION")
	case "openai-direct":
		if out.APIKey == "" {
			out.APIKey = cfg.APIKeys["openai"]
		}
		if out.BaseURL == "" && cfg.OpenAIConfig != nil {
			out.BaseURL = cfg.OpenAIConfig.BaseURL
		}
		applyEnv(&out.APIKey, "OPENAI_API_KEY")
		applyEnv(&out.BaseURL, "OPENAI_BASE_URL")
	case "openai-azure":
		if cfg.AzureOpenAIConfig != nil {
			if out.APIKey == "" {
				out.APIKey = cfg.AzureOpenAIConfig.APIKey
			}
			if out.Endpoint == "" {
				out.Endpoint = cfg.AzureOpenAIConfig.Endpoint
			}
			if out.APIVersion == "" {
				out.APIVersion = cfg.AzureOpenAIConfig.APIVersion
			}
		}
		applyEnv(&out.APIKey, "AZURE_OPENAI_API_KEY")
		applyEnv(&out.Endpoint, "AZURE_OPENAI_ENDPOINT")
		applyEnv(&out.APIVersion, "AZURE_OPENAI_API_VERSION")
	case "gemini-direct":
		if out.APIKey == "" {
			out.APIKey = cfg.APIKeys["gemini"]
		}
		if out.BaseURL == "" && cfg.GeminiConfig != nil {
			out.BaseURL = cfg.GeminiConfig.BaseURL
		}
		applyEnv(&out.APIKey, "GEMINI_API_KEY")
	case "gemini-vertex":
		if cfg.VertexConfig != nil {
			if out.ProjectID == "" {
				out.ProjectID = cfg.VertexConfig.ProjectID
			}
			if out.Region == "" {
				out.Region = cfg.VertexConfig.Region
			}
		}
		applyEnv(&out.ProjectID, "VERTEX_PROJECT_ID")
		applyEnv(&out.Region, "VERTEX_REGION")
	case "grok-direct":
		if out.APIKey == "" {
			out.APIKey = cfg.APIKeys["grok"]
		}
		if out.BaseURL == "" && cfg.GrokConfig != nil {
			out.BaseURL = cfg.GrokConfig.BaseURL
		}
		applyEnv(&out.APIKey, "XAI_API_KEY")
		applyEnv(&out.BaseURL, "XAI_BASE_URL")
	case "openrouter":
		if out.APIKey == "" {
			out.APIKey = cfg.APIKeys["openrouter"]
		}
		if out.BaseURL == "" && cfg.OpenRouterConfig != nil {
			out.BaseURL = cfg.OpenRouterConfig.BaseURL
		}
		applyEnv(&out.APIKey, "OPENROUTER_API_KEY")
		applyEnv(&out.BaseURL, "OPENROUTER_BASE_URL")
	case "ollama-local":
		if out.BaseURL == "" && cfg.OllamaConfig != nil {
			out.BaseURL = cfg.OllamaConfig.BaseURL
		}
		applyEnv(&out.BaseURL, "OLLAMA_BASE_URL")
	case "apple-local":
		if out.BaseURL == "" && cfg.AppleConfig != nil {
			out.BaseURL = cfg.AppleConfig.BaseURL
		}
		applyEnv(&out.BaseURL, "APPLE_FM_BASE_URL")
	}
	return out
}

func cloneLangdagStringMap(in map[string]string) map[string]string {
	if len(in) == 0 {
		return nil
	}
	out := make(map[string]string, len(in))
	for key, value := range in {
		out[key] = value
	}
	return out
}

// buildRouter creates a Router from routing and fallback configuration.
func buildRouter(ctx context.Context, cfg Config, globalRetry internalprovider.RetryConfig) (internalprovider.Provider, error) {
	providerCache := map[string]internalprovider.Provider{}

	getOrCreate := func(name string) (internalprovider.Provider, error) {
		if p, ok := providerCache[name]; ok {
			return p, nil
		}
		p, err := createSingleProvider(ctx, name, cfg)
		if err != nil {
			// Silently drop unavailable providers in routing (consistent with api/server.go)
			log.Printf("langdag: skipping unavailable provider %q: %v", name, err)
			return nil, nil //nolint:nilerr
		}
		providerCache[name] = p
		return p, nil
	}

	// Build routing entries
	var entries []internalprovider.RouteEntry
	for _, re := range cfg.Routing {
		p, err := getOrCreate(re.Provider)
		if err != nil {
			return nil, err
		}
		if p == nil {
			continue
		}
		entryCfg := resolveRetryConfig(re.Retry)
		wrapped := internalprovider.WithRetry(internalprovider.WithServerToolFilter(p), entryCfg)
		entries = append(entries, internalprovider.RouteEntry{Provider: wrapped, Weight: re.Weight})
		log.Printf("langdag: routing: %s (weight=%d)", re.Provider, re.Weight)
	}

	// Build fallback chain
	var fallbackProviders []internalprovider.Provider
	for _, name := range cfg.FallbackOrder {
		p, err := getOrCreate(name)
		if err != nil {
			return nil, err
		}
		if p == nil {
			continue
		}
		wrapped := internalprovider.WithRetry(internalprovider.WithServerToolFilter(p), globalRetry)
		fallbackProviders = append(fallbackProviders, wrapped)
		log.Printf("langdag: fallback: %s", name)
	}

	return internalprovider.NewRouter(entries, fallbackProviders)
}

// DefaultModelCatalog returns the model catalog embedded with the library.
// It contains model names, pricing (per 1M tokens), context window sizes,
// and max output tokens for all supported providers.
func DefaultModelCatalog() (*ModelCatalog, error) {
	return models.DefaultCatalog()
}

// LoadModelCatalog loads the model catalog from a cache file, falling back to
// the embedded default if the file does not exist or is invalid JSON.
func LoadModelCatalog(cachePath string) (*ModelCatalog, error) {
	return models.LoadCatalog(cachePath)
}

// LoadModelCatalogWithOptions loads a usable model catalog immediately,
// preferring a valid cache and falling back to embedded catalog data with
// diagnostics.
func LoadModelCatalogWithOptions(opts CatalogLoadOptions) (*CatalogLoadResult, error) {
	return models.LoadCatalogWithOptions(opts)
}

// LoadRuntimeModelCatalog loads the model catalog used by prompt/runtime
// routing from the embedded catalog generated from the published model-catalog
// branch.
func LoadRuntimeModelCatalog() (*CatalogLoadResult, error) {
	return models.LoadRuntimeCatalog(models.CatalogLoadOptions{})
}

// LoadRuntimeModelCatalogWithOptions loads the prompt/runtime catalog with
// explicit load options. If opts.CachePath is empty, the embedded catalog is
// used and no user cache path is read implicitly.
func LoadRuntimeModelCatalogWithOptions(opts CatalogLoadOptions) (*CatalogLoadResult, error) {
	return models.LoadRuntimeCatalog(opts)
}

// LoadRemoteModelCatalog fetches the published remote model catalog and
// validates it without reading or writing any local cache file.
func LoadRemoteModelCatalog(ctx context.Context, opts CatalogRefreshOptions) (*CatalogRefreshResult, error) {
	return models.LoadRemoteCatalog(ctx, opts)
}

// RefreshModelCatalogCache fetches the published remote catalog artifact. If
// opts.CachePath is non-empty, the fetched catalog replaces that cache.
// Invalid, stale, or partial remote data does not replace an existing cache.
func RefreshModelCatalogCache(ctx context.Context, opts CatalogRefreshOptions) (*CatalogRefreshResult, error) {
	return models.RefreshCatalogCache(ctx, opts)
}

// FetchModelCatalog fetches the latest model catalog from official provider
// documentation pages (OpenAI, Anthropic, Google, xAI).
// This does not require any API keys.
//
// If cachePath is non-empty, the fetched catalog is saved to that path
// so it can be loaded with LoadModelCatalog in future sessions.
func FetchModelCatalog(ctx context.Context, cachePath string) (*ModelCatalog, error) {
	catalog, err := models.FetchLatest(ctx)
	if err != nil {
		return nil, err
	}
	if cachePath != "" {
		if saveErr := models.SaveCatalog(catalog, cachePath); saveErr != nil {
			return catalog, fmt.Errorf("langdag: catalog fetched but save failed: %w", saveErr)
		}
	}
	return catalog, nil
}

// resolveRetryConfig converts a *RetryConfig to internalprovider.RetryConfig,
// falling back to the default when nil.
func resolveRetryConfig(rc *RetryConfig) internalprovider.RetryConfig {
	defaults := internalprovider.DefaultRetryConfig()
	if rc == nil {
		return defaults
	}
	cfg := defaults
	if rc.MaxRetries > 0 {
		cfg.MaxRetries = rc.MaxRetries
	}
	if rc.BaseDelay > 0 {
		cfg.BaseDelay = rc.BaseDelay
	}
	if rc.MaxDelay > 0 {
		cfg.MaxDelay = rc.MaxDelay
	}
	cfg.OnRetry = rc.OnRetry
	return cfg
}
