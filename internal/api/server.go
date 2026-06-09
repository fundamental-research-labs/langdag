// Package api provides an HTTP API server for langdag.
package api

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net/http"
	"strings"
	"time"

	"langdag.com/langdag/internal/config"
	"langdag.com/langdag/internal/conversation"
	"langdag.com/langdag/internal/models"
	"langdag.com/langdag/internal/provider"
	"langdag.com/langdag/internal/provider/anthropic"
	geminiprovider "langdag.com/langdag/internal/provider/gemini"
	mockprovider "langdag.com/langdag/internal/provider/mock"
	openaiprovider "langdag.com/langdag/internal/provider/openai"
	"langdag.com/langdag/internal/storage/sqlite"
)

// Server represents the HTTP API server.
type Server struct {
	httpServer *http.Server
	store      *sqlite.SQLiteStorage
	convMgr    *conversation.Manager
	apiKey     string
}

// Config holds server configuration.
type Config struct {
	Addr   string
	APIKey string // Optional API key for authentication
}

// New creates a new API server.
func New(cfg *Config, appConfig *config.Config) (*Server, error) {
	ctx := context.Background()

	// Initialize storage
	storagePath := appConfig.Storage.Path
	if storagePath == "./langdag.db" {
		storagePath = config.GetDefaultStoragePath()
	}

	if err := config.EnsureStorageDir(storagePath); err != nil {
		return nil, err
	}

	store, err := sqlite.New(storagePath)
	if err != nil {
		return nil, err
	}

	if err := store.Init(ctx); err != nil {
		store.Close()
		return nil, err
	}

	// Create provider (may return a Router when routing is configured)
	prov, err := createProvider(ctx, appConfig)
	if err != nil {
		store.Close()
		return nil, err
	}

	// Create managers
	convMgr := conversation.NewManager(store, prov)

	s := &Server{
		store:   store,
		convMgr: convMgr,
		apiKey:  cfg.APIKey,
	}

	// Setup routes
	mux := http.NewServeMux()

	// Health check
	mux.HandleFunc("GET /health", s.handleHealth)

	// Prompt endpoints
	mux.HandleFunc("POST /prompt", s.authMiddleware(s.handlePrompt))
	mux.HandleFunc("POST /nodes/{id}/prompt", s.authMiddleware(s.handleNodePrompt))

	// Node endpoints
	mux.HandleFunc("GET /nodes", s.authMiddleware(s.handleListNodes))
	mux.HandleFunc("GET /nodes/{id}", s.authMiddleware(s.handleGetNode))
	mux.HandleFunc("GET /nodes/{id}/tree", s.authMiddleware(s.handleGetTree))
	mux.HandleFunc("DELETE /nodes/{id}", s.authMiddleware(s.handleDeleteNode))

	// Alias endpoints
	mux.HandleFunc("PUT /nodes/{id}/aliases/{alias}", s.authMiddleware(s.handleCreateAlias))
	mux.HandleFunc("GET /nodes/{id}/aliases", s.authMiddleware(s.handleListAliases))
	mux.HandleFunc("DELETE /aliases/{alias}", s.authMiddleware(s.handleDeleteAlias))

	s.httpServer = &http.Server{
		Addr:         cfg.Addr,
		Handler:      s.corsMiddleware(mux),
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 0, // Disable for SSE streaming
		IdleTimeout:  120 * time.Second,
	}

	return s, nil
}

// Start starts the HTTP server.
func (s *Server) Start() error {
	log.Printf("Starting API server on %s", s.httpServer.Addr)
	return s.httpServer.ListenAndServe()
}

// Shutdown gracefully shuts down the server.
func (s *Server) Shutdown(ctx context.Context) error {
	s.store.Close()
	return s.httpServer.Shutdown(ctx)
}

// Addr returns the server address.
func (s *Server) Addr() string {
	return s.httpServer.Addr
}

// authMiddleware checks for API key authentication if configured.
func (s *Server) authMiddleware(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if s.apiKey != "" {
			auth := r.Header.Get("Authorization")
			if auth == "" {
				auth = r.Header.Get("X-API-Key")
			} else {
				auth = strings.TrimPrefix(auth, "Bearer ")
			}

			if auth != s.apiKey {
				writeError(w, http.StatusUnauthorized, "unauthorized")
				return
			}
		}
		next(w, r)
	}
}

// corsMiddleware adds CORS headers.
func (s *Server) corsMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization, X-API-Key")

		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusNoContent)
			return
		}

		next.ServeHTTP(w, r)
	})
}

// handleHealth returns a simple health check response.
func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

// writeJSON writes a JSON response.
func writeJSON(w http.ResponseWriter, status int, data interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(data)
}

// writeError writes a JSON error response.
func writeError(w http.ResponseWriter, status int, message string) {
	writeJSON(w, status, map[string]string{"error": message})
}

// decodeJSON decodes JSON from the request body.
func decodeJSON(r *http.Request, v interface{}) error {
	return json.NewDecoder(r.Body).Decode(v)
}

// parseRetryConfig parses a config.RetryConfig into a provider.RetryConfig,
// falling back to the global retry config for unset fields.
func parseRetryConfig(rc config.RetryConfig, global provider.RetryConfig) provider.RetryConfig {
	cfg := global
	if rc.MaxRetries > 0 {
		cfg.MaxRetries = rc.MaxRetries
	}
	if rc.BaseDelay != "" {
		if d, err := time.ParseDuration(rc.BaseDelay); err == nil {
			cfg.BaseDelay = d
		}
	}
	if rc.MaxDelay != "" {
		if d, err := time.ParseDuration(rc.MaxDelay); err == nil {
			cfg.MaxDelay = d
		}
	}
	return cfg
}

// providerFactory is a function that creates a provider.
type providerFactory func(ctx context.Context, appConfig *config.Config) (provider.Provider, error)

// providerRegistry maps provider names to their factory functions.
var providerRegistry = map[string]providerFactory{
	"anthropic": func(_ context.Context, c *config.Config) (provider.Provider, error) {
		if c.Providers.Anthropic.APIKey == "" {
			return nil, fmt.Errorf("ANTHROPIC_API_KEY not set")
		}
		return anthropic.New(c.Providers.Anthropic.APIKey), nil
	},
	"anthropic-vertex": func(ctx context.Context, c *config.Config) (provider.Provider, error) {
		vc := c.Providers.AnthropicVertex
		if vc.ProjectID == "" || vc.Region == "" {
			return nil, fmt.Errorf("VERTEX_PROJECT_ID and VERTEX_REGION must be set for anthropic-vertex")
		}
		return anthropic.NewVertex(ctx, vc.Region, vc.ProjectID)
	},
	"anthropic-bedrock": func(ctx context.Context, c *config.Config) (provider.Provider, error) {
		return anthropic.NewBedrock(ctx, c.Providers.AnthropicBedrock.Region)
	},
	"openai": func(_ context.Context, c *config.Config) (provider.Provider, error) {
		if c.Providers.OpenAI.APIKey == "" {
			return nil, fmt.Errorf("OPENAI_API_KEY not set")
		}
		return openaiprovider.New(c.Providers.OpenAI.APIKey, c.Providers.OpenAI.BaseURL), nil
	},
	"openai-azure": func(_ context.Context, c *config.Config) (provider.Provider, error) {
		ac := c.Providers.OpenAIAzure
		if ac.APIKey == "" || ac.Endpoint == "" {
			return nil, fmt.Errorf("AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT must be set for openai-azure")
		}
		return openaiprovider.NewAzure(ac.APIKey, ac.Endpoint, ac.APIVersion), nil
	},
	"grok": func(_ context.Context, c *config.Config) (provider.Provider, error) {
		if c.Providers.Grok.APIKey == "" {
			return nil, fmt.Errorf("XAI_API_KEY not set")
		}
		return openaiprovider.NewGrok(c.Providers.Grok.APIKey, c.Providers.Grok.BaseURL), nil
	},
	"openrouter": func(_ context.Context, c *config.Config) (provider.Provider, error) {
		if c.Providers.OpenRouter.APIKey == "" {
			return nil, fmt.Errorf("OPENROUTER_API_KEY not set")
		}
		return openaiprovider.NewOpenRouter(c.Providers.OpenRouter.APIKey, c.Providers.OpenRouter.BaseURL), nil
	},
	"ollama": func(_ context.Context, c *config.Config) (provider.Provider, error) {
		return openaiprovider.NewOllama(c.Providers.Ollama.BaseURL), nil
	},
	"apple": func(_ context.Context, c *config.Config) (provider.Provider, error) {
		return openaiprovider.NewApple(c.Providers.Apple.BaseURL), nil
	},
	"gemini": newGeminiProvider,
	"gemma":  newGeminiProvider,
	"gemini-vertex": func(ctx context.Context, c *config.Config) (provider.Provider, error) {
		vc := c.Providers.GeminiVertex
		if vc.ProjectID == "" || vc.Region == "" {
			return nil, fmt.Errorf("VERTEX_PROJECT_ID and VERTEX_REGION must be set for gemini-vertex")
		}
		return geminiprovider.NewVertex(ctx, vc.ProjectID, vc.Region)
	},
	"mock": func(_ context.Context, c *config.Config) (provider.Provider, error) {
		cfg := mockprovider.Config{
			Mode:             c.Providers.Mock.Mode,
			FixedResponse:    c.Providers.Mock.FixedResponse,
			ErrorAfterChunks: c.Providers.Mock.ErrorAfterChunks,
		}
		if c.Providers.Mock.ErrorMessage != "" {
			cfg.Error = fmt.Errorf("%s", c.Providers.Mock.ErrorMessage)
		}
		if c.Providers.Mock.Delay != "" {
			d, err := time.ParseDuration(c.Providers.Mock.Delay)
			if err != nil {
				return nil, fmt.Errorf("invalid mock delay: %w", err)
			}
			cfg.Delay = d
		}
		if c.Providers.Mock.ChunkDelay != "" {
			d, err := time.ParseDuration(c.Providers.Mock.ChunkDelay)
			if err != nil {
				return nil, fmt.Errorf("invalid mock chunk_delay: %w", err)
			}
			cfg.ChunkDelay = d
		}
		return mockprovider.New(cfg), nil
	},
}

// newGeminiProvider creates a Gemini provider using Google AI Studio credentials.
func newGeminiProvider(_ context.Context, c *config.Config) (provider.Provider, error) {
	if c.Providers.Gemini.APIKey == "" {
		return nil, fmt.Errorf("GEMINI_API_KEY not set")
	}
	return geminiprovider.New(c.Providers.Gemini.APIKey), nil
}

// createProvider creates the LLM provider based on configuration.
// When routing config is present, it builds a Router with weighted selection
// and fallback. Otherwise, it creates a single provider with global retry.
func createProvider(ctx context.Context, appConfig *config.Config) (provider.Provider, error) {
	globalRetry := parseRetryConfig(appConfig.Retry, provider.DefaultRetryConfig())

	if len(appConfig.Deployments) > 0 || appConfig.Routing != nil {
		return createDeploymentAwareProvider(ctx, appConfig, globalRetry)
	}

	// If routing is configured, build a Router
	if len(appConfig.Providers.Routing) > 0 {
		return createRouter(ctx, appConfig, globalRetry)
	}

	// Single-provider mode (backward compatible)
	name := appConfig.Providers.Default
	factory, ok := providerRegistry[name]
	if !ok {
		return nil, fmt.Errorf("unknown provider: %s", name)
	}

	prov, err := factory(ctx, appConfig)
	if err != nil {
		return nil, err
	}

	log.Printf("Using provider: %s", name)
	return provider.WithRetry(provider.WithServerToolFilter(prov), globalRetry), nil
}

func createDeploymentAwareProvider(ctx context.Context, appConfig *config.Config, globalRetry provider.RetryConfig) (provider.Provider, error) {
	catalogResult, err := models.LoadRuntimeCatalog(models.CatalogLoadOptions{})
	if err != nil {
		return nil, err
	}
	compiled, err := models.CompileCatalogV1(catalogResult.Catalog)
	if err != nil {
		return nil, err
	}
	deploymentIDs := apiDeploymentIDsForConfig(appConfig)
	adapters := map[string]provider.DeploymentAdapter{}
	var adapterErrors []error
	for _, deploymentID := range deploymentIDs {
		adapter, err := createDeploymentAdapter(ctx, deploymentID, appConfig, globalRetry)
		if err != nil {
			adapterErrors = append(adapterErrors, fmt.Errorf("%s: %w", deploymentID, err))
			log.Printf("Skipping unavailable deployment: %s: %v", deploymentID, err)
			continue
		}
		adapters[deploymentID] = adapter
	}
	if len(adapters) == 0 {
		if len(adapterErrors) > 0 {
			return nil, fmt.Errorf("no configured deployments are available: %w", errors.Join(adapterErrors...))
		}
		return nil, fmt.Errorf("no configured deployments are available")
	}
	return provider.NewDeploymentRouter(provider.DeploymentRouterOptions{
		Catalog:     compiled,
		Deployments: adapters,
		Routing:     apiRoutingPolicy(appConfig.Routing),
	})
}

func apiDeploymentIDsForConfig(appConfig *config.Config) []string {
	seen := map[string]bool{}
	var ids []string
	add := func(id string) {
		if id == "" || seen[id] {
			return
		}
		seen[id] = true
		ids = append(ids, id)
	}
	for deploymentID := range appConfig.Deployments {
		add(deploymentID)
	}
	if appConfig.Routing != nil {
		collect := func(stages []config.RoutingStage) {
			for _, stage := range stages {
				for _, choice := range stage.Deployments {
					add(choice.DeploymentID)
				}
			}
		}
		collect(appConfig.Routing.Default)
		for _, stages := range appConfig.Routing.Providers {
			collect(stages)
		}
		for _, stages := range appConfig.Routing.Models {
			collect(stages)
		}
	}
	return ids
}

func apiRoutingPolicy(routing *config.RoutingPolicy) provider.RoutingPolicy {
	if routing == nil {
		return provider.RoutingPolicy{}
	}
	return provider.RoutingPolicy{
		Default:   apiRoutingStages(routing.Default),
		Providers: apiRoutingStageMap(routing.Providers),
		Models:    apiRoutingStageMap(routing.Models),
	}
}

func apiRoutingStageMap(in map[string][]config.RoutingStage) map[string][]provider.RoutingStage {
	if len(in) == 0 {
		return nil
	}
	out := make(map[string][]provider.RoutingStage, len(in))
	for key, stages := range in {
		out[key] = apiRoutingStages(stages)
	}
	return out
}

func apiRoutingStages(in []config.RoutingStage) []provider.RoutingStage {
	if in == nil {
		return nil
	}
	out := make([]provider.RoutingStage, len(in))
	for i, stage := range in {
		out[i].Retries = stage.Retries
		for _, choice := range stage.Deployments {
			out[i].Deployments = append(out[i].Deployments, provider.DeploymentChoice{
				DeploymentID: choice.DeploymentID,
				Weight:       choice.Weight,
			})
		}
	}
	return out
}

func createDeploymentAdapter(ctx context.Context, deploymentID string, appConfig *config.Config, globalRetry provider.RetryConfig) (provider.DeploymentAdapter, error) {
	cfg := apiDeploymentConfigForID(deploymentID, appConfig)
	var prov provider.Provider
	var err error
	switch deploymentID {
	case "anthropic-direct":
		if cfg.APIKey == "" {
			return provider.DeploymentAdapter{}, fmt.Errorf("ANTHROPIC_API_KEY not set")
		}
		prov = anthropic.New(cfg.APIKey)
	case "anthropic-bedrock":
		prov, err = anthropic.NewBedrock(ctx, cfg.Region)
	case "anthropic-vertex":
		if cfg.ProjectID == "" || cfg.Region == "" {
			return provider.DeploymentAdapter{}, fmt.Errorf("VERTEX_PROJECT_ID and VERTEX_REGION must be set for anthropic-vertex")
		}
		prov, err = anthropic.NewVertex(ctx, cfg.Region, cfg.ProjectID)
	case "openai-direct":
		if cfg.APIKey == "" {
			return provider.DeploymentAdapter{}, fmt.Errorf("OPENAI_API_KEY not set")
		}
		prov = openaiprovider.New(cfg.APIKey, cfg.BaseURL)
	case "openai-azure":
		if cfg.APIKey == "" || cfg.Endpoint == "" {
			return provider.DeploymentAdapter{}, fmt.Errorf("AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT must be set for openai-azure")
		}
		prov = openaiprovider.NewAzure(cfg.APIKey, cfg.Endpoint, cfg.APIVersion)
	case "gemini-direct":
		if cfg.APIKey == "" {
			return provider.DeploymentAdapter{}, fmt.Errorf("GEMINI_API_KEY not set")
		}
		prov = geminiprovider.New(cfg.APIKey)
	case "gemini-vertex":
		if cfg.ProjectID == "" || cfg.Region == "" {
			return provider.DeploymentAdapter{}, fmt.Errorf("VERTEX_PROJECT_ID and VERTEX_REGION must be set for gemini-vertex")
		}
		prov, err = geminiprovider.NewVertex(ctx, cfg.ProjectID, cfg.Region)
	case "grok-direct":
		if cfg.APIKey == "" {
			return provider.DeploymentAdapter{}, fmt.Errorf("XAI_API_KEY not set")
		}
		prov = openaiprovider.NewGrok(cfg.APIKey, cfg.BaseURL)
	case "openrouter":
		if cfg.APIKey == "" {
			return provider.DeploymentAdapter{}, fmt.Errorf("OPENROUTER_API_KEY not set")
		}
		prov = openaiprovider.NewOpenRouter(cfg.APIKey, cfg.BaseURL)
	case "ollama-local":
		prov = openaiprovider.NewOllama(cfg.BaseURL)
	case "apple-local":
		prov = openaiprovider.NewApple(cfg.BaseURL)
	default:
		return provider.DeploymentAdapter{}, fmt.Errorf("unknown deployment: %s", deploymentID)
	}
	if err != nil {
		return provider.DeploymentAdapter{}, err
	}
	prov = provider.WithRetry(prov, globalRetry)
	return provider.DeploymentAdapter{
		DeploymentID:  deploymentID,
		Provider:      prov,
		ModelMappings: cfg.ModelMappings,
	}, nil
}

func apiDeploymentConfigForID(deploymentID string, appConfig *config.Config) config.DeploymentConfig {
	cfg := appConfig.Deployments[deploymentID]
	switch deploymentID {
	case "anthropic-direct":
		if cfg.APIKey == "" {
			cfg.APIKey = appConfig.Providers.Anthropic.APIKey
		}
	case "anthropic-bedrock":
		if cfg.Region == "" {
			cfg.Region = appConfig.Providers.AnthropicBedrock.Region
		}
	case "anthropic-vertex":
		if cfg.ProjectID == "" {
			cfg.ProjectID = appConfig.Providers.AnthropicVertex.ProjectID
		}
		if cfg.Region == "" {
			cfg.Region = appConfig.Providers.AnthropicVertex.Region
		}
	case "openai-direct":
		if cfg.APIKey == "" {
			cfg.APIKey = appConfig.Providers.OpenAI.APIKey
		}
		if cfg.BaseURL == "" {
			cfg.BaseURL = appConfig.Providers.OpenAI.BaseURL
		}
	case "openai-azure":
		if cfg.APIKey == "" {
			cfg.APIKey = appConfig.Providers.OpenAIAzure.APIKey
		}
		if cfg.Endpoint == "" {
			cfg.Endpoint = appConfig.Providers.OpenAIAzure.Endpoint
		}
		if cfg.APIVersion == "" {
			cfg.APIVersion = appConfig.Providers.OpenAIAzure.APIVersion
		}
	case "gemini-direct":
		if cfg.APIKey == "" {
			cfg.APIKey = appConfig.Providers.Gemini.APIKey
		}
	case "gemini-vertex":
		if cfg.ProjectID == "" {
			cfg.ProjectID = appConfig.Providers.GeminiVertex.ProjectID
		}
		if cfg.Region == "" {
			cfg.Region = appConfig.Providers.GeminiVertex.Region
		}
	case "grok-direct":
		if cfg.APIKey == "" {
			cfg.APIKey = appConfig.Providers.Grok.APIKey
		}
		if cfg.BaseURL == "" {
			cfg.BaseURL = appConfig.Providers.Grok.BaseURL
		}
	case "openrouter":
		if cfg.APIKey == "" {
			cfg.APIKey = appConfig.Providers.OpenRouter.APIKey
		}
		if cfg.BaseURL == "" {
			cfg.BaseURL = appConfig.Providers.OpenRouter.BaseURL
		}
	case "ollama-local":
		if cfg.BaseURL == "" {
			cfg.BaseURL = appConfig.Providers.Ollama.BaseURL
		}
	case "apple-local":
		if cfg.BaseURL == "" {
			cfg.BaseURL = appConfig.Providers.Apple.BaseURL
		}
	}
	return cfg
}

// createRouter builds a Router from the routing and fallback config.
func createRouter(ctx context.Context, appConfig *config.Config, globalRetry provider.RetryConfig) (provider.Provider, error) {
	// Build a cache of providers by name (so we don't create duplicates)
	providerCache := map[string]provider.Provider{}

	getOrCreate := func(name string) (provider.Provider, error) {
		if p, ok := providerCache[name]; ok {
			return p, nil
		}
		factory, ok := providerRegistry[name]
		if !ok {
			return nil, fmt.Errorf("unknown provider in routing config: %s", name)
		}
		p, err := factory(ctx, appConfig)
		if err != nil {
			return nil, nil // silently drop unavailable providers
		}
		providerCache[name] = p
		return p, nil
	}

	// Build routing entries
	var entries []provider.RouteEntry
	for _, re := range appConfig.Providers.Routing {
		p, err := getOrCreate(re.Provider)
		if err != nil {
			return nil, err
		}
		if p == nil {
			log.Printf("Skipping unavailable provider in routing: %s", re.Provider)
			continue
		}
		// Wrap with server tool filter and per-provider retry
		retryCfg := parseRetryConfig(re.Retry, globalRetry)
		wrapped := provider.WithRetry(provider.WithServerToolFilter(p), retryCfg)
		entries = append(entries, provider.RouteEntry{Provider: wrapped, Weight: re.Weight})
		log.Printf("Routing: %s (weight=%d)", re.Provider, re.Weight)
	}

	// Build fallback chain
	var fallbackProviders []provider.Provider
	for _, name := range appConfig.Providers.FallbackOrder {
		p, err := getOrCreate(name)
		if err != nil {
			return nil, err
		}
		if p == nil {
			log.Printf("Skipping unavailable provider in fallback: %s", name)
			continue
		}
		wrapped := provider.WithRetry(provider.WithServerToolFilter(p), globalRetry)
		fallbackProviders = append(fallbackProviders, wrapped)
		log.Printf("Fallback: %s", name)
	}

	return provider.NewRouter(entries, fallbackProviders)
}
