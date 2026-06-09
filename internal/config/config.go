// Package config handles configuration loading for langdag.
package config

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/spf13/viper"
)

// Config represents the application configuration.
type Config struct {
	Storage     StorageConfig               `mapstructure:"storage"`
	Providers   ProvidersConfig             `mapstructure:"providers"`
	Deployments map[string]DeploymentConfig `mapstructure:"deployments"`
	Routing     *RoutingPolicy              `mapstructure:"routing"`
	Server      ServerConfig                `mapstructure:"server"`
	Logging     LoggingConfig               `mapstructure:"logging"`
	Retry       RetryConfig                 `mapstructure:"retry"`
}

// StorageConfig represents storage configuration.
type StorageConfig struct {
	Driver     string `mapstructure:"driver"`
	Path       string `mapstructure:"path"`
	Connection string `mapstructure:"connection"`
}

// ProvidersConfig represents provider configurations.
type ProvidersConfig struct {
	Default string `mapstructure:"default"`

	// Direct providers
	Anthropic  ProviderConfig     `mapstructure:"anthropic"`
	OpenAI     ProviderConfig     `mapstructure:"openai"`
	Gemini     ProviderConfig     `mapstructure:"gemini"`
	Grok       ProviderConfig     `mapstructure:"grok"`
	OpenRouter ProviderConfig     `mapstructure:"openrouter"`
	Ollama     ProviderConfig     `mapstructure:"ollama"`
	Apple      ProviderConfig     `mapstructure:"apple"`
	Mock       MockProviderConfig `mapstructure:"mock"`

	// Cloud platform variants
	AnthropicVertex  VertexConfig  `mapstructure:"anthropic-vertex"`
	AnthropicBedrock BedrockConfig `mapstructure:"anthropic-bedrock"`
	OpenAIAzure      AzureConfig   `mapstructure:"openai-azure"`
	GeminiVertex     VertexConfig  `mapstructure:"gemini-vertex"`

	// Routing and fallback
	Routing       []RoutingEntry `mapstructure:"routing"`
	FallbackOrder []string       `mapstructure:"fallback_order"`
}

// ProviderConfig represents a single provider configuration.
type ProviderConfig struct {
	APIKey  string `mapstructure:"api_key"`
	BaseURL string `mapstructure:"base_url"`
}

// VertexConfig represents Vertex AI provider configuration.
type VertexConfig struct {
	ProjectID string `mapstructure:"project_id"`
	Region    string `mapstructure:"region"`
}

// BedrockConfig represents AWS Bedrock provider configuration.
type BedrockConfig struct {
	Region string `mapstructure:"region"`
}

// AzureConfig represents Azure OpenAI provider configuration.
type AzureConfig struct {
	APIKey     string `mapstructure:"api_key"`
	Endpoint   string `mapstructure:"endpoint"`
	APIVersion string `mapstructure:"api_version"`
}

// DeploymentConfig represents deployment-scoped runtime configuration.
type DeploymentConfig struct {
	APIKey        string            `mapstructure:"api_key"`
	BaseURL       string            `mapstructure:"base_url"`
	Endpoint      string            `mapstructure:"endpoint"`
	APIVersion    string            `mapstructure:"api_version"`
	ProjectID     string            `mapstructure:"project_id"`
	Region        string            `mapstructure:"region"`
	ModelMappings map[string]string `mapstructure:"model_mappings"`
}

// RoutingPolicy represents deployment-aware routing configuration.
type RoutingPolicy struct {
	Default   []RoutingStage            `mapstructure:"default"`
	Providers map[string][]RoutingStage `mapstructure:"providers"`
	Models    map[string][]RoutingStage `mapstructure:"models"`
}

type RoutingStage struct {
	Deployments []DeploymentChoice `mapstructure:"deployments"`
	Retries     int                `mapstructure:"retries"`
}

type DeploymentChoice struct {
	DeploymentID string `mapstructure:"deployment_id"`
	Weight       int    `mapstructure:"weight"`
}

// RoutingEntry represents a single entry in the routing configuration.
type RoutingEntry struct {
	Provider string      `mapstructure:"provider"`
	Weight   int         `mapstructure:"weight"`
	Retry    RetryConfig `mapstructure:"retry"`
}

// MockProviderConfig represents mock provider configuration.
type MockProviderConfig struct {
	Mode             string `mapstructure:"mode"`               // random, echo, fixed, error, stream_error
	FixedResponse    string `mapstructure:"fixed_response"`     // response for fixed mode
	Delay            string `mapstructure:"delay"`              // delay before responding
	ChunkDelay       string `mapstructure:"chunk_delay"`        // delay between stream chunks
	ErrorMessage     string `mapstructure:"error_message"`      // error text for error/stream_error modes
	ErrorAfterChunks int    `mapstructure:"error_after_chunks"` // chunks before error in stream_error mode
}

// ServerConfig represents server configuration.
type ServerConfig struct {
	Host        string   `mapstructure:"host"`
	Port        int      `mapstructure:"port"`
	CORSOrigins []string `mapstructure:"cors_origins"`
}

// LoggingConfig represents logging configuration.
type LoggingConfig struct {
	Level  string `mapstructure:"level"`
	Format string `mapstructure:"format"`
}

// RetryConfig represents retry configuration for LLM provider calls.
type RetryConfig struct {
	MaxRetries int    `mapstructure:"max_retries"`
	BaseDelay  string `mapstructure:"base_delay"`
	MaxDelay   string `mapstructure:"max_delay"`
}

// Load loads the configuration from files and environment variables.
func Load() (*Config, error) {
	v := viper.New()

	// Set defaults
	setDefaults(v)

	// Set config name and paths
	v.SetConfigName("config")
	v.SetConfigType("yaml")

	// Add config paths
	v.AddConfigPath(".")
	v.AddConfigPath("./langdag")

	// Add user config directory
	homeDir, err := os.UserHomeDir()
	if err == nil {
		v.AddConfigPath(filepath.Join(homeDir, ".config", "langdag"))
	}

	// Read config file (optional)
	if err := v.ReadInConfig(); err != nil {
		if _, ok := err.(viper.ConfigFileNotFoundError); !ok {
			return nil, fmt.Errorf("error reading config file: %w", err)
		}
		// Config file not found is OK, we'll use defaults and env vars
	}

	// Bind environment variables
	v.SetEnvPrefix("LANGDAG")
	v.AutomaticEnv()

	// Also support direct env var names
	v.BindEnv("providers.default", "LANGDAG_PROVIDER")
	v.BindEnv("providers.anthropic.api_key", "ANTHROPIC_API_KEY")
	v.BindEnv("providers.openai.api_key", "OPENAI_API_KEY")
	v.BindEnv("providers.openai.base_url", "OPENAI_BASE_URL")
	v.BindEnv("providers.gemini.api_key", "GEMINI_API_KEY")
	v.BindEnv("providers.grok.api_key", "XAI_API_KEY")
	v.BindEnv("providers.grok.base_url", "XAI_BASE_URL")
	v.BindEnv("providers.openrouter.api_key", "OPENROUTER_API_KEY")
	v.BindEnv("providers.openrouter.base_url", "OPENROUTER_BASE_URL")
	v.BindEnv("providers.ollama.base_url", "OLLAMA_BASE_URL")
	v.BindEnv("providers.apple.base_url", "APPLE_FM_BASE_URL")
	v.BindEnv("providers.mock.mode", "LANGDAG_MOCK_MODE")
	v.BindEnv("providers.mock.fixed_response", "LANGDAG_MOCK_RESPONSE")
	v.BindEnv("providers.mock.delay", "LANGDAG_MOCK_DELAY")
	v.BindEnv("providers.mock.chunk_delay", "LANGDAG_MOCK_CHUNK_DELAY")
	v.BindEnv("providers.mock.error_message", "LANGDAG_MOCK_ERROR_MESSAGE")
	v.BindEnv("providers.mock.error_after_chunks", "LANGDAG_MOCK_ERROR_AFTER_CHUNKS")
	v.BindEnv("storage.path", "LANGDAG_STORAGE_PATH")
	v.BindEnv("retry.max_retries", "LANGDAG_RETRY_MAX")
	v.BindEnv("retry.base_delay", "LANGDAG_RETRY_BASE_DELAY")
	v.BindEnv("retry.max_delay", "LANGDAG_RETRY_MAX_DELAY")

	// Provider variant env vars
	v.BindEnv("providers.anthropic-vertex.project_id", "VERTEX_PROJECT_ID")
	v.BindEnv("providers.anthropic-vertex.region", "VERTEX_REGION")
	v.BindEnv("providers.anthropic-bedrock.region", "AWS_REGION")
	v.BindEnv("providers.openai-azure.api_key", "AZURE_OPENAI_API_KEY")
	v.BindEnv("providers.openai-azure.endpoint", "AZURE_OPENAI_ENDPOINT")
	v.BindEnv("providers.openai-azure.api_version", "AZURE_OPENAI_API_VERSION")
	v.BindEnv("providers.gemini-vertex.project_id", "VERTEX_PROJECT_ID")
	v.BindEnv("providers.gemini-vertex.region", "VERTEX_REGION")

	// Unmarshal config
	var cfg Config
	if err := v.Unmarshal(&cfg); err != nil {
		return nil, fmt.Errorf("error unmarshaling config: %w", err)
	}

	// Expand environment variables in paths
	cfg.Storage.Path = os.ExpandEnv(cfg.Storage.Path)

	// Parse LANGDAG_ROUTING env var (JSON array)
	if routingJSON := os.Getenv("LANGDAG_ROUTING"); routingJSON != "" {
		var entries []RoutingEntry
		if err := json.Unmarshal([]byte(routingJSON), &entries); err != nil {
			return nil, fmt.Errorf("error parsing LANGDAG_ROUTING: %w", err)
		}
		cfg.Providers.Routing = entries
	}

	// Parse LANGDAG_FALLBACK_ORDER env var (comma-separated)
	if fallbackStr := os.Getenv("LANGDAG_FALLBACK_ORDER"); fallbackStr != "" {
		cfg.Providers.FallbackOrder = strings.Split(fallbackStr, ",")
		for i, s := range cfg.Providers.FallbackOrder {
			cfg.Providers.FallbackOrder[i] = strings.TrimSpace(s)
		}
	}

	return &cfg, nil
}

// setDefaults sets default configuration values.
func setDefaults(v *viper.Viper) {
	// Storage defaults
	v.SetDefault("storage.driver", "sqlite")
	v.SetDefault("storage.path", "./langdag.db")

	// Provider defaults
	v.SetDefault("providers.default", "anthropic")
	v.SetDefault("providers.mock.mode", "random")

	// Server defaults
	v.SetDefault("server.host", "0.0.0.0")
	v.SetDefault("server.port", 8080)
	v.SetDefault("server.cors_origins", []string{"*"})

	// Logging defaults
	v.SetDefault("logging.level", "info")
	v.SetDefault("logging.format", "text")

	// Retry defaults
	v.SetDefault("retry.max_retries", 3)
	v.SetDefault("retry.base_delay", "1s")
	v.SetDefault("retry.max_delay", "30s")
}

// GetDefaultStoragePath returns the default storage path.
func GetDefaultStoragePath() string {
	homeDir, err := os.UserHomeDir()
	if err != nil {
		return "./langdag.db"
	}
	return filepath.Join(homeDir, ".config", "langdag", "langdag.db")
}

// EnsureConfigDir ensures the config directory exists.
func EnsureConfigDir() error {
	homeDir, err := os.UserHomeDir()
	if err != nil {
		return err
	}
	configDir := filepath.Join(homeDir, ".config", "langdag")
	return os.MkdirAll(configDir, 0755)
}

// EnsureStorageDir ensures the directory for the storage path exists.
func EnsureStorageDir(storagePath string) error {
	dir := filepath.Dir(storagePath)
	return os.MkdirAll(dir, 0755)
}
