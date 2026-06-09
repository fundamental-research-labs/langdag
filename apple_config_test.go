package langdag

import (
	"context"
	"testing"

	internalprovider "langdag.com/langdag/internal/provider"
)

func TestAppleDeploymentMappingAndEnvFallback(t *testing.T) {
	t.Setenv("APPLE_FM_BASE_URL", "http://127.0.0.1:2999")

	if got := deploymentIDForProviderName("apple"); got != "apple-local" {
		t.Fatalf("deploymentIDForProviderName(apple) = %q", got)
	}
	cfg := Config{}
	deployment := deploymentConfigForID("apple-local", cfg)
	if deployment.BaseURL != "http://127.0.0.1:2999" {
		t.Fatalf("apple-local BaseURL = %q", deployment.BaseURL)
	}
	adapter, err := createDeploymentAdapter(context.Background(), "apple-local", cfg, internalprovider.RetryConfig{})
	if err != nil {
		t.Fatal(err)
	}
	if adapter.DeploymentID != "apple-local" || adapter.Provider.Name() != "apple" {
		t.Fatalf("adapter = %+v provider=%q", adapter, adapter.Provider.Name())
	}
}

func TestAppleSingleProvider(t *testing.T) {
	provider, err := createSingleProvider(context.Background(), "apple", Config{
		AppleConfig: &AppleConfig{BaseURL: "http://127.0.0.1:2999"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if provider.Name() != "apple" {
		t.Fatalf("provider name = %q", provider.Name())
	}
}
