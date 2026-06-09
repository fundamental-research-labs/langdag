package provider

import (
	"context"
	"fmt"
	"log"
	"math/rand"
	"sort"
	"strings"

	"langdag.com/langdag/internal/models"
	"langdag.com/langdag/types"
)

// DeploymentChoice is one weighted deployment option within a routing stage.
type DeploymentChoice struct {
	DeploymentID string `json:"deployment_id" mapstructure:"deployment_id"`
	Weight       int    `json:"weight" mapstructure:"weight"`
}

// RoutingStage contains deployment choices that are retried before falling
// through to the next explicit stage.
type RoutingStage struct {
	Deployments []DeploymentChoice `json:"deployments" mapstructure:"deployments"`
	Retries     int                `json:"retries,omitempty" mapstructure:"retries"`
}

// RoutingPolicy selects stages by exact canonical model, model owner provider,
// then default. Model/provider overrides are scoped and authoritative: a
// matching route that has no eligible deployment does not cascade, while
// unrelated models keep using automatic eligible deployment resolution unless
// an explicit default route is configured.
type RoutingPolicy struct {
	Default   []RoutingStage            `json:"default,omitempty" mapstructure:"default"`
	Providers map[string][]RoutingStage `json:"providers,omitempty" mapstructure:"providers"`
	Models    map[string][]RoutingStage `json:"models,omitempty" mapstructure:"models"`
}

// DeploymentAdapter binds a known deployment ID to the adapter that can execute
// requests against that deployment.
type DeploymentAdapter struct {
	DeploymentID   string
	Provider       Provider
	ModelMappings  map[string]string
	DefaultRetries int
}

// DeploymentRouterOptions configures a deployment-aware Provider.
type DeploymentRouterOptions struct {
	Catalog     *models.CompiledCatalogV1
	Deployments map[string]DeploymentAdapter
	Routing     RoutingPolicy
}

// DeploymentRouter resolves canonical model IDs to deployment offerings, copies
// requests to native model IDs, and routes across explicit deployment stages.
type DeploymentRouter struct {
	catalog       *models.CompiledCatalogV1
	deployments   map[string]DeploymentAdapter
	routing       RoutingPolicy
	defaultStages []RoutingStage
}

type modelTarget struct {
	CanonicalModelID      string
	NativeModelIDHint     string
	SyntheticDeploymentID string
}

// NewDeploymentRouter creates a catalog-backed deployment router.
func NewDeploymentRouter(opts DeploymentRouterOptions) (*DeploymentRouter, error) {
	if opts.Catalog == nil {
		return nil, fmt.Errorf("deployment router: catalog is required")
	}
	if len(opts.Deployments) == 0 {
		return nil, fmt.Errorf("deployment router: at least one deployment must be configured")
	}
	deployments := make(map[string]DeploymentAdapter, len(opts.Deployments))
	for id, adapter := range opts.Deployments {
		if adapter.DeploymentID == "" {
			adapter.DeploymentID = id
		}
		if adapter.DeploymentID != id {
			return nil, fmt.Errorf("deployment router: deployment map key %q does not match adapter deployment %q", id, adapter.DeploymentID)
		}
		if adapter.Provider == nil {
			return nil, fmt.Errorf("deployment router: deployment %q has nil provider", id)
		}
		if opts.Catalog.DeploymentsByID[id] == nil {
			return nil, fmt.Errorf("deployment router: deployment %q is not in catalog", id)
		}
		adapter.ModelMappings = cloneStringMap(adapter.ModelMappings)
		deployments[id] = adapter
	}
	defaultStages := defaultStagesForDeployments(deployments)
	if opts.Routing.Default != nil {
		defaultStages = opts.Routing.Default
	}
	return &DeploymentRouter{
		catalog:       opts.Catalog,
		deployments:   deployments,
		routing:       cloneRoutingPolicy(opts.Routing),
		defaultStages: cloneRoutingStages(defaultStages),
	}, nil
}

// Name returns a route-level diagnostic name. Served deployments are recorded
// per response in CompletionResponse.Provider and ModelResolutionMetadata.
func (r *DeploymentRouter) Name() string {
	return "deployment-router"
}

// Models returns canonical model rows that at least one configured deployment
// can serve.
func (r *DeploymentRouter) Models() []types.ModelInfo {
	seen := map[string]bool{}
	var out []types.ModelInfo
	for _, deploymentID := range sortedDeploymentIDs(r.deployments) {
		for _, offering := range r.catalog.OfferingsByDeployment[deploymentID] {
			if !seen[offering.CanonicalModelID] {
				out = append(out, modelInfoFromOffering(offering))
				seen[offering.CanonicalModelID] = true
			}
		}
		for _, template := range r.catalog.OfferingTemplatesByDeployment[deploymentID] {
			adapter := r.deployments[deploymentID]
			if adapter.ModelMappings[template.CanonicalModelID] == "" || seen[template.CanonicalModelID] {
				continue
			}
			offering, err := template.Materialize(adapter.ModelMappings[template.CanonicalModelID])
			if err != nil {
				continue
			}
			out = append(out, modelInfoFromOffering(&offering))
			seen[template.CanonicalModelID] = true
		}
		for _, offering := range r.discoveredOfferings(deploymentID, "", "") {
			if !seen[offering.CanonicalModelID] {
				out = append(out, modelInfoFromOffering(offering))
				seen[offering.CanonicalModelID] = true
			}
		}
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	return out
}

// Complete resolves the request model to a routeable offering, calls the
// selected deployment adapter with its native model ID, and attaches served
// identity/pricing metadata to the response.
func (r *DeploymentRouter) Complete(ctx context.Context, req *types.CompletionRequest) (*types.CompletionResponse, error) {
	target, err := r.resolveModelTarget(req.Model)
	if err != nil {
		return nil, err
	}
	stages := r.routeFor(target.CanonicalModelID)
	var lastErr error
	for stageIndex, stage := range stages {
		candidates := r.eligibleChoices(target, stage, req)
		if len(candidates) == 0 {
			lastErr = fmt.Errorf("deployment router: stage %d has no eligible deployments for %q", stageIndex, target.CanonicalModelID)
			continue
		}
		attempts := stage.Retries + 1
		if attempts < 1 {
			attempts = 1
		}
		for attempt := 0; attempt < attempts; attempt++ {
			choice := selectDeploymentChoice(candidates)
			resp, err := r.completeWithDeployment(ctx, req, target, choice.DeploymentID)
			if err == nil {
				return resp, nil
			}
			lastErr = err
			log.Printf("deployment router: deployment %q failed for %q: %v", choice.DeploymentID, target.CanonicalModelID, err)
		}
	}
	if lastErr == nil {
		lastErr = fmt.Errorf("deployment router: no route configured for %q", target.CanonicalModelID)
	}
	return nil, fmt.Errorf("deployment router: all deployments failed for %q: %w", target.CanonicalModelID, lastErr)
}

// Stream resolves and routes a streaming request. Fallback is attempted only
// until output has been emitted; after a delta/content block, stream errors are
// surfaced without switching deployments.
func (r *DeploymentRouter) Stream(ctx context.Context, req *types.CompletionRequest) (<-chan types.StreamEvent, error) {
	target, err := r.resolveModelTarget(req.Model)
	if err != nil {
		return nil, err
	}
	stages := r.routeFor(target.CanonicalModelID)
	out := make(chan types.StreamEvent, 100)
	go func() {
		defer close(out)
		var lastErr error
		for stageIndex, stage := range stages {
			candidates := r.eligibleChoices(target, stage, req)
			if len(candidates) == 0 {
				lastErr = fmt.Errorf("deployment router: stage %d has no eligible deployments for %q", stageIndex, target.CanonicalModelID)
				continue
			}
			attempts := stage.Retries + 1
			if attempts < 1 {
				attempts = 1
			}
			for attempt := 0; attempt < attempts; attempt++ {
				choice := selectDeploymentChoice(candidates)
				shouldFallback, err := r.streamWithDeployment(ctx, out, req, target, choice.DeploymentID)
				if err == nil {
					return
				}
				lastErr = err
				if !shouldFallback {
					out <- types.StreamEvent{Type: types.StreamEventError, Error: err}
					return
				}
				log.Printf("deployment router: deployment %q stream failed before output for %q: %v", choice.DeploymentID, target.CanonicalModelID, err)
			}
		}
		if lastErr == nil {
			lastErr = fmt.Errorf("deployment router: no route configured for %q", target.CanonicalModelID)
		}
		out <- types.StreamEvent{
			Type:  types.StreamEventError,
			Error: fmt.Errorf("deployment router: all deployments failed for %q: %w", target.CanonicalModelID, lastErr),
		}
	}()
	return out, nil
}

func (r *DeploymentRouter) completeWithDeployment(ctx context.Context, req *types.CompletionRequest, target modelTarget, deploymentID string) (*types.CompletionResponse, error) {
	offering, adapter, err := r.resolveOffering(target, deploymentID)
	if err != nil {
		return nil, err
	}
	nativeReq := requestForOffering(req, offering, adapter)
	resp, err := adapter.Provider.Complete(ctx, nativeReq)
	if err != nil {
		return nil, err
	}
	enrichResponseFromOffering(resp, offering)
	enrichResponseProtocolFromRequest(resp, nativeReq)
	return resp, nil
}

func (r *DeploymentRouter) streamWithDeployment(ctx context.Context, out chan<- types.StreamEvent, req *types.CompletionRequest, target modelTarget, deploymentID string) (bool, error) {
	offering, adapter, err := r.resolveOffering(target, deploymentID)
	if err != nil {
		return true, err
	}
	nativeReq := requestForOffering(req, offering, adapter)
	ch, err := adapter.Provider.Stream(ctx, nativeReq)
	if err != nil {
		return true, err
	}
	emittedOutput := false
	sawDone := false
	var buffered []types.StreamEvent
	flushBuffered := func() {
		for _, bufferedEvent := range buffered {
			out <- bufferedEvent
		}
		buffered = nil
	}
	for event := range ch {
		switch event.Type {
		case types.StreamEventDelta:
			if event.Content != "" {
				emittedOutput = true
				flushBuffered()
			}
		case types.StreamEventContentDone:
			emittedOutput = true
			flushBuffered()
		case types.StreamEventDone:
			sawDone = true
			enrichResponseFromOffering(event.Response, offering)
			enrichResponseProtocolFromRequest(event.Response, nativeReq)
			flushBuffered()
		case types.StreamEventError:
			if emittedOutput {
				return false, event.Error
			}
			if event.Error == nil {
				return true, fmt.Errorf("deployment router: stream from %q returned an error event", deploymentID)
			}
			return true, event.Error
		}
		if emittedOutput || sawDone {
			out <- event
		} else {
			buffered = append(buffered, event)
		}
		if sawDone {
			return false, nil
		}
	}
	if sawDone {
		return false, nil
	}
	if emittedOutput {
		return false, fmt.Errorf("deployment router: stream from %q ended after output without completion", deploymentID)
	}
	return true, fmt.Errorf("deployment router: stream from %q ended without completion", deploymentID)
}

func (r *DeploymentRouter) resolveModelTarget(requested string) (modelTarget, error) {
	if requested == "" {
		return modelTarget{}, fmt.Errorf("deployment router: model is required")
	}
	if canonical, ok := r.catalog.CanonicalModelForAliasOrID(requested); ok {
		return modelTarget{CanonicalModelID: canonical}, nil
	}
	var matches []string
	for _, offering := range r.catalog.Catalog.Offerings {
		if offering.NativeModelID == requested || offering.ID == requested {
			matches = append(matches, offering.CanonicalModelID)
		}
	}
	for _, adapter := range r.deployments {
		for canonical, native := range adapter.ModelMappings {
			if native == requested {
				matches = append(matches, canonical)
			}
		}
	}
	matches = uniqueStrings(matches)
	if len(matches) == 1 {
		return modelTarget{CanonicalModelID: matches[0]}, nil
	}
	if len(matches) > 1 {
		sort.Strings(matches)
		return modelTarget{}, fmt.Errorf("deployment router: model %q is ambiguous across canonical models: %s", requested, strings.Join(matches, ", "))
	}
	discoveredTargets := r.discoveredTargetsForNativeID(requested)
	if len(discoveredTargets) == 1 {
		return discoveredTargets[0], nil
	}
	if len(discoveredTargets) > 1 {
		var canonicalIDs []string
		for _, target := range discoveredTargets {
			canonicalIDs = append(canonicalIDs, target.CanonicalModelID)
		}
		sort.Strings(canonicalIDs)
		return modelTarget{}, fmt.Errorf("deployment router: model %q is ambiguous across discovered deployments: %s", requested, strings.Join(canonicalIDs, ", "))
	}
	if len(r.deployments) == 1 {
		for deploymentID := range r.deployments {
			deployment := r.catalog.DeploymentsByID[deploymentID]
			if deployment == nil {
				continue
			}
			nativeID := syntheticNativeModelID(deployment, requested)
			return modelTarget{
				CanonicalModelID:      syntheticCanonicalModelID(deployment, nativeID),
				NativeModelIDHint:     nativeID,
				SyntheticDeploymentID: deploymentID,
			}, nil
		}
	}
	if strings.Contains(requested, "/") {
		return modelTarget{CanonicalModelID: requested, NativeModelIDHint: requested}, nil
	}
	return modelTarget{}, fmt.Errorf("deployment router: model %q is not in the catalog", requested)
}

func (r *DeploymentRouter) discoveredTargetsForNativeID(requested string) []modelTarget {
	var targets []modelTarget
	for deploymentID, adapter := range r.deployments {
		deployment := r.catalog.DeploymentsByID[deploymentID]
		if deployment == nil || deployment.NativeModelIDSource != models.NativeModelIDDiscovered {
			continue
		}
		for _, model := range adapter.Provider.Models() {
			if model.ID != requested {
				continue
			}
			targets = append(targets, modelTarget{
				CanonicalModelID:      syntheticCanonicalModelID(deployment, requested),
				NativeModelIDHint:     requested,
				SyntheticDeploymentID: deploymentID,
			})
			break
		}
	}
	return targets
}

func (r *DeploymentRouter) routeFor(canonicalModelID string) []RoutingStage {
	if stages, ok := r.routing.Models[canonicalModelID]; ok {
		return cloneRoutingStages(stages)
	}
	providerID := r.providerIDForCanonical(canonicalModelID)
	if providerID != "" {
		if stages, ok := r.routing.Providers[providerID]; ok {
			return cloneRoutingStages(stages)
		}
		for key, stages := range r.routing.Providers {
			if canonicalProviderID(key) == providerID {
				return cloneRoutingStages(stages)
			}
		}
	}
	return cloneRoutingStages(r.defaultStages)
}

func (r *DeploymentRouter) providerIDForCanonical(canonicalModelID string) string {
	if model := r.catalog.ModelsByID[canonicalModelID]; model != nil {
		return model.ProviderID
	}
	owner, _, ok := strings.Cut(canonicalModelID, "/")
	if ok {
		return canonicalProviderID(owner)
	}
	return ""
}

func (r *DeploymentRouter) eligibleChoices(target modelTarget, stage RoutingStage, req *types.CompletionRequest) []DeploymentChoice {
	var choices []DeploymentChoice
	var capable []DeploymentChoice
	requiredServerTools := requestedServerTools(req)
	for _, choice := range stage.Deployments {
		if choice.DeploymentID == "" || choice.Weight <= 0 {
			continue
		}
		if _, ok := r.deployments[choice.DeploymentID]; !ok {
			continue
		}
		offering, adapter, err := r.resolveOffering(target, choice.DeploymentID)
		if err != nil {
			continue
		}
		choices = append(choices, choice)
		if len(requiredServerTools) == 0 || offeringSupportsServerTools(offering, adapter, requiredServerTools) {
			capable = append(capable, choice)
		}
	}
	if len(requiredServerTools) > 0 && len(capable) > 0 {
		return capable
	}
	return choices
}

func (r *DeploymentRouter) resolveOffering(target modelTarget, deploymentID string) (*models.ModelOfferingV1, DeploymentAdapter, error) {
	adapter, ok := r.deployments[deploymentID]
	if !ok {
		return nil, DeploymentAdapter{}, fmt.Errorf("deployment %q is not configured", deploymentID)
	}
	if offering, ok := r.catalog.OfferingForDeployment(target.CanonicalModelID, deploymentID); ok {
		return offering, adapter, nil
	}
	for _, template := range r.catalog.OfferingTemplatesByCanonicalModel[target.CanonicalModelID] {
		if template.DeploymentID != deploymentID {
			continue
		}
		nativeID := adapter.ModelMappings[target.CanonicalModelID]
		if nativeID == "" {
			return nil, DeploymentAdapter{}, fmt.Errorf("deployment %q requires model_mappings[%q]", deploymentID, target.CanonicalModelID)
		}
		offering, err := template.Materialize(nativeID)
		if err != nil {
			return nil, DeploymentAdapter{}, err
		}
		return &offering, adapter, nil
	}
	for _, offering := range r.discoveredOfferings(deploymentID, target.CanonicalModelID, target.NativeModelIDHint) {
		if offering.CanonicalModelID == target.CanonicalModelID {
			return offering, adapter, nil
		}
	}
	if target.SyntheticDeploymentID == deploymentID && target.NativeModelIDHint != "" {
		deployment := r.catalog.DeploymentsByID[deploymentID]
		if deployment != nil {
			return r.syntheticOffering(deployment, target.CanonicalModelID, target.NativeModelIDHint), adapter, nil
		}
	}
	if target.SyntheticDeploymentID == "" && target.NativeModelIDHint != "" {
		deployment := r.catalog.DeploymentsByID[deploymentID]
		if r.deploymentCanServeUnknownCanonical(deployment, target.CanonicalModelID) {
			return r.syntheticOffering(deployment, target.CanonicalModelID, syntheticNativeModelID(deployment, target.CanonicalModelID)), adapter, nil
		}
	}
	return nil, DeploymentAdapter{}, fmt.Errorf("deployment %q cannot serve %q", deploymentID, target.CanonicalModelID)
}

func (r *DeploymentRouter) deploymentCanServeUnknownCanonical(deployment *models.DeploymentV1, canonicalModelID string) bool {
	if deployment == nil || deployment.NativeModelIDSource != models.NativeModelIDCatalogKnown {
		return false
	}
	owner, _, ok := strings.Cut(canonicalModelID, "/")
	if !ok || owner == "" {
		return false
	}
	return canonicalProviderID(owner) == deployment.ProviderID
}

func (r *DeploymentRouter) discoveredOfferings(deploymentID, onlyCanonical, nativeHint string) []*models.ModelOfferingV1 {
	adapter, ok := r.deployments[deploymentID]
	if !ok {
		return nil
	}
	deployment := r.catalog.DeploymentsByID[deploymentID]
	if deployment == nil || deployment.NativeModelIDSource != models.NativeModelIDDiscovered {
		return nil
	}
	switch deploymentID {
	case "openrouter":
		if onlyCanonical != "" && strings.Contains(onlyCanonical, "/") {
			nativeID := nativeHint
			if nativeID == "" || strings.HasPrefix(onlyCanonical, "openrouter/") {
				nativeID = openRouterNativeModelID(onlyCanonical)
			}
			return []*models.ModelOfferingV1{r.syntheticOffering(deployment, onlyCanonical, nativeID)}
		}
		var out []*models.ModelOfferingV1
		for _, model := range adapter.Provider.Models() {
			if model.ID == "" {
				continue
			}
			canonicalID := syntheticCanonicalModelID(deployment, model.ID)
			out = append(out, r.syntheticOfferingFromModelInfo(deployment, canonicalID, model.ID, model))
		}
		return out
	case "ollama-local":
		var out []*models.ModelOfferingV1
		for _, model := range adapter.Provider.Models() {
			canonicalID := syntheticCanonicalModelID(deployment, model.ID)
			if onlyCanonical != "" && canonicalID != onlyCanonical && model.ID != onlyCanonical {
				continue
			}
			out = append(out, r.syntheticOfferingFromModelInfo(deployment, canonicalID, model.ID, model))
		}
		return out
	case "apple-local":
		var out []*models.ModelOfferingV1
		for _, model := range adapter.Provider.Models() {
			canonicalID := syntheticCanonicalModelID(deployment, model.ID)
			if onlyCanonical != "" && canonicalID != onlyCanonical && model.ID != onlyCanonical {
				continue
			}
			out = append(out, r.syntheticOfferingFromModelInfo(deployment, canonicalID, model.ID, model))
		}
		return out
	}
	return nil
}

func (r *DeploymentRouter) syntheticOffering(deployment *models.DeploymentV1, canonicalID, nativeID string) *models.ModelOfferingV1 {
	ownerProviderID, _, ok := strings.Cut(canonicalID, "/")
	if !ok || ownerProviderID == "" {
		ownerProviderID = deployment.ProviderID
	}
	offering := &models.ModelOfferingV1{
		ID:               deployment.ID + ":" + nativeID,
		CanonicalModelID: canonicalID,
		DeploymentID:     deployment.ID,
		NativeModelID:    nativeID,
		Capabilities: models.CapabilitySetV1{
			ServerTools: map[string]models.CapabilityState{types.ServerToolWebSearch: models.CapabilityUnknown},
		},
		Pricing:     models.PricingV1{Status: models.PricingUnknown, Currency: "USD", Source: "discovered"},
		Deployment:  deployment,
		APIProtocol: deployment.APIProtocol,
	}
	if deployment.ID == "ollama-local" || deployment.ID == "apple-local" {
		offering.Pricing = models.PricingV1{
			Status:     models.PricingFree,
			Currency:   "USD",
			RatesPer1M: map[string]float64{"input_tokens": 0, "output_tokens": 0},
			Source:     "local",
		}
	}
	if model := r.catalog.ModelsByID[canonicalID]; model != nil {
		offering.Model = model
		offering.Provider = model.Provider
	} else if provider := r.catalog.ProvidersByID[ownerProviderID]; provider != nil {
		offering.Provider = provider
	}
	return offering
}

func (r *DeploymentRouter) syntheticOfferingFromModelInfo(deployment *models.DeploymentV1, canonicalID, nativeID string, info types.ModelInfo) *models.ModelOfferingV1 {
	offering := r.syntheticOffering(deployment, canonicalID, nativeID)
	if info.Name != "" || info.ContextWindow != 0 || info.MaxOutput != 0 {
		ownerProviderID, _, _ := strings.Cut(canonicalID, "/")
		offering.Model = &models.ModelV1{
			ID:            canonicalID,
			ProviderID:    ownerProviderID,
			Name:          info.Name,
			ContextWindow: info.ContextWindow,
			MaxOutput:     info.MaxOutput,
			Provider:      offering.Provider,
		}
		if offering.Model.Name == "" {
			offering.Model.Name = nativeID
		}
	}
	for _, tool := range info.ServerTools {
		if tool != "" {
			offering.Capabilities.ServerTools[tool] = models.CapabilitySupported
		}
	}
	if info.SupportsFunctionCalling {
		offering.Capabilities.FunctionCalling = models.CapabilitySupported
	}
	if info.SupportsExplicitThinkingBudget {
		offering.Capabilities.ExplicitThinkingBudget = models.CapabilitySupported
	}
	return offering
}

func syntheticCanonicalModelID(deployment *models.DeploymentV1, nativeID string) string {
	if deployment == nil {
		return nativeID
	}
	if deployment.ID == "openrouter" {
		if strings.Contains(nativeID, "/") {
			return nativeID
		}
		return "openrouter/" + nativeID
	}
	return deployment.ProviderID + "/" + nativeID
}

func syntheticNativeModelID(deployment *models.DeploymentV1, requested string) string {
	if deployment == nil {
		return requested
	}
	if deployment.ID == "openrouter" {
		if nativeID, ok := strings.CutPrefix(requested, "openrouter/"); ok {
			return nativeID
		}
		return requested
	}
	if nativeID, ok := strings.CutPrefix(requested, deployment.ProviderID+"/"); ok {
		return nativeID
	}
	return requested
}

func openRouterNativeModelID(canonicalID string) string {
	if nativeID, ok := strings.CutPrefix(canonicalID, "openrouter/"); ok {
		return nativeID
	}
	return canonicalID
}

func requestForOffering(req *types.CompletionRequest, offering *models.ModelOfferingV1, adapter DeploymentAdapter) *types.CompletionRequest {
	copied := *req
	copied.Model = offering.NativeModelID
	if copied.APIProtocolID == "" {
		copied.APIProtocolID = apiProtocolIDForOffering(offering)
	}
	return filterToolsForOffering(&copied, offering, adapter)
}

func apiProtocolIDForOffering(offering *models.ModelOfferingV1) string {
	if offering == nil {
		return ""
	}
	if offering.APIProtocolID != "" {
		return offering.APIProtocolID
	}
	if offering.DeploymentID == "openai-direct" {
		return "openai-responses"
	}
	if offering.APIProtocol != nil {
		return offering.APIProtocol.ID
	}
	if offering.Deployment != nil {
		return offering.Deployment.APIProtocolID
	}
	return ""
}

func filterToolsForOffering(req *types.CompletionRequest, offering *models.ModelOfferingV1, adapter DeploymentAdapter) *types.CompletionRequest {
	if len(req.Tools) == 0 {
		return req
	}
	supported := map[string]bool{}
	for name, state := range offering.Capabilities.ServerTools {
		if state == models.CapabilitySupported {
			supported[name] = true
		}
	}
	if offering.DeploymentID == "ollama-local" && len(supported) == 0 {
		for _, model := range adapter.Provider.Models() {
			if model.ID != offering.NativeModelID {
				continue
			}
			for _, tool := range model.ServerTools {
				supported[tool] = true
			}
		}
	}
	needsFilter := false
	for _, tool := range req.Tools {
		if !tool.IsClientTool() && !supported[tool.Name] {
			needsFilter = true
			break
		}
	}
	if !needsFilter {
		return req
	}
	filtered := *req
	filtered.Tools = make([]types.ToolDefinition, 0, len(req.Tools))
	for _, tool := range req.Tools {
		if tool.IsClientTool() || supported[tool.Name] {
			filtered.Tools = append(filtered.Tools, tool)
		}
	}
	return &filtered
}

func requestedServerTools(req *types.CompletionRequest) []string {
	if req == nil || len(req.Tools) == 0 {
		return nil
	}
	seen := map[string]bool{}
	var out []string
	for _, tool := range req.Tools {
		if tool.IsClientTool() || tool.Name == "" || seen[tool.Name] {
			continue
		}
		seen[tool.Name] = true
		out = append(out, tool.Name)
	}
	sort.Strings(out)
	return out
}

func offeringSupportsServerTools(offering *models.ModelOfferingV1, adapter DeploymentAdapter, required []string) bool {
	if len(required) == 0 {
		return true
	}
	supported := map[string]bool{}
	for name, state := range offering.Capabilities.ServerTools {
		if state == models.CapabilitySupported {
			supported[name] = true
		}
	}
	if offering.DeploymentID == "ollama-local" && len(supported) == 0 {
		for _, model := range adapter.Provider.Models() {
			if model.ID != offering.NativeModelID {
				continue
			}
			for _, tool := range model.ServerTools {
				supported[tool] = true
			}
		}
	}
	for _, name := range required {
		if !supported[name] {
			return false
		}
	}
	return true
}

func enrichResponseFromOffering(resp *types.CompletionResponse, offering *models.ModelOfferingV1) {
	if resp == nil || offering == nil {
		return
	}
	if resp.Model == "" {
		resp.Model = offering.NativeModelID
	}
	resp.Provider = offering.DeploymentID
	if resp.ModelResolution == nil {
		resolution := models.ModelResolutionMetadataFromOffering(offering)
		resp.ModelResolution = &resolution
	}
	if resp.PricingSnapshot == nil {
		snapshot := models.PricingSnapshotFromPricingV1(offering.Pricing)
		resp.PricingSnapshot = &snapshot
	}
	resp.EnsureNormalizedUsage()
}

func enrichResponseProtocolFromRequest(resp *types.CompletionResponse, req *types.CompletionRequest) {
	if resp == nil || req == nil || req.APIProtocolID == "" || resp.ModelResolution == nil {
		return
	}
	resp.ModelResolution.APIProtocolID = req.APIProtocolID
}

func modelInfoFromOffering(offering *models.ModelOfferingV1) types.ModelInfo {
	info := types.ModelInfo{ID: offering.CanonicalModelID, Name: offering.CanonicalModelID}
	if offering.Model != nil {
		info.Name = offering.Model.Name
		info.ContextWindow = offering.Model.ContextWindow
		info.MaxOutput = offering.Model.MaxOutput
	}
	for name, state := range offering.Capabilities.ServerTools {
		if state == models.CapabilitySupported {
			info.ServerTools = append(info.ServerTools, name)
		}
	}
	sort.Strings(info.ServerTools)
	info.SupportsFunctionCalling = offering.Capabilities.FunctionCalling == models.CapabilitySupported
	info.SupportsExplicitThinkingBudget = offering.Capabilities.ExplicitThinkingBudget == models.CapabilitySupported
	return info
}

func selectDeploymentChoice(choices []DeploymentChoice) DeploymentChoice {
	if len(choices) == 1 {
		return choices[0]
	}
	total := 0
	for _, choice := range choices {
		total += choice.Weight
	}
	if total <= 0 {
		return choices[0]
	}
	n := rand.Intn(total)
	for _, choice := range choices {
		n -= choice.Weight
		if n < 0 {
			return choice
		}
	}
	return choices[len(choices)-1]
}

func defaultStagesForDeployments(deployments map[string]DeploymentAdapter) []RoutingStage {
	var choices []DeploymentChoice
	maxRetries := 0
	for _, deploymentID := range sortedDeploymentIDs(deployments) {
		adapter := deployments[deploymentID]
		retries := adapter.DefaultRetries
		if retries < 0 {
			retries = 0
		}
		if retries > maxRetries {
			maxRetries = retries
		}
		choices = append(choices, DeploymentChoice{DeploymentID: deploymentID, Weight: 100})
		if len(deployments) == 1 {
			return []RoutingStage{{Deployments: choices, Retries: retries}}
		}
	}
	return []RoutingStage{{Deployments: choices, Retries: maxRetries}}
}

func sortedDeploymentIDs(deployments map[string]DeploymentAdapter) []string {
	ids := make([]string, 0, len(deployments))
	for id := range deployments {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	return ids
}

func uniqueStrings(values []string) []string {
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

func canonicalProviderID(providerID string) string {
	switch providerID {
	case "gemini":
		return "google"
	case "grok":
		return "xai"
	default:
		return providerID
	}
}

func cloneRoutingPolicy(policy RoutingPolicy) RoutingPolicy {
	return RoutingPolicy{
		Default:   cloneRoutingStages(policy.Default),
		Providers: cloneRoutingStageMap(policy.Providers),
		Models:    cloneRoutingStageMap(policy.Models),
	}
}

func cloneRoutingStageMap(in map[string][]RoutingStage) map[string][]RoutingStage {
	if len(in) == 0 {
		return nil
	}
	out := make(map[string][]RoutingStage, len(in))
	for key, stages := range in {
		out[key] = cloneRoutingStages(stages)
	}
	return out
}

func cloneRoutingStages(stages []RoutingStage) []RoutingStage {
	if len(stages) == 0 {
		return nil
	}
	out := make([]RoutingStage, len(stages))
	for i, stage := range stages {
		out[i].Retries = stage.Retries
		out[i].Deployments = append([]DeploymentChoice(nil), stage.Deployments...)
	}
	return out
}

func cloneStringMap(in map[string]string) map[string]string {
	if len(in) == 0 {
		return nil
	}
	out := make(map[string]string, len(in))
	for key, value := range in {
		out[key] = value
	}
	return out
}
