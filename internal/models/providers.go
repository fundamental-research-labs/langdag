package models

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"
)

// Source URLs for official provider data (variables for testability).
var (
	openAISourceURL       = "https://developers.openai.com/api/docs/models/all"
	openAIModelDetailsURL = "https://developers.openai.com/api/docs/models"
	anthropicSourceURL    = "https://platform.claude.com/docs/en/docs/about-claude/models"
	anthropicModelsAPIURL = "https://api.anthropic.com/v1/models"
	geminiSourceURL       = "https://ai.google.dev/pricing"
	geminiSpecBaseURL     = "https://ai.google.dev/gemini-api/docs/models"
	geminiDeprecationsURL = "https://ai.google.dev/gemini-api/docs/deprecations"
	grokSourceURL         = "https://docs.x.ai/docs/models"

	providerNow = func() time.Time { return time.Now().UTC() }
)

// --- Shared helpers ---

const providerUserAgent = "langdag-model-catalog/1.0"

func providerHTTPGet(ctx context.Context, url string) (string, error) {
	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return "", err
	}
	req.Header.Set("User-Agent", providerUserAgent)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("HTTP %d from %s", resp.StatusCode, url)
	}
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}
	return string(body), nil
}

var (
	htmlTagRe    = regexp.MustCompile(`<[^>]*>`)
	whitespaceRe = regexp.MustCompile(`\s+`)
	tableRe      = regexp.MustCompile(`(?s)<table[^>]*>(.*?)</table>`)
	rowRe        = regexp.MustCompile(`(?s)<tr[^>]*>(.*?)</tr>`)
	cellRe       = regexp.MustCompile(`(?s)<t[dh][^>]*>(.*?)</t[dh]>`)
	tokenCountRe = regexp.MustCompile(`([\d.]+)\s*([KkMm])\s*tokens`)
	sizeRe       = regexp.MustCompile(`([\d.]+)\s*([KkMm])`)
	dollarRe     = regexp.MustCompile(`\$\s*([\d.]+)`)
)

func stripHTMLTags(s string) string {
	s = htmlTagRe.ReplaceAllString(s, " ")
	s = strings.ReplaceAll(s, "&amp;", "&")
	s = strings.ReplaceAll(s, "&lt;", "<")
	s = strings.ReplaceAll(s, "&gt;", ">")
	s = strings.ReplaceAll(s, "&quot;", `"`)
	s = strings.ReplaceAll(s, "&#39;", "'")
	s = strings.ReplaceAll(s, "&nbsp;", " ")
	s = strings.ReplaceAll(s, `\$`, "$")
	s = whitespaceRe.ReplaceAllString(s, " ")
	return strings.TrimSpace(s)
}

// parseHTMLTables extracts tables from HTML as [table][row][cell] strings.
func parseHTMLTables(html string) [][][]string {
	var tables [][][]string
	for _, tm := range tableRe.FindAllStringSubmatch(html, -1) {
		var rows [][]string
		for _, rm := range rowRe.FindAllStringSubmatch(tm[1], -1) {
			var cells []string
			for _, cm := range cellRe.FindAllStringSubmatch(rm[1], -1) {
				cells = append(cells, stripHTMLTags(cm[1]))
			}
			if len(cells) > 0 {
				rows = append(rows, cells)
			}
		}
		if len(rows) > 0 {
			tables = append(tables, rows)
		}
	}
	return tables
}

func parseTokenCount(s string) int {
	m := tokenCountRe.FindStringSubmatch(s)
	if m == nil {
		return 0
	}
	return parseSizeValue(m[1], m[2])
}

func parseSizeStr(s string) int {
	m := sizeRe.FindStringSubmatch(s)
	if m == nil {
		return 0
	}
	return parseSizeValue(m[1], m[2])
}

func parseSizeValue(numStr, unit string) int {
	val, _ := strconv.ParseFloat(numStr, 64)
	switch strings.ToUpper(unit) {
	case "K":
		return int(val * 1000)
	case "M":
		return int(val * 1000000)
	}
	return int(val)
}

func parseDollarAmount(s string) float64 {
	m := dollarRe.FindStringSubmatch(s)
	if m == nil {
		return 0
	}
	v, _ := strconv.ParseFloat(m[1], 64)
	return v
}

// --- OpenAI ---

func fetchOpenAIModels(ctx context.Context) ([]ModelPricing, error) {
	body, err := providerHTTPGet(ctx, openAISourceURL)
	if err != nil {
		return nil, err
	}
	modelURLs := parseOpenAIModelLinks(body, openAISourceURL)
	if len(modelURLs) == 0 {
		return parseOpenAIText(body)
	}
	return fetchOpenAIModelDetails(ctx, modelURLs)
}

var (
	openAIPricingRowRe = regexp.MustCompile(`(?m)^\|\s*([a-z0-9][\w.-]*)\s*\|\s*([\d.]+)\s*\|[^|]*\|\s*([\d.]+)\s*\|`)
	openAISnapshotRe   = regexp.MustCompile(`(?m)^###\s+([\w.-]+)`)
	openAICtxWindowRe  = regexp.MustCompile(`(?m)^-\s*Context window size:\s*(\d+)`)
	openAIMaxOutputRe  = regexp.MustCompile(`(?m)^-\s*Maximum output tokens:\s*(\d+)`)
	openAIModelLinkRe  = regexp.MustCompile(`(?i)href\s*=\s*["']([^"']*/api/docs/models/([a-z0-9][a-z0-9._-]*))/?(?:#[^"']*)?["']`)
	openAIContextRe    = regexp.MustCompile(`(?i)([\d,]+)\s+context window`)
	openAIMaxOutputRe2 = regexp.MustCompile(`(?i)([\d,]+)\s+max output tokens`)
	openAITextPriceRe  = regexp.MustCompile(`(?i)Text tokens\s+.*?\bInput\s+\$([\d.]+)(?:\s+Cached input\s+(?:\$[\d.]+|-))?\s+Output\s+\$([\d.]+)`)
	openAIModelIDRe    = regexp.MustCompile(`\b(?:gpt|o[0-9]|computer-use|chatgpt|chat-latest|codex|babbage|davinci|omni|text)[a-z0-9._-]*\b`)
)

func parseOpenAIModelLinks(html, sourceURL string) []string {
	seen := make(map[string]bool)
	var urls []string
	for _, m := range openAIModelLinkRe.FindAllStringSubmatch(html, -1) {
		slug := strings.Trim(strings.ToLower(m[2]), "/")
		if !isOpenAIModelSlug(slug) {
			continue
		}
		detailURL := resolveOpenAIModelURL(sourceURL, m[1])
		if detailURL == "" {
			detailURL = strings.TrimRight(openAIModelDetailsURL, "/") + "/" + slug
		}
		if seen[detailURL] {
			continue
		}
		seen[detailURL] = true
		urls = append(urls, detailURL)
	}
	return urls
}

func isOpenAIModelSlug(slug string) bool {
	switch slug {
	case "", "all", "compare", "model-archive":
		return false
	}
	if strings.HasPrefix(slug, "model-") {
		return false
	}
	return strings.HasPrefix(slug, "gpt-") ||
		strings.HasPrefix(slug, "o1") ||
		strings.HasPrefix(slug, "o3") ||
		strings.HasPrefix(slug, "o4") ||
		strings.HasPrefix(slug, "computer-use") ||
		strings.HasPrefix(slug, "chatgpt") ||
		strings.HasPrefix(slug, "chat-latest") ||
		strings.HasPrefix(slug, "codex") ||
		strings.HasPrefix(slug, "babbage") ||
		strings.HasPrefix(slug, "davinci")
}

func resolveOpenAIModelURL(sourceURL, href string) string {
	base, err := url.Parse(sourceURL)
	if err != nil {
		return ""
	}
	ref, err := url.Parse(href)
	if err != nil {
		return ""
	}
	return base.ResolveReference(ref).String()
}

func fetchOpenAIModelDetails(ctx context.Context, modelURLs []string) ([]ModelPricing, error) {
	type result struct {
		models []ModelPricing
		err    error
	}

	results := make([]result, len(modelURLs))
	var wg sync.WaitGroup
	sem := make(chan struct{}, 6)
	for i, modelURL := range modelURLs {
		wg.Add(1)
		go func(i int, modelURL string) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()

			body, err := providerHTTPGet(ctx, modelURL)
			if err != nil {
				results[i].err = err
				return
			}
			modelID := openAIModelIDFromURL(modelURL)
			results[i].models = parseOpenAIModelDetail(modelID, body)
		}(i, modelURL)
	}
	wg.Wait()

	models := make(map[string]ModelPricing)
	var errs []string
	for i, r := range results {
		if r.err != nil {
			errs = append(errs, fmt.Sprintf("%s: %v", modelURLs[i], r.err))
			continue
		}
		for _, m := range r.models {
			models[m.ID] = m
		}
	}
	if len(errs) > 0 {
		return nil, fmt.Errorf("OpenAI model detail fetch failed: %s", strings.Join(errs, "; "))
	}
	allModels := make([]ModelPricing, 0, len(models))
	for _, m := range models {
		allModels = append(allModels, m)
	}
	if len(allModels) == 0 {
		return nil, fmt.Errorf("no models found in OpenAI model docs")
	}
	return allModels, nil
}

func openAIModelIDFromURL(modelURL string) string {
	u, err := url.Parse(modelURL)
	if err != nil {
		return ""
	}
	parts := strings.Split(strings.Trim(u.Path, "/"), "/")
	if len(parts) == 0 {
		return ""
	}
	return strings.ToLower(parts[len(parts)-1])
}

func parseOpenAIModelDetail(modelID, html string) []ModelPricing {
	modelID = strings.ToLower(strings.TrimSpace(modelID))
	if modelID == "" {
		return nil
	}
	text := stripHTMLTags(html)
	ctxWindow := parseOpenAINumberAfter(openAIContextRe, text)
	maxOutput := parseOpenAINumberAfter(openAIMaxOutputRe2, text)
	inputPrice, outputPrice := parseOpenAITextPrices(text)
	if ctxWindow == 0 || maxOutput == 0 || (inputPrice == 0 && outputPrice == 0) {
		return nil
	}

	ids := openAIModelIDsFromDetail(modelID, text)
	models := make([]ModelPricing, 0, len(ids))
	for _, id := range ids {
		models = append(models, ModelPricing{
			ID:               id,
			InputPricePer1M:  inputPrice,
			OutputPricePer1M: outputPrice,
			ContextWindow:    ctxWindow,
			MaxOutput:        maxOutput,
		})
	}
	return models
}

func parseOpenAINumberAfter(re *regexp.Regexp, text string) int {
	m := re.FindStringSubmatch(text)
	if m == nil {
		return 0
	}
	n, _ := strconv.Atoi(strings.ReplaceAll(m[1], ",", ""))
	return n
}

func parseOpenAITextPrices(text string) (float64, float64) {
	m := openAITextPriceRe.FindStringSubmatch(text)
	if m == nil {
		return 0, 0
	}
	input, _ := strconv.ParseFloat(m[1], 64)
	output, _ := strconv.ParseFloat(m[2], 64)
	return roundPrice(input), roundPrice(output)
}

func openAIModelIDsFromDetail(modelID, text string) []string {
	ids := []string{modelID}
	seen := map[string]bool{modelID: true}

	section := text
	if idx := strings.Index(section, "Snapshots"); idx >= 0 {
		section = section[idx:]
		if end := strings.Index(section, "Rate limits"); end >= 0 {
			section = section[:end]
		}
		for _, m := range openAIModelIDRe.FindAllString(strings.ToLower(section), -1) {
			if m != modelID && !strings.HasPrefix(m, modelID+"-") {
				continue
			}
			if seen[m] {
				continue
			}
			seen[m] = true
			ids = append(ids, m)
		}
	}
	return ids
}

func parseOpenAIText(text string) ([]ModelPricing, error) {
	models := make(map[string]*ModelPricing)

	// Parse pricing table (## Text tokens section)
	if idx := strings.Index(text, "## Text tokens"); idx >= 0 {
		section := text[idx:]
		if end := strings.Index(section[1:], "\n## "); end >= 0 {
			section = section[:end+1]
		}
		for _, line := range strings.Split(section, "\n") {
			if strings.Contains(line, "(batch)") {
				continue
			}
			m := openAIPricingRowRe.FindStringSubmatch(line)
			if m == nil {
				continue
			}
			name := strings.TrimSpace(m[1])
			input, _ := strconv.ParseFloat(m[2], 64)
			output, _ := strconv.ParseFloat(m[3], 64)
			models[name] = &ModelPricing{
				ID:               name,
				InputPricePer1M:  roundPrice(input),
				OutputPricePer1M: roundPrice(output),
			}
		}
	}

	// Parse model metadata sections for context window and max output
	parts := openAISnapshotRe.Split(text, -1)
	names := openAISnapshotRe.FindAllStringSubmatch(text, -1)
	for i, name := range names {
		section := parts[i+1]
		modelID := name[1]

		var ctxWindow, maxOutput int
		if m := openAICtxWindowRe.FindStringSubmatch(section); m != nil {
			ctxWindow, _ = strconv.Atoi(m[1])
		}
		if m := openAIMaxOutputRe.FindStringSubmatch(section); m != nil {
			maxOutput, _ = strconv.Atoi(m[1])
		}
		if ctxWindow == 0 && maxOutput == 0 {
			continue
		}
		if existing, ok := models[modelID]; ok {
			existing.ContextWindow = ctxWindow
			existing.MaxOutput = maxOutput
		} else {
			models[modelID] = &ModelPricing{
				ID:            modelID,
				ContextWindow: ctxWindow,
				MaxOutput:     maxOutput,
			}
		}
	}

	result := make([]ModelPricing, 0, len(models))
	for _, m := range models {
		result = append(result, *m)
	}
	if len(result) == 0 {
		return nil, fmt.Errorf("no models found in OpenAI text")
	}
	return result, nil
}

// --- Anthropic ---

func fetchAnthropicModels(ctx context.Context) ([]ModelPricing, error) {
	fetchHTMLOnly := func() ([]ModelPricing, error) {
		body, err := providerHTTPGet(ctx, anthropicSourceURL)
		if err != nil {
			return nil, err
		}
		models, err := parseAnthropicHTML(body)
		if err != nil {
			return nil, err
		}
		return mergeAnthropicAPIAndHTML(nil, models), nil
	}

	apiKey := strings.TrimSpace(os.Getenv("ANTHROPIC_API_KEY"))
	if apiKey == "" {
		if requireAnthropicModelsAPI() {
			return nil, fmt.Errorf("ANTHROPIC_API_KEY is required for Anthropic Models API catalog refresh")
		}
		return fetchHTMLOnly()
	}

	apiModels, err := fetchAnthropicModelsAPI(ctx, apiKey)
	if err != nil {
		if requireAnthropicModelsAPI() {
			return nil, err
		}
		return fetchHTMLOnly()
	}

	body, err := providerHTTPGet(ctx, anthropicSourceURL)
	if err != nil {
		return mergeAnthropicAPIAndHTML(apiModels, nil), nil
	}
	htmlModels, err := parseAnthropicHTML(body)
	if err != nil {
		return mergeAnthropicAPIAndHTML(apiModels, nil), nil
	}
	return mergeAnthropicAPIAndHTML(apiModels, htmlModels), nil
}

var anthropicPriceRe = regexp.MustCompile(`(?s)\$\s*([\d.]+)\s*/\s*input.*?\$\s*([\d.]+)\s*/\s*output`)

type anthropicModelsAPIResponse struct {
	Data    []anthropicModelsAPIModel `json:"data"`
	HasMore bool                      `json:"has_more"`
	LastID  string                    `json:"last_id"`
}

type anthropicModelsAPIModel struct {
	ID             string          `json:"id"`
	MaxInputTokens int             `json:"max_input_tokens"`
	MaxTokens      int             `json:"max_tokens"`
	Capabilities   json.RawMessage `json:"capabilities"`
	// TODO: Persist capabilities when the catalog schema has a place for them.
}

func fetchAnthropicModelsAPI(ctx context.Context, apiKey string) ([]ModelPricing, error) {
	var models []ModelPricing
	afterID := ""

	for {
		pageURL, err := anthropicModelsAPIPageURL(afterID)
		if err != nil {
			return nil, err
		}

		req, err := http.NewRequestWithContext(ctx, "GET", pageURL, nil)
		if err != nil {
			return nil, err
		}
		req.Header.Set("User-Agent", providerUserAgent)
		req.Header.Set("x-api-key", apiKey)
		req.Header.Set("anthropic-version", "2023-06-01")
		req.Header.Set("Accept", "application/json")

		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			return nil, err
		}
		body, readErr := io.ReadAll(resp.Body)
		closeErr := resp.Body.Close()
		if readErr != nil {
			return nil, readErr
		}
		if closeErr != nil {
			return nil, closeErr
		}
		if resp.StatusCode != http.StatusOK {
			return nil, fmt.Errorf("HTTP %d from %s", resp.StatusCode, pageURL)
		}

		var page anthropicModelsAPIResponse
		if err := json.Unmarshal(body, &page); err != nil {
			return nil, err
		}
		for _, apiModel := range page.Data {
			id := strings.TrimSpace(apiModel.ID)
			if id == "" {
				continue
			}
			models = append(models, ModelPricing{
				ID:                  id,
				ContextWindow:       apiModel.MaxInputTokens,
				MaxOutput:           apiModel.MaxTokens,
				AllowUnknownPricing: true,
			})
		}

		if !page.HasMore {
			break
		}
		lastID := strings.TrimSpace(page.LastID)
		if lastID == "" {
			return nil, fmt.Errorf("Anthropic Models API pagination missing last_id")
		}
		if lastID == afterID {
			return nil, fmt.Errorf("Anthropic Models API pagination repeated last_id %q", lastID)
		}
		afterID = lastID
	}

	if len(models) == 0 {
		return nil, fmt.Errorf("no models found in Anthropic Models API")
	}
	return models, nil
}

func anthropicModelsAPIPageURL(afterID string) (string, error) {
	u, err := url.Parse(anthropicModelsAPIURL)
	if err != nil {
		return "", err
	}
	q := u.Query()
	q.Set("limit", "1000")
	if afterID != "" {
		q.Set("after_id", afterID)
	} else {
		q.Del("after_id")
	}
	u.RawQuery = q.Encode()
	return u.String(), nil
}

func requireAnthropicModelsAPI() bool {
	return os.Getenv("LANGDAG_REQUIRE_ANTHROPIC_MODELS_API") == "1"
}

func mergeAnthropicAPIAndHTML(apiModels, htmlModels []ModelPricing) []ModelPricing {
	htmlByID := make(map[string]ModelPricing, len(htmlModels))
	for _, model := range htmlModels {
		htmlByID[model.ID] = model
	}

	seen := make(map[string]bool, len(apiModels)+len(htmlModels))
	merged := make([]ModelPricing, 0, len(apiModels)+len(htmlModels))
	for _, model := range apiModels {
		if model.ID == "" || seen[model.ID] {
			continue
		}
		if htmlModel, ok := htmlByID[model.ID]; ok {
			if anthropicHasPricing(htmlModel) {
				model.InputPricePer1M = htmlModel.InputPricePer1M
				model.OutputPricePer1M = htmlModel.OutputPricePer1M
			}
			if model.ContextWindow == 0 {
				model.ContextWindow = htmlModel.ContextWindow
			}
			if model.MaxOutput == 0 {
				model.MaxOutput = htmlModel.MaxOutput
			}
		}
		applyAnthropicDirectPriceOverride(&model)
		merged = append(merged, model)
		seen[model.ID] = true
	}

	for _, model := range htmlModels {
		if model.ID == "" || seen[model.ID] {
			continue
		}
		applyAnthropicDirectPriceOverride(&model)
		merged = append(merged, model)
		seen[model.ID] = true
	}
	return merged
}

func anthropicHasPricing(model ModelPricing) bool {
	return model.InputPricePer1M > 0 || model.OutputPricePer1M > 0
}

func applyAnthropicDirectPriceOverride(model *ModelPricing) {
	if model.InputPricePer1M > 0 && model.OutputPricePer1M > 0 {
		return
	}
	input, output, ok := anthropicDirectPriceOverride(model.ID)
	if !ok {
		return
	}
	model.InputPricePer1M = input
	model.OutputPricePer1M = output
}

func anthropicDirectPriceOverride(modelID string) (float64, float64, bool) {
	switch {
	case anthropicModelIDMatchesFamily(modelID, "claude-fable-5"),
		anthropicModelIDMatchesFamily(modelID, "claude-mythos-5"):
		return 10, 50, true
	case anthropicModelIDMatchesFamily(modelID, "claude-opus-4-8"),
		anthropicModelIDMatchesFamily(modelID, "claude-opus-4-7"),
		anthropicModelIDMatchesFamily(modelID, "claude-opus-4-6"),
		anthropicModelIDMatchesFamily(modelID, "claude-opus-4-5"):
		return 5, 25, true
	case anthropicModelIDMatchesFamily(modelID, "claude-sonnet-4-6"),
		anthropicModelIDMatchesFamily(modelID, "claude-sonnet-4-5"):
		return 3, 15, true
	case anthropicModelIDMatchesFamily(modelID, "claude-haiku-4-5"):
		return 1, 5, true
	default:
		return 0, 0, false
	}
}

func anthropicModelIDMatchesFamily(modelID, family string) bool {
	return modelID == family || strings.HasPrefix(modelID, family+"-")
}

func parseAnthropicHTML(html string) ([]ModelPricing, error) {
	tables := parseHTMLTables(html)

	var all []ModelPricing
	for _, table := range tables {
		all = append(all, parseAnthropicTable(table)...)
	}
	if len(all) == 0 {
		return nil, fmt.Errorf("no models found in Anthropic HTML")
	}
	return all, nil
}

// parseAnthropicTable handles the transposed table format where models are
// columns and attributes (API ID, Pricing, Context window, Max output) are rows.
func parseAnthropicTable(rows [][]string) []ModelPricing {
	if len(rows) < 2 {
		return nil
	}

	var modelIDs []string
	var aliases []string
	inputPrices := make(map[int]float64)
	outputPrices := make(map[int]float64)
	contextWindows := make(map[int]int)
	maxOutputs := make(map[int]int)

	for _, row := range rows {
		if len(row) < 2 {
			continue
		}
		label := strings.ToLower(row[0])

		switch {
		case strings.Contains(label, "api id") && !strings.Contains(label, "alias"):
			for _, cell := range row[1:] {
				modelIDs = append(modelIDs, strings.TrimSpace(cell))
			}

		case strings.Contains(label, "api") && strings.Contains(label, "alias"):
			for _, cell := range row[1:] {
				aliases = append(aliases, strings.TrimSpace(cell))
			}

		case strings.Contains(label, "pricing") || (strings.Contains(label, "price") && strings.Contains(label, "input")):
			for j, cell := range row[1:] {
				if m := anthropicPriceRe.FindStringSubmatch(cell); m != nil {
					inputPrices[j], _ = strconv.ParseFloat(m[1], 64)
					outputPrices[j], _ = strconv.ParseFloat(m[2], 64)
				}
			}

		case strings.Contains(label, "context window"):
			for j, cell := range row[1:] {
				if v := parseTokenCount(cell); v > 0 {
					contextWindows[j] = v
				}
			}

		case strings.Contains(label, "max output"):
			for j, cell := range row[1:] {
				if v := parseTokenCount(cell); v > 0 {
					maxOutputs[j] = v
				}
			}
		}
	}

	if len(modelIDs) == 0 {
		return nil
	}

	isDash := func(s string) bool {
		return s == "" || s == "—" || s == "-" || s == "–"
	}

	buildModel := func(id string, idx int) ModelPricing {
		return ModelPricing{
			ID:               id,
			InputPricePer1M:  inputPrices[idx],
			OutputPricePer1M: outputPrices[idx],
			ContextWindow:    contextWindows[idx],
			MaxOutput:        maxOutputs[idx],
		}
	}

	var models []ModelPricing
	for i, id := range modelIDs {
		if isDash(id) {
			continue
		}
		models = append(models, buildModel(id, i))

		// Also add alias if different from ID
		if i < len(aliases) && !isDash(aliases[i]) && aliases[i] != id {
			models = append(models, buildModel(aliases[i], i))
		}
	}
	return models
}

// --- Gemini ---

func fetchGeminiModels(ctx context.Context) ([]ModelPricing, error) {
	// Step 1: Get pricing from pricing page
	body, err := providerHTTPGet(ctx, geminiSourceURL)
	if err != nil {
		return nil, err
	}
	models, err := parseGeminiHTML(body)
	if err != nil {
		return nil, err
	}

	// Step 2: Fetch spec pages and deprecations page concurrently
	var wg sync.WaitGroup
	for i := range models {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			specURL := geminiSpecBaseURL + "/" + models[i].ID
			specBody, err := providerHTTPGet(ctx, specURL)
			if err != nil {
				return // spec page not available; leave fields as zero
			}
			ctxWin, maxOut := parseGeminiSpecPage(specBody)
			if ctxWin > 0 {
				models[i].ContextWindow = ctxWin
			}
			if maxOut > 0 {
				models[i].MaxOutput = maxOut
			}
		}(i)
	}

	var shutdownModels map[string]time.Time
	wg.Add(1)
	go func() {
		defer wg.Done()
		depBody, err := providerHTTPGet(ctx, geminiDeprecationsURL)
		if err != nil {
			return // deprecations page not available; skip filtering
		}
		shutdownModels = parseGeminiDeprecations(depBody)
	}()

	wg.Wait()

	// Step 3: Filter out models once their shutdown date has arrived.
	if len(shutdownModels) > 0 {
		models = filterGeminiModelsByShutdown(models, shutdownModels, providerNow())
	}

	return models, nil
}

var geminiModelRe = regexp.MustCompile(`(?i)Gemini\s+([\d.]+)\s+([\w-]+)`)

// geminiSkipVariants are model variant names that are not text LLMs.
var geminiSkipVariants = map[string]bool{
	"nano": true, "embeddings": true, "embedding": true,
	"computer": true, "deep": true, "robotics": true,
}

func parseGeminiHTML(html string) ([]ModelPricing, error) {
	text := stripHTMLTags(html)

	indices := geminiModelRe.FindAllStringIndex(text, -1)
	names := geminiModelRe.FindAllStringSubmatch(text, -1)
	if len(indices) == 0 {
		return nil, fmt.Errorf("no models found in Gemini HTML")
	}

	seen := make(map[string]bool)
	var models []ModelPricing

	for i, name := range names {
		version := name[1]
		variant := strings.ToLower(name[2])

		if geminiSkipVariants[variant] {
			continue
		}

		// Check text after the match for non-LLM suffixes
		afterMatch := ""
		if indices[i][1] < len(text) {
			end := indices[i][1] + 40
			if end > len(text) {
				end = len(text)
			}
			afterMatch = strings.TrimSpace(text[indices[i][1]:end])
		}
		if strings.HasPrefix(afterMatch, "Image") || strings.HasPrefix(afterMatch, "TTS") {
			continue
		}

		modelID := "gemini-" + version + "-" + variant
		// Append "-preview" if the heading includes "Preview"
		afterLower := strings.ToLower(afterMatch)
		if strings.HasPrefix(afterLower, "preview") || strings.HasPrefix(afterLower, "- preview") {
			modelID += "-preview"
		}

		if seen[modelID] {
			continue
		}

		// Section: text from end of this match to start of next
		start := indices[i][1]
		end := len(text)
		if i+1 < len(indices) {
			end = indices[i+1][0]
		}
		section := text[start:end]

		inputPrice := findFirstPrice(section, "input")
		outputPrice := findFirstPrice(section, "output")
		if inputPrice <= 0 && outputPrice <= 0 {
			continue
		}

		seen[modelID] = true
		models = append(models, ModelPricing{
			ID:               modelID,
			InputPricePer1M:  roundPrice(inputPrice),
			OutputPricePer1M: roundPrice(outputPrice),
		})
	}

	if len(models) == 0 {
		return nil, fmt.Errorf("no pricing found in Gemini HTML")
	}
	return models, nil
}

var (
	geminiInputLimitRe  = regexp.MustCompile(`(?i)Input\s+token\s+limit\D+([\d,]+)`)
	geminiOutputLimitRe = regexp.MustCompile(`(?i)Output\s+token\s+limit\D+([\d,]+)`)
)

// parseGeminiSpecPage extracts token limits from a Gemini model documentation page.
func parseGeminiSpecPage(html string) (contextWindow int, maxOutput int) {
	text := stripHTMLTags(html)
	if m := geminiInputLimitRe.FindStringSubmatch(text); m != nil {
		contextWindow = parseCommaNumber(m[1])
	}
	if m := geminiOutputLimitRe.FindStringSubmatch(text); m != nil {
		maxOutput = parseCommaNumber(m[1])
	}
	return
}

// geminiDeprecationRowRe matches table rows in the Gemini deprecations page.
// It captures the model ID (column 1) and shutdown date text (column 3).
var geminiDeprecationRowRe = regexp.MustCompile(
	`(?i)<tr[^>]*>\s*<td[^>]*>(.*?)</td>\s*<td[^>]*>.*?</td>\s*<td[^>]*>(.*?)</td>`,
)

// parseGeminiDeprecations extracts announced shutdown dates from the Gemini
// deprecations page. Models remain usable until their shutdown date arrives.
func parseGeminiDeprecations(html string) map[string]time.Time {
	shutdownModels := make(map[string]time.Time)
	for _, match := range geminiDeprecationRowRe.FindAllStringSubmatch(html, -1) {
		modelID := strings.TrimSpace(stripHTMLTags(match[1]))
		shutdownDate := strings.TrimSpace(stripHTMLTags(match[2]))
		if modelID == "" || shutdownDate == "" {
			continue
		}
		// Keep models that have no shutdown date announced.
		if strings.Contains(strings.ToLower(shutdownDate), "no shutdown date") {
			continue
		}
		if parsed, ok := parseGeminiShutdownDate(shutdownDate); ok {
			shutdownModels[modelID] = parsed
		}
	}
	return shutdownModels
}

func parseGeminiShutdownDate(value string) (time.Time, bool) {
	value = strings.TrimSpace(value)
	for _, layout := range []string{"January 2, 2006", "Jan 2, 2006", "2006-01-02"} {
		if parsed, err := time.Parse(layout, value); err == nil {
			return parsed, true
		}
	}
	return time.Time{}, false
}

func filterGeminiModelsByShutdown(models []ModelPricing, shutdownModels map[string]time.Time, now time.Time) []ModelPricing {
	if len(shutdownModels) == 0 {
		return models
	}
	filtered := models[:0]
	for _, m := range models {
		shutdown, hasShutdown := shutdownModels[m.ID]
		if !hasShutdown || shutdown.After(now) {
			filtered = append(filtered, m)
		}
	}
	return filtered
}

func parseCommaNumber(s string) int {
	s = strings.ReplaceAll(s, ",", "")
	v, _ := strconv.Atoi(s)
	return v
}

// findFirstPrice finds the first dollar amount near a keyword in a text section.
func findFirstPrice(section, keyword string) float64 {
	lower := strings.ToLower(section)
	idx := strings.Index(lower, keyword)
	if idx < 0 {
		return 0
	}
	end := idx + 100
	if end > len(section) {
		end = len(section)
	}
	m := dollarRe.FindStringSubmatch(section[idx:end])
	if m == nil {
		return 0
	}
	v, _ := strconv.ParseFloat(m[1], 64)
	return v
}

// --- Grok ---

func fetchGrokModels(ctx context.Context) ([]ModelPricing, error) {
	body, err := providerHTTPGet(ctx, grokSourceURL)
	if err != nil {
		return nil, err
	}
	return parseGrokPage(body)
}

// Grok model data is embedded in the page as escaped JSON within the RSC payload.
// Each model entry has fields like:
//
//	"name":"grok-3","promptTextTokenPrice":"$n30000","completionTextTokenPrice":"$n150000","maxPromptLength":131072
//
// Prices use "$nXXXXX" format where the integer / 10000 = dollars per 1M tokens.
// Models appear once per cluster (us-east-1, eu-west-1, etc.) so we deduplicate.
var (
	grokModelRe       = regexp.MustCompile(`"name":"(grok-[^"]+)"`)
	grokInputPriceRe  = regexp.MustCompile(`"promptTextTokenPrice":"\$n(\d+)"`)
	grokOutputPriceRe = regexp.MustCompile(`"completionTextTokenPrice":"\$n(\d+)"`)
	grokContextRe     = regexp.MustCompile(`"maxPromptLength":(\d+)`)
)

func parseGrokPage(html string) ([]ModelPricing, error) {
	// Unescape the RSC payload: \" → "
	text := strings.ReplaceAll(html, `\"`, `"`)

	// Find all model entries with their surrounding context
	nameMatches := grokModelRe.FindAllStringSubmatchIndex(text, -1)
	if len(nameMatches) == 0 {
		return nil, fmt.Errorf("no models found in Grok page")
	}

	seen := make(map[string]bool)
	var models []ModelPricing

	for _, loc := range nameMatches {
		name := text[loc[2]:loc[3]]

		// Skip image/video models
		if strings.Contains(name, "imagine") || strings.Contains(name, "video") {
			continue
		}
		if seen[name] {
			continue
		}

		// Extract data from the region after the name match (fields follow in sequence)
		start := loc[1]
		end := start + 500
		if end > len(text) {
			end = len(text)
		}
		region := text[start:end]

		var inputPrice, outputPrice float64
		var contextWindow int

		if m := grokInputPriceRe.FindStringSubmatch(region); m != nil {
			v, _ := strconv.ParseFloat(m[1], 64)
			inputPrice = roundPrice(v / 10000)
		}
		if m := grokOutputPriceRe.FindStringSubmatch(region); m != nil {
			v, _ := strconv.ParseFloat(m[1], 64)
			outputPrice = roundPrice(v / 10000)
		}
		if m := grokContextRe.FindStringSubmatch(region); m != nil {
			contextWindow, _ = strconv.Atoi(m[1])
		}

		if inputPrice > 0 || outputPrice > 0 || contextWindow > 0 {
			seen[name] = true
			models = append(models, ModelPricing{
				ID:               name,
				InputPricePer1M:  inputPrice,
				OutputPricePer1M: outputPrice,
				ContextWindow:    contextWindow,
			})
		}
	}

	if len(models) == 0 {
		return nil, fmt.Errorf("no models found in Grok page")
	}
	return models, nil
}

// gemmaHardcodedModels returns Gemma model metadata for the free Google AI Studio API.
func gemmaHardcodedModels() []ModelPricing {
	return []ModelPricing{
		{ID: "gemma-3-1b-it", Free: true, ContextWindow: 32768, MaxOutput: 8192},
		{ID: "gemma-3-4b-it", Free: true, ContextWindow: 131072, MaxOutput: 8192},
		{ID: "gemma-3-12b-it", Free: true, ContextWindow: 131072, MaxOutput: 8192},
		{ID: "gemma-3-27b-it", Free: true, ContextWindow: 131072, MaxOutput: 8192},
		{ID: "gemma-4-31b-it", Free: true, ContextWindow: 262144, MaxOutput: 8192},
		{ID: "gemma-4-26b-a4b-it", Free: true, ContextWindow: 262144, MaxOutput: 8192},
	}
}

// fetchGeminiAndGemmaModels fetches Gemini models from the pricing page and
// appends the hardcoded Gemma models (same Google AI Studio endpoint).
func fetchGeminiAndGemmaModels(ctx context.Context) ([]ModelPricing, error) {
	geminiModels, err := fetchGeminiModels(ctx)
	if err != nil {
		return nil, err
	}
	return append(geminiModels, gemmaHardcodedModels()...), nil
}
