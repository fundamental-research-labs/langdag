package models

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"sort"
)

func strictJSONUnmarshal(data []byte, dst any) error {
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	if err := dec.Decode(dst); err != nil {
		return err
	}
	var extra any
	if err := dec.Decode(&extra); err != io.EOF {
		if err == nil {
			return fmt.Errorf("unexpected trailing JSON")
		}
		return err
	}
	return nil
}

func NormalizeCatalogV1(catalog *CatalogV1) {
	if catalog == nil {
		return
	}
	mergeReferenceRuntimePlaceholdersV1(catalog)
	normalizeKnownDeploymentProtocolsV1(catalog)
	for _, provider := range catalog.Providers {
		if provider != nil {
			sort.Strings(provider.Aliases)
		}
	}
	for _, model := range catalog.Models {
		if model != nil {
			sort.Strings(model.Aliases)
		}
	}
	sort.SliceStable(catalog.Offerings, func(i, j int) bool {
		if catalog.Offerings[i].CanonicalModelID != catalog.Offerings[j].CanonicalModelID {
			return catalog.Offerings[i].CanonicalModelID < catalog.Offerings[j].CanonicalModelID
		}
		return catalog.Offerings[i].ID < catalog.Offerings[j].ID
	})
	sort.SliceStable(catalog.OfferingTemplates, func(i, j int) bool {
		if catalog.OfferingTemplates[i].CanonicalModelID != catalog.OfferingTemplates[j].CanonicalModelID {
			return catalog.OfferingTemplates[i].CanonicalModelID < catalog.OfferingTemplates[j].CanonicalModelID
		}
		return catalog.OfferingTemplates[i].ID < catalog.OfferingTemplates[j].ID
	})
	for i := range catalog.Offerings {
		normalizePricingV1(&catalog.Offerings[i].Pricing)
	}
	for i := range catalog.OfferingTemplates {
		normalizePricingV1(&catalog.OfferingTemplates[i].Pricing)
	}
}

func normalizeKnownDeploymentProtocolsV1(catalog *CatalogV1) {
	if catalog.APIProtocols == nil {
		catalog.APIProtocols = map[string]*APIProtocolV1{}
	}
	if catalog.APIProtocols["openai-responses"] == nil {
		catalog.APIProtocols["openai-responses"] = &APIProtocolV1{ID: "openai-responses", Name: "OpenAI Responses"}
	}
	if catalog.APIProtocols["openai-chat-completions"] == nil {
		catalog.APIProtocols["openai-chat-completions"] = &APIProtocolV1{ID: "openai-chat-completions", Name: "OpenAI Chat Completions"}
	}
	deployment := catalog.Deployments["openai-direct"]
	if deployment == nil {
		return
	}
	deployment.APIProtocolID = "openai-responses"
	deployment.APIProtocolIDs = appendUniqueStringsV1(deployment.APIProtocolIDs, "openai-responses", "openai-chat-completions")
}

func appendUniqueStringsV1(values []string, additions ...string) []string {
	seen := map[string]bool{}
	out := make([]string, 0, len(values)+len(additions))
	for _, value := range values {
		if value == "" || seen[value] {
			continue
		}
		seen[value] = true
		out = append(out, value)
	}
	for _, value := range additions {
		if value == "" || seen[value] {
			continue
		}
		seen[value] = true
		out = append(out, value)
	}
	sort.Strings(out)
	return out
}

func normalizePricingV1(pricing *PricingV1) {
	if pricing == nil {
		return
	}
	sort.Strings(pricing.MissingDimensions)
	sort.Strings(pricing.Notes)
}
