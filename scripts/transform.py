import requests
import json
from collections import defaultdict
from datetime import datetime

LITELLM_JSON_URL = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
OUTPUT_PATH = "dist/models.json"

# Supported Modes (Primary Functions):
# chat
# embedding
# completion
# audio_transcription
# audio_speech
# image_generation
# moderation
# rerank

CAPABILITY_MAP = {
    "supports_function_calling": "tool_calling",
    "supports_parallel_function_calling": "parallel_tools",
    "supports_vision": "vision",
    "supports_audio_input": "audio_transcription",
    "supports_audio_output": "audio_speech",
    "supports_prompt_caching": "caching",
    "supports_response_schema": "json_mode",
    "supports_system_messages": "system_messages",
    "supports_reasoning": "reasoning",
    "supports_web_search": "web_search"
}

def transform_data():
    """
    Fetches the flat model list from LiteLLM and transforms it into a structure
    grouped by provider, adding a status for deprecated models.
    """
    print("Fetching latest model data from LiteLLM...")
    response = requests.get(LITELLM_JSON_URL)
    response.raise_for_status()
    data = response.json()

    provider_map = defaultdict(list)
    today = datetime.now().date()
    
    print("Transforming data...")
    for model_name, model_info in data.items():
        if model_name == "sample_spec":
            continue

        provider = model_info.get("litellm_provider")
        
        deprecation_date_str = model_info.get("deprecation_date")
        status = "active"
        if deprecation_date_str:
            try:
                deprecation_date = datetime.strptime(deprecation_date_str, "%Y-%m-%d").date()
                if deprecation_date < today:
                    status = "deprecated"
            except ValueError:
                pass
            
        capabilities = []
        
        for key, tag in CAPABILITY_MAP.items():
            if model_info.get(key) is True:
                capabilities.append(tag)

        if provider:
            final_model_data = {
                "model_name": model_name,
                "provider": provider,
                "status": status,
                "capabilities": sorted(capabilities),
                
                "mode": model_info.get("mode"),
                "max_tokens": model_info.get("max_tokens"),
                "max_input_tokens": model_info.get("max_input_tokens"),
                "max_output_tokens": model_info.get("max_output_tokens"),
                "input_cost_per_token": model_info.get("input_cost_per_token"),
                "output_cost_per_token": model_info.get("output_cost_per_token"),
                "output_cost_per_reasoning_token": model_info.get("output_cost_per_reasoning_token"),
                "file_search_cost_per_1k_calls": model_info.get("file_search_cost_per_1k_calls"),
                "file_search_cost_per_gb_per_day": model_info.get("file_search_cost_per_gb_per_day"),
                "vector_store_cost_per_gb_per_day": model_info.get("vector_store_cost_per_gb_per_day"),
                "computer_use_input_cost_per_1k_tokens": model_info.get("computer_use_input_cost_per_1k_tokens"),
                "computer_use_output_cost_per_1k_tokens": model_info.get("computer_use_output_cost_per_1k_tokens"),
                "code_interpreter_cost_per_session": model_info.get("code_interpreter_cost_per_session"),
                "supported_regions": model_info.get("supported_regions"),
                "deprecation_date": model_info.get("deprecation_date"),
                
                "advanced_costs": {}
            }
            
            if model_info.get("search_context_cost_per_query"):
                final_model_data["advanced_costs"]["search_context_cost_per_query"] = model_info.get("search_context_cost_per_query")

            provider_map[provider].append(final_model_data)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(provider_map, f, indent=2)
    
    print(f"Successfully transformed data and saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    transform_data()
