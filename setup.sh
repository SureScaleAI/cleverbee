#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- UI & Colors ---
# Define ANSI color codes
RED=$'\033[0;31m'
GREEN=$'\033[0;32m'
YELLOW=$'\033[0;33m'
BLUE=$'\033[0;34m'
MAGENTA=$'\033[0;35m'
CYAN=$'\033[0;36m'
GRAY=$'\033[0;37m'
BOLD=$'\033[1m'
UNDERLINE=$'\033[4m'
RESET=$'\033[0m'

# Color support check
if [[ -t 1 ]]; then
    # Check if terminal supports colors
    ncolors=$(tput colors 2>/dev/null || echo 0)
    if [[ -n "$ncolors" && $ncolors -ge 8 ]]; then
        HAS_COLORS=true
    else
        HAS_COLORS=false
        # Reset color variables to empty strings if colors aren't supported
        RED='' GREEN='' YELLOW='' BLUE='' MAGENTA='' CYAN='' GRAY='' BOLD='' UNDERLINE='' RESET=''
    fi
else
    HAS_COLORS=false
    RED='' GREEN='' YELLOW='' BLUE='' MAGENTA='' CYAN='' GRAY='' BOLD='' UNDERLINE='' RESET=''
fi

# --- Configuration Variables ---
PYTHON_CMD="python3"
VENV_DIR="venv"
MCP_JSON_FILE="mcp.json"
MODELS_DIR="models"
HUGGINGFACE_CLI_CMD="huggingface-cli" # Use variable for potential path issues
CONFIG_YAML="config.yaml"

# --- Helper Function Definitions ---

print_section_header() {
    local title="$1"
    local width=70
    local title_len=${#title}
    local padding_total=$((width - title_len - 2))
    local padding_left=$((padding_total / 2))
    local padding_right=$((padding_total - padding_left))
    if [ $padding_left -lt 0 ]; then padding_left=0; fi
    if [ $padding_right -lt 0 ]; then padding_right=0; fi
    echo ""
    echo "${BLUE}┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓${RESET}"
    printf "${BLUE}┃%${padding_left}s${RESET}${BOLD}%s${RESET}%${padding_right}s${BLUE}┃${RESET}\\n" "" "$title" ""
    echo "${BLUE}┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛${RESET}"
    echo ""
}

print_info() { echo "${GREEN}[INFO]${RESET} $1"; }
print_warning() { echo "${YELLOW}[WARNING]${RESET} $1"; }
print_error() { echo "${RED}[ERROR]${RESET} $1" >&2; }

print_progress() {
    local percent=$1
    local width=50
    local completed=$((percent * width / 100))
    local remaining=$((width - completed))
    printf "${BLUE}[${GREEN}"
    printf "%0.s█" $(seq 1 $completed)
    printf "%0.s░" $(seq 1 $remaining)
    printf "${BLUE}] ${YELLOW}%d%%${RESET}\\r" $percent
}

print_model_choice() {
    local number=$1
    local name=$2
    local description=$3
    local recommended=$4
    if [ "$recommended" = "true" ]; then
        echo "${GREEN}${BOLD}[$number]${RESET} ${CYAN}${BOLD}$name${RESET} ${GREEN}(Recommended)${RESET}"
    else
        echo "${GREEN}${BOLD}[$number]${RESET} ${CYAN}${BOLD}$name${RESET}"
    fi
    echo "    ${GRAY}$description${RESET}"
}

print_summary_box() {
    local title="$1"
    shift
    local lines=("$@")
    local width=70
    echo "${YELLOW}┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓${RESET}"
    printf "${YELLOW}┃${RESET}${BOLD}%$(( (width - ${#title}) / 2 ))s%s%$(( (width - ${#title} + 1) / 2 ))s${RESET}${YELLOW}┃${RESET}\\n" "" "$title" ""
    echo "${YELLOW}┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫${RESET}"
    for line in "${lines[@]}"; do
        printf "${YELLOW}┃${RESET} %-$((width-2))s ${YELLOW}┃${RESET}\\n" "$line"
    done
    echo "${YELLOW}┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛${RESET}"
}

get_system_ram() {
    if [[ "$(uname)" == "Darwin" ]]; then
        local ram_bytes=$(sysctl -n hw.memsize)
        echo $((ram_bytes / 1024 / 1024 / 1024))
    elif [[ "$(uname)" == "Linux" ]]; then
        local ram_kb=$(grep MemTotal /proc/meminfo | awk '{print $2}')
        echo $((ram_kb / 1024 / 1024))
    else
        echo 8 # Default fallback
    fi
}

detect_apple_silicon_model() {
    if [[ "$(uname)" == "Darwin" ]]; then
        local model=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "")
        if [[ "$model" == *"Apple"* ]]; then
            local chip_info=""
            local grep_exit_status=1
            if system_profiler SPHardwareDataType 2>/dev/null | grep -q "Chip"; then
                grep_exit_status=0
                chip_info=$(system_profiler SPHardwareDataType 2>/dev/null | grep "Chip" | awk -F': ' '{print $2}')
            fi
            if [ $grep_exit_status -eq 0 ]; then echo "$chip_info"; else echo "Apple Silicon (unknown)"; fi
        else echo "Not Apple Silicon"; fi
    else echo "Not macOS"; fi
}

has_nvidia_gpu() {
    if command -v nvidia-smi &> /dev/null; then return 0; else return 1; fi
}

get_nvidia_vram() {
    if command -v nvidia-smi &> /dev/null; then
        local vram_mb=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1 | tr -d '[:space:]')
        echo $((vram_mb / 1024))
    else echo 0; fi
}

get_apple_unified_memory() {
    if [[ "$(uname)" == "Darwin" ]]; then
        if system_profiler SPHardwareDataType 2>/dev/null | grep -q "Memory"; then
            local memory=$(system_profiler SPHardwareDataType 2>/dev/null | grep "Memory" | awk -F': ' '{print $2}' | sed 's/ GB//')
            echo "$memory"
        else echo "$(get_system_ram)"; fi
    else echo 0; fi
}

# GGUF Model Download Helper
_huggingface_download_gguf() {
    local repo_id=$1
    local gguf_filename=$2
    local target_dir=$3
    local model_url=$4 # Optional URL for error messages
    local target_path="${target_dir}/${gguf_filename}"
    local marker_file="${target_dir}/.${gguf_filename}.download_complete"

    print_info "Attempting download: Repo=${repo_id}, File=${gguf_filename}, Target=${target_dir}"
    mkdir -p "$target_dir"

    if [ ! -f "$marker_file" ] && [ -f "$target_path" ]; then
        print_warning "Found potentially incomplete GGUF file: ${target_path}. Removing."
        rm -f "$target_path"
    fi

    if [ -f "$marker_file" ]; then
        print_info "GGUF file ${gguf_filename} already downloaded, skipping."
        return 0
    fi

    # Ensure VENV_DIR and HUGGINGFACE_CLI_CMD are accessible
    if $VENV_DIR/bin/$HUGGINGFACE_CLI_CMD download "$repo_id" "$gguf_filename" --local-dir "$target_dir" --local-dir-use-symlinks False; then
        if [ -f "$target_path" ]; then
            print_info "GGUF file downloaded successfully: ${target_path}"
            touch "$marker_file"
            return 0
        else
            print_error "huggingface-cli download succeeded but file not found: ${target_path}"
            rm -f "$target_path"
            return 1
        fi
    else
        local exit_code=$?
        print_error "huggingface-cli download failed (exit code ${exit_code}) for ${repo_id}/${gguf_filename}."
        if [ $exit_code -eq 1 ]; then
            print_warning "May need to accept terms or request access on Hugging Face."
            if [ -n "$model_url" ]; then
                 echo "Visit: ${model_url}"
                 echo "Log in, accept terms, ensure access, then re-run setup."
                 if [[ "$(uname)" == "Darwin" ]]; then open "${model_url}"; fi
            else echo "Check Hugging Face repository ${repo_id} for access requirements."; fi
            fi
        rm -f "$target_path"
        return 1
    fi
}

# Function to determine if local models should be offered (based only on NVIDIA)
should_offer_local_models() {
    if has_nvidia_gpu; then echo true; else echo false; fi
}

# Simplified function to ask user about model preference based on hardware
# Sets global _ask_model_preference_result
ask_model_preference() {
    echo ""
    echo "${YELLOW}${BOLD}Checking Hardware Compatibility...${RESET}"
    if ! has_nvidia_gpu; then
        print_warning "NVIDIA GPU not detected."
        if [[ "$(uname)" == "Darwin" ]]; then print_warning "Local models not recommended on macOS."; fi
        print_info "Only Cloud Models will be configured."
        _ask_model_preference_result=false # Set global var
        return
    fi
    
    print_info "NVIDIA GPU detected ✓"
    echo ""
    echo "${CYAN}Cloud Models (Claude/Gemini) are always available.${RESET}"
    echo "${GRAY}Compatible Local Models (70B+, requires >=24GB VRAM) can also be set up.${RESET}"
    echo ""
    
    # Check for quick install mode
    if [ "$QUICK_INSTALL" = true ]; then
        local vram_gb=$(get_nvidia_vram)
        if [ "$vram_gb" -ge 24 ]; then
            print_info "${GREEN}Auto-enabling setup for Local Models (${vram_gb}GB VRAM detected).${RESET}"
            _ask_model_preference_result=true # Set global var
        else
            print_info "${YELLOW}Auto-disabling setup for Local Models (insufficient VRAM: ${vram_gb}GB).${RESET}"
            _ask_model_preference_result=false # Set global var
        fi
        return
    fi
    
    # Interactive mode
    local user_choice
    read -p "${BLUE}Enable setup for Local Models? (Requires login, downloads) [y/${GREEN}N${BLUE}]: ${RESET}" user_choice
    if [[ "$user_choice" == "y" || "$user_choice" == "Y" ]]; then
        echo ""; print_info "${GREEN}Local Model setup enabled.${RESET}"; 
        _ask_model_preference_result=true # Set global var
    else
        echo ""; print_info "${YELLOW}Local Model setup disabled. Only Cloud Models configured.${RESET}"; 
        _ask_model_preference_result=false # Set global var
    fi
}

# Summarizer Model Selection (Simplified - Always Flash)
# Sets global _selected_model_result
select_summarizer_model() {
    # Assumes CONFIG_YAML accessible
    print_section_header "SUMMARIZER MODEL CONFIGURATION"
    local MODEL_GEMINI_FLASH="gemini-2.0-flash"  # Change to gemini-2.5-flash-preview-04-17 when ready
    
    # Always auto-select Gemini Flash regardless of quick/interactive mode
    print_info "Defaulting Summarizer Model to: ${CYAN}${MODEL_GEMINI_FLASH}${RESET} (Cloud)"
    print_info "Updating config.yaml..."
    local update_ok=true
    # Always use yq now
    yq -i ".SUMMARIZER_MODEL = \"$MODEL_GEMINI_FLASH\" | .CHUNK_SIZE = 0 | .USE_LOCAL_SUMMARIZER_MODEL = false" "$CONFIG_YAML" || update_ok=false
    
    if [ "$update_ok" = true ]; then
        print_info "Summarizer configured in config.yaml ✓"
    else
        print_error "Failed to update config.yaml for summarizer!"
    fi
    
    # Set a global-like variable instead of echoing
    _selected_model_result="$MODEL_GEMINI_FLASH" 
}

# Primary Model Selection (Simplified - No Associative Arrays)
# Sets global _selected_model_result
select_primary_model() {
    # Assumes USE_LOCAL_MODELS, CONFIG_YAML accessible
    print_section_header "PRIMARY REASONING MODEL SELECTION"

    # Define model names and descriptions in parallel arrays
    local model_names=()
    local model_descriptions=()
    local default_model_name=""

    # Add Cloud Models
    local model_claude="claude-3.7-sonnet-cloud"
    model_names+=("$model_claude")
    model_descriptions+=("Cloud: Claude 3.7 Sonnet - 200K context, strong reasoning (API Key)")
    
    local model_gemini="gemini-2.5-pro-cloud"
    model_names+=("$model_gemini")
    model_descriptions+=("Cloud: Gemini 2.5 Pro - 1M+ context, multimodal (API Key)")
    default_model_name="$model_gemini" # Default to Gemini Cloud

    # Define 70B+ Local Models from table
    local model_llama33_70b="Llama-3.3-70B-Instruct-Q4_K_M.gguf"
    local model_qwen_72b="Qwen2.5-72B-Instruct-Q4_K_M.gguf"
    local model_deepseek_70b="DeepSeek-R1-Distill-Llama-70B-IQ4_NL.gguf"
    local model_xwin_70b="xwin-lm-70b-v0.1.Q4_K_M.gguf"
    local local_models_added=false 

    # Offer 70B+ Local Models only if local models enabled AND sufficient VRAM (>= 24GB)
    if [ "$USE_LOCAL_MODELS" = true ]; then
        NVIDIA_VRAM=$(get_nvidia_vram)
        print_info "Checking VRAM for 70B+ Local Models (Detected: ${NVIDIA_VRAM}GB)"
        
        # Check for 70B+ models (>= 24GB VRAM)
        if [ "$NVIDIA_VRAM" -ge 24 ]; then
             print_info "Offering 70B+ Local Models (>=24GB VRAM required)."
             # Add the models from the table with updated descriptions
             model_names+=("$model_llama33_70b")
             model_descriptions+=("Local: Llama-3.3 70B (128k ctx) - Best all-round reasoning (~5 tok/s Q4)")
             model_names+=("$model_qwen_72b")
             model_descriptions+=("Local: Qwen2.5 72B (128k ctx) - Balanced, multilingual (~6 tok/s Q4)")
             model_names+=("$model_deepseek_70b")
             model_descriptions+=("Local: DeepSeek-R1 70B (128k ctx) - Science-grade CoT (IQ4, slow ~3 tok/s)")
             model_names+=("$model_xwin_70b")
             model_descriptions+=("Local: Xwin-LM 70B v0.1 (8k ctx) - Alignment champ, instruction following")
             
             local_models_added=true
             # For 70B+ models, set Llama 3.3 as default if available
             if [ "$USE_LOCAL_MODELS" = true ] && [ "$NVIDIA_VRAM" -ge 24 ]; then
                 default_model_name="$model_llama33_70b"
             fi
        else
             print_warning "Insufficient VRAM (${NVIDIA_VRAM}GB) for 70B+ Local Models. Only Cloud models offered."
        fi
    else
        print_info "Local models disabled, only offering cloud models."
    fi

    # Find index of default model
    local default_index=1 # Start assuming first model is default
    for i in "${!model_names[@]}"; do
        if [[ "${model_names[$i]}" = "$default_model_name" ]]; then
            default_index=$((i + 1))
            break
        fi
    done

    # Quick install mode - auto-select the default model
    if [ "$QUICK_INSTALL" = true ]; then
        local selected_primary_model="${model_names[$((default_index - 1))]}"
        print_info "Auto-selecting primary model: ${CYAN}${selected_primary_model}${RESET}"
    else
        # Interactive mode - Print selection header and menu
        echo ""
        echo "${CYAN}${BOLD}Choose a primary reasoning model:${RESET}"
        echo "${GRAY}Plans research, uses tools, writes reports${RESET}"
        echo ""
        
        for i in "${!model_names[@]}"; do
            idx=$((i + 1))
            model="${model_names[$i]}"
            description="${model_descriptions[$i]}"
            is_recommended=false
            if [ "$idx" -eq $default_index ]; then is_recommended=true; fi
            print_model_choice "$idx" "$model" "$description" "$is_recommended"
        done
        echo ""

        # Get user choice
        local primary_user_choice
        # Use printf for consistent prompting
        printf "%s [%s%s%s]: %s" "${BLUE}Enter selection" "${GREEN}" "${default_index}" "${BLUE}" "${RESET}"
        read primary_user_choice # Use plain read to wait for input
        echo "" # Add newline after input
        primary_user_choice=${primary_user_choice:-$default_index} # Default if empty

        # Validate choice
        if ! [[ "$primary_user_choice" =~ ^[0-9]+$ ]] || [ "$primary_user_choice" -lt 1 ] || [ "$primary_user_choice" -gt ${#model_names[@]} ]; then
            print_warning "Invalid selection. Using default: ${default_model_name}"
            primary_user_choice=$default_index
        fi

        # Get selected model name using the chosen index (0-based for array)
        local selected_primary_model="${model_names[$((primary_user_choice - 1))]}"
        print_info "Selected primary model: ${CYAN}${selected_primary_model}${RESET}"
    fi

    # Rest of the function is the same for both modes
    # Determine final config values based on selection
    local final_primary_model_type=""
    local final_local_model_name=""
    local final_local_quant_level=""

    if [[ "$selected_primary_model" == *"cloud"* ]]; then
        final_local_model_name=""      # Clear local name for cloud
        final_local_quant_level=""     # Clear quant level for cloud
        if [[ "$selected_primary_model" == *"claude"* ]]; then final_primary_model_type="claude";
        elif [[ "$selected_primary_model" == *"gemini"* ]]; then final_primary_model_type="gemini";
        else final_primary_model_type="unknown_cloud"; fi
    elif [[ "$selected_primary_model" == *"$model_llama33_70b"* ]]; then
        # Known local Q4_K_M model
        final_primary_model_type="local"
        final_local_model_name="$selected_primary_model"
        final_local_quant_level="Q4_K_M"
    elif [[ "$selected_primary_model" == *"$model_qwen_72b"* ]]; then
        # Known local Q4_K_M model
        final_primary_model_type="local"
        final_local_model_name="$selected_primary_model"
        final_local_quant_level="Q4_K_M"
    elif [[ "$selected_primary_model" == *"$model_deepseek_70b"* ]]; then
        # Known local IQ4_NL model
        final_primary_model_type="local"
        final_local_model_name="$selected_primary_model"
        final_local_quant_level="IQ4_NL"
    elif [[ "$selected_primary_model" == *"$model_xwin_70b"* ]]; then
        # Known local Q4_K_M model
        final_primary_model_type="local"
        final_local_model_name="$selected_primary_model"
        final_local_quant_level="Q4_K_M"
    else
        # Fallback for unexpected cases
        print_error "Error: Unknown model type selected: ${selected_primary_model}"
        final_primary_model_type="unknown"
        final_local_model_name="$selected_primary_model"
        final_local_quant_level=""
    fi

    print_info "Updating config.yaml for primary model..."
    # Assume CONFIG_YAML is accessible
    local update_ok=true
    # Always use yq now
    # Use yq: set local fields to null if cloud, otherwise use final values
    if [[ "$final_primary_model_type" == "claude" || "$final_primary_model_type" == "gemini" ]]; then
        yq -i ".PRIMARY_MODEL_TYPE = \"$final_primary_model_type\" | .LOCAL_MODEL_NAME = null | .LOCAL_MODEL_QUANT_LEVEL = null" "$CONFIG_YAML" || update_ok=false
    else # Local model
        yq -i ".PRIMARY_MODEL_TYPE = \"$final_primary_model_type\" | .LOCAL_MODEL_NAME = \"$final_local_model_name\" | .LOCAL_MODEL_QUANT_LEVEL = \"$final_local_quant_level\"" "$CONFIG_YAML" || update_ok=false
    fi

    if [ "$update_ok" = true ]; then 
        print_info "Primary model configured in config.yaml ✓"
    else
        print_error "Failed to update config.yaml for primary model!"
    fi
    # Set the global variable instead of echoing
    _selected_model_result="$selected_primary_model"
}

# Configure Additional Options
configure_additional_options() {
    local SELECTED_PRIMARY_MODEL="$1" # Passed from main
    # Assumes CONFIG_YAML accessible
    print_section_header "ADDITIONAL CONFIGURATION OPTIONS"

    # 1. Memory Capacity
    echo "${CYAN}${BOLD}Memory Capacity Setting${RESET}"
    local CURRENT_MEMORY_TOKENS=$(grep -E "^CONVERSATION_MEMORY_MAX_TOKENS:" "$CONFIG_YAML" | awk '{print $2}' || echo "900000")
    # Note: Suggested values leave a buffer within the model's max context window to ensure space for the generated output.
    local SUGGESTED_MEMORY_TOKENS="900000"; local MEMORY_EXPLANATION="Gemini default (1M+ context)"
    if [[ "$SELECTED_PRIMARY_MODEL" == *"claude"* ]]; then SUGGESTED_MEMORY_TOKENS="120000"; MEMORY_EXPLANATION="Claude (~200K context, 120K leaves buffer for output)";
    elif [[ "$SELECTED_PRIMARY_MODEL" == *"llama-3.3-70b"* ]]; then SUGGESTED_MEMORY_TOKENS="110000"; MEMORY_EXPLANATION="Llama 3 70B (128K context, 110K leaves buffer for output)"; fi
    echo ""; echo "Current: ${YELLOW}${CURRENT_MEMORY_TOKENS}${RESET}, Suggested: ${GREEN}${SUGGESTED_MEMORY_TOKENS}${RESET} (${MEMORY_EXPLANATION})"; echo ""
    
    local UPDATE_MEMORY
    if [ "$QUICK_INSTALL" = true ]; then
        print_info "Auto-updating memory capacity to recommended value: ${SUGGESTED_MEMORY_TOKENS}"
        UPDATE_MEMORY="Y"
    else
        printf "%s [%s%s%s/%s] %s" "${BLUE}Update memory capacity?" "${GREEN}" "Y" "${BLUE}" "n" "${RESET}" # Use printf for consistent prompting
        read UPDATE_MEMORY # Use plain read to wait for input
        echo "" # Add newline after input
        UPDATE_MEMORY=${UPDATE_MEMORY:-Y} # Default to Y if empty
    fi
    
    if [[ "$UPDATE_MEMORY" == "y" || "$UPDATE_MEMORY" == "Y" ]]; then
        # Always use yq now
        yq -i ".CONVERSATION_MEMORY_MAX_TOKENS = $SUGGESTED_MEMORY_TOKENS" "$CONFIG_YAML" || print_error "yq update failed for memory capacity"
        print_info "Memory capacity updated."
    else print_info "Keeping current memory capacity."; fi

    # 2. Enhanced Reasoning
    echo ""; echo "${CYAN}${BOLD}Enhanced Reasoning Setting${RESET}"; echo "${GRAY}Enables explicit 'thinking' steps${RESET}"
    local CURRENT_THINKING=$(grep -E "^ENABLE_THINKING:" "$CONFIG_YAML" | awk '{print $2}' || echo "false")
    local SUGGESTED_THINKING="true"; local THINKING_EXPLANATION="Recommended for capable models"
    echo ""; echo "Current: ${YELLOW}${CURRENT_THINKING}${RESET}, Suggested: ${GREEN}${SUGGESTED_THINKING}${RESET} (${THINKING_EXPLANATION})"; echo ""
    
    local UPDATE_THINKING
    if [ "$QUICK_INSTALL" = true ]; then
        print_info "Auto-enabling enhanced reasoning for better performance."
        UPDATE_THINKING="Y"
    else
        printf "%s [%s%s%s/%s] %s" "${BLUE}Update enhanced reasoning?" "${GREEN}" "Y" "${BLUE}" "n" "${RESET}" # Use printf for consistent prompting
        read UPDATE_THINKING # Use plain read to wait for input
        echo "" # Add newline after input
        UPDATE_THINKING=${UPDATE_THINKING:-Y} # Default to Y if empty
    fi
    
    if [[ "$UPDATE_THINKING" == "y" || "$UPDATE_THINKING" == "Y" ]]; then
        # Always use yq now
        yq -i ".ENABLE_THINKING = $SUGGESTED_THINKING" "$CONFIG_YAML" || print_error "yq update failed for enhanced reasoning"
        print_info "Enhanced reasoning updated."
    else
        yq -i ".ENABLE_THINKING = false" "$CONFIG_YAML" || print_error "yq update failed for enhanced reasoning"
        print_info "Enhanced reasoning set to false."
    fi
    
    # Research Depth settings are managed directly in config.yaml
    print_info "Research depth settings (MIN/MAX_REGULAR_WEB_PAGES) are kept as defined in config.yaml."
}

# Optimal GPU Layers Calculation (Simplified)
get_optimal_gpu_layers() { 
    local model="$1"
    local has_nvidia="$4" # Only 4th arg needed now
    local gpu_layers=0

    if [ "$has_nvidia" = true ]; then
        NVIDIA_VRAM=$(get_nvidia_vram)
        print_info "Calculating optimal GPU layers for $model on NVIDIA (${NVIDIA_VRAM}GB VRAM)"

        # Check for 70B / 72B models
        if [[ "$model" == *"70b"* || "$model" == *"70B"* || "$model" == *"72b"* || "$model" == *"72B"* ]]; then
            # 70B/72B Model Logic
            if [ "$NVIDIA_VRAM" -ge 48 ]; then gpu_layers=64
            elif [ "$NVIDIA_VRAM" -ge 40 ]; then gpu_layers=50
            elif [ "$NVIDIA_VRAM" -ge 32 ]; then gpu_layers=40
            elif [ "$NVIDIA_VRAM" -ge 24 ]; then gpu_layers=32
            else gpu_layers=10; print_warning "Low VRAM ($NVIDIA_VRAM GB) for 70B+ model."; fi
            print_info "Recommended GPU layers for 70B+: $gpu_layers"
        else 
            # Fallback for unknown local model
            gpu_layers=10; print_warning "Unknown local model size ($model), using default GPU layers: $gpu_layers"; 
        fi
    else
        # Non-NVIDIA or Cloud
        gpu_layers=0
        print_info "No NVIDIA GPU detected or Cloud model selected, setting GPU layers to 0."
    fi
    echo "$gpu_layers"
}

# Function to update N_GPU_LAYERS in config.yaml
update_gpu_layers() {
    local gpu_layers=$1
    # Assumes CONFIG_YAML accessible
    print_info "Setting N_GPU_LAYERS to $gpu_layers in config.yaml"
    # Always use yq now
    yq -i ".N_GPU_LAYERS = $gpu_layers" "$CONFIG_YAML" || print_error "yq update failed for N_GPU_LAYERS"
}

# Check and Set Environment Variables
check_and_set_env_var() {
    local key_name="$1"
    local description="$2"
    local key_url="$3"
    local is_required="$4"  # New parameter to indicate if key is mandatory
    local env_file=".env"
    local current_value
    local key_exists=false
    local key_has_value=false
    local is_placeholder=false

    # Ensure .env file exists before trying to read/write
    if [ ! -f "$env_file" ]; then
        # Try creating from example if possible
        if [ -f ".env.example" ]; then
             print_info "Creating $env_file from .env.example for key check..."
             cp .env.example "$env_file"
        else
             print_warning "$env_file not found, cannot check/set ${key_name}."
             return
        fi
    fi

    if grep -qE "^${key_name}=" "$env_file"; then
        key_exists=true
        current_value=$(grep -E "^${key_name}=" "$env_file" | head -n 1 | cut -d '=' -f 2- | sed -e 's/^[\\"\\ \\"]*//' -e 's/[\\"\\ \\"]*$//')
        if [ -n "$current_value" ]; then
            key_has_value=true
            # Check if it's a placeholder value
            if [[ "$current_value" == "api_key_here" ]]; then
                is_placeholder=true
            fi
        fi
    fi

    if [ "$key_has_value" = true ] && [ "$is_placeholder" = false ]; then
        # Mask the key for display (show first 4 and last 4 chars)
        local masked_value
        local length=${#current_value}
        if [ $length -le 8 ]; then
            # If key is too short, just show all asterisks
            masked_value=$(printf '%*s' "$length" | tr ' ' '*')
        else
            # Show first 4 and last 4 chars with asterisks in between
            local first_part="${current_value:0:4}"
            local last_part="${current_value: -4}"
            local middle_length=$((length - 8))
            local middle_part=$(printf '%*s' "$middle_length" | tr ' ' '*')
            masked_value="${first_part}${middle_part}${last_part}"
        fi
        print_info "${key_name} found: ${CYAN}${masked_value}${RESET}"
        echo "Current ${key_name} appears to be set."

        local update_key="N"
        # In quick install mode, keep existing key values
        if [ "$QUICK_INSTALL" = true ]; then
            print_info "Keeping existing ${key_name} in quick install mode ✓"
        else
            # Interactive mode: Ask to update
            printf "%s [%s%s%s/%s] %s" "${YELLOW}Update this key?" "${YELLOW}" "y" "${GREEN}" "N" "${RESET}" # Note: Default is N
            read update_key
            echo "" # Add newline
            update_key=${update_key:-N} # Default to N if empty
        fi

        if [[ "$update_key" == "y" || "$update_key" == "Y" ]]; then
            read -s -p "${YELLOW}Enter new value for ${key_name}: ${RESET}" user_key; echo
            if [ -n "$user_key" ]; then
                # Escape potential special characters in key for sed/echo
                escaped_user_key=$(echo "$user_key" | sed 's/[&/\\#]/\\\\&/g') # Escape &, /, \, #
                # Ensure value is quoted in .env
                quoted_key_value="\"${escaped_user_key}\""
                print_info "Updating ${key_name} in $env_file..."
                # Use # as separator for sed to avoid issues with / in keys/paths
                sed -i.bak "s#^${key_name}=.*#${key_name}=${quoted_key_value}#" "$env_file" && rm -f "${env_file}.bak" || print_error "sed update failed for ${key_name}"
                print_info "${key_name} has been updated in $env_file ✓"
            else
                print_warning "Empty key provided, keeping existing value."
            fi
        else
            # Only print this if we didn't update (i.e., quick mode or user chose 'N')
            if [ "$QUICK_INSTALL" = false ]; then
                 print_info "Keeping existing ${key_name} ✓"
            fi
        fi
    else
        # Key is missing, empty, or contains placeholder
        if [ "$key_exists" = true ] && [ "$is_placeholder" = true ]; then
            print_warning "${key_name} found in $env_file but contains placeholder value."
        elif [ "$key_exists" = true ]; then
            print_warning "${key_name} found in $env_file but appears to be empty."
        else
            print_warning "${key_name} not found in $env_file."
        fi

        echo "${CYAN}This key is needed for: ${description}${RESET}"
        echo "You can obtain one here: ${BLUE}${UNDERLINE}${key_url}${RESET}"

        # In quick install mode, just warn if required, don't prompt
        if [ "$QUICK_INSTALL" = true ]; then
             if [ "$is_required" = true ]; then
                print_info "${YELLOW}This key is required. Please add it to the .env file manually later.${RESET}"
             else
                print_info "${YELLOW}Optional key not set. Can be added to .env file later if needed.${RESET}"
             fi
             return
        fi

        # Interactive mode: Prompt for the key
        local user_key=""
        local valid_key=false

        # If required, loop until valid key is provided
        while [ "$valid_key" = false ]; do
            # Use standard read -s for sensitive input
            read -s -p "${YELLOW}Enter value for ${key_name}: ${RESET}" user_key; echo

            if [ -n "$user_key" ]; then
                valid_key=true
            elif [ "$is_required" = true ]; then
                print_error "This API key is required for the selected model. Please provide a valid key."
            else
                print_warning "Empty key provided. Continuing without ${key_name}."
                valid_key=true  # Allow empty if not required
            fi
        done

        if [ -n "$user_key" ]; then
            # Escape potential special characters in key for sed/echo
            escaped_user_key=$(echo "$user_key" | sed 's/[&/\\#]/\\\\&/g') # Escape &, /, \, #
            # Ensure value is quoted in .env
            quoted_key_value="\"${escaped_user_key}\""
            if [ "$key_exists" = true ]; then
                print_info "Updating ${key_name} in $env_file..."
                # Use # as separator for sed to avoid issues with / in keys/paths
                sed -i.bak "s#^${key_name}=.*#${key_name}=${quoted_key_value}#" "$env_file" && rm -f "${env_file}.bak" || print_error "sed update failed for ${key_name}"
            else
                print_info "Adding ${key_name} to $env_file..."
                # Append the new key=value pair
                echo "${key_name}=${quoted_key_value}" >> "$env_file"
            fi
            print_info "${key_name} has been set in $env_file ✓"
        else
            print_warning "Skipped setting ${key_name}. Application might require it later."
        fi
    fi
}

# Display Configuration Summary (Simplified)
display_configuration_summary() {
    # Assumes these are passed or accessible: SELECTED_MODEL_FILENAME, SELECTED_PRIMARY_MODEL, OPTIMAL_GPU_LAYERS, CONFIG_YAML, VENV_DIR
    print_section_header "FINAL CONFIGURATION"
    local summary_lines=(
        "Summarizer Model: ${CYAN}${SELECTED_MODEL_FILENAME}${RESET}" # Always Flash now
        "- ${GREEN}Using cloud-based Gemini Flash model${RESET}"
        "- ${GREEN}Chunking disabled (CHUNK_SIZE: 0)${RESET}"
        "- ${GREEN}USE_LOCAL_SUMMARIZER_MODEL: false${RESET}" ""
        "Primary Reasoning Model: ${CYAN}${SELECTED_PRIMARY_MODEL}${RESET}"
    )
    local FINAL_PRIMARY_MODEL_TYPE=$(grep -E "^PRIMARY_MODEL_TYPE:" "$CONFIG_YAML" | awk '{print $2}' | sed 's/\"//g')
    local FINAL_GPU_LAYERS=$(grep -E "^N_GPU_LAYERS:" "$CONFIG_YAML" | awk '{print $2}' || echo "0") # Read final value
    local FINAL_QUANT_LEVEL=$(grep -E "^LOCAL_MODEL_QUANT_LEVEL:" "$CONFIG_YAML" | awk '{print $2}' | sed 's/\"//g')

    if [[ "$FINAL_PRIMARY_MODEL_TYPE" == "claude" ]]; then
            summary_lines+=("- ${GREEN}Using cloud-based Claude model${RESET}")
            summary_lines+=("- ${GRAY}Requires ANTHROPIC_API_KEY in .env file${RESET}")
    elif [[ "$FINAL_PRIMARY_MODEL_TYPE" == "gemini" ]]; then
            summary_lines+=("- ${GREEN}Using cloud-based Gemini model${RESET}")
            summary_lines+=("- ${GRAY}Requires GEMINI_API_KEY in .env file${RESET}")
    elif [[ "$FINAL_PRIMARY_MODEL_TYPE" == "local" && "$SELECTED_PRIMARY_MODEL" == *"llama-3.3-70b"* ]]; then
        summary_lines+=("- ${YELLOW}Using local Llama 3 70B GGUF model${RESET}")
        summary_lines+=("- ${GRAY}LOCAL_MODEL_NAME: $SELECTED_PRIMARY_MODEL${RESET}")
        summary_lines+=("- ${GRAY}LOCAL_MODEL_QUANT_LEVEL: ${FINAL_QUANT_LEVEL}${RESET}")
        summary_lines+=("- ${GRAY}N_GPU_LAYERS: ${FINAL_GPU_LAYERS}${RESET}")
    else summary_lines+=("- ${RED}Unknown primary model configuration${RESET}"); fi

    # --- Add Next Step Model to Summary ---
    local FINAL_NEXT_STEP_MODEL=$(grep -E "^NEXT_STEP_MODEL:" "$CONFIG_YAML" | awk '{print $2}' | sed 's/\"//g')
    local FINAL_NEXT_STEP_THINKING=$(grep -E "^NEXT_STEP_THINKING_DEFAULT:" "$CONFIG_YAML" | awk '{print $2}' | sed 's/\"//g')
    summary_lines+=("" "Next Step Model: ${CYAN}${FINAL_NEXT_STEP_MODEL}${RESET}")
    summary_lines+=("- Default Thinking Mode: ${YELLOW}${FINAL_NEXT_STEP_THINKING}${RESET}")

    summary_lines+=("" "--- Additional Settings ---")
    local MEMORY_TOKENS=$(grep -E "^CONVERSATION_MEMORY_MAX_TOKENS:" "$CONFIG_YAML" | awk '{print $2}' || echo "Unknown")
    local THINKING_ENABLED=$(grep -E "^ENABLE_THINKING:" "$CONFIG_YAML" | awk '{print $2}' || echo "Unknown")
    local MIN_PAGES=$(grep -E "^MIN_REGULAR_WEB_PAGES:" "$CONFIG_YAML" | awk '{print $2}' || echo "Unknown")
    local MAX_PAGES=$(grep -E "^MAX_REGULAR_WEB_PAGES:" "$CONFIG_YAML" | awk '{print $2}' || echo "Unknown")
    summary_lines+=("Memory Capacity: ${GRAY}${MEMORY_TOKENS} tokens${RESET}")
    summary_lines+=("Enhanced Reasoning: ${GRAY}${THINKING_ENABLED}${RESET}")
    summary_lines+=("Research Depth: ${GRAY}${MIN_PAGES}-${MAX_PAGES} web pages per topic${RESET}")
    
    print_summary_box "FINAL CONFIGURATION" "${summary_lines[@]}"
    echo ""
    echo "${GREEN}${BOLD}Setup Steps Completed!${RESET}"
    echo ""
    echo "To activate the virtual environment, run: ${CYAN}source $VENV_DIR/bin/activate${RESET}"
    if [[ "$FINAL_PRIMARY_MODEL_TYPE" == "claude" ]] || [[ "$FINAL_PRIMARY_MODEL_TYPE" == "gemini" ]] || [[ "$SELECTED_MODEL_FILENAME" == *"gemini"* ]]; then
        echo "Remember to add required API keys to ${CYAN}.env${RESET} file."
    fi
    echo "Run the application using: ${CYAN}./run.sh${RESET}"
    echo ""
}

# --- GGUF Download Functions for 70B+ Models ---

download_llama33_70b_gguf() {
    local target_dir=$1
    local repo_id="unsloth/Llama-3.3-70B-Instruct-GGUF"
    local gguf_filename="Llama-3.3-70B-Instruct-Q4_K_M.gguf"
    local model_url="https://huggingface.co/${repo_id}"
    print_info "Attempting download from: ${repo_id} / ${gguf_filename}"
    if _huggingface_download_gguf "$repo_id" "$gguf_filename" "$target_dir" "$model_url"; then return 0; else print_error "Failed download: ${gguf_filename}"; return 1; fi
}

download_qwen_72b_gguf() {
    local target_dir=$1
    local repo_id="Qwen/Qwen2.5-72B-Instruct-GGUF"
    local gguf_filename="Qwen2.5-72B-Instruct-Q4_K_M.gguf"
    local model_url="https://huggingface.co/${repo_id}"
    print_info "Attempting download from: ${repo_id} / ${gguf_filename}"
    if _huggingface_download_gguf "$repo_id" "$gguf_filename" "$target_dir" "$model_url"; then return 0; else print_error "Failed download: ${gguf_filename}"; return 1; fi
}

download_deepseek_70b_gguf() {
    local target_dir=$1
    local repo_id="unsloth/DeepSeek-R1-Distill-Llama-70B-GGUF"
    local gguf_filename="DeepSeek-R1-Distill-Llama-70B-IQ4_NL.gguf" # Note the IQ4_NL quant
    local model_url="https://huggingface.co/${repo_id}"
    print_info "Attempting download from: ${repo_id} / ${gguf_filename}"
    if _huggingface_download_gguf "$repo_id" "$gguf_filename" "$target_dir" "$model_url"; then return 0; else print_error "Failed download: ${gguf_filename}"; return 1; fi
}

download_xwin_70b_gguf() {
    local target_dir=$1
    local repo_id="TheBloke/Xwin-LM-70B-V0.1-GGUF"
    local gguf_filename="xwin-lm-70b-v0.1.Q4_K_M.gguf"
    local model_url="https://huggingface.co/${repo_id}"
    print_info "Attempting download from: ${repo_id} / ${gguf_filename}"
    if _huggingface_download_gguf "$repo_id" "$gguf_filename" "$target_dir" "$model_url"; then return 0; else print_error "Failed download: ${gguf_filename}"; return 1; fi
}

# --- Main Execution Function ---
main() {
    # --- Define variables needed in this scope ---
    local SELECTED_MODEL_FILENAME="N/A"
    local SELECTED_PRIMARY_MODEL="N/A"
    local USE_LOCAL_MODELS=false
    local LLAMA_CPP_PYTHON_FLAGS="N/A"
    local OPTIMAL_GPU_LAYERS=0
    local CURRENT_GPU_LAYERS="N/A"
    local UPDATE_GPU_LAYERS="n"
    local PRIMARY_QUANT_LEVEL="N/A"
    local IS_MAC=$([ "$(uname)" == "Darwin" ] && echo true || echo false)
    local HAS_NVIDIA=$(has_nvidia_gpu && echo true || echo false) # Check GPU early
    local MODEL_LLAMA3_70B="llama-3.3-70b-instruct.Q4_K_M.gguf" # Define needed model const
    local _selected_model_result="" # Variable to capture function results
    local _ask_model_preference_result="" # Variable to capture ask_model_preference result
    
    # --- New setup mode variable: quick vs interactive ---
    local QUICK_INSTALL=true
    
    # --- Print setup header ---
    echo ""
    echo "${BLUE}${BOLD}==================================================================${RESET}"
    echo "${BLUE}${BOLD}             Python AI Research Agent - Setup Script              ${RESET}"
    echo "${BLUE}${BOLD}==================================================================${RESET}"
    echo ""
    
    # --- Welcome message and setup mode selection ---
    echo "${CYAN}${BOLD}Welcome to the Deep Researcher Setup Script${RESET}"
    echo ""
    echo "This script will set up your environment with the following components:"
    echo "  • ${GREEN}Python virtual environment${RESET}"
    echo "  • ${GREEN}Required Python packages${RESET}"
    echo "  • ${GREEN}Patchright web browser automation${RESET}"
    echo "  • ${GREEN}MCP API tools configuration${RESET}"
    echo "  • ${GREEN}API keys for AI services${RESET}"
    
    if [ "$HAS_NVIDIA" = true ]; then
        local vram_gb=$(get_nvidia_vram)
        if [ "$vram_gb" -ge 24 ]; then
            echo "  • ${GREEN}Local LLM models (NVIDIA GPU ${vram_gb}GB detected)${RESET}"
        else
            echo "  • ${YELLOW}Cloud models only (NVIDIA GPU has insufficient VRAM: ${vram_gb}GB)${RESET}"
        fi
    elif [ "$IS_MAC" = true ]; then
        local mac_model=$(detect_apple_silicon_model)
        if [[ "$mac_model" == *"Apple"* ]]; then
            echo "  • ${YELLOW}Cloud models only (Apple Silicon: ${mac_model})${RESET}"
        else
            echo "  • ${YELLOW}Cloud models only (Intel Mac)${RESET}"
        fi
    else
        echo "  • ${YELLOW}Cloud models only (No NVIDIA GPU detected)${RESET}"
    fi
    
    echo ""
    echo "The script can run in one of two modes:"
    echo "  1. ${CYAN}Quick Install:${RESET} Automatically install with defaults for your hardware"
    echo "  2. ${CYAN}Interactive:${RESET} Ask for confirmation at each step"
    echo ""
    
    local setup_mode
    printf "%s [%s%s%s/%s] %s" "${BLUE}Choose setup mode" "${GREEN}" "1" "${BLUE}" "2" "${RESET}" # Use printf for consistent prompting
    read setup_mode # Use plain read to wait for input
    echo "" # Add newline after input

    setup_mode=${setup_mode:-1} # Default to 1 if empty

    if [ "$setup_mode" = "2" ]; then
        QUICK_INSTALL=false
        echo "${YELLOW}Selected: Interactive installation${RESET}"
    else
        QUICK_INSTALL=true
        echo "${GREEN}Selected: Quick installation with defaults${RESET}"
    fi
    
    # --- Comprehensive prerequisite checking ---
    print_section_header "CHECKING PREREQUISITES"
    
    local missing_prerequisites=false
    
    # Essential commands
    echo "${CYAN}Checking essential commands...${RESET}"
    for cmd in "$PYTHON_CMD" "jq" "git"; do
        if ! command -v $cmd &> /dev/null; then
            print_error "$cmd is required but not found on your system."
            missing_prerequisites=true
            case $cmd in
                "$PYTHON_CMD")
                    echo "Please install Python 3 from https://www.python.org/downloads/"
                    ;;
                "jq")
                    if [ "$IS_MAC" = true ]; then
                        echo "On macOS, you can install jq with: brew install jq"
                    else
                        echo "On Linux, you can install jq with: sudo apt install jq (Debian/Ubuntu)"
                        echo "or: sudo dnf install jq (Fedora)"
                    fi
                    ;;
                "git")
                    if [ "$IS_MAC" = true ]; then
                        echo "On macOS, you can install git with: brew install git"
                    else
                        echo "On Linux, you can install git with: sudo apt install git (Debian/Ubuntu)"
                        echo "or: sudo dnf install git (Fedora)"
                    fi
                    ;;
            esac
        else
            print_info "$cmd found ✓"
            
            # Check Python version if the command is Python
            if [ "$cmd" = "$PYTHON_CMD" ]; then
                local python_version=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2)
                local python_major=$(echo $python_version | cut -d. -f1)
                local python_minor=$(echo $python_version | cut -d. -f2)
                
                print_info "Python version: $python_version"
                
                if [ "$python_major" -lt 3 ] || ([ "$python_major" -eq 3 ] && [ "$python_minor" -lt 8 ]); then
                    print_warning "Python 3.8+ is recommended, you have $python_version"
                    echo "The script might work with your version, but some features may be unavailable."
                fi
            fi
        fi
    done
    
    # Check for yq, which we will attempt to install if missing
    if ! command -v yq &> /dev/null; then
        print_warning "yq (YAML processor) not found - we'll try to install it later."
    else
        print_info "yq found ✓"
    fi
    
    # Check for required files
    echo ""
    echo "${CYAN}Checking required files...${RESET}"
    for file in "$MCP_JSON_FILE" "$CONFIG_YAML"; do
        if [ ! -f "$file" ]; then
            print_error "$file not found."
            missing_prerequisites=true
            
            if [ "$file" = "$CONFIG_YAML" ]; then
                echo "Is config.yaml.example available to copy from?"
                if [ -f "${file}.example" ]; then
                    echo "Attempting to create $file from ${file}.example..."
                    cp "${file}.example" "$file" && print_info "Created $file from example ✓" || print_error "Failed to create $file"
                else
                    echo "No ${file}.example found. Please create $file before continuing."
                fi
            fi
        else
            print_info "$file found ✓"
        fi
    done
    
    # Hardware detection information
    echo ""
    echo "${CYAN}Hardware detection...${RESET}"
    if [ "$IS_MAC" = true ]; then
        local mac_model=$(detect_apple_silicon_model)
        print_info "Running on macOS: $mac_model"
        
        if [[ "$mac_model" == *"Apple"* ]]; then
            local unified_memory=$(get_apple_unified_memory)
            print_info "Apple Silicon with ${unified_memory}GB unified memory detected."
        fi
    else
        print_info "Running on Linux ($(uname -m))"
    fi
    
    if [ "$HAS_NVIDIA" = true ]; then
        local vram_gb=$(get_nvidia_vram)
        print_info "NVIDIA GPU detected with ${vram_gb}GB VRAM"
    else
        print_info "No NVIDIA GPU detected."
    fi
    
    local ram_gb=$(get_system_ram)
    print_info "System RAM: ${ram_gb}GB"
    
    # Exit if critical prerequisites are missing
    if [ "$missing_prerequisites" = true ]; then
        print_error "Some critical prerequisites are missing. Please install them and try again."
        exit 1
    fi

    # Make global config paths accessible if helper functions need them without args
    # (Already defined globally, but ensure functions can see them if needed)
    # VENV_DIR="venv"; CONFIG_YAML="config.yaml"; MODELS_DIR="models"; MCP_JSON_FILE="mcp.json"
    # PYTHON_CMD="python3"

    # --- Step 0: Ensure uv (uvx) is installed ---
    print_section_header "CHECKING FOR 'uv' (uvx) TOOL"
    if ! command -v uvx &> /dev/null; then
        print_warning "'uvx' (uv) not found. Installing via official script..."
        # Use the official install script for Mac/Linux (ARM64 safe)
        curl -LsSf https://astral.sh/uv/install.sh | sh
        # Ensure ~/.local/bin is on PATH for this session
        export PATH="$HOME/.local/bin:$PATH"
        print_info "'uv' (uvx) installed. If you still get 'command not found', add 'export PATH=\"$HOME/.local/bin:$PATH\"' to your shell profile."
    else
        print_info "'uvx' (uv) found ✓"
    fi

    # --- Step 1: Prerequisite Checks ---
    print_section_header "SYSTEM CHECKS & PREFERENCES"
    if ! command -v $PYTHON_CMD &> /dev/null; then print_error "Python 3 not found."; exit 1; else print_info "Python 3 found ✓"; fi
    if ! command -v jq &> /dev/null; then print_error "jq not found."; exit 1; else print_info "jq found ✓"; fi

    # Check for yq, install if missing
    if ! command -v yq &> /dev/null; then
        print_warning "yq YAML processor not found."
        if [ "$IS_MAC" = true ] && command -v brew &> /dev/null; then
            print_info "Attempting to install yq using Homebrew..."
            if brew install yq; then
                print_info "yq installed successfully ✓"
            else
                print_error "Failed to install yq using Homebrew. Please install it manually and re-run setup."
                exit 1
            fi
        elif [[ "$(uname)" == "Linux" ]]; then
            print_info "Attempting to install yq using common Linux package managers (sudo required)..."
            if command -v snap &> /dev/null; then
                print_info "Trying snap..."
                if sudo snap install yq; then print_info "yq installed via snap ✓"; else print_warning "snap install failed."; fi
            elif command -v apt &> /dev/null; then
                print_info "Trying apt..."
                if sudo apt update > /dev/null && sudo apt install -y yq; then print_info "yq installed via apt ✓"; else print_warning "apt install failed."; fi
            elif command -v dnf &> /dev/null; then
                print_info "Trying dnf..."
                if sudo dnf install -y yq; then print_info "yq installed via dnf ✓"; else print_warning "dnf install failed."; fi
            elif command -v yum &> /dev/null; then
                print_info "Trying yum..."
                if sudo yum install -y yq; then print_info "yq installed via yum ✓"; else print_warning "yum install failed."; fi
            else
                print_warning "Could not find snap, apt, dnf, or yum to install yq automatically."
            fi
            # Final check after attempting Linux installs
            if ! command -v yq &> /dev/null; then
                 print_error "Failed to install yq automatically. Please install it using your system's package manager (e.g., 'sudo apt install yq', 'sudo dnf install yq', 'sudo snap install yq') and re-run setup."
                 exit 1
            fi
        else
            print_error "yq not found and cannot automatically install on this OS. Please install yq manually and re-run setup."
            exit 1
        fi
    else
        print_info "yq found ✓"
    fi

    if [ ! -f "$MCP_JSON_FILE" ]; then print_error "$MCP_JSON_FILE not found."; exit 1; else print_info "MCP config file found ✓"; fi
    if [ ! -f "$CONFIG_YAML" ]; then print_error "$CONFIG_YAML not found. Please create from example."; exit 1; else print_info "$CONFIG_YAML found ✓"; fi

    # --- Step 2: Ask User Preference (Hardware Aware) ---
    # Call function directly, it sets _ask_model_preference_result
    ask_model_preference 
    USE_LOCAL_MODELS="$_ask_model_preference_result" # Assign result
    echo "DEBUG: USE_LOCAL_MODELS set to: [$USE_LOCAL_MODELS]" # Add debug

    # --- Step 3: Conditional Prereq Check (CMake) ---
    if [ "$USE_LOCAL_MODELS" = true ]; then
        if ! command -v cmake &> /dev/null; then print_error "CMake not found (required for local models)."; exit 1; else print_info "CMake found ✓"; fi
    fi

    # --- Step 4: Setup Virtual Environment ---
    print_section_header "PYTHON VIRTUAL ENVIRONMENT"
    if [ -d "$VENV_DIR" ]; then print_info "Virtual environment '$VENV_DIR' already exists ✓"; else print_info "Creating Python venv..."; $PYTHON_CMD -m venv $VENV_DIR && print_info "Venv created ✓" || { print_error "Venv creation failed."; exit 1; }; fi
    local PYTHON_VENV="$VENV_DIR/bin/python"
    local PIP_VENV="$VENV_DIR/bin/pip"
    local PLAYWRIGHT_VENV="$VENV_DIR/bin/playwright"
    print_info "Upgrading pip..."
    $PIP_VENV install --upgrade pip > /dev/null

    # --- Step 5: Install Base Dependencies ---
    print_section_header "BASE DEPENDENCIES"
    print_info "Installing Python dependencies..."

    # Check if uvx is available for faster dependency resolution
    if command -v uvx &> /dev/null; then
        print_info "Using UV for faster dependency resolution..."
        if uvx pip install -r requirements.txt; then
            print_info "Base dependencies installed with UV ✓"
        else
            print_warning "UV installation failed, falling back to pip..."
            $PYTHON_VENV -m pip install -r requirements.txt || print_error "Failed to install dependencies with pip"
            print_info "Base dependencies installed with pip ✓"
        fi
    else
        print_warning "UV not found, using standard pip (slower)..."
        $PYTHON_VENV -m pip install -r requirements.txt || print_error "Failed to install dependencies with pip"
        print_info "Consider installing UV for faster dependency resolution: curl -LsSf https://astral.sh/uv/install.sh | sh"
        print_info "Base dependencies installed with pip ✓"
    fi

    # --- Step 6: Conditional Local Model Setup (llama install, HF login, models dir) ---
    if [ "$USE_LOCAL_MODELS" = true ]; then
        print_section_header "LOCAL MODEL SETUP ACTIONS"

        # Install additional ML dependencies only needed for local models
        print_info "Installing additional packages required for local models..."
        local LOCAL_MODEL_PACKAGES=(
            "huggingface-hub>=0.19.0"
            "accelerate>=0.27.0"
            "bitsandbytes>=0.41.0"
            "safetensors>=0.4.0"
        )
        
        for package in "${LOCAL_MODEL_PACKAGES[@]}"; do
            print_info "Installing $package..."
            $PIP_VENV install "$package" > /dev/null || print_warning "Failed to install $package"
        done
        
        # Install PyTorch with CUDA for NVIDIA GPUs
        if [ "$HAS_NVIDIA" = true ]; then
            print_info "Installing PyTorch with CUDA support for NVIDIA GPUs..."
            # Use a proper PyTorch CUDA version based on system configuration
            if command -v nvcc &> /dev/null; then
                CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
                CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)
                CUDA_MINOR=$(echo "$CUDA_VERSION" | cut -d. -f2)
                
                if [ "$CUDA_MAJOR" -ge 12 ]; then
                    # CUDA 12.x
                    print_info "Using torch with CUDA 12.x support"
                    $PIP_VENV install torch torchvision --index-url https://download.pytorch.org/whl/cu121 > /dev/null || print_warning "Failed to install PyTorch with CUDA 12.1"
                elif [ "$CUDA_MAJOR" -eq 11 ] && [ "$CUDA_MINOR" -ge 8 ]; then
                    # CUDA 11.8+
                    print_info "Using torch with CUDA 11.8+ support"
                    $PIP_VENV install torch torchvision --index-url https://download.pytorch.org/whl/cu118 > /dev/null || print_warning "Failed to install PyTorch with CUDA 11.8"
                else
                    # Older CUDA or unknown version
                    print_info "Using torch with default CUDA support"
                    $PIP_VENV install torch torchvision > /dev/null || print_warning "Failed to install PyTorch with default CUDA"
                fi
            else
                # No nvcc, use default CUDA
                print_info "Using torch with default CUDA support (nvcc not found)"
                $PIP_VENV install torch torchvision > /dev/null || print_warning "Failed to install PyTorch with default CUDA"
            fi
        else
            # Non-NVIDIA system, install CPU-only PyTorch
            if [ "$IS_MAC" = true ] && [[ "$(uname -m)" == "arm64" ]]; then
                # Apple Silicon - native package
                print_info "Installing PyTorch for Apple Silicon"
                $PIP_VENV install torch torchvision > /dev/null || print_warning "Failed to install PyTorch for Apple Silicon"
            else
                # CPU-only package
                print_info "Installing CPU-only PyTorch"
                $PIP_VENV install torch torchvision --index-url https://download.pytorch.org/whl/cpu > /dev/null || print_warning "Failed to install CPU-only PyTorch"
            fi
        fi
        
        print_info "Additional local model dependencies installed ✓"

        # Install llama-cpp-python with proper acceleration
        print_info "Installing llama-cpp-python with hardware acceleration..."
        if [ "$IS_MAC" = true ]; then
            # macOS - try Apple Silicon Metal
            if [[ "$(uname -m)" == "arm64" ]]; then
                print_info "Detected Apple Silicon, using Metal acceleration."
                # For Apple Silicon, use Metal acceleration
                LLAMA_CPP_PYTHON_FLAGS="CMAKE_ARGS=\"-DGGML_METAL=ON\""
                print_info "Running: $LLAMA_CPP_PYTHON_FLAGS $PIP_VENV install --force-reinstall --no-cache-dir llama-cpp-python"
                
                # Create a variable for the installation command
                INSTALL_CMD="$LLAMA_CPP_PYTHON_FLAGS $PIP_VENV install --force-reinstall --no-cache-dir llama-cpp-python"
                
                # Try with a timeout to avoid hanging
                if command -v gtimeout &> /dev/null; then
                    # Use gtimeout if available (from coreutils)
                    eval "gtimeout 600 $INSTALL_CMD" || {
                        print_error "llama-cpp-python install with Metal failed or timed out."
                        print_warning "Trying without Metal acceleration..."
                        # Fallback to CPU-only
                        $PIP_VENV install --force-reinstall --no-cache-dir llama-cpp-python || print_error "CPU fallback also failed."
                    }
                else
                    # No gtimeout available
                    eval "$INSTALL_CMD" || {
                        print_error "llama-cpp-python install with Metal failed."
                        print_warning "Trying without Metal acceleration..."
                        # Fallback to CPU-only
                        $PIP_VENV install --force-reinstall --no-cache-dir llama-cpp-python || print_error "CPU fallback also failed."
                    }
                fi
            else
                # Intel Mac - use AVX if available
                print_info "Detected Intel Mac, using AVX acceleration."
                LLAMA_CPP_PYTHON_FLAGS=""
                $PIP_VENV install --force-reinstall --no-cache-dir llama-cpp-python || print_error "llama-cpp-python install failed on Intel Mac."
            fi
        else
            # Linux - check for NVIDIA and CPU architecture
            if [ "$HAS_NVIDIA" = true ]; then
                print_info "Using NVIDIA CUDA acceleration."
                LLAMA_CPP_PYTHON_FLAGS="CMAKE_ARGS=\"-DGGML_CUDA=on\""
                
                # Check CUDA version if available
                if command -v nvcc &> /dev/null; then
                    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
                    print_info "Detected CUDA version: $CUDA_VERSION"
                    # Check for CUDA compatibility issues
                    if [[ $(echo "$CUDA_VERSION" | cut -d. -f1) -lt 11 ]]; then
                        print_warning "CUDA version < 11.0 may have compatibility issues with llama-cpp-python"
                    fi
                fi
                
                # Try with a timeout to avoid hanging
                if command -v timeout &> /dev/null; then
                    eval "timeout 600 $LLAMA_CPP_PYTHON_FLAGS $PIP_VENV install --force-reinstall --no-cache-dir llama-cpp-python" || {
                        print_error "llama-cpp-python install with CUDA failed or timed out."
                        print_warning "Trying to install CPU-only version as fallback..."
                        $PIP_VENV install --force-reinstall --no-cache-dir llama-cpp-python || print_error "CPU fallback also failed."
                    }
                else
                    eval "$LLAMA_CPP_PYTHON_FLAGS $PIP_VENV install --force-reinstall --no-cache-dir llama-cpp-python" || {
                        print_error "llama-cpp-python install with CUDA failed."
                        print_warning "Trying to install CPU-only version as fallback..."
                        $PIP_VENV install --force-reinstall --no-cache-dir llama-cpp-python || print_error "CPU fallback also failed."
                    }
                fi
            else
                # Check for ARM architecture on Linux
                if [[ "$(uname -m)" == "aarch64" ]]; then
                    print_info "Detected ARM64 Linux. Using optimized build."
                    # ARM-specific optimizations if needed
                    $PIP_VENV install --force-reinstall --no-cache-dir llama-cpp-python || print_error "llama-cpp-python install failed on ARM Linux."
                else
                    # Standard x86_64 Linux without NVIDIA
                    print_info "Using CPU-only mode (no GPU acceleration detected)."
                    $PIP_VENV install --force-reinstall --no-cache-dir llama-cpp-python || print_error "llama-cpp-python install failed on Linux."
                fi
            fi
        fi
        
        # Verify llama-cpp-python installation
        if $PYTHON_VENV -c "import llama_cpp" &>/dev/null; then
            print_info "llama-cpp-python installed and imported successfully ✓"
            # Try to verify hardware acceleration if possible
            if [ "$IS_MAC" = true ] && [[ "$(uname -m)" == "arm64" ]]; then
                # Check if Metal is available in llama_cpp
                if $PYTHON_VENV -c "from llama_cpp import Llama; print('Metal acceleration available' if Llama.contains_metal() else 'Metal not enabled')" 2>&1 | grep -q "Metal acceleration available"; then
                    print_info "Metal GPU acceleration confirmed ✓"
                else
                    print_warning "Metal acceleration may not be enabled."
                fi
            elif [ "$HAS_NVIDIA" = true ]; then
                # Check if CUDA is available in llama_cpp
                if $PYTHON_VENV -c "from llama_cpp import Llama; print('CUDA acceleration available' if Llama.contains_cuda() else 'CUDA not enabled')" 2>&1 | grep -q "CUDA acceleration available"; then
                    print_info "CUDA GPU acceleration confirmed ✓"
                else
                    print_warning "CUDA acceleration may not be enabled."
                fi
            fi
        else
            print_error "llama-cpp-python installed but failed to import. Local models may not function correctly."
            # Continue execution despite error - don't return/exit
        fi

        # Check hugging face login
        print_info "Checking Hugging Face authentication..."
        $PIP_VENV install -q huggingface_hub # Ensure installed
        local HUGGINGFACE_CLI_CMD_PATH="$VENV_DIR/bin/huggingface-cli"
        if ! $HUGGINGFACE_CLI_CMD_PATH whoami &>/dev/null; then
            print_warning "Login to Hugging Face required..."
            $HUGGINGFACE_CLI_CMD_PATH login
        else print_info "Already authenticated with Hugging Face ✓"; fi
        # local HUGGINGFACE_CLI_CMD="huggingface-cli" # Reset in _huggingface_download_gguf if needed

        # Check models directory
        if [ ! -d "$MODELS_DIR" ]; then mkdir -p "$MODELS_DIR" && print_info "Created models directory ✓"; else print_info "Models directory exists ✓"; fi
    else
        # Cloud-only setup steps (if any were needed here)
        print_info "Skipping local model setup actions (llama install, HF login, models dir) ✓"
        LLAMA_CPP_PYTHON_FLAGS="N/A (Cloud models selected)"
    fi

    # --- Step 7: Setup Patchright Browser (only Chrome) ---
    print_section_header "PATCHRIGHT BROWSER"
    print_info "Setting up Patchright with Chrome browser..."

    if $PYTHON_VENV -c "import patchright" &>/dev/null; then
        print_info "Installing Chrome browser for Patchright..."
        
        # Use timeout to prevent hanging during installation (use gtimeout on Mac if available)
        if command -v gtimeout &> /dev/null; then
            TIMEOUT_CMD="gtimeout"
        elif command -v timeout &> /dev/null; then
            TIMEOUT_CMD="timeout"
        else
            print_warning "Neither timeout nor gtimeout available. Installation might hang indefinitely."
            TIMEOUT_CMD=""
        fi
        
        # Create a temporary file to capture the output
        TEMP_OUTPUT=$(mktemp)
        
        # Temporarily disable exit on error for this section
        set +e
        
        if [ -n "$TIMEOUT_CMD" ]; then
            $TIMEOUT_CMD 300 $VENV_DIR/bin/patchright install chrome > "$TEMP_OUTPUT" 2>&1
            INSTALL_RESULT=$?
        else
            $VENV_DIR/bin/patchright install chrome > "$TEMP_OUTPUT" 2>&1
            INSTALL_RESULT=$?
        fi
        
        # Re-enable exit on error
        set -e
        
        # Check if output contains the "already installed" message
        if grep -q "\"chrome\" is already installed on the system" "$TEMP_OUTPUT"; then
            print_info "Chrome is already installed on your system ✓"
        elif [ $INSTALL_RESULT -eq 0 ]; then
            print_info "Patchright Chrome installation completed successfully ✓"
        else
            print_warning "Patchright Chrome installation timed out or failed. You may need to run it manually."
            cat "$TEMP_OUTPUT" | tail -10  # Show the last few lines of output for debugging
        fi
        
        # Clean up temp file
        rm -f "$TEMP_OUTPUT"
        
        print_info "Patchright Chrome setup complete ✓"
    else
        print_warning "Patchright not found or failed to import. Web automation may be detected."
    fi

    # --- Step 8: MCP Tool Setup (on-demand installation) ---
    print_section_header "MCP TOOLS"
    print_info "Setting up MCP tools..."
    MCP_JSON_FILE="$VENV_DIR/lib/python*/site-packages/mcp/tools.json"

    # Check if Node.js and npm are available for YouTube transcript tool (don't install, just check)
    if command -v node &> /dev/null && command -v npm &> /dev/null; then
        print_info "Node.js and npm available for YouTube transcript tool ✓"
        HAS_NODE=true
    else
        print_warning "Node.js or npm not found. YouTube transcript tool will be set up on first use."
        HAS_NODE=false
    fi

    # Check if uvx is available for pubmedmcp tool
    if command -v uvx &> /dev/null; then
        print_info "uvx available for pubmedmcp tool ✓"
        HAS_UVX=true
    else
        print_warning "uvx not found. The pubmedmcp tool may not function properly."
        HAS_UVX=false
    fi

    # Update MCP tools.json if it exists
    if [ -n "$(find $VENV_DIR -path "$MCP_JSON_FILE" 2>/dev/null)" ]; then
        MCP_JSON_PATH=$(find $VENV_DIR -path "$MCP_JSON_FILE" 2>/dev/null | head -n 1)
        print_info "Checking MCP configuration at $MCP_JSON_PATH"
        
        # Check if jq is available for JSON manipulation
        if command -v jq &> /dev/null; then
            # Check if YouTube Transcript tool is missing and needs to be added
            if ! grep -q "youtube-transcript" "$MCP_JSON_PATH"; then
                print_info "Adding YouTube Transcript MCP tool to configuration..."
                jq '. += [{
                    "name": "youtube-transcript",
                    "description": "YouTube Transcript Extractor",
                    "function": "get_transcript",
                    "requirements": "youtube-transcript-api"
                }]' "$MCP_JSON_PATH" > "$MCP_JSON_PATH.tmp" && mv "$MCP_JSON_PATH.tmp" "$MCP_JSON_PATH"
            fi
            
            # Note about pubmedmcp
            print_info "pubmedmcp will be used as an MCP server via the installed package"
        else
            print_warning "jq not available. MCP tools will need to be configured manually."
        fi
        
        print_info "MCP tools configuration complete ✓"
    else
        print_warning "MCP configuration file not found. Tools will be configured on first use."
    fi

    # --- Step 9: Model Configuration ---
    echo "DEBUG: Entering Step 9: Model Configuration"
    # Call function directly, it will set _selected_model_result
    echo "DEBUG: Calling select_summarizer_model..."
    select_summarizer_model 
    SELECTED_MODEL_FILENAME="$_selected_model_result" # Assign result
    echo "DEBUG: Returned from select_summarizer_model. Filename: ${SELECTED_MODEL_FILENAME}"

    # Call function directly, it will set _selected_model_result
    echo "DEBUG: Calling select_primary_model..."
    select_primary_model 
    SELECTED_PRIMARY_MODEL="$_selected_model_result" # Assign result
    echo "DEBUG: Returned from select_primary_model. Model: ${SELECTED_PRIMARY_MODEL}"
    echo "DEBUG: Exiting Step 9: Model Configuration"

    # --- Step 10: Conditional Download & Optimization for Primary Model ---
    # Check if a local model was selected AND local models are enabled
    if [ "$USE_LOCAL_MODELS" = true ] && [[ "$SELECTED_PRIMARY_MODEL" == *".gguf" ]]; then
        print_section_header "LOCAL PRIMARY MODEL DOWNLOAD & OPTIMIZATION"

        # Determine which download function to use
        local PRIMARY_DOWNLOAD_FUNCTION=""
        case "$SELECTED_PRIMARY_MODEL" in
            *"Llama-3.3-70B-Instruct"*)
                PRIMARY_DOWNLOAD_FUNCTION="download_llama33_70b_gguf"
                ;;
            *"Qwen2.5-72B-Instruct"*)
                PRIMARY_DOWNLOAD_FUNCTION="download_qwen_72b_gguf"
                ;;
            *"DeepSeek-R1-Distill-Llama-70B"*)
                PRIMARY_DOWNLOAD_FUNCTION="download_deepseek_70b_gguf"
                ;;
            *"xwin-lm-70b-v0.1"*)
                PRIMARY_DOWNLOAD_FUNCTION="download_xwin_70b_gguf"
                ;;
            *)
                print_error "Unknown local model selected for download: $SELECTED_PRIMARY_MODEL"
                # Optionally exit or just skip download
                # exit 1 
                ;;
        esac

        # Download the selected model if a function was determined
        if [ -n "$PRIMARY_DOWNLOAD_FUNCTION" ]; then
            local MODEL_TARGET_DIR="$MODELS_DIR"
            # Use specific filename for marker to avoid conflicts if quant changes
            local MODEL_FILENAME_ONLY=$(basename "$SELECTED_PRIMARY_MODEL") 
            local PRIMARY_MARKER_FILE="${MODEL_TARGET_DIR}/.${MODEL_FILENAME_ONLY}.download_complete"
            
            if [ -f "$PRIMARY_MARKER_FILE" ]; then
                print_info "$SELECTED_PRIMARY_MODEL already downloaded ✓"
            else
                print_info "Downloading $SELECTED_PRIMARY_MODEL using function $PRIMARY_DOWNLOAD_FUNCTION..."
                # Ensure VENV_DIR and HUGGINGFACE_CLI_CMD are available to the download function
                if $PRIMARY_DOWNLOAD_FUNCTION "$MODEL_TARGET_DIR"; then 
                    print_info "Download succeeded ✓"
                else 
                    print_error "Download failed for $SELECTED_PRIMARY_MODEL."
                    # Don't create marker if download failed
                    rm -f "$PRIMARY_MARKER_FILE" 
                fi
            fi
        else
            print_warning "Could not determine download function for $SELECTED_PRIMARY_MODEL. Skipping download."
        fi

        # Hardware optimization (GPU layers) - This part remains the same
        OPTIMAL_GPU_LAYERS=$(get_optimal_gpu_layers "$SELECTED_PRIMARY_MODEL" "" "" "$HAS_NVIDIA") # Pass HAS_NVIDIA
        CURRENT_GPU_LAYERS=$(grep -E "^N_GPU_LAYERS:" config.yaml | awk '{print $2}' || echo "0")
        echo ""; print_info "Hardware Optimization for Primary Model ($SELECTED_PRIMARY_MODEL)"
        echo "Current GPU layers: ${CURRENT_GPU_LAYERS:-0}"; echo "Recommended: $OPTIMAL_GPU_LAYERS"

        local UPDATE_GPU_LAYERS="Y"
        # Quick install mode: Auto-update GPU layers
        if [ "$QUICK_INSTALL" = true ]; then
            print_info "Auto-updating N_GPU_LAYERS to recommended value: $OPTIMAL_GPU_LAYERS"
        else
            # Interactive mode: Ask to update GPU layers
            printf "%s %s? [%s%s%s/%s] %s" "${BLUE}Update N_GPU_LAYERS to" "$OPTIMAL_GPU_LAYERS" "${GREEN}" "Y" "${BLUE}" "n" "${RESET}" # Use printf for consistent prompting
            read UPDATE_GPU_LAYERS
            echo "" # Add newline
            UPDATE_GPU_LAYERS=${UPDATE_GPU_LAYERS:-Y} # Default to Y if empty
        fi

        if [[ "$UPDATE_GPU_LAYERS" == "y" || "$UPDATE_GPU_LAYERS" == "Y" ]]; then
            update_gpu_layers $OPTIMAL_GPU_LAYERS # Function updates config
            print_info "N_GPU_LAYERS updated."
        else
            OPTIMAL_GPU_LAYERS=$CURRENT_GPU_LAYERS # Use current value for summary
            print_info "Keeping N_GPU_LAYERS at $CURRENT_GPU_LAYERS."
        fi
    # Check if cloud model was selected OR local models were disabled entirely
    elif [ "$USE_LOCAL_MODELS" = false ] || [[ "$SELECTED_PRIMARY_MODEL" == *"cloud"* ]]; then
        print_info "Setting N_GPU_LAYERS to 0 (Cloud or no local primary model)"
        update_gpu_layers 0
        OPTIMAL_GPU_LAYERS=0 # For summary
    fi

    # --- Step 11: Configure Additional Options ---
    configure_additional_options "$SELECTED_PRIMARY_MODEL"
    
    # --- Step 12: API Key / .env Setup ---
    print_section_header "API KEY CONFIGURATION"
    if [ ! -f ".env" ] && [ -f ".env.example" ]; then
        print_info "Creating .env from .env.example..."
        cp .env.example .env
        print_info "IMPORTANT: Add required API Keys to .env file."
    elif [ ! -f ".env" ]; then print_warning ".env AND .env.example not found."; fi
    # Check keys based on final primary model type from config
    local FINAL_PRIMARY_MODEL_TYPE=$(grep -E "^PRIMARY_MODEL_TYPE:" "$CONFIG_YAML" | awk '{print $2}' | sed 's/\"//g')
    if [[ "$FINAL_PRIMARY_MODEL_TYPE" == "claude" ]] ; then check_and_set_env_var "ANTHROPIC_API_KEY" "Claude models" "https://console.anthropic.com/settings/keys" true; fi
    # Check Gemini key if summarizer OR primary is Gemini
    local FINAL_SUMMARIZER=$(grep -E "^SUMMARIZER_MODEL:" "$CONFIG_YAML" | awk '{print $2}' | sed 's/\"//g')
    if [[ "$FINAL_SUMMARIZER" == *"gemini"* ]] || [[ "$FINAL_PRIMARY_MODEL_TYPE" == "gemini" ]]; then check_and_set_env_var "GEMINI_API_KEY" "Gemini models" "https://ai.google.dev/tutorials/setup" true; fi

    # --- Step 13: Make run script executable ---
    chmod +x run.sh || true

    # --- Step 14: Display final configuration summary ---
    # Read final quant level from config for summary
    if [[ "$SELECTED_PRIMARY_MODEL" == *"$MODEL_LLAMA3_70B"* ]]; then
         PRIMARY_QUANT_LEVEL=$(grep -E "^LOCAL_MODEL_QUANT_LEVEL:" "$CONFIG_YAML" | awk '{print $2}' | sed 's/\"//g')
    fi
    # Pass necessary info to summary function if not using globals
    display_configuration_summary # Assumes variables are accessible in scope

    # --- Step 9: Install Patchright Browser (Chrome only) ---
    print_section_header "PATCHRIGHT BROWSER"
    print_info "Setting up Patchright browser (Chrome only)..."

    # Determine timeout command based on system
    TIMEOUT_CMD="timeout"
    if command -v gtimeout &> /dev/null; then
        TIMEOUT_CMD="gtimeout"
    elif ! command -v timeout &> /dev/null; then
        print_warning "Neither 'gtimeout' nor 'timeout' command found. Installation might hang indefinitely."
    fi

    # Install Chrome browser only with timeout to prevent hanging
    print_info "Installing Chrome browser..."
    if ! $TIMEOUT_CMD 300 npx playwright install chrome; then
        if [ $? -eq 124 ]; then
            print_warning "Chrome installation timed out after 300 seconds."
        else
            print_warning "Chrome installation failed. You may need to install it manually or it may already be installed."
        fi
    else
        print_info "Chrome browser installed successfully ✓"
    fi

    # --- Step 10: Check MCP Tool Setup ---
    print_section_header "MCP TOOL SETUP"
    print_info "Verifying MCP tool configuration..."

    # Instead of checking MCP version directly (which can hang), verify configuration files
    if [ -f "mcp.json" ]; then
        print_info "Found mcp.json configuration file ✓"
        
        # Check if the file contains the necessary tool configurations
        if grep -q "pubmedmcp" "mcp.json" && grep -q "youtube-transcript" "mcp.json"; then
            print_info "MCP tools are properly configured in mcp.json ✓"
        else
            print_warning "mcp.json exists but may be missing some tool configurations."
            print_info "You can manually verify the configuration in mcp.json."
        fi
        
        # Check if uvx is available for pubmedmcp
        if command -v uvx &> /dev/null; then
            print_info "uvx is available for pubmedmcp tool ✓"
        else
            print_warning "uvx not found. Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
        fi
        
        # Check if npx is available for youtube-transcript
        if command -v npx &> /dev/null; then
            print_info "npx is available for youtube-transcript tool ✓"
        else
            print_warning "npx not found. Install Node.js to use the YouTube transcript tool."
        fi
    else
        print_warning "mcp.json configuration file not found in the current directory."
        print_info "You may need to create or copy mcp.json to enable MCP tools."
    fi
}


# --- Final Execution ---
# Only the call to main should be here at the bottom
main
SETUP_EXIT_CODE=$?

# --- Run the Application ---
if [ $SETUP_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "${GREEN}${BOLD}Setup completed successfully! Starting the application now...${RESET}"
    echo ""

    # Ask user if they want to run the application now
    START_APP="Y" # Removed 'local' keyword as this is outside a function
    # Use -n 1 to read a single character, often more reliable
    read -n 1 -p "${YELLOW}Do you want to start the application now? [${GREEN}Y${YELLOW}/n] ${RESET}" START_APP
    echo "" # Add newline after single character read
    START_APP=${START_APP:-Y} # Default to Y if empty (user pressed Enter)

    if [[ "$START_APP" == "y" || "$START_APP" == "Y" ]]; then
        # Activate virtual environment for the run script if it's not already active
        if [ -z "$VIRTUAL_ENV" ]; then
            echo "${YELLOW}Activating virtual environment for application...${RESET}"
            source "${VENV_DIR}/bin/activate"
        fi

        # Run the application
        ./run.sh

        # Note: The run.sh script handles deactivation of the virtual environment
    else
        echo ""
        echo "${BLUE}You can start the application later with: ${CYAN}./run.sh${RESET}"
    fi
else
    echo ""
    echo "${RED}${BOLD}Setup had some issues. Please resolve them before running the application.${RESET}"
    echo "${BLUE}After fixing any issues, you can run the application with: ${CYAN}./run.sh${RESET}"
    echo ""
fi