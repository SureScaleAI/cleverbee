#!/bin/bash

# Ensure we are in the script's directory (project root)
cd "$(dirname "$0")"

# --- Argument Parsing ---
NO_CACHE_FLAG=false
CHAINLIT_ARGS="" # Store arguments for chainlit run

while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --no-cache)
      NO_CACHE_FLAG=true
      shift # past argument
      ;;
    *) # unknown option, assume it's for chainlit run
      # Quote the argument to handle spaces correctly
      CHAINLIT_ARGS+="$1"" " 
      shift # past argument
      ;;
  esac
done
# --- End Argument Parsing ---

# Check if we're already in a virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
  # Activate virtual environment if it exists (optional, adjust path if needed)
  ACTIVATED_VENV=false
  if [ -d "venv" ]; then
    echo "Activating virtual environment (venv)..."
    source venv/bin/activate
    ACTIVATED_VENV=true
  elif [ -d ".venv" ]; then
    echo "Activating virtual environment (.venv)..."
    source .venv/bin/activate
    ACTIVATED_VENV=true
  fi
else
  echo "Already in virtual environment: $VIRTUAL_ENV"
  ACTIVATED_VENV=false
fi

# Check if chainlit is installed within the environment
if ! command -v chainlit &> /dev/null
then
    echo "Warning: 'chainlit' command not found in the current environment."
    
    # Check if requirements.txt exists
    if [ -f "requirements.txt" ]; then
        echo "Attempting to install dependencies from requirements.txt..."
        
        # Check if pip is available
        if ! command -v pip &> /dev/null
        then
            echo "Error: 'pip' command not found. Cannot install dependencies."
            if [ "$ACTIVATED_VENV" = true ] && type deactivate &>/dev/null; then deactivate; fi
            exit 1
        fi

        # Run pip install
        pip install -r requirements.txt
        
        # Check the exit code of pip install
        if [ $? -ne 0 ]; then
            echo "Error: 'pip install -r requirements.txt' failed. Please check the output above and fix any issues manually."
            if [ "$ACTIVATED_VENV" = true ] && type deactivate &>/dev/null; then deactivate; fi
            exit 1
        else
            echo "Dependencies installed successfully."
            # Re-check if chainlit is available NOW
            if ! command -v chainlit &> /dev/null
            then
                 # Attempt to find chainlit executable path directly (might help in some cases)
                 CHAINLIT_PATH=$(which chainlit)
                 if [ -z "$CHAINLIT_PATH" ] || ! command -v chainlit &> /dev/null ; then
                     echo "Error: 'chainlit' command still not found after installing dependencies."
                     echo "Please check requirements.txt and ensure 'chainlit' is listed."
                     echo "Also check pip install output for errors."
                     if [ "$ACTIVATED_VENV" = true ] && type deactivate &>/dev/null; then deactivate; fi
                     exit 1
                 fi
                 # If which found it, try proceeding (though command -v should ideally work)
                 echo "'chainlit' command found after install (using which). Proceeding..."
            else
                 echo "'chainlit' command is now available. Proceeding to run the app..."
            fi
            # Chainlit is now available (or assumed available if `which` found it), script will continue
        fi
    else
        echo "Error: requirements.txt not found in the project root."
        echo "Cannot install dependencies. Please create requirements.txt or install manually."
        if [ "$ACTIVATED_VENV" = true ] && type deactivate &>/dev/null; then deactivate; fi
        exit 1
    fi
fi

# Check for critical dependencies
echo "Checking for critical dependencies..."
if ! python -c "import fitz; print(f'PyMuPDF successfully imported as fitz')" &> /dev/null; then
    echo "Warning: 'PyMuPDF' module not found. Installing it now..."
    pip install pymupdf==1.23.8
    
    if [ $? -ne 0 ]; then
        echo "Error: Failed to install PyMuPDF. Please install it manually with: pip install pymupdf==1.23.8"
        if [ "$ACTIVATED_VENV" = true ] && type deactivate &>/dev/null; then deactivate; fi
        exit 1
    else
        echo "PyMuPDF installed successfully."
    fi
fi

# --- Cache Deletion Logic ---
DB_FILE=".langchain.db"
if [ "$NO_CACHE_FLAG" = true ]; then
    if [ -f "$DB_FILE" ]; then
        echo "Deleting cache file: $DB_FILE"
        rm -f "$DB_FILE"
        if [ $? -ne 0 ]; then
            echo "Warning: Failed to delete $DB_FILE. Proceeding anyway."
        fi
    else
        echo "Cache file $DB_FILE not found. Skipping deletion."
    fi
fi
# --- End Cache Deletion Logic ---

# If chainlit command was found (either initially or after install), run the application
# -w flag enables auto-reloading, remove if not desired for production
# Use eval to correctly handle quoted arguments in CHAINLIT_ARGS
echo "Starting Chainlit app (src/chainlit_app.py) with args: $CHAINLIT_ARGS..."
eval chainlit run src/chainlit_app.py $CHAINLIT_ARGS

# Deactivate virtual environment (if applicable)
# Check if deactivate function exists before calling
if [ "$ACTIVATED_VENV" = true ] && type deactivate &>/dev/null; then
  echo "Deactivating virtual environment."
  deactivate
fi

echo "Chainlit app stopped." 