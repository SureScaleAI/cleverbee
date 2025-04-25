# Python AI Research Agent - Windows Setup Script (BETA)
# This script is experimental and may require adjustments for your specific Windows environment

# Function to display colorful messages
function Write-ColorOutput {
    param(
        [Parameter(Mandatory=$true)]
        [string]$Message,
        
        [Parameter(Mandatory=$false)]
        [string]$ForegroundColor = "White"
    )
    
    Write-Host $Message -ForegroundColor $ForegroundColor
}

# Function to create section headers
function Show-SectionHeader {
    param(
        [Parameter(Mandatory=$true)]
        [string]$Title
    )
    
    Write-Host ""
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
    Write-Host "  $Title" -ForegroundColor White
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
    Write-Host ""
}

# Function to check if a command exists
function Test-Command {
    param(
        [Parameter(Mandatory=$true)]
        [string]$Command
    )
    
    return (Get-Command $Command -ErrorAction SilentlyContinue)
}

# Function to check prerequisites
function Check-Prerequisites {
    Show-SectionHeader "CHECKING PREREQUISITES"
    
    $missingPrerequisites = $false
    
    # Check for Python
    if (Test-Command "python") {
        $pythonVersion = (python --version 2>&1).ToString()
        Write-ColorOutput "Python found: $pythonVersion" "Green"
        
        # Check Python version
        if ($pythonVersion -match "Python (\d+)\.(\d+)\.(\d+)") {
            $major = [int]$Matches[1]
            $minor = [int]$Matches[2]
            
            if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 8)) {
                Write-ColorOutput "Python 3.8+ is recommended, you have $pythonVersion" "Yellow"
                Write-Host "The script might work with your version, but some features may be unavailable."
            }
        }
    } else {
        Write-ColorOutput "Python 3 is required but not found on your system." "Red"
        Write-Host "Please install Python 3 from https://www.python.org/downloads/"
        $missingPrerequisites = $true
    }
    
    # Check for git
    if (Test-Command "git") {
        Write-ColorOutput "git found ✓" "Green"
    } else {
        Write-ColorOutput "git is required but not found on your system." "Red"
        Write-Host "Please install git from https://git-scm.com/download/win"
        $missingPrerequisites = $true
    }
    
    # Check for Node.js and npm
    if (Test-Command "node" -and Test-Command "npm") {
        $nodeVersion = (node --version).Trim()
        Write-ColorOutput "Node.js found: $nodeVersion ✓" "Green"
    } else {
        Write-ColorOutput "Node.js and npm are recommended but not found." "Yellow"
        Write-Host "Some features may not work. Install from https://nodejs.org/"
    }
    
    # Check for config.yaml
    if (Test-Path "config.yaml") {
        Write-ColorOutput "config.yaml found ✓" "Green"
    } elseif (Test-Path "config.yaml.example") {
        Write-ColorOutput "config.yaml not found. Creating from example..." "Yellow"
        Copy-Item "config.yaml.example" "config.yaml"
        Write-ColorOutput "Created config.yaml from example ✓" "Green"
    } else {
        Write-ColorOutput "config.yaml not found and no example available." "Red"
        $missingPrerequisites = $true
    }
    
    # Check for mcp.json
    if (Test-Path "mcp.json") {
        Write-ColorOutput "mcp.json found ✓" "Green"
    } else {
        Write-ColorOutput "mcp.json not found." "Red"
        $missingPrerequisites = $true
    }
    
    # Hardware detection for NVIDIA GPUs (Windows-specific approach)
    $hasNvidia = $false
    $nvidiaSmiPath = "C:\Windows\System32\nvidia-smi.exe"
    
    if (Test-Path $nvidiaSmiPath) {
        try {
            $nvidiaSmiOutput = & $nvidiaSmiPath
            $hasNvidia = $true
            
            # Try to extract VRAM info
            $vramPattern = "(?i)(\d+)MiB\s+/\s+(\d+)MiB"
            if ($nvidiaSmiOutput -match $vramPattern) {
                $totalVramMB = [int]$Matches[2]
                $vramGB = [math]::Round($totalVramMB / 1024, 1)
                Write-ColorOutput "NVIDIA GPU detected with ${vramGB}GB VRAM" "Green"
            } else {
                Write-ColorOutput "NVIDIA GPU detected but couldn't determine VRAM size" "Yellow"
            }
        } catch {
            Write-ColorOutput "Error running nvidia-smi: $_" "Red"
        }
    } else {
        Write-ColorOutput "No NVIDIA GPU detected." "Yellow"
    }
    
    # Return prerequisite check status
    return @{
        Success = -not $missingPrerequisites
        HasNvidia = $hasNvidia
    }
}

# Function to create and setup Python virtual environment
function Setup-VirtualEnvironment {
    Show-SectionHeader "PYTHON VIRTUAL ENVIRONMENT"
    
    $venvDir = "venv"
    
    if (Test-Path $venvDir) {
        Write-ColorOutput "Virtual environment '$venvDir' already exists ✓" "Green"
    } else {
        Write-ColorOutput "Creating Python venv..." "Green"
        try {
            python -m venv $venvDir
            Write-ColorOutput "Venv created ✓" "Green"
        } catch {
            Write-ColorOutput "Venv creation failed: $_" "Red"
            return $false
        }
    }
    
    # Upgrade pip
    Write-ColorOutput "Upgrading pip..." "Green"
    & "$venvDir\Scripts\python.exe" -m pip install --upgrade pip | Out-Null
    
    return $true
}

# Function to install base dependencies
function Install-BaseDependencies {
    Show-SectionHeader "BASE DEPENDENCIES"
    Write-ColorOutput "Installing Python dependencies..." "Green"
    
    try {
        & "venv\Scripts\pip.exe" install -r requirements.txt
        Write-ColorOutput "Base dependencies installed ✓" "Green"
        return $true
    } catch {
        Write-ColorOutput "Failed to install dependencies: $_" "Red"
        return $false
    }
}

# Function to set up Playwright
function Setup-Playwright {
    Show-SectionHeader "PLAYWRIGHT SETUP"
    
    # Install Playwright
    Write-ColorOutput "Installing Chrome browser for Playwright..." "Green"
    try {
        # Use Start-Process to avoid blocking the script
        Start-Process -FilePath "venv\Scripts\playwright.exe" -ArgumentList "install", "chrome" -Wait -NoNewWindow
        Write-ColorOutput "Playwright Chrome installation completed ✓" "Green"
        
        # Configure anti-detection settings
        Configure-PlaywrightForStealth
    } catch {
        Write-ColorOutput "Playwright Chrome installation failed: $_" "Yellow"
        Write-Host "You may need to run it manually."
    }
}

# Function to configure Playwright with stealth settings to prevent detection
function Configure-PlaywrightForStealth {
    Write-ColorOutput "Configuring Playwright to avoid detection..." "Green"
    
    # Create a helper module for anti-detection
    $stealthJsContent = @"
// playwright-stealth.js - Anti-detection configuration for Playwright
module.exports = {
    // Browser launch options to prevent detection
    launchOptions: {
        headless: false, // Always use headed mode on Windows to prevent detection
        args: [
            '--disable-blink-features=AutomationControlled',
            '--disable-extensions',
            '--disable-component-extensions-with-background-pages',
            '--disable-infobars',
            '--no-default-browser-check',
            '--no-first-run',
            '--disable-features=IsolateOrigins,site-per-process',
            '--window-size=1920,1080'
        ]
    },
    
    // Browser context options
    contextOptions: {
        userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        viewport: { width: 1280, height: 720 },
        bypassCSP: true,
        ignoreHTTPSErrors: true,
        permissions: ['geolocation'],
        locale: 'en-US',
        timezoneId: 'America/New_York',
        deviceScaleFactor: 1
    },
    
    // JavaScript to inject that helps avoid detection
    evasions: `
        // Remove webdriver property
        delete Object.getPrototypeOf(navigator).webdriver;
        
        // Overwrite the plugins to report a normal amount
        if (navigator.plugins) {
            Object.defineProperty(navigator, 'plugins', {
                get: () => Array(3).fill().map(() => {
                    return { length: 1 }
                })
            });
        }
        
        // Overwrite permissions API
        if (navigator.permissions) {
            const originalPermissions = navigator.permissions;
            navigator.permissions = {
                ...originalPermissions,
                query: async (parameters) => {
                    return { state: 'prompt', onchange: null };
                }
            };
        }
    `
};
"@

    # Create an example implementation to help users
    $exampleJsContent = @"
// example-stealth-usage.js - Example of using the anti-detection configuration
const { chromium } = require('playwright');
const stealthConfig = require('./playwright-stealth.js');

(async () => {
  // Launch browser with anti-detection settings
  const browser = await chromium.launch(stealthConfig.launchOptions);
  
  // Create context with stealth settings
  const context = await browser.newContext(stealthConfig.contextOptions);
  
  // Add script to avoid detection
  await context.addInitScript(stealthConfig.evasions);
  
  // Open a new page
  const page = await context.newPage();
  
  // Test if anti-detection is working
  try {
    console.log('Testing navigation to bot detection site...');
    await page.goto('https://bot.sannysoft.com', { timeout: 60000 });
    console.log('Page loaded successfully! Taking screenshot...');
    await page.screenshot({ path: 'bot-test.png' });
    console.log('Screenshot saved as bot-test.png');
    
    console.log('Press Enter to close browser...');
    // Keep browser open to inspect results
    require('readline').createInterface({
      input: process.stdin,
      output: process.stdout
    }).question('', () => {
      browser.close();
      process.exit(0);
    });
  } catch (error) {
    console.error('Error during navigation:', error);
    await browser.close();
  }
})();
"@

    # Create directory for the modules if it doesn't exist
    $modulesDir = ".\playwright-tools"
    if (-not (Test-Path $modulesDir)) {
        New-Item -ItemType Directory -Path $modulesDir | Out-Null
    }
    
    # Write the files
    $stealthJsContent | Out-File -FilePath "$modulesDir\playwright-stealth.js" -Encoding utf8
    $exampleJsContent | Out-File -FilePath "$modulesDir\example-stealth-usage.js" -Encoding utf8
    
    # Create a simple batch file to run the test
    $batchContent = @"
@echo off
echo Running Playwright with anti-detection settings...
cd %~dp0
venv\Scripts\node.exe playwright-tools\example-stealth-usage.js
"@
    $batchContent | Out-File -FilePath "test-stealth.bat" -Encoding utf8
    
    # Update run.ps1 to include the stealth configurations
    $runContent = Get-Content "run.ps1" -Raw
    $modified = $false
    
    if (-not $runContent.Contains("# Anti-detection settings")) {
        $insertPoint = $runContent.IndexOf("Write-Host `"Starting application...`"")
        if ($insertPoint -gt 0) {
            $newContent = $runContent.Insert($insertPoint, @"
    # Anti-detection settings - these are required for Windows to avoid headless detection
    Write-Host "Configuring environment variables to prevent headless detection..."
    `$env:PLAYWRIGHT_HEADLESS_MODE = "false" # Always use headed mode
    `$env:PLAYWRIGHT_DISABLE_AUTOMATION = "true" # Try to disable automation flags
    
"@)
            $newContent | Set-Content "run.ps1" -Force
            $modified = $true
        }
    }
    
    Write-ColorOutput "Playwright anti-detection configuration created successfully!" "Green"
    Write-Host "To test if websites can detect your Playwright automation:"
    Write-Host "1. Run: .\test-stealth.bat"
    Write-Host "2. Check the displayed bot detection test results"
    Write-Host ""
    Write-Host "NOTE: On Windows, always use non-headless mode to prevent detection"
    Write-Host "      This is automatically configured in the updated run.ps1 script"
}

# Function to prompt for API keys
function Setup-ApiKeys {
    Show-SectionHeader "API KEY CONFIGURATION"
    
    # Create .env file if it doesn't exist
    if (-not (Test-Path ".env") -and (Test-Path ".env.example")) {
        Write-ColorOutput "Creating .env from .env.example..." "Green"
        Copy-Item ".env.example" ".env"
    }
    
    # Function to check and set an API key
    function Check-ApiKey {
        param(
            [string]$KeyName,
            [string]$Description,
            [string]$KeyUrl,
            [bool]$IsRequired = $false
        )
        
        $envContent = Get-Content ".env" -ErrorAction SilentlyContinue
        $keyExists = $false
        $keyHasValue = $false
        $isPlaceholder = $false
        $currentValue = ""
        
        # Check if key exists and has a value
        foreach ($line in $envContent) {
            if ($line -match "^$KeyName=(.*)$") {
                $keyExists = $true
                $currentValue = $Matches[1].Trim('"').Trim("'")
                
                if (-not [string]::IsNullOrWhiteSpace($currentValue)) {
                    $keyHasValue = $true
                    if ($currentValue -eq "api_key_here") {
                        $isPlaceholder = $true
                    }
                }
                break
            }
        }
        
        if ($keyHasValue -and -not $isPlaceholder) {
            # Mask the key for display
            $length = $currentValue.Length
            $maskedValue = ""
            
            if ($length -le 8) {
                $maskedValue = "*" * $length
            } else {
                $firstPart = $currentValue.Substring(0, 4)
                $lastPart = $currentValue.Substring($length - 4)
                $middlePart = "*" * ($length - 8)
                $maskedValue = "${firstPart}${middlePart}${lastPart}"
            }
            
            Write-ColorOutput "$KeyName found: $maskedValue" "Cyan"
            Write-Host "Current $KeyName appears to be set."
            
            # Ask to update the key
            $updateKey = Read-Host "Update this key? [y/N]"
            if ($updateKey -eq "y" -or $updateKey -eq "Y") {
                $secureString = Read-Host "Enter new value for $KeyName" -AsSecureString
                $newKey = [System.Runtime.InteropServices.Marshal]::PtrToStringAuto(
                    [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureString)
                )
                
                if (-not [string]::IsNullOrWhiteSpace($newKey)) {
                    # Update the key in .env
                    $newEnvContent = @()
                    foreach ($line in $envContent) {
                        if ($line -match "^$KeyName=") {
                            $newEnvContent += "$KeyName=`"$newKey`""
                        } else {
                            $newEnvContent += $line
                        }
                    }
                    $newEnvContent | Set-Content ".env"
                    Write-ColorOutput "$KeyName has been updated in .env ✓" "Green"
                } else {
                    Write-ColorOutput "Empty key provided, keeping existing value." "Yellow"
                }
            } else {
                Write-ColorOutput "Keeping existing $KeyName ✓" "Green"
            }
        } else {
            # Key is missing, empty, or contains placeholder
            if ($keyExists -and $isPlaceholder) {
                Write-ColorOutput "$KeyName found in .env but contains placeholder value." "Yellow"
            } elseif ($keyExists) {
                Write-ColorOutput "$KeyName found in .env but appears to be empty." "Yellow"
            } else {
                Write-ColorOutput "$KeyName not found in .env." "Yellow"
            }
            
            Write-Host "This key is needed for: $Description"
            Write-Host "You can obtain one here: $KeyUrl"
            
            # Prompt for the key
            $secureString = Read-Host "Enter value for $KeyName" -AsSecureString
            $newKey = [System.Runtime.InteropServices.Marshal]::PtrToStringAuto(
                [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureString)
            )
            
            if (-not [string]::IsNullOrWhiteSpace($newKey)) {
                if ($keyExists) {
                    # Update existing key
                    $newEnvContent = @()
                    foreach ($line in $envContent) {
                        if ($line -match "^$KeyName=") {
                            $newEnvContent += "$KeyName=`"$newKey`""
                        } else {
                            $newEnvContent += $line
                        }
                    }
                    $newEnvContent | Set-Content ".env"
                } else {
                    # Add new key
                    Add-Content ".env" "$KeyName=`"$newKey`""
                }
                Write-ColorOutput "$KeyName has been set in .env ✓" "Green"
            } elseif ($IsRequired) {
                Write-ColorOutput "This API key is required. Please add it manually to .env later." "Red"
            } else {
                Write-ColorOutput "Empty key provided. Continuing without $KeyName." "Yellow"
            }
        }
    }
    
    # Check for all necessary API keys based on config
    $configYaml = Get-Content "config.yaml" -Raw
    
    # Check if GEMINI_API_KEY is needed (either for summarizer or primary model)
    if ($configYaml -match "SUMMARIZER_MODEL:\s*[""']?gemini" -or $configYaml -match "PRIMARY_MODEL_TYPE:\s*[""']?gemini") {
        Check-ApiKey "GEMINI_API_KEY" "Gemini models" "https://ai.google.dev/tutorials/setup" $true
    }
    
    # Check if ANTHROPIC_API_KEY is needed
    if ($configYaml -match "PRIMARY_MODEL_TYPE:\s*[""']?claude") {
        Check-ApiKey "ANTHROPIC_API_KEY" "Claude models" "https://console.anthropic.com/settings/keys" $true
    }
}

# Function to update config.yaml for model settings
function Update-ConfigFile {
    param(
        [bool]$HasNvidia
    )
    
    Show-SectionHeader "MODEL CONFIGURATION"
    
    # Determine default model based on hardware
    $defaultModel = "gemini"  # Default to cloud model
    
    if ($HasNvidia) {
        # Check VRAM
        $nvidiaSmiPath = "C:\Windows\System32\nvidia-smi.exe"
        $hasEnoughVram = $false
        
        try {
            $nvidiaSmiOutput = & $nvidiaSmiPath
            $vramPattern = "(?i)(\d+)MiB\s+/\s+(\d+)MiB"
            if ($nvidiaSmiOutput -match $vramPattern) {
                $totalVramMB = [int]$Matches[2]
                $vramGB = [math]::Round($totalVramMB / 1024, 1)
                
                # Check if enough VRAM for local models
                if ($vramGB -ge 24) {
                    $hasEnoughVram = $true
                    Write-ColorOutput "Detected ${vramGB}GB VRAM - sufficient for local models ✓" "Green"
                } else {
                    Write-ColorOutput "Detected ${vramGB}GB VRAM - insufficient for large local models" "Yellow"
                    Write-Host "Using cloud models instead."
                }
            }
        } catch {
            Write-ColorOutput "Error checking VRAM: $_" "Red"
        }
        
        # Use local models only if enough VRAM
        if ($hasEnoughVram) {
            Write-ColorOutput "Would you like to use local models? (requires downloading large models)" "Cyan"
            $useLocalModels = Read-Host "Enable local models? [y/N]"
            
            if ($useLocalModels -eq "y" -or $useLocalModels -eq "Y") {
                Write-ColorOutput "Local models enabled. Will need to be downloaded separately." "Green"
                $defaultModel = "local"
                
                # Update config.yaml for local models
                # This is a simplified version - would need to be expanded for full functionality
                (Get-Content "config.yaml") | ForEach-Object {
                    $_ -replace "PRIMARY_MODEL_TYPE:\s*[""']?\w+[""']?", "PRIMARY_MODEL_TYPE: `"local`"" `
                       -replace "N_GPU_LAYERS:\s*\d+", "N_GPU_LAYERS: 40" `
                       -replace "LOCAL_MODEL_NAME:\s*[""']?.*[""']?", "LOCAL_MODEL_NAME: `"Llama-3.3-70B-Instruct-Q4_K_M.gguf`"" `
                       -replace "LOCAL_MODEL_QUANT_LEVEL:\s*[""']?.*[""']?", "LOCAL_MODEL_QUANT_LEVEL: `"Q4_K_M`""
                } | Set-Content "config.yaml"
                
                Write-ColorOutput "Config updated for local models. You will need to download the models separately." "Yellow"
                Write-Host "Download from: https://huggingface.co/unsloth/Llama-3.3-70B-Instruct-GGUF"
            } else {
                $defaultModel = "gemini"
                Write-ColorOutput "Using cloud models (Gemini/Claude) instead." "Green"
            }
        }
    } else {
        Write-ColorOutput "No NVIDIA GPU detected. Using cloud models." "Yellow"
    }
    
    # Update config.yaml for cloud models if needed
    if ($defaultModel -eq "gemini") {
        (Get-Content "config.yaml") | ForEach-Object {
            $_ -replace "PRIMARY_MODEL_TYPE:\s*[""']?\w+[""']?", "PRIMARY_MODEL_TYPE: `"gemini`"" `
               -replace "SUMMARIZER_MODEL:\s*[""']?.*[""']?", "SUMMARIZER_MODEL: `"gemini-2.0-flash`"" `
               -replace "USE_LOCAL_SUMMARIZER_MODEL:\s*\w+", "USE_LOCAL_SUMMARIZER_MODEL: false" `
               -replace "CHUNK_SIZE:\s*\d+", "CHUNK_SIZE: 0"
        } | Set-Content "config.yaml"
        
        Write-ColorOutput "Config updated for Gemini cloud models. You will need to provide API keys." "Green"
    }
}

# Function to create run.ps1 script
function Create-RunScript {
    Show-SectionHeader "CREATE RUN SCRIPT"
    
    $runScriptContent = @"
# Python AI Research Agent - Windows Run Script
# Activate virtual environment and run the application

# Check for --no-cache flag
param (
    [switch]`$NoCache
)

# Remove cache if requested
if (`$NoCache -and (Test-Path ".langchain.db")) {
    Write-Host "Removing cache as requested..."
    Remove-Item ".langchain.db" -Force
}

# Activate virtual environment and run application
try {
    Write-Host "Activating virtual environment..."
    & "venv\Scripts\Activate.ps1"
    
    Write-Host "Starting application..."
    python main.py
} catch {
    Write-Host "Error running the application: `$_" -ForegroundColor Red
} finally {
    # Deactivate virtual environment if active
    if (`$env:VIRTUAL_ENV) {
        Write-Host "Deactivating virtual environment..."
        deactivate
    }
}
"@
    
    # Write the run script
    $runScriptContent | Set-Content "run.ps1"
    Write-ColorOutput "Created run.ps1 script ✓" "Green"
}

# Main execution
function main {
    Write-Host ""
    Write-Host "=================================================================" -ForegroundColor Cyan
    Write-Host "             Python AI Research Agent - Setup Script (Windows)   " -ForegroundColor Cyan
    Write-Host "=================================================================" -ForegroundColor Cyan
    Write-Host ""
    
    Write-ColorOutput "Welcome to the Deep Researcher Setup Script (Windows Beta)" "Cyan"
    Write-Host ""
    Write-Host "This script will set up your environment with the following components:"
    Write-Host "  • Python virtual environment"
    Write-Host "  • Required Python packages"
    Write-Host "  • Playwright web browser automation"
    Write-Host "  • API keys for AI services"
    Write-Host "  • Configuration for cloud models (local model setup is limited on Windows)"
    Write-Host ""
    Write-Host "IMPORTANT: This Windows script is in BETA and may require adjustments."
    Write-Host "For best results, consider using Windows Subsystem for Linux (WSL) with the bash script."
    Write-Host ""
    
    # Ask for confirmation to proceed
    $confirmation = Read-Host "Do you want to proceed with setup? [Y/n]"
    if ($confirmation -eq "n" -or $confirmation -eq "N") {
        Write-ColorOutput "Setup cancelled." "Yellow"
        return
    }
    
    # Check prerequisites
    $prereqResults = Check-Prerequisites
    if (-not $prereqResults.Success) {
        Write-ColorOutput "Some critical prerequisites are missing. Please install them and try again." "Red"
        return
    }
    
    # Setup virtual environment
    if (-not (Setup-VirtualEnvironment)) {
        Write-ColorOutput "Failed to set up virtual environment. Exiting." "Red"
        return
    }
    
    # Install base dependencies
    if (-not (Install-BaseDependencies)) {
        Write-ColorOutput "Failed to install base dependencies. Exiting." "Red"
        return
    }
    
    # Update config file
    Update-ConfigFile -HasNvidia $prereqResults.HasNvidia
    
    # Setup Playwright
    Setup-Playwright
    
    # Setup API Keys
    Setup-ApiKeys
    
    # Create run script
    Create-RunScript
    
    # Final messages
    Show-SectionHeader "SETUP COMPLETE"
    Write-ColorOutput "Setup completed successfully!" "Green"
    Write-Host ""
    Write-Host "To run the application:"
    Write-Host "1. Open PowerShell and navigate to this directory"
    Write-Host "2. Run: .\run.ps1"
    Write-Host ""
    Write-Host "Note: If you encounter script execution policy errors, you may need to run:"
    Write-Host "Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass"
    Write-Host ""
    
    # Ask if user wants to run the application now
    $runNow = Read-Host "Do you want to start the application now? [Y/n]"
    if ($runNow -ne "n" -and $runNow -ne "N") {
        Write-ColorOutput "Starting application..." "Green"
        & ".\run.ps1"
    }
}

# Run the main function
main 