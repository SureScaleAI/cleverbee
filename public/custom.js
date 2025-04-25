// Wait for document to be ready
document.addEventListener('DOMContentLoaded', function() {
    // Create the honeycomb background elements
    setupHoneycombBackground();
    
    // Initialize the animation with a small delay to ensure CSS is fully loaded
    setTimeout(() => {
        initHoneycombAnimation();
    }, 100);

    // Set up message animation observer for new messages with updated selectors
    setupMessageAnimation();
});

// Function to set up the honeycomb background elements
function setupHoneycombBackground() {
    // Create container divs for the honeycomb background
    const honeycombBackground = document.createElement('div');
    honeycombBackground.className = 'honeycomb-background';
    
    const honeycombShapesLayer = document.createElement('div');
    honeycombShapesLayer.className = 'honeycomb-shapes-layer';
    
    // Create canvas elements
    const lightCanvas = document.createElement('canvas');
    lightCanvas.id = 'light-source-canvas';
    
    const shapeCanvas = document.createElement('canvas');
    shapeCanvas.id = 'shape-canvas';
    shapeCanvas.style.opacity = '0'; // Start hidden to prevent flash
    shapeCanvas.style.visibility = 'hidden'; // Add visibility:hidden for extra safety
    
    // Create dot canvas if it doesn't exist yet
    const dotCanvas = document.createElement('canvas');
    dotCanvas.id = 'dot-canvas';
    
    // Append elements to the DOM
    honeycombBackground.appendChild(lightCanvas);
    honeycombShapesLayer.appendChild(shapeCanvas);
    honeycombBackground.appendChild(dotCanvas);
    
    // Add both layers to the body
    document.body.prepend(honeycombBackground);
    document.body.prepend(honeycombShapesLayer);
}

// Setup message animation - Updated for new Chainlit UI
function setupMessageAnimation() {
    // Find the messages container in the new Chainlit UI
    const messagesContainer = document.querySelector('.flex.flex-col.flex-grow.overflow-y-auto');
    if (!messagesContainer) {
        // If not found yet, retry after a short delay
        setTimeout(setupMessageAnimation, 200);
        return;
    }

    // Set up mutation observer to watch for new messages
    const observer = new MutationObserver((mutations) => {
        mutations.forEach(mutation => {
            if (mutation.type === 'childList' && mutation.addedNodes.length > 0) {
                // Process any new message steps
                processNewMessageSteps();
            }
        });
    });

    // Start observing with configuration
    observer.observe(messagesContainer, {
        childList: true, // observe direct children
        subtree: true    // observe all descendants
    });

    // Initial activation for any messages already in the DOM
    processNewMessageSteps();
}

// Process and animate message steps
function processNewMessageSteps() {
    // Find all message steps
    const steps = document.querySelectorAll('.step.py-2:not(.active)');
    
    // Add active class with staggered delay
    steps.forEach((step, index) => {
        // Set a CSS variable for staggered animation
        step.style.setProperty('--index', index);
        
        // Add active class after a small delay
        setTimeout(() => {
            step.classList.add('active');
        }, 50 + (index * 120));
    });
}

// Main function to initialize the honeycomb animation
function initHoneycombAnimation() {
    const shapeCanvas = document.getElementById('shape-canvas');
    const dotCanvas = document.getElementById('dot-canvas');
    const lightCanvas = document.getElementById('light-source-canvas');
    
    if (!shapeCanvas || !dotCanvas || !lightCanvas) {
        console.warn('Canvas elements not found, retrying in 200ms');
        setTimeout(initHoneycombAnimation, 200);
        return;
    }
    
    // Don't need to set opacity again as it's already set in setupHoneycombBackground
    
    const shapeCtx = shapeCanvas.getContext('2d');
    const lightCtx = lightCanvas.getContext('2d');
    const dotCtx = dotCanvas.getContext('2d');
    
    if (!shapeCtx || !lightCtx || !dotCtx) {
        console.warn('Canvas contexts not available, retrying in 200ms');
        setTimeout(initHoneycombAnimation, 200);
        return;
    }

    // --- Configuration --- (Match website/index.html)
    const cellWidth = 150;
    const cellHeight = cellWidth * 0.866; // Ideal height
    const gap = 2;
    // Actual spacing includes the gap
    const actualVerticalSpacing = cellHeight + gap;
    const actualHorizontalSpacing = cellWidth * 0.75 + gap * Math.cos(Math.PI / 6);
    // Ideal spacing assumes no gap
    const idealVerticalSpacing = cellHeight;
    const idealHorizontalSpacing = cellWidth * 0.75;
    
    // --- Retrieve Colors from CSS Variables (like website) ---
    function getCssVariable(name, fallback) {
        const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
        return value || fallback;
    }
    const shapeFillColor = getCssVariable('--shape-fill-color', '#1A1A1A');
    const glowColor = getCssVariable('--primary-yellow-glow', 'rgba(250, 204, 21, 0.6)');
    const coreDotColor = getCssVariable('--primary-yellow', '#facc15');
    const accentBlueGlow = getCssVariable('--accent-blue-glow', 'rgba(59, 130, 246, 0.6)');
    const accentPurpleGlow = getCssVariable('--accent-purple-glow', 'rgba(139, 92, 246, 0.6)');
    // --- End Color Retrieval ---

    // Light Source / Trail configuration (Match website/index.html)
    const lightSourceSpeed = 0.5; // Website speed
    const glowRadius = 100; // Website glow radius
    const dotCoreSize = 2; // Website core dot size
    const trailLength = 25; // Website trail length
    const trailPointBaseSize = 1.5; // Website trail point size

    // --- State --- (Match website/index.html)
    let vertexMap = new Map(); // Map IDEAL vertex key ("x.xx,y.yy") to { x: actual_x, y: actual_y, connections: Set<string> }
    let dots = [];
    let animationFrameId = null;
    let lastTime = 0;

    // Function to determine how many dots to show based on screen size (Match website/index.html)
    function getDotCountForScreen() {
        const w = window.innerWidth;
        if (w >= 1000) return 10; // Website dot count
        if (w >= 600) return 6;
        if (w < 400) return 2;
        return 3;
    }

    // Create state for each dot (Match website/index.html)
    function makeDotState() {
        // Randomly select a color scheme
        const colorScheme = Math.random() < 0.7 ? 'yellow' : (Math.random() < 0.5 ? 'blue' : 'purple');
        
        return {
            lightSource: { x: 0, y: 0, progress: 0 },
            currentVertex: null,
            previousVertexKey: null,
            targetVertex: null,
            trailPoints: [],
            dotInitialized: false,
            colorScheme: colorScheme,
            dotSize: 1.5 + Math.random() * 1.5, // Website random size
            pulsePhase: Math.random() * Math.PI * 2 // Website random phase
        };
    }

    // Resize canvas to match window size (Same as before)
    function resizeCanvas() {
        lightCanvas.width = window.innerWidth;
        lightCanvas.height = window.innerHeight;
        shapeCanvas.width = window.innerWidth;
        shapeCanvas.height = window.innerHeight;
    }

    // Parse RGBA color (Same as website/index.html)
    function parseRGBAColor(color) {
        // Try rgba() format first
        let match = color.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)(?:,\s*([\d.]+))?\)/);
        if (match) {
            return {
                r: parseInt(match[1]),
                g: parseInt(match[2]),
                b: parseInt(match[3]),
                a: match[4] !== undefined ? parseFloat(match[4]) : 1
            };
        }
        // Try hex format
        if (color.startsWith('#')) {
            let hex = color.slice(1);
            if (hex.length === 3) hex = hex.split('').map(c => c + c).join('');
            if (hex.length !== 6) return null;
            const r = parseInt(hex.slice(0, 2), 16);
            const g = parseInt(hex.slice(2, 4), 16);
            const b = parseInt(hex.slice(4, 6), 16);
            return { r, g, b, a: 1 };
        }
        return null;
    }

    // Create honeycomb grid (Same as website/index.html)
    function createHoneycombGrid() {
        vertexMap.clear();
        const canvasWidth = lightCanvas.width;
        const canvasHeight = lightCanvas.height;
        const cols = Math.ceil(canvasWidth / idealHorizontalSpacing) + 2;
        const rows = Math.ceil(canvasHeight / idealVerticalSpacing) + 2;
        const offsetX = (canvasWidth - (cols - 1) * idealHorizontalSpacing) / 2;
        const offsetY = (canvasHeight - (rows - 1) * idealVerticalSpacing) / 2;

        for (let row = -1; row < rows + 1; row++) {
            for (let col = -1; col < cols + 1; col++) {
                const idealX = col * idealHorizontalSpacing;
                const idealY = row * idealVerticalSpacing + (col % 2 === 0 ? 0 : idealVerticalSpacing / 2);
                const actualX = col * actualHorizontalSpacing + offsetX;
                const actualY = row * actualVerticalSpacing + (col % 2 === 0 ? 0 : actualVerticalSpacing / 2) + offsetY;
                const vertexKey = `${idealX.toFixed(2)},${idealY.toFixed(2)}`;
                const vertex = {
                    x: actualX,
                    y: actualY,
                    connections: new Set()
                };
                vertexMap.set(vertexKey, vertex);
                
                const directions = [
                    { col: 0, row: -1 }, { col: 1, row: col % 2 === 0 ? -1 : 0 },
                    { col: 1, row: col % 2 === 0 ? 0 : 1 }, { col: 0, row: 1 },
                    { col: -1, row: col % 2 === 0 ? 0 : 1 }, { col: -1, row: col % 2 === 0 ? -1 : 0 }
                ];
                
                for (const dir of directions) {
                    const neighCol = col + dir.col;
                    const neighRow = row + dir.row;
                    const neighIdealX = neighCol * idealHorizontalSpacing;
                    const neighIdealY = neighRow * idealVerticalSpacing + (neighCol % 2 === 0 ? 0 : idealVerticalSpacing / 2);
                    const neighKey = `${neighIdealX.toFixed(2)},${neighIdealY.toFixed(2)}`;
                    vertex.connections.add(neighKey);
                }
            }
        }
        drawHoneycomb(shapeCtx);
    }

    // Draw honeycomb grid (Same as website/index.html, removed alpha modification)
    function drawHoneycomb(ctx) {
        ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
        ctx.fillStyle = shapeFillColor;
        
        // Draw the shapes
        for (const [key, vertex] of vertexMap.entries()) {
            const { x, y } = vertex;
            ctx.beginPath();
            for (let i = 0; i < 6; i++) {
                const angle = (Math.PI / 3) * i;
                const px = x + cellWidth / 2 * Math.cos(angle);
                const py = y + cellWidth / 2 * Math.sin(angle);
                if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
            }
            ctx.closePath();
            ctx.fill();
        }
        
        // Show the shape canvas now that it's fully rendered
        // Wait until next animation frame to ensure rendering is complete
        requestAnimationFrame(() => {
            const shapeCanvas = document.getElementById('shape-canvas');
            if (shapeCanvas) {
                shapeCanvas.style.visibility = 'visible'; // Make visible first
                shapeCanvas.style.opacity = '1';
                shapeCanvas.style.transition = 'opacity 0.5s ease-in'; // Longer transition
            }
        });
    }
    
    // Get the color for a dot based on its color scheme (Same as website/index.html)
    function getDotColors(dotState) {
        switch(dotState.colorScheme) {
            case 'blue': return { glow: accentBlueGlow, core: '#3B82F6' };
            case 'purple': return { glow: accentPurpleGlow, core: '#8B5CF6' };
            case 'yellow': default: return { glow: glowColor, core: coreDotColor };
        }
    }

    // Initialize and reset all dots (Same as website/index.html)
    function initDots() {
        const dotCount = getDotCountForScreen();
        dots = [];
        for (let i = 0; i < dotCount; i++) {
            dots.push(makeDotState());
        }
    }

    // Choose a random vertex from the grid (Same as website/index.html)
    function getRandomVertex() {
        const vertexKeys = Array.from(vertexMap.keys());
        if (vertexKeys.length === 0) return null;
        const randomIndex = Math.floor(Math.random() * vertexKeys.length);
        const randomKey = vertexKeys[randomIndex];
        return { key: randomKey, vertex: vertexMap.get(randomKey) };
    }

    // Choose a random connected vertex (Same as website/index.html)
    function getRandomConnectedVertex(vertexKey, excludeKey = null) {
        const vertex = vertexMap.get(vertexKey);
        if (!vertex || vertex.connections.size === 0) return null;
        const possibleConnections = Array.from(vertex.connections).filter(key => key !== excludeKey && vertexMap.has(key));
        if (possibleConnections.length === 0) return null;
        const randomIndex = Math.floor(Math.random() * possibleConnections.length);
        const targetKey = possibleConnections[randomIndex];
        return { key: targetKey, vertex: vertexMap.get(targetKey) };
    }

    // Update a single dot's position and trail (Match website/index.html)
    function updateDot(dotState, deltaTime) {
        if (!dotState.dotInitialized || !dotState.currentVertex) {
            const randomVertex = getRandomVertex();
            if (!randomVertex) return;
            dotState.currentVertex = randomVertex;
            dotState.dotInitialized = true;
            dotState.previousVertexKey = null;
            dotState.targetVertex = null;
            dotState.lightSource = { x: randomVertex.vertex.x, y: randomVertex.vertex.y, progress: 0 };
            dotState.trailPoints = [];
        }
        
        if (!dotState.targetVertex) {
            const newTarget = getRandomConnectedVertex(dotState.currentVertex.key, dotState.previousVertexKey);
            if (!newTarget) return;
            dotState.targetVertex = newTarget;
            dotState.lightSource.progress = 0;
        }
        
        // Use website speed calculation (no speedFactor)
        dotState.lightSource.progress += (lightSourceSpeed / 1000) * deltaTime;
        
        if (dotState.lightSource.progress >= 1) {
            dotState.previousVertexKey = dotState.currentVertex.key;
            dotState.currentVertex = dotState.targetVertex;
            dotState.targetVertex = null;
            dotState.lightSource.progress = 0;
        } else {
            const t = dotState.lightSource.progress;
            const start = dotState.currentVertex.vertex;
            const end = dotState.targetVertex.vertex;
            dotState.lightSource.x = start.x + (end.x - start.x) * t;
            dotState.lightSource.y = start.y + (end.y - start.y) * t;
        }
        
        dotState.trailPoints.unshift({ x: dotState.lightSource.x, y: dotState.lightSource.y, age: 0 });
        
        for (let i = dotState.trailPoints.length - 1; i >= 0; i--) {
            const point = dotState.trailPoints[i];
            point.age += deltaTime;
            // Use website trail expiration time (trailLength * 16)
            if (point.age > trailLength * 16) {
                dotState.trailPoints.splice(i, 1);
            }
        }
    }

    // Draw a single dot and its trail (Match website/index.html)
    function drawDot(ctx, dotState) {
        const { glow, core } = getDotColors(dotState);
        
        // Draw trail first (Match website logic)
        for (let i = dotState.trailPoints.length - 1; i >= 0; i--) {
            const point = dotState.trailPoints[i];
            const ageRatio = Math.min(point.age / (trailLength * 16), 1); // Website age ratio
            const alpha = 1 - ageRatio;
            const size = trailPointBaseSize * (1 - ageRatio); // Website size reduction
            
            if (alpha > 0) {
                const glowColorObj = parseRGBAColor(glow);
                if (glowColorObj) {
                    const adjustedGlow = `rgba(${glowColorObj.r}, ${glowColorObj.g}, ${glowColorObj.b}, ${glowColorObj.a * alpha * 0.5})`; // Website alpha
                    const gradient = ctx.createRadialGradient(point.x, point.y, 0, point.x, point.y, glowRadius * size);
                    gradient.addColorStop(0, adjustedGlow);
                    gradient.addColorStop(1, 'rgba(0,0,0,0)');
                    ctx.fillStyle = gradient;
                    ctx.beginPath();
                    ctx.arc(point.x, point.y, glowRadius * size, 0, Math.PI * 2);
                    ctx.fill();
                }
            }
        }
        
        // Draw main light source glow (Match website logic - simple pulse)
        const glowColorObj = parseRGBAColor(glow);
        if (glowColorObj) {
            const pulseScale = 1 + 0.2 * Math.sin(dotState.pulsePhase + performance.now() / 1000); // Website pulse
            const gradient = ctx.createRadialGradient(
                dotState.lightSource.x, dotState.lightSource.y, 0,
                dotState.lightSource.x, dotState.lightSource.y, glowRadius * dotState.dotSize * pulseScale
            );
            gradient.addColorStop(0, glow);
            gradient.addColorStop(1, 'rgba(0,0,0,0)');
            ctx.fillStyle = gradient;
            ctx.beginPath();
            ctx.arc(dotState.lightSource.x, dotState.lightSource.y, glowRadius * dotState.dotSize * pulseScale, 0, Math.PI * 2);
            ctx.fill();
        }
        
        // Draw light source center dot (Same as website)
        ctx.fillStyle = core;
        ctx.beginPath();
        ctx.arc(dotState.lightSource.x, dotState.lightSource.y, dotCoreSize * dotState.dotSize, 0, Math.PI * 2);
        ctx.fill();
    }

    // Main animation loop (Same as website/index.html)
    function animate(timestamp) {
        if (!lastTime) lastTime = timestamp;
        const deltaTime = timestamp - lastTime;
        lastTime = timestamp;

        // --- REMOVED Logging Added for Debugging ---
        // // Log approximately once per second to avoid flooding
        // if (!window._debugLastLogTime) window._debugLastLogTime = 0;
        // const now = performance.now();
        // if (now - window._debugLastLogTime > 1000) {
        //   console.log('[Honeycomb Debug] deltaTime:', deltaTime);
        //   window._debugLastLogTime = now;
        // }
        // --- End Logging ---

        lightCtx.clearRect(0, 0, lightCanvas.width, lightCanvas.height);
        for (const dotState of dots) {
            updateDot(dotState, deltaTime);
            drawDot(lightCtx, dotState);
        }
        animationFrameId = requestAnimationFrame(animate);
    }

    // Handle window resize (Same as website/index.html)
    function handleResize() {
        resizeCanvas();
        createHoneycombGrid();
        // Re-init dots on resize to adjust count if needed
        initDots(); 
    }

    // Initialize the animation (Same as website/index.html)
    function initialize() {
        resizeCanvas();
        
        // Draw honeycomb first, then initialize dots
        createHoneycombGrid();
        
        // Delay starting the animation until after the shapes are fully rendered
        setTimeout(() => {
            initDots();
            if (animationFrameId) cancelAnimationFrame(animationFrameId);
            lastTime = 0;
            animationFrameId = requestAnimationFrame(animate);
        }, 100);
        
        window.addEventListener('resize', handleResize);
    }

    // Start the animation
    initialize();
} 

// Function to create and show the loader
function showLoader() {
  // Prevent creating multiple loaders
  if (document.getElementById('initial-loader')) {
    return;
  }

  const loader = document.createElement('div');
  loader.id = 'initial-loader';
  loader.innerHTML = `
    <div class="loader-spinner"></div>
    <p>Initializing CleverBee...</p>
  `;
  document.body.appendChild(loader);
  console.log('Initial loader shown.');
}

// Function to hide the loader
function hideLoader() {
  const loader = document.getElementById('initial-loader');
  if (loader) {
    loader.classList.add('hidden');
    // Optional: Remove the element after transition
    setTimeout(() => {
      if (loader.parentNode) {
        loader.parentNode.removeChild(loader);
        console.log('Initial loader removed from DOM.');
      }
    }, 600); // Match CSS transition duration + buffer
    console.log('Initial loader hidden.');
  }
}

// --- Main Logic ---

// Immediately show the loader when the script runs
showLoader();

// Function to check if Chainlit UI is ready
function checkChainlitReady() {
  // Check for chat input or the welcome message
  const chatInputContainer = document.querySelector('#chat-input-container');
  const welcomeMsg = Array.from(document.querySelectorAll('.message-content'))
    .find(el => el.textContent && el.textContent.includes('Welcome to CleverBee!'));

  if (chatInputContainer || welcomeMsg) {
    hideLoader();
    if (checkInterval) clearInterval(checkInterval);
  }
}

// Poll every 150ms to check if Chainlit is ready (more responsive)
// Start checking after a short delay to allow the initial DOM to settle
let checkInterval = null;
setTimeout(() => {
   checkInterval = setInterval(checkChainlitReady, 150); 
}, 100); 

// Fallback: Hide loader after a timeout (e.g., 15 seconds) 
// in case the target element never appears (error condition)
setTimeout(() => {
  if (document.getElementById('initial-loader') && !document.getElementById('initial-loader').classList.contains('hidden')) {
    console.warn('Loader fallback timeout reached. Hiding loader forcefully.');
    hideLoader();
    if (checkInterval) {
      clearInterval(checkInterval);
    }
  }
}, 15000); // 15 seconds timeout 

/**
 * Estimated reading time in seconds saved by collapsing reasoning sections
 * Used to show users how much time they saved by using the collapsible interface
 */
let totalTimeSavedInSeconds = 0;
const wordsPerMinute = 200; // Average reading speed

/**
 * Enhances details/summary elements for reasoning sections
 * @param {HTMLElement} element - The message element to process
 */
function enhanceReasoningSections(element) {
  // Find all details elements within this message
  const detailsElements = element.querySelectorAll('details');
  
  // Process each details element
  detailsElements.forEach(details => {
    // Calculate approximate reading time of the content
    const content = details.textContent.replace(details.querySelector('summary').textContent, '');
    const wordCount = content.split(/\s+/).length;
    const readingTimeSeconds = Math.round(wordCount / (wordsPerMinute / 60));
    
    // Add reading time to the summary
    const summary = details.querySelector('summary');
    const timeLabel = document.createElement('span');
    timeLabel.className = 'reading-time';
    timeLabel.textContent = `${readingTimeSeconds}s`;
    timeLabel.style.marginLeft = 'auto';
    timeLabel.style.fontSize = '0.75em';
    timeLabel.style.opacity = '0.7';
    summary.appendChild(timeLabel);
    
    // Set up event listeners for tracking time saved
    details.addEventListener('toggle', () => {
      if (!details.open) {
        totalTimeSavedInSeconds += readingTimeSeconds;
        updateTimeSavedDisplay();
      }
    });
    
    // Start with details closed to save reading time
    details.open = false;
  });
}

/**
 * Updates the display showing how much reading time was saved
 */
function updateTimeSavedDisplay() {
  let timeSavedDisplay = document.getElementById('time-saved-display');
  
  // Create the display if it doesn't exist
  if (!timeSavedDisplay) {
    timeSavedDisplay = document.createElement('div');
    timeSavedDisplay.id = 'time-saved-display';
    timeSavedDisplay.style.position = 'fixed';
    timeSavedDisplay.style.bottom = '10px';
    timeSavedDisplay.style.right = '10px';
    timeSavedDisplay.style.backgroundColor = 'rgba(0,0,0,0.6)';
    timeSavedDisplay.style.color = '#FACC15';
    timeSavedDisplay.style.padding = '5px 10px';
    timeSavedDisplay.style.borderRadius = '5px';
    timeSavedDisplay.style.fontSize = '12px';
    timeSavedDisplay.style.fontWeight = '500';
    timeSavedDisplay.style.zIndex = '1000';
    timeSavedDisplay.style.backdropFilter = 'blur(5px)';
    timeSavedDisplay.style.transition = 'opacity 0.3s ease';
    document.body.appendChild(timeSavedDisplay);
  }
  
  // Format the time saved
  const minutes = Math.floor(totalTimeSavedInSeconds / 60);
  const seconds = totalTimeSavedInSeconds % 60;
  let displayText = 'Time saved: ';
  
  if (minutes > 0) {
    displayText += `${minutes}m `;
  }
  
  displayText += `${seconds}s`;
  timeSavedDisplay.textContent = displayText;
  
  // Make it visible
  timeSavedDisplay.style.opacity = '1';
  
  // Hide after 3 seconds
  setTimeout(() => {
    timeSavedDisplay.style.opacity = '0';
  }, 3000);
}

/**
 * Initialize the observer to enhance new messages as they are added
 */
function initMessageObserver() {
  // Select the container where messages will be inserted
  const messagesContainer = document.querySelector('.flex.flex-col.overflow-auto.px-4');
  
  if (!messagesContainer) {
    // If container not found, retry after a short delay
    console.log('Messages container not found, retrying in 500ms');
    setTimeout(initMessageObserver, 500);
    return;
  }
  
  // Create observer to watch for new messages
  const observer = new MutationObserver(mutations => {
    mutations.forEach(mutation => {
      // Check if nodes were added
      if (mutation.addedNodes.length) {
        mutation.addedNodes.forEach(node => {
          // Only process element nodes
          if (node.nodeType === Node.ELEMENT_NODE) {
            // Check if this is a message element
            if (node.classList.contains('ai-message') || 
                node.hasAttribute('data-step-type') && 
                node.getAttribute('data-step-type') === 'user_message') {
              // Process after a slight delay to ensure content is fully rendered
              setTimeout(() => enhanceReasoningSections(node), 100);
            }
          }
        });
      }
    });
  });
  
  // Start observing changes to the messages container
  observer.observe(messagesContainer, { childList: true, subtree: true });
  console.log('Message observer initialized');
}

// Initialize when the page is loaded
window.addEventListener('load', () => {
  console.log('Custom JS loaded');
  // Use a small delay to ensure Chainlit has initialized
  setTimeout(initMessageObserver, 500);
}); 