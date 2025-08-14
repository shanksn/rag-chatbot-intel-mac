// API base URL - use relative path to work from any host
const API_URL = '/api';

// Global state
let currentSessionId = null;

// DOM elements
let chatMessages, chatInput, sendButton, totalCourses, courseTitles, newChatButton, themeToggle;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    // Get DOM elements after page loads
    chatMessages = document.getElementById('chatMessages');
    chatInput = document.getElementById('chatInput');
    sendButton = document.getElementById('sendButton');
    totalCourses = document.getElementById('totalCourses');
    courseTitles = document.getElementById('courseTitles');
    newChatButton = document.getElementById('newChatButton');
    themeToggle = document.getElementById('themeToggle');
    
    setupEventListeners();
    createNewSession();
    loadCourseStats();
    initializeToggle();
});

// Event Listeners
function setupEventListeners() {
    // Chat functionality
    sendButton.addEventListener('click', sendMessage);
    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendMessage();
    });
    
    // New Chat button
    newChatButton.addEventListener('click', clearCurrentChat);
    
    // Theme Toggle button
    if (themeToggle) {
        themeToggle.addEventListener('click', toggleTheme);
        themeToggle.addEventListener('keydown', handleToggleKeydown);
    }
    
    // Suggested questions
    document.querySelectorAll('.suggested-item').forEach(button => {
        button.addEventListener('click', (e) => {
            const question = e.target.getAttribute('data-question');
            chatInput.value = question;
            sendMessage();
        });
    });
}


// Chat Functions
async function sendMessage() {
    const query = chatInput.value.trim();
    if (!query) return;

    // Disable input
    chatInput.value = '';
    chatInput.disabled = true;
    sendButton.disabled = true;

    // Add user message
    addMessage(query, 'user');

    // Add loading message - create a unique container for it
    const loadingMessage = createLoadingMessage();
    chatMessages.appendChild(loadingMessage);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    try {
        const response = await fetch(`${API_URL}/query`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query: query,
                session_id: currentSessionId
            })
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Query failed (${response.status}): ${errorText}`);
        }

        const data = await response.json();
        
        // Update session ID if new
        if (!currentSessionId) {
            currentSessionId = data.session_id;
        }

        // Replace loading message with response
        loadingMessage.remove();
        addMessage(data.answer, 'assistant', data.sources);

    } catch (error) {
        console.error('Query error:', error);
        // Replace loading message with error
        loadingMessage.remove();
        addMessage(`Error: ${error.message}`, 'assistant');
    } finally {
        chatInput.disabled = false;
        sendButton.disabled = false;
        chatInput.focus();
    }
}

function createLoadingMessage() {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant';
    messageDiv.innerHTML = `
        <div class="message-content">
            <div class="loading">
                <span></span>
                <span></span>
                <span></span>
            </div>
        </div>
    `;
    return messageDiv;
}

function addMessage(content, type, sources = null, isWelcome = false) {
    const messageId = Date.now();
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}${isWelcome ? ' welcome-message' : ''}`;
    messageDiv.id = `message-${messageId}`;
    
    // Convert markdown to HTML for assistant messages
    const displayContent = type === 'assistant' ? marked.parse(content) : escapeHtml(content);
    
    let html = `<div class="message-content">${displayContent}</div>`;
    
    if (sources && sources.length > 0) {
        const sourcesList = sources.map((source, index) => {
            const number = index + 1;
            if (source && source.link) {
                return `<div class="source-item">${number}. <a href="${source.link}" target="_blank" rel="noopener noreferrer">${source.title}</a></div>`;
            } else if (source && source.title) {
                return `<div class="source-item">${number}. ${source.title}</div>`;
            } else {
                // Handle case where source is still a string (backward compatibility)
                return `<div class="source-item">${number}. ${typeof source === 'string' ? source : JSON.stringify(source)}</div>`;
            }
        }).join('');
        
        html += `
            <details class="sources-collapsible">
                <summary class="sources-header">Sources</summary>
                <div class="sources-content">${sourcesList}</div>
            </details>
        `;
    }
    
    messageDiv.innerHTML = html;
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    return messageId;
}

// Helper function to escape HTML for user messages
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Removed removeMessage function - no longer needed since we handle loading differently

async function createNewSession() {
    currentSessionId = null;
    chatMessages.innerHTML = '';
    addMessage('Welcome to the Course Materials Assistant! I can help you with questions about courses, lessons and specific content. What would you like to know?', 'assistant', null, true);
}

function clearCurrentChat() {
    // Clear chat messages
    chatMessages.innerHTML = '';
    
    // Reset session
    currentSessionId = null;
    
    // Clear input and re-enable if disabled
    chatInput.value = '';
    chatInput.disabled = false;
    sendButton.disabled = false;
    
    // Show welcome message
    addMessage('Welcome to the Course Materials Assistant! I can help you with questions about courses, lessons and specific content. What would you like to know?', 'assistant', null, true);
    
    // Focus on input for immediate use
    chatInput.focus();
}

// Load course statistics
async function loadCourseStats() {
    try {
        console.log('Loading course stats...');
        const response = await fetch(`${API_URL}/courses`);
        if (!response.ok) throw new Error('Failed to load course stats');
        
        const data = await response.json();
        console.log('Course data received:', data);
        
        // Update stats in UI
        if (totalCourses) {
            totalCourses.textContent = data.total_courses;
        }
        
        // Update course titles
        if (courseTitles) {
            if (data.course_titles && data.course_titles.length > 0) {
                courseTitles.innerHTML = data.course_titles
                    .map(title => `<div class="course-title-item">${title}</div>`)
                    .join('');
            } else {
                courseTitles.innerHTML = '<span class="no-courses">No courses available</span>';
            }
        }
        
    } catch (error) {
        console.error('Error loading course stats:', error);
        // Set default values on error
        if (totalCourses) {
            totalCourses.textContent = '0';
        }
        if (courseTitles) {
            courseTitles.innerHTML = '<span class="error">Failed to load courses</span>';
        }
    }
}

// Toggle Button Functions
function toggleTheme() {
    const currentState = themeToggle.getAttribute('aria-checked') === 'true';
    const newState = !currentState;
    
    // Update ARIA state
    themeToggle.setAttribute('aria-checked', newState.toString());
    
    // Update label for screen readers
    const label = newState ? 'Switch to dark theme' : 'Switch to light theme';
    themeToggle.setAttribute('aria-label', label);
    
    // Toggle the actual theme
    document.body.classList.toggle('light-theme', newState);
    
    // Save theme preference to localStorage
    localStorage.setItem('theme', newState ? 'light' : 'dark');
    
    console.log(`Theme toggled to: ${newState ? 'light' : 'dark'} theme`);
    
    // Announce the change to screen readers
    announceToggleState(newState);
}

function handleToggleKeydown(event) {
    // Handle Space and Enter keys for accessibility
    if (event.key === ' ' || event.key === 'Enter') {
        event.preventDefault();
        toggleTheme();
    }
    
    // Handle arrow keys for additional accessibility
    if (event.key === 'ArrowLeft') {
        event.preventDefault();
        if (themeToggle.getAttribute('aria-checked') === 'true') {
            toggleTheme();
        }
    }
    
    if (event.key === 'ArrowRight') {
        event.preventDefault();
        if (themeToggle.getAttribute('aria-checked') === 'false') {
            toggleTheme();
        }
    }
}

function announceToggleState(isOn) {
    // Create a temporary element to announce the state change to screen readers
    const announcement = document.createElement('div');
    announcement.setAttribute('aria-live', 'polite');
    announcement.setAttribute('aria-atomic', 'true');
    announcement.className = 'sr-only';
    announcement.textContent = `Theme switched to ${isOn ? 'light' : 'dark'} mode`;
    
    document.body.appendChild(announcement);
    
    // Remove the announcement element after it's been read
    setTimeout(() => {
        document.body.removeChild(announcement);
    }, 1000);
}

// Initialize toggle state
function initializeToggle() {
    if (themeToggle) {
        // Check for saved theme preference or system preference
        const savedTheme = localStorage.getItem('theme');
        const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
        const isLightTheme = savedTheme === 'light' || (!savedTheme && !prefersDark);
        
        // Set initial state
        themeToggle.setAttribute('aria-checked', isLightTheme.toString());
        const label = isLightTheme ? 'Switch to dark theme' : 'Switch to light theme';
        themeToggle.setAttribute('aria-label', label);
        
        // Apply the theme
        document.body.classList.toggle('light-theme', isLightTheme);
    }
}