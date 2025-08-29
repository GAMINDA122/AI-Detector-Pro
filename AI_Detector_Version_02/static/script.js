// Global variables
let currentAnalysisData = null;
let isAnalyzing = false;

// Utility functions
function showStatus(message, type = 'info') {
    const banner = document.getElementById('statusBanner');
    const messageEl = document.getElementById('statusMessage');
    
    banner.className = `mb-6 p-4 border rounded-lg ${type === 'error' ? 'status-error' : type === 'success' ? 'status-success' : type === 'warning' ? 'status-warning' : 'bg-blue-100 border-blue-300'}`;
    messageEl.textContent = message;
    banner.classList.remove('hidden');
    
    // Auto-hide after 5 seconds
    setTimeout(() => {
        banner.classList.add('hidden');
    }, 5000);
}

function setButtonLoading(button, loading, originalText) {
    if (loading) {
        button.disabled = true;
        button.innerHTML = `<div class="loading-spinner"></div>${originalText}`;
    } else {
        button.disabled = false;
        button.innerHTML = originalText;
    }
}

// Tab switching functionality
document.getElementById('textTab').addEventListener('click', function() {
    document.getElementById('textInput').classList.remove('hidden');
    document.getElementById('fileInput').classList.add('hidden');
    this.classList.add('tab-active');
    document.getElementById('fileTab').classList.remove('tab-active');
});

document.getElementById('fileTab').addEventListener('click', function() {
    document.getElementById('fileInput').classList.remove('hidden');
    document.getElementById('textInput').classList.add('hidden');
    this.classList.add('tab-active');
    document.getElementById('textTab').classList.remove('tab-active');
});

// Character counter
document.getElementById('contentText').addEventListener('input', function() {
    document.getElementById('charCount').textContent = this.value.length;
});

// File upload handling
document.getElementById('fileUpload').addEventListener('change', function() {
    const fileList = document.getElementById('fileList');
    fileList.innerHTML = '';

    Array.from(this.files).forEach(file => {
        const fileItem = document.createElement('div');
        fileItem.className = 'flex items-center justify-between p-3 bg-gray-50 rounded-lg';
        fileItem.innerHTML = `
            <div class="flex items-center space-x-3">
                <i class="fas fa-file-alt text-blue-500"></i>
                <div>
                    <p class="font-medium text-gray-800">${file.name}</p>
                    <p class="text-sm text-gray-500">${(file.size / 1024).toFixed(1)} KB</p>
                </div>
            </div>
            <button class="text-red-500 hover:text-red-700" onclick="this.parentElement.remove()">
                <i class="fas fa-times"></i>
            </button>
        `;
        fileList.appendChild(fileItem);

        const reader = new FileReader();

        // PDF files
        if (file.type === 'application/pdf') {
            reader.onload = function(e) {
                const typedarray = new Uint8Array(e.target.result);
                pdfjsLib.getDocument(typedarray).promise.then(pdf => {
                    let pdfText = '';
                    const pagePromises = [];
                    for (let i = 1; i <= pdf.numPages; i++) {
                        pagePromises.push(
                            pdf.getPage(i).then(page => 
                                page.getTextContent().then(content => {
                                    const textItems = content.items.map(item => item.str);
                                    pdfText += textItems.join(' ') + '\n';
                                })
                            )
                        );
                    }
                    Promise.all(pagePromises).then(() => {
                        document.getElementById('contentText').value = pdfText;
                        document.getElementById('charCount').textContent = pdfText.length;
                    });
                });
            };
            reader.readAsArrayBuffer(file);
        } 
        // DOC/DOCX files
        else if (file.name.endsWith('.doc') || file.name.endsWith('.docx')) {
            reader.onload = function(e) {
                const arrayBuffer = e.target.result;
                mammoth.extractRawText({ arrayBuffer })
                    .then(result => {
                        document.getElementById('contentText').value = result.value;
                        document.getElementById('charCount').textContent = result.value.length;
                    })
                    .catch(err => console.error('Error reading Word file:', err));
            };
            reader.readAsArrayBuffer(file);
        } 
        // TXT and other text files
        else {
            reader.onload = function(e) {
                document.getElementById('contentText').value = e.target.result;
                document.getElementById('charCount').textContent = e.target.result.length;
            };
            reader.readAsText(file);
        }
    });
});


// Main analyze button functionality
document.getElementById('analyzeBtn').addEventListener('click', async function() {
    const text = document.getElementById('contentText').value;
    const contentType = document.getElementById('contentType').value;
    
    if (!text.trim()) {
        showStatus('Please enter some text to analyze', 'error');
        return;
    }
    
    if (text.length < 10) {
        showStatus('Please provide at least 10 characters of text for analysis', 'error');
        return;
    }
    
    if (isAnalyzing) {
        return;
    }
    
    isAnalyzing = true;
    const originalText = '<i class="fas fa-search mr-2"></i>Analyze Content';
    setButtonLoading(this, true, 'Analyzing...');
    
    try {
        showStatus('Analyzing content with AI models...', 'info');
        
        const response = await fetch('/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                text: text,
                content_type: contentType
            })
        });
        
        const data = await response.json();
        
        if (data.error) {
            showStatus(data.error, 'error');
            return;
        }
        
        currentAnalysisData = data;
        updateResults(data);
        showStatus('Analysis completed successfully!', 'success');
        
    } catch (error) {
        console.error('Analysis error:', error);
        showStatus('Analysis failed. Please try again.', 'error');
    } finally {
        isAnalyzing = false;
        setButtonLoading(this, false, originalText);
    }
});

// Update results display
function updateResults(data) {
    // Update progress circle
    const circle = document.getElementById('progressCircle');
    const scoreText = document.getElementById('scoreText');
    const circumference = 2 * Math.PI * 56;
    const offset = circumference - (data.overall_score / 100) * circumference;
    
    circle.style.strokeDashoffset = offset;
    scoreText.textContent = data.overall_score + '%';
    
    // Update color based on score
    if (data.overall_score >= 70) {
        circle.style.stroke = '#ef4444'; // Red
        scoreText.className = 'text-2xl font-bold text-red-500';
    } else if (data.overall_score >= 40) {
        circle.style.stroke = '#f59e0b'; // Yellow
        scoreText.className = 'text-2xl font-bold text-yellow-500';
    } else {
        circle.style.stroke = '#10b981'; // Green
        scoreText.className = 'text-2xl font-bold text-green-500';
    }
    
    // Update breakdown percentages
    document.getElementById('humanPercent').textContent = data.breakdown.human + '%';
    document.getElementById('likelyPercent').textContent = data.breakdown.likely_ai + '%';
    document.getElementById('aiPercent').textContent = data.breakdown.ai_generated + '%';
    
    // Animate progress bars
    setTimeout(() => {
        document.getElementById('humanBar').style.width = data.breakdown.human + '%';
        document.getElementById('likelyBar').style.width = data.breakdown.likely_ai + '%';
        document.getElementById('aiBar').style.width = data.breakdown.ai_generated + '%';
    }, 100);
    
    // Update model detection
    document.getElementById('gptPercent').textContent = data.model_detection.chatgpt + '%';
    document.getElementById('claudePercent').textContent = data.model_detection.claude + '%';
    document.getElementById('geminiPercent').textContent = data.model_detection.gemini + '%';
    
    // Show highlighted text analysis
    if (data.highlighted_sentences) {
        showHighlightedText(data.highlighted_sentences);
    }
    
    // Show additional results
    document.getElementById('additionalResults').classList.remove('hidden');
}

// Show highlighted text analysis
function showHighlightedText(highlightedSentences) {
    let highlightedContent = '';
    
    highlightedSentences.forEach(sentence => {
        highlightedContent += `<span class="${sentence.class} sentence-highlight px-1 rounded" title="${sentence.label} (${sentence.probability}%)">${sentence.text}</span> `;
    });
    
    document.getElementById('highlightedText').innerHTML = highlightedContent;
}

// Download report functionality
document.getElementById('downloadBtn').addEventListener('click', function() {
    if (!currentAnalysisData) {
        showStatus('Please analyze some content first', 'error');
        return;
    }
    
    const button = this;
    const originalText = '<i class="fas fa-download mr-2"></i>Download Report';
    setButtonLoading(button, true, 'Generating PDF...');
    
    setTimeout(() => {
        try {
            // Create PDF using jsPDF
            const { jsPDF } = window.jspdf;
            const doc = new jsPDF();
            
            // Set font and colors
            doc.setFont("helvetica");
            
            // Header
            doc.setFontSize(20);
            doc.setTextColor(102, 126, 234);
            doc.text('AI CONTENT DETECTION REPORT', 20, 30);
            
            // Date
            doc.setFontSize(10);
            doc.setTextColor(100, 100, 100);
            doc.text(`Generated: ${new Date().toLocaleDateString()} at ${new Date().toLocaleTimeString()}`, 20, 40);
            
            // Analysis Results Section
            doc.setFontSize(14);
            doc.setTextColor(0, 0, 0);
            doc.text('ANALYSIS RESULTS', 20, 60);
            
            doc.setFontSize(11);
            doc.text(`AI Detection Score: ${currentAnalysisData.overall_score}%`, 25, 75);
            doc.text(`Human-like: ${currentAnalysisData.breakdown.human}%`, 25, 85);
            doc.text(`Likely AI: ${currentAnalysisData.breakdown.likely_ai}%`, 25, 95);
            doc.text(`AI Generated: ${currentAnalysisData.breakdown.ai_generated}%`, 25, 105);
            
            // Model Detection Section
            doc.setFontSize(14);
            doc.text('MODEL DETECTION', 20, 125);
            
            doc.setFontSize(11);
            doc.text(`GPT-4: ${currentAnalysisData.model_detection.chatgpt}%`, 25, 140);
            doc.text(`Claude-3: ${currentAnalysisData.model_detection.claude}%`, 25, 150);
            doc.text(`Gemini Pro: ${currentAnalysisData.model_detection.gemini}%`, 25, 160);
            
            // Content Type
            doc.setFontSize(14);
            doc.text('CONTENT TYPE', 20, 180);
            doc.setFontSize(11);
            doc.text(currentAnalysisData.content_type.toUpperCase(), 25, 195);
            
            // Analyzed Text Section
            doc.setFontSize(14);
            doc.text('ANALYZED TEXT', 20, 215);
            
            // Split text into lines that fit the page
            const text = document.getElementById('contentText').value;
            doc.setFontSize(9);
            const splitText = doc.splitTextToSize(text, 170);
            let yPosition = 230;
            
            splitText.forEach((line, index) => {
                if (yPosition > 270) {
                    doc.addPage();
                    yPosition = 20;
                }
                doc.text(line, 25, yPosition);
                yPosition += 5;
            });
            
            // Footer
            const pageCount = doc.internal.getNumberOfPages();
            for (let i = 1; i <= pageCount; i++) {
                doc.setPage(i);
                doc.setFontSize(8);
                doc.setTextColor(150, 150, 150);
                doc.text('Report generated by AI Content Detector Pro (Powered by Gemini AI)', 20, 285);
                doc.text(`Page ${i} of ${pageCount}`, 170, 285);
            }
            
            // Generate filename with timestamp
            const now = new Date();
            const filename = `AI_Detection_Report_${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')}_${String(now.getHours()).padStart(2, '0')}-${String(now.getMinutes()).padStart(2, '0')}.pdf`;
            
            // Download the PDF
            doc.save(filename);
            
            // Success feedback
            button.innerHTML = '<i class="fas fa-check mr-2"></i>PDF Downloaded!';
            button.className = 'w-full bg-green-500 text-white p-3 rounded-lg transition-colors text-left';
            
            setTimeout(() => {
                button.innerHTML = originalText;
                button.className = 'w-full bg-blue-500 text-white p-3 rounded-lg hover:bg-blue-600 transition-colors text-left';
                button.disabled = false;
            }, 3000);
            
            showStatus('PDF report downloaded successfully!', 'success');
            
        } catch (error) {
            console.error('PDF generation failed:', error);
            showStatus('PDF generation failed. Please try again.', 'error');
            setButtonLoading(button, false, originalText);
        }
    }, 500);
});

// Share functionality
document.getElementById('shareBtn').addEventListener('click', function() {
    if (!currentAnalysisData) {
        showStatus('Please analyze some content first', 'error');
        return;
    }
    document.getElementById('shareModal').classList.remove('hidden');
});

document.getElementById('closeModal').addEventListener('click', function() {
    document.getElementById('shareModal').classList.add('hidden');
});

document.getElementById('shareModal').addEventListener('click', function(e) {
    if (e.target === this) {
        this.classList.add('hidden');
    }
});

document.getElementById('shareEmail').addEventListener('click', function() {
    if (!currentAnalysisData) return;
    
    const results = `AI Detection Results:
- Overall Score: ${currentAnalysisData.overall_score}%
- Human-like: ${currentAnalysisData.breakdown.human}%
- Likely AI: ${currentAnalysisData.breakdown.likely_ai}%
- AI Generated: ${currentAnalysisData.breakdown.ai_generated}%

Model Detection:
- GPT-4: ${currentAnalysisData.model_detection.chatgpt}%
- Claude-3: ${currentAnalysisData.model_detection.claude}%
- Gemini Pro: ${currentAnalysisData.model_detection.gemini}%

Generated by AI Content Detector Pro (Powered by Gemini AI)`;
    
    const subject = encodeURIComponent('AI Content Detection Report');
    const body = encodeURIComponent(results);
    window.open(`mailto:?subject=${subject}&body=${body}`, '_blank');
    document.getElementById('shareModal').classList.add('hidden');
});

document.getElementById('shareWhatsApp').addEventListener('click', function() {
    if (!currentAnalysisData) return;
    
    const results = `🤖 *AI Content Detection Report*

📊 *Overall Score:* ${currentAnalysisData.overall_score}%

📈 *Breakdown:*
• Human-like: ${currentAnalysisData.breakdown.human}%
• Likely AI: ${currentAnalysisData.breakdown.likely_ai}%
• AI Generated: ${currentAnalysisData.breakdown.ai_generated}%

🔍 *Model Detection:*
• GPT-4: ${currentAnalysisData.model_detection.chatgpt}%
• Claude-3: ${currentAnalysisData.model_detection.claude}%
• Gemini Pro: ${currentAnalysisData.model_detection.gemini}%

Generated by AI Content Detector Pro ✨`;
    
    const message = encodeURIComponent(results);
    window.open(`https://wa.me/?text=${message}`, '_blank');
    document.getElementById('shareModal').classList.add('hidden');
});

document.getElementById('copyResults').addEventListener('click', function() {
    if (!currentAnalysisData) return;
    
    const results = `AI Detection: ${currentAnalysisData.overall_score}% | Human: ${currentAnalysisData.breakdown.human}% | AI: ${currentAnalysisData.breakdown.ai_generated}%`;
    navigator.clipboard.writeText(results).then(() => {
        this.innerHTML = '<i class="fas fa-check mr-3"></i>Copied!';
        setTimeout(() => {
            this.innerHTML = '<i class="fas fa-copy mr-3"></i>Copy Results';
        }, 2000);
        showStatus('Results copied to clipboard!', 'success');
    });
});

// Theme toggle functionality
document.getElementById('themeToggle').addEventListener('click', function() {
    const body = document.body;
    const themeIcon = document.getElementById('themeIcon');
    
    if (body.classList.contains('dark')) {
        // Switch to light theme
        body.classList.remove('dark');
        themeIcon.classList.remove('fa-sun');
        themeIcon.classList.add('fa-moon');
        localStorage.setItem('theme', 'light');
    } else {
        // Switch to dark theme
        body.classList.add('dark');
        themeIcon.classList.remove('fa-moon');
        themeIcon.classList.add('fa-sun');
        localStorage.setItem('theme', 'dark');
    }
});

// Load saved theme on page load
document.addEventListener('DOMContentLoaded', function() {
    const savedTheme = localStorage.getItem('theme');
    const body = document.body;
    const themeIcon = document.getElementById('themeIcon');
    
    if (savedTheme === 'dark') {
        body.classList.add('dark');
        themeIcon.classList.remove('fa-moon');
        themeIcon.classList.add('fa-sun');
    }
    
    // Show welcome message
    showStatus('AI Content Detector Pro is ready! Powered by Gemini AI.', 'success');
});

// Keyboard shortcuts
document.addEventListener('keydown', function(e) {
    // Ctrl/Cmd + Enter to analyze
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        document.getElementById('analyzeBtn').click();
    }
    
    // Escape to close modal
    if (e.key === 'Escape') {
        document.getElementById('shareModal').classList.add('hidden');
    }
});

// Remove any saved text on page load to prevent persistence
document.addEventListener('DOMContentLoaded', function() {
    localStorage.removeItem('savedText'); // remove previous saved text
    document.getElementById('contentText').value = '';
    document.getElementById('charCount').textContent = 0;

    // Show welcome message
    showStatus('AI Content Detector Pro is ready! Powered by Gemini AI.', 'success');
});