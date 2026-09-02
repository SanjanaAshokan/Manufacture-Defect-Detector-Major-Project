/* ============================================================
   Manufacturing Defect Detector — Custom Frontend Logic
   ============================================================ */

document.addEventListener('DOMContentLoaded', () => {
    // ---- Navbar Scroll Effect ----
    const navbar = document.getElementById('navbar');
    window.addEventListener('scroll', () => {
        if (window.scrollY > 50) {
            navbar.classList.add('scrolled');
        } else {
            navbar.classList.remove('scrolled');
        }
    });

    // ---- Responsive Mobile Menu Toggle ----
    const navToggle = document.getElementById('navToggle');
    const navLinks = document.getElementById('navLinks');
    if (navToggle && navLinks) {
        navToggle.addEventListener('click', () => {
            navLinks.classList.toggle('active');
            // Toggle hamburger animation
            navToggle.classList.toggle('active');
        });
    }

    // ---- Stat Counter Animation (for Landing Page) ----
    const counters = document.querySelectorAll('.stat-value');
    if (counters.length > 0) {
        const observerOptions = {
            threshold: 0.5,
            rootMargin: "0px"
        };
        
        const counterObserver = new IntersectionObserver((entries, observer) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const counter = entry.target;
                    const target = parseFloat(counter.getAttribute('data-target'));
                    const duration = 1500; // ms
                    const startTime = performance.now();
                    
                    const updateCounter = (currentTime) => {
                        const elapsed = currentTime - startTime;
                        const progress = Math.min(elapsed / duration, 1);
                        // Easing out function
                        const easeProgress = 1 - Math.pow(1 - progress, 3);
                        const currentValue = (easeProgress * target).toFixed(1);
                        
                        counter.textContent = currentValue;
                        
                        if (progress < 1) {
                            requestAnimationFrame(updateCounter);
                        } else {
                            counter.textContent = target; // Ensure exact final value
                        }
                    };
                    
                    requestAnimationFrame(updateCounter);
                    observer.unobserve(counter);
                }
            });
        }, observerOptions);

        counters.forEach(counter => counterObserver.observe(counter));
    }

    // ---- Detection / Upload Page Logic ----
    const uploadZone = document.getElementById('uploadZone');
    const fileInput = document.getElementById('fileInput');
    const uploadContent = document.getElementById('uploadContent');
    const uploadPreview = document.getElementById('uploadPreview');
    const previewImage = document.getElementById('previewImage');
    const clearImage = document.getElementById('clearImage');
    const analyzeBtn = document.getElementById('analyzeBtn');
    
    const thresholdSlider = document.getElementById('thresholdSlider');
    const thresholdValue = document.getElementById('thresholdValue');
    const modelSelect = document.getElementById('modelSelect');

    const resultsPanel = document.getElementById('resultsPanel');
    const resultsEmpty = document.getElementById('resultsEmpty');
    const resultsLoading = document.getElementById('resultsLoading');
    const resultsContent = document.getElementById('resultsContent');
    const resultsError = document.getElementById('resultsError');
    const errorMessage = document.getElementById('errorMessage');
    const retryBtn = document.getElementById('retryBtn');
    
    const analyzeAnotherBtn = document.getElementById('analyzeAnotherBtn');
    
    // Result details elements
    const predictionBadge = document.getElementById('predictionBadge');
    const predictionIcon = document.getElementById('predictionIcon');
    const predictionLabel = document.getElementById('predictionLabel');
    const confidenceValue = document.getElementById('confidenceValue');
    const confidenceFill = document.getElementById('confidenceFill');
    const probGood = document.getElementById('probGood');
    const probGoodValue = document.getElementById('probGoodValue');
    const probDefective = document.getElementById('probDefective');
    const probDefectiveValue = document.getElementById('probDefectiveValue');
    const gradcamImage = document.getElementById('gradcamImage');
    
    const metaModel = document.getElementById('metaModel');
    const metaThreshold = document.getElementById('metaThreshold');
    const metaCalibrated = document.getElementById('metaCalibrated');
    const uncertainBadge = document.getElementById('uncertainBadge');

    // State object for caching prediction response image sources
    let activeImages = {
        original: '',
        heatmap: '',
        overlay: ''
    };
    let selectedFile = null;

    if (uploadZone) {
        // Trigger file browser on click
        uploadZone.addEventListener('click', (e) => {
            // Prevent click propagation if clicked on clear button
            if (e.target !== clearImage && !clearImage.contains(e.target)) {
                fileInput.click();
            }
        });

        // Drag events
        ['dragenter', 'dragover'].forEach(eventName => {
            uploadZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                uploadZone.classList.add('dragover');
            }, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            uploadZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                uploadZone.classList.remove('dragover');
            }, false);
        });

        // Drop handler
        uploadZone.addEventListener('drop', (e) => {
            const dt = e.dataTransfer;
            const files = dt.files;
            if (files.length > 0) {
                handleFile(files[0]);
            }
        });

        // File input change handler
        fileInput.addEventListener('change', () => {
            if (fileInput.files.length > 0) {
                handleFile(fileInput.files[0]);
            }
        });

        // Clear image preview handler
        clearImage.addEventListener('click', (e) => {
            e.stopPropagation();
            resetUploadZone();
        });

        // Example image clicks
        const exampleThumbs = document.querySelectorAll('.example-thumb');
        exampleThumbs.forEach(thumb => {
            thumb.addEventListener('click', (e) => {
                e.stopPropagation();
                const imgSrc = thumb.getAttribute('data-src');
                fetchImageAndLoad(imgSrc);
            });
        });

        // Live threshold slider value update
        thresholdSlider.addEventListener('input', () => {
            thresholdValue.textContent = parseFloat(thresholdSlider.value).toFixed(2);
        });

        // Analyze button handler
        analyzeBtn.addEventListener('click', () => {
            if (selectedFile) {
                runInference();
            }
        });

        // Grad-CAM Tab Switching
        const tabs = document.querySelectorAll('.gradcam-tab');
        tabs.forEach(tab => {
            tab.addEventListener('click', () => {
                tabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                
                const activeTab = tab.getAttribute('data-tab');
                if (activeImages[activeTab]) {
                    gradcamImage.src = activeImages[activeTab];
                }
            });
        });

        // Analyze another button handler
        analyzeAnotherBtn.addEventListener('click', () => {
            resetUploadZone();
            showResultState('empty');
        });

        // Error retry button
        retryBtn.addEventListener('click', () => {
            if (selectedFile) {
                runInference();
            }
        });
    }

    // Helper: Reset upload zone to upload prompt state
    function resetUploadZone() {
        selectedFile = null;
        fileInput.value = '';
        previewImage.src = '';
        uploadPreview.style.display = 'none';
        uploadContent.style.display = 'block';
        analyzeBtn.disabled = true;
    }

    // Helper: Validate file size & type, then load preview
    function handleFile(file) {
        // Validate type
        const allowedTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/bmp', 'image/tiff'];
        if (!allowedTypes.includes(file.type)) {
            alert('Error: Only PNG, JPEG, JPG, BMP, or TIFF images are allowed.');
            return;
        }

        // Validate size (max 16MB)
        if (file.size > 16 * 1024 * 1024) {
            alert('Error: Image size exceeds the 16MB limit.');
            return;
        }

        selectedFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            uploadContent.style.display = 'none';
            uploadPreview.style.display = 'flex';
            analyzeBtn.disabled = false;
            
            // Auto scroll to make button visible on mobile
            analyzeBtn.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        };
        reader.readAsDataURL(file);
    }

    // Helper: Fetch remote example image (as Blob) and load into upload zone
    function fetchImageAndLoad(url) {
        showResultState('loading');
        fetch(url)
            .then(res => {
                if (!res.ok) throw new Error('Failed to fetch example image');
                return res.blob();
            })
            .then(blob => {
                const filename = url.substring(url.lastIndexOf('/') + 1);
                const file = new File([blob], filename, { type: blob.type || 'image/png' });
                handleFile(file);
                showResultState('empty');
            })
            .catch(err => {
                console.error(err);
                showError('Could not load the example image. Please check path structure.');
            });
    }

    // Helper: Switch results panel views ('empty', 'loading', 'content', 'error')
    function showResultState(state) {
        resultsEmpty.style.display = state === 'empty' ? 'flex' : 'none';
        resultsLoading.style.display = state === 'loading' ? 'flex' : 'none';
        resultsContent.style.display = state === 'content' ? 'flex' : 'none';
        resultsError.style.display = state === 'error' ? 'flex' : 'none';
    }

    // Helper: Display validation/runtime error inside panel
    function showError(msg) {
        errorMessage.textContent = msg;
        showResultState('error');
    }

    // Main API caller
    function runInference() {
        if (!selectedFile) return;

        showResultState('loading');
        
        // Disable button states
        analyzeBtn.disabled = true;
        const btnText = analyzeBtn.querySelector('.btn-text');
        const btnLoader = analyzeBtn.querySelector('.btn-loader');
        btnText.style.display = 'none';
        btnLoader.style.display = 'inline-flex';

        // Prepare multi-part Form Data
        const formData = new FormData();
        formData.append('image', selectedFile);
        formData.append('architecture', modelSelect.value);
        formData.append('threshold', thresholdSlider.value);

        fetch('/api/predict', {
            method: 'POST',
            body: formData
        })
        .then(res => {
            if (!res.ok) {
                return res.json().then(data => {
                    throw new Error(data.error || 'Server error occurred during prediction.');
                });
            }
            return res.json();
        })
        .then(data => {
            if (data.success) {
                renderResults(data);
                updateRecentDetections(data);
            } else {
                throw new Error(data.error || 'Prediction process failed.');
            }
        })
        .catch(err => {
            console.error(err);
            showError(err.message || 'Network connection error. Server might be offline.');
        })
        .finally(() => {
            // Restore button states
            analyzeBtn.disabled = false;
            btnText.style.display = 'inline-flex';
            btnLoader.style.display = 'none';
        });
    }

    // Render prediction results to UI
    function renderResults(data) {
        // Cache images
        activeImages.original = `data:image/png;base64,${data.original_image}`;
        activeImages.heatmap = `data:image/png;base64,${data.heatmap}`;
        activeImages.overlay = `data:image/png;base64,${data.overlay}`;

        // Reset active tab to overlay
        const tabs = document.querySelectorAll('.gradcam-tab');
        tabs.forEach(t => t.classList.remove('active'));
        document.getElementById('tabOverlay').classList.add('active');
        gradcamImage.src = activeImages.overlay;

        // Prediction details
        const isDefective = data.is_defective;
        const displayLabel = isDefective ? 'Defective' : 'Good';
        
        // Toggle prediction class tags
        predictionBadge.className = 'prediction-badge ' + (isDefective ? 'defective' : 'good');
        predictionIcon.textContent = isDefective ? '⚠' : '✓';
        predictionLabel.textContent = displayLabel;

        // Flag predictions that are too close to the decision threshold to
        // trust at face value, and flag when the model behind this result
        // has never actually been trained (raw ImageNet fallback weights).
        if (uncertainBadge) {
            uncertainBadge.style.display = data.is_uncertain ? 'flex' : 'none';
        }
        if (data.model_trained === false) {
            predictionBadge.classList.add('untrained-model');
        }

        // Confidence details
        const confPercent = Math.round(data.confidence * 100);
        confidenceValue.textContent = `${confPercent}%`;
        confidenceFill.style.width = `${confPercent}%`;

        // Class probability details
        const probGoodPercent = Math.round(data.probabilities.good * 100);
        const probDefectPercent = Math.round(data.probabilities.defective * 100);

        probGood.style.width = `${probGoodPercent}%`;
        probGoodValue.textContent = `${probGoodPercent}%`;
        
        probDefective.style.width = `${probDefectPercent}%`;
        probDefectiveValue.textContent = `${probDefectPercent}%`;

        // Update meta items
        const selectedModelLabel = modelSelect.options[modelSelect.selectedIndex].text.split(' ')[0];
        metaModel.textContent = `Model: ${selectedModelLabel}${data.model_trained === false ? ' (untrained fallback)' : ''}`;
        metaThreshold.textContent = `Threshold: ${parseFloat(data.threshold).toFixed(2)}`;
        if (metaCalibrated) {
            metaCalibrated.style.display = data.calibrated ? 'inline' : 'none';
        }

        // Switch to content layout
        showResultState('content');
        resultsPanel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
});

    // ---- Recent Detections Logic ----
    function updateRecentDetections(resultData) {
        let recent = JSON.parse(localStorage.getItem('recentDetections') || '[]');
        recent.unshift({
            image: resultData.original_image.startsWith('data:image') ? resultData.original_image : 'data:image/png;base64,' + resultData.original_image,
            prediction: resultData.prediction,
            confidence: resultData.confidence,
            threshold: resultData.threshold,
            time: new Date().toLocaleTimeString()
        });
        if (recent.length > 5) recent.pop();
        localStorage.setItem('recentDetections', JSON.stringify(recent));
        renderRecentDetections();
    }
    
    function renderRecentDetections() {
        const recentList = document.getElementById('recentList');
        if (!recentList) return;
        
        const recent = JSON.parse(localStorage.getItem('recentDetections') || '[]');
        if (recent.length === 0) {
            recentList.innerHTML = '<p class="text-muted">No recent detections.</p>';
            return;
        }
        
        recentList.innerHTML = '';
        recent.forEach(item => {
            const isGood = item.prediction.toLowerCase() === 'good';
            const badgeClass = isGood ? 'good' : 'defective';
            const threshText = item.threshold !== undefined ? `<span style="font-size: 0.85em; color: #8a9bb4; margin-left: 6px;">(Thr: ${parseFloat(item.threshold).toFixed(2)})</span>` : '';
            recentList.innerHTML += `
                <div class="recent-item">
                    <img src="${item.image}" alt="thumb">
                    <div class="recent-item-info">
                        <strong>${item.time}</strong><br>
                        <span class="recent-badge ${badgeClass}">${item.prediction}</span>
                        <span>${(item.confidence * 100).toFixed(1)}%</span>${threshText}
                    </div>
                </div>
            `;
        });
    }
    
    // Call render on load
    renderRecentDetections();
