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
    const autoOrientToggle = document.getElementById('autoOrientToggle');
    const modelWarning = document.getElementById('modelWarning');

    // Orientation view + diagnosis elements (Detect page)
    const orientCard = document.getElementById('orientationCard');
    const orientWrap = document.getElementById('orientWrap');
    const orientImg = document.getElementById('orientImg');
    const orientBodyBox = document.getElementById('orientBodyBox');
    const orientLeadBox = document.getElementById('orientLeadBox');
    const orientText = document.getElementById('orientText');
    const orientReplay = document.getElementById('orientReplay');
    const diagnosisCard = document.getElementById('diagnosisCard');
    const diagTitle = document.getElementById('diagTitle');
    const diagNote = document.getElementById('diagNote');
    const diagBox = document.getElementById('diagBox');
    let lastOrientation = null;
    let lastDiagnosis = null;

    const resultsPanel = document.getElementById('resultsPanel');
    const resultsEmpty = document.getElementById('resultsEmpty');
    const resultsLoading = document.getElementById('resultsLoading');
    const resultsContent = document.getElementById('resultsContent');
    const resultsError = document.getElementById('resultsError');
    const errorMessage = document.getElementById('errorMessage');
    const retryBtn = document.getElementById('retryBtn');
    const oodWarning = document.getElementById('oodWarning');
    
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

        // Warn when the selected model has no trained checkpoint on this server
        if (modelSelect && modelWarning) {
            const syncModelWarning = () => {
                const opt = modelSelect.options[modelSelect.selectedIndex];
                modelWarning.classList.toggle('show', !!opt && opt.dataset.trained === 'false');
            };
            modelSelect.addEventListener('change', syncModelWarning);
            syncModelWarning();
        }

        if (orientReplay) {
            orientReplay.addEventListener('click', () => playOrientation(lastOrientation));
        }
        if (gradcamImage) {
            gradcamImage.addEventListener('load', positionDiagBox);
        }
        window.addEventListener('resize', positionDiagBox);

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
        formData.append('auto_orient', (!autoOrientToggle || autoOrientToggle.checked) ? 'true' : 'false');

        fetch('/api/predict', {
            method: 'POST',
            body: formData
        })
        .then(res => {
            if (!res.ok) {
                return res.json().then(data => {
                    const err = new Error(data.error || 'Server error occurred during prediction.');
                    err.oodRejected = !!data.ood_rejected;
                    throw err;
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
            showError(err.message || 'Network connection error. Server might be offline.', !!err.oodRejected);
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

        if (oodWarning) {
            const msgs = [];
            if (data.ood && data.ood.warning) msgs.push(data.ood.warning);
            oodWarning.textContent = msgs.join(' ');
            oodWarning.style.display = msgs.length ? 'block' : 'none';
        }
        renderOrientation(data.orientation);
        renderDiagnosis(data.diagnosis, data.is_defective);

        // Switch to content layout
        showResultState('content');
        requestAnimationFrame(positionDiagBox);   // image sizes are only known once the panel is visible
        resultsPanel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    // ---- Orientation view: animate the upload turning upright ----
    function setBox(el, box) {
        if (!el) return;
        if (!box) { el.style.display = 'none'; return; }
        el.style.display = 'block';
        el.style.left = (box[0] * 100) + '%';
        el.style.top = (box[1] * 100) + '%';
        el.style.width = ((box[2] - box[0]) * 100) + '%';
        el.style.height = ((box[3] - box[1]) * 100) + '%';
    }

    function playOrientation(o) {
        if (!orientWrap || !o) return;
        orientWrap.style.transition = 'none';
        orientWrap.style.transform = 'rotate(0deg)';
        void orientWrap.offsetWidth;   // restart the transition
        if (o.applied) {
            setTimeout(() => {
                orientWrap.style.transition = '';
                // server reports the counter-clockwise turn; CSS rotates clockwise
                orientWrap.style.transform = `rotate(${-o.rotation_deg}deg)`;
            }, 700);
        }
    }

    function renderOrientation(o) {
        lastOrientation = o || null;
        if (!orientCard) return;
        if (!o) { orientCard.style.display = 'none'; return; }
        orientCard.style.display = 'block';
        orientImg.src = previewImage.src;
        setBox(orientBodyBox, o.body_box);
        setBox(orientLeadBox, o.lead_box);

        let text;
        if (o.applied) {
            const deg = Math.round(o.rotation_deg * 10) / 10;
            const quarter = Math.abs(deg % 90) < 0.01;
            text = (quarter ? 'Sideways transistor detected' : 'Tilted transistor detected') +
                   `: the leads pointed ${o.leads_pointing}. ` +
                   `The image was turned ${deg}° counter-clockwise so the model sees it upright.`;
        } else if (o.leads_pointing === 'down') {
            text = 'Upright already (leads point down) — no rotation needed.';
        } else if (autoOrientToggle && !autoOrientToggle.checked) {
            text = 'Auto-orient is off — the image was analysed exactly as uploaded.';
        } else {
            text = 'Orientation could not be determined confidently — the image was analysed as uploaded.';
        }
        orientText.textContent = text;
        if (orientImg.complete) playOrientation(o); else orientImg.onload = () => playOrientation(o);
    }

    // ---- Suspected-defect box drawn over the Grad-CAM image ----
    function positionDiagBox() {
        if (!diagBox || !gradcamImage) return;
        const d = lastDiagnosis;
        if (!d || !d.box || !gradcamImage.clientWidth) { diagBox.style.display = 'none'; return; }
        diagBox.style.display = 'block';
        diagBox.style.left = (gradcamImage.offsetLeft + d.box[0] * gradcamImage.clientWidth) + 'px';
        diagBox.style.top = (gradcamImage.offsetTop + d.box[1] * gradcamImage.clientHeight) + 'px';
        diagBox.style.width = ((d.box[2] - d.box[0]) * gradcamImage.clientWidth) + 'px';
        diagBox.style.height = ((d.box[3] - d.box[1]) * gradcamImage.clientHeight) + 'px';
    }

    function renderDiagnosis(d, isDefective) {
        lastDiagnosis = (isDefective && d) ? d : null;
        if (!diagnosisCard) return;
        if (!lastDiagnosis) { diagnosisCard.style.display = 'none'; positionDiagBox(); return; }
        diagnosisCard.style.display = 'block';
        diagTitle.textContent = d.label;
        diagNote.textContent = d.note;
        positionDiagBox();
    }
});

    // ---- Recent Detections Logic ----
    function updateRecentDetections(resultData) {
        let recent = [];
        try { recent = JSON.parse(localStorage.getItem('recentDetections') || '[]'); } catch (e) { recent = []; }
        recent.unshift({
            image: resultData.original_image.startsWith('data:image') ? resultData.original_image : 'data:image/png;base64,' + resultData.original_image,
            prediction: resultData.prediction,
            confidence: resultData.confidence,
            probabilities: resultData.probabilities || null,
            is_defective: resultData.is_defective,
            is_uncertain: resultData.is_uncertain,
            threshold: resultData.threshold,
            architecture: resultData.architecture,
            model_trained: resultData.model_trained,
            calibrated: resultData.calibrated,
            ood: resultData.ood || null,
            orientation: resultData.orientation || null,
            diagnosis: resultData.diagnosis || null,
            time: new Date().toLocaleString()
        });
        if (recent.length > 5) recent.pop();
        try { localStorage.setItem('recentDetections', JSON.stringify(recent)); } catch (e) { /* storage full or blocked */ }
        renderRecentDetections();
    }
    
    function renderRecentDetections() {
        const recentList = document.getElementById('recentList');
        if (!recentList) return;
        
        let recent = [];
        try { recent = JSON.parse(localStorage.getItem('recentDetections') || '[]'); } catch (e) { recent = []; }
        if (recent.length === 0) {
            recentList.innerHTML = '<p class="text-muted">No recent detections.</p>';
            return;
        }
        
        recentList.innerHTML = '';
        recent.forEach((item, index) => {
            const isGood = (item.prediction || '').toLowerCase() === 'good';
            const badgeClass = isGood ? 'good' : 'defective';
            const button = document.createElement('button');
            button.type = 'button';
            button.className = 'recent-item';
            button.dataset.recentIndex = String(index);
            button.setAttribute('aria-haspopup', 'dialog');
            button.setAttribute('aria-label', `View details for ${item.prediction || 'recent'} detection from ${item.time || 'unknown time'}`);

            const image = document.createElement('img');
            image.src = item.image || '';
            image.alt = 'Detection thumbnail';
            const details = document.createElement('div');
            details.className = 'recent-item-info';
            const time = document.createElement('strong');
            time.textContent = item.time || 'Recent detection';
            const line = document.createElement('div');
            const model = document.createElement('span');
            model.className = 'recent-item-model';
            const modelNames = {
                convnext_tiny: 'ConvNeXt-Tiny',
                efficientnet_b0: 'EfficientNet-B0',
                resnet50: 'ResNet-50'
            };
            model.textContent = modelNames[item.architecture] || item.architecture || 'Model unavailable';
            const badge = document.createElement('span');
            badge.className = `recent-badge ${badgeClass}`;
            badge.textContent = item.prediction || 'Unknown';
            const confidence = document.createElement('span');
            confidence.textContent = Number.isFinite(Number(item.confidence)) ? ` ${(Number(item.confidence) * 100).toFixed(1)}%` : '';
            line.append(model, badge, confidence);
            if (item.threshold !== undefined && Number.isFinite(Number(item.threshold))) {
                const threshold = document.createElement('span');
                threshold.className = 'recent-item-threshold';
                threshold.textContent = `(Thr: ${Number(item.threshold).toFixed(2)})`;
                line.appendChild(threshold);
            }
            details.append(time, line);
            button.append(image, details);
            recentList.appendChild(button);
        });
    }

    function showRecentDetectionDetails(item, index) {
        const dialog = document.getElementById('recentDetectionModal');
        const image = document.getElementById('recentDetailImage');
        const title = document.getElementById('recentDetailTitle');
        const verdict = document.getElementById('recentDetailVerdict');
        const fields = document.getElementById('recentDetailFields');
        const notes = document.getElementById('recentDetailNotes');
        if (!dialog || !item) return;

        const isGood = (item.prediction || '').toLowerCase() === 'good';
        image.src = item.image || '';
        title.textContent = `Detection ${index + 1}`;
        verdict.textContent = item.prediction || 'Unknown';
        verdict.className = `recent-badge ${isGood ? 'good' : 'defective'}`;
        fields.replaceChildren();

        const isNumber = value => value !== null && value !== undefined && value !== '' && Number.isFinite(Number(value));
        const percent = value => isNumber(value) ? `${(Number(value) * 100).toFixed(1)}%` : '—';
        const rows = [
            ['Confidence', percent(item.confidence)],
            ['Good probability', percent(item.probabilities && item.probabilities.good)],
            ['Defective probability', percent(item.probabilities && item.probabilities.defective)],
            ['Decision threshold', isNumber(item.threshold) ? Number(item.threshold).toFixed(2) : '—'],
            ['Model', item.architecture || '—'],
            ['Checkpoint', item.model_trained === true ? 'Trained' : item.model_trained === false ? 'Untrained fallback' : '—'],
            ['Confidence calibration', item.calibrated === true ? 'Applied' : item.calibrated === false ? 'Not applied' : '—'],
            ['Review flag', item.is_uncertain === true ? 'Manual review suggested' : 'No low-margin flag'],
            ['Time', item.time || '—']
        ];
        rows.forEach(([label, value]) => {
            const row = document.createElement('div');
            row.className = 'recent-detail-field';
            const labelEl = document.createElement('span');
            labelEl.textContent = label;
            const valueEl = document.createElement('strong');
            valueEl.textContent = value;
            row.append(labelEl, valueEl);
            fields.appendChild(row);
        });

        notes.replaceChildren();
        const noteItems = [];
        if (item.ood && item.ood.warning) noteItems.push({ text: item.ood.warning, warning: true });
        if (item.orientation && item.orientation.reason) noteItems.push({ text: `Orientation: ${item.orientation.reason}` });
        if (item.diagnosis && (item.diagnosis.label || item.diagnosis.note)) {
            noteItems.push({ text: `Suspected defect: ${[item.diagnosis.label, item.diagnosis.note].filter(Boolean).join(' — ')}`, warning: true });
        }
        noteItems.forEach(note => {
            const el = document.createElement('div');
            el.className = `recent-detail-note${note.warning ? ' warning' : ''}`;
            el.textContent = note.text;
            notes.appendChild(el);
        });
        notes.hidden = noteItems.length === 0;

        if (typeof dialog.showModal === 'function') dialog.showModal();
        else dialog.setAttribute('open', '');
    }

    function setupRecentDetectionDetails() {
        const recentList = document.getElementById('recentList');
        const dialog = document.getElementById('recentDetectionModal');
        if (recentList) {
            recentList.addEventListener('click', event => {
                const button = event.target.closest('[data-recent-index]');
                if (!button) return;
                let recent = [];
                try { recent = JSON.parse(localStorage.getItem('recentDetections') || '[]'); } catch (e) { recent = []; }
                const index = Number(button.dataset.recentIndex);
                if (Number.isInteger(index) && index >= 0 && index < recent.length) {
                    showRecentDetectionDetails(recent[index], index);
                }
            });
        }
        if (!dialog) return;
        const closeButton = dialog.querySelector('[data-close-recent-detail]');
        if (closeButton) closeButton.addEventListener('click', () => dialog.close ? dialog.close() : dialog.removeAttribute('open'));
        dialog.addEventListener('click', event => {
            if (event.target === dialog) dialog.close ? dialog.close() : dialog.removeAttribute('open');
        });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', setupRecentDetectionDetails, { once: true });
    } else {
        setupRecentDetectionDetails();
    }
    
    // Call render on load
    renderRecentDetections();
