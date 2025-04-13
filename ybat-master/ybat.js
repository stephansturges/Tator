(() => {
    "use strict";

    // -----------------------------------------
    // NEW CODE: Add a global dictionary for pending bboxes, and a UUID generator.
    // -----------------------------------------
    let pendingApiBboxes = {};

    function generateUUID() {
        // Modern browsers: use crypto.randomUUID().
        // Fallback if unavailable.
        if (window.crypto && window.crypto.randomUUID) {
            return crypto.randomUUID();
        }
        // Fallback: timestamp + random
        return (
            Date.now().toString(36) +
            Math.random().toString(36).substring(2, 12)
        );
    }

    // 1) Define a palette of 100 colors spread around the hue wheel
    //    (First ~20 are roughly 0°, 18°, 36°, ..., 342°, then continuing).
    //    We'll dynamically reduce alpha for fills. Also random offset.
    const colorPalette = [];
    for (let i = 0; i < 100; i++) {
        const baseHue = i * 20; // increment by 20° each time
        const randomOffset = Math.random() * 0.3;  // up to ~6°
        const hue = (baseHue + randomOffset) % 360;
        colorPalette.push(`hsla(${hue}, 100%, 45%, 1)`);
    }

    // 2) Function that picks the correct color from the palette
    //    by using the YOLO class index stored in classes[className].
    //    We then cycle through the palette if classes are ≥100.
    function getColorFromClass(className) {
        const index = classes[className] % 100; // fallback cycle if >100 classes
        return colorPalette[index];
    }

    // 3) A little helper to convert e.g. "hsla(240,100%,45%,1)"
    //    into the same HSLA color but with alpha=0.2 for the fill.
    function withAlpha(color, alpha) {
        return color.replace(/(\d?\.?\d+)\)$/, `${alpha})`);
    }

    // Global flags for “auto mode”, “SAM mode”, “point mode”
    // plus computed “samAutoMode” (sam + auto) and “samPointAutoMode” (point + auto).
    let autoMode = false;
    let samMode = false;
    let pointMode = false;
    let samAutoMode = false;
    let samPointAutoMode = false;

    document.addEventListener("DOMContentLoaded", () => {
        const autoModeCheckbox = document.getElementById("autoMode");
        if (autoModeCheckbox) {
            autoModeCheckbox.addEventListener("change", () => {
                autoMode = autoModeCheckbox.checked;
                samAutoMode = samMode && autoMode;
                samPointAutoMode = pointMode && autoMode;
                console.log("autoMode =>", autoMode, 
                            "samAutoMode =>", samAutoMode, 
                            "samPointAutoMode =>", samPointAutoMode);
            });
        }

        const samModeCheckbox = document.getElementById("samMode");
        if (samModeCheckbox) {
            samModeCheckbox.addEventListener("change", () => {
                samMode = samModeCheckbox.checked;
                samAutoMode = samMode && autoMode;
                console.log("SAM mode =>", samMode, 
                            "samAutoMode =>", samAutoMode);
            });
        }

        const pointModeCheckbox = document.getElementById("pointMode");
        if (pointModeCheckbox) {
            pointModeCheckbox.addEventListener("change", () => {
                pointMode = pointModeCheckbox.checked;
                samPointAutoMode = pointMode && autoMode;
                console.log("pointMode =>", pointMode, 
                            "samPointAutoMode =>", samPointAutoMode);
            });
        }
    });

    // Helper that extracts base64 from currentImage
    async function extractBase64Image() {
        const offCan = document.createElement("canvas");
        offCan.width = currentImage.width;
        offCan.height = currentImage.height;
        const ctx = offCan.getContext("2d");
        ctx.drawImage(currentImage.object, 0, 0);
        const dataUrl = offCan.toDataURL("image/jpeg");
        return dataUrl.split(",")[1]; // base64 only
    }

    /*****************************************************
     * Existing SAM / CLIP calls (unchanged)
     *****************************************************/
    async function sam2BboxPrompt(bbox) {
        const base64Img = await extractBase64Image();
        const bodyData = {
            image_base64: base64Img,
            bbox_left: bbox.x,
            bbox_top: bbox.y,
            bbox_width: bbox.width,
            bbox_height: bbox.height,
            uuid: bbox.uuid
        };
        try {
            const resp = await fetch("http://localhost:8000/sam2_bbox", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(bodyData)
            });
            if (!resp.ok) {
                throw new Error("Response not OK: " + resp.statusText);
            }
            const result = await resp.json();
            const returnedUUID = result.uuid;
            const targetBbox = pendingApiBboxes[returnedUUID];
            if (!targetBbox) {
                console.warn("No pending bbox found for uuid:", returnedUUID);
                return;
            }
            currentBbox = {
                bbox: targetBbox,
                index: -1,
                originalX: targetBbox.x,
                originalY: targetBbox.y,
                originalWidth: targetBbox.width,
                originalHeight: targetBbox.height,
                moving: false,
                resizing: null
            };

            if (result.bbox) {
                const [cx, cy, ww, hh] = result.bbox;
                const absW = ww * currentImage.width;
                const absH = hh * currentImage.height;
                const absX = cx * currentImage.width - absW / 2;
                const absY = cy * currentImage.height - absH / 2;
                targetBbox.x = absX;
                targetBbox.y = absY;
                targetBbox.width = absW;
                targetBbox.height = absH;
                updateBboxAfterTransform();
                console.log("Updated SAM bounding box:", absX, absY, absW, absH);
            } else {
                console.warn("No 'bbox' field returned from sam2_bbox. Full response:", result);
            }
            delete pendingApiBboxes[returnedUUID];
        } catch (err) {
            console.error("sam2_bbox error:", err);
            alert("sam2_bbox call failed: " + err);
        }
    }

    async function sam2PointPrompt(px, py) {
        try {
            const base64Img = await extractBase64Image();
            const bodyData = {
                image_base64: base64Img,
                point_x: px,
                point_y: py,
                uuid: currentBbox ? currentBbox.bbox.uuid : null
            };
            const resp = await fetch("http://localhost:8000/sam2_point", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(bodyData)
            });
            if (!resp.ok) {
                throw new Error("sam2_point failed: " + resp.statusText);
            }
            const result = await resp.json();
            const returnedUUID = result.uuid;
            const targetBbox = pendingApiBboxes[returnedUUID];
            if (!targetBbox) {
                console.warn("No pending bbox found for uuid:", returnedUUID);
                return;
            }
            currentBbox = {
                bbox: targetBbox,
                index: -1,
                originalX: targetBbox.x,
                originalY: targetBbox.y,
                originalWidth: targetBbox.width,
                originalHeight: targetBbox.height,
                moving: false,
                resizing: null
            };

            if (!result.bbox) {
                console.warn("No 'bbox' field in sam2_point response:", result);
                return;
            }
            const [cx, cy, wNorm, hNorm] = result.bbox;
            const absW = wNorm * currentImage.width;
            const absH = hNorm * currentImage.height;
            const absX = cx * currentImage.width - absW / 2;
            const absY = cy * currentImage.height - absH / 2;
            targetBbox.x = absX;
            targetBbox.y = absY;
            targetBbox.width = absW;
            targetBbox.height = absH;
            updateBboxAfterTransform();
            console.log("Updated existing bbox from point mode:", absX, absY, absW, absH);
            delete pendingApiBboxes[returnedUUID];
        } catch (err) {
            console.error("sam2PointPrompt error:", err);
            alert("sam2PointPrompt call failed: " + err);
        }
    }

    async function autoPredictNewCrop(bbox) {
        const offCanvas = document.createElement("canvas");
        offCanvas.width = bbox.width;
        offCanvas.height = bbox.height;
        const ctx = offCanvas.getContext("2d");
        ctx.drawImage(
            currentImage.object,
            bbox.x,
            bbox.y,
            bbox.width,
            bbox.height,
            0,
            0,
            bbox.width,
            bbox.height
        );
        const base64String = offCanvas.toDataURL("image/jpeg");
        const base64Data = base64String.split(",")[1];
        const resp = await fetch("http://localhost:8000/predict_base64", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                image_base64: base64Data,
                uuid: bbox.uuid
            })
        });
        const data = await resp.json();
        console.log("autoPredictNewCrop =>", data);
        if (!data.uuid) {
            alert("Auto mode error: you probably don't have a trained .pkl file for CLIP!");
            removeBbox(bbox);
            return;
        }
        const predictedClass = data.prediction;
        const returnedUUID = data.uuid;
        const targetBbox = pendingApiBboxes[returnedUUID];
        if (!targetBbox) {
            console.warn("No pending bbox found for uuid:", returnedUUID);
            return;
        }
        currentBbox = {
            bbox: targetBbox,
            index: -1,
            originalX: targetBbox.x,
            originalY: targetBbox.y,
            originalWidth: targetBbox.width,
            originalHeight: targetBbox.height,
            moving: false,
            resizing: null
        };
        const oldClass = targetBbox.class;
        const oldArr = bboxes[currentImage.name][oldClass];
        const idx = oldArr.indexOf(targetBbox);
        if (idx !== -1) oldArr.splice(idx, 1);
        if (!bboxes[currentImage.name][predictedClass]) {
            bboxes[currentImage.name][predictedClass] = [];
        }
        targetBbox.class = predictedClass;
        bboxes[currentImage.name][predictedClass].push(targetBbox);
        delete pendingApiBboxes[returnedUUID];
    }

    async function sam2BboxAutoPrompt(bbox) {
        const base64Img = await extractBase64Image();
        const bodyData = {
            image_base64: base64Img,
            bbox_left: bbox.x,
            bbox_top: bbox.y,
            bbox_width: bbox.width,
            bbox_height: bbox.height,
            uuid: bbox.uuid
        };
        const resp = await fetch("http://localhost:8000/sam2_bbox_auto", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(bodyData)
        });
        const result = await resp.json();
        console.log("sam2_bbox_auto =>", result);
        if (!result.uuid || !result.bbox || result.bbox.length < 4) {
            alert("Auto mode error: missing 'uuid' or invalid 'bbox' in /sam2_bbox_auto response.");
            removeBbox(bbox);
            return;
        }
        const returnedUUID = result.uuid;
        const targetBbox = pendingApiBboxes[returnedUUID];
        if (!targetBbox) {
            console.warn("No pending bbox found for uuid:", returnedUUID);
            return;
        }
        currentBbox = {
            bbox: targetBbox,
            index: -1,
            originalX: targetBbox.x,
            originalY: targetBbox.y,
            originalWidth: targetBbox.width,
            originalHeight: targetBbox.height,
            moving: false,
            resizing: null
        };
        const [cx, cy, wNorm, hNorm] = result.bbox;
        const absW = wNorm * currentImage.width;
        const absH = hNorm * currentImage.height;
        const absX = cx * currentImage.width - absW / 2;
        const absY = cy * currentImage.height - absH / 2;
        const oldClass = targetBbox.class;
        const oldArr = bboxes[currentImage.name][oldClass];
        const idx = oldArr.indexOf(targetBbox);
        if (idx !== -1) oldArr.splice(idx, 1);
        const newClass = result.prediction;
        if (!bboxes[currentImage.name][newClass]) {
            bboxes[currentImage.name][newClass] = [];
        }
        targetBbox.class = newClass;
        bboxes[currentImage.name][newClass].push(targetBbox);
        targetBbox.x = absX;
        targetBbox.y = absY;
        targetBbox.width = absW;
        targetBbox.height = absH;
        updateBboxAfterTransform();
        delete pendingApiBboxes[returnedUUID];
    }

    async function sam2PointAutoPrompt(px, py) {
        const base64Img = await extractBase64Image();
        const bodyData = {
            image_base64: base64Img,
            point_x: px,
            point_y: py,
            uuid: currentBbox ? currentBbox.bbox.uuid : null
        };
        const resp = await fetch("http://localhost:8000/sam2_point_auto", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(bodyData)
        });
        const result = await resp.json();
        console.log("sam2_point_auto =>", result);
        if (!result.uuid || !result.bbox || result.bbox.length < 4) {
            alert("Auto mode error: missing 'uuid' or invalid 'bbox' in /sam2_point_auto response.");
            removeBbox(currentBbox.bbox);
            return;
        }
        const returnedUUID = result.uuid;
        const targetBbox = pendingApiBboxes[returnedUUID];
        if (!targetBbox) {
            console.warn("No pending bbox found for uuid:", returnedUUID);
            return;
        }
        currentBbox = {
            bbox: targetBbox,
            index: -1,
            originalX: targetBbox.x,
            originalY: targetBbox.y,
            originalWidth: targetBbox.width,
            originalHeight: targetBbox.height,
            moving: false,
            resizing: null
        };
        const [cx, cy, wNorm, hNorm] = result.bbox;
        const absW = wNorm * currentImage.width;
        const absH = wNorm * currentImage.height;
        const absX = cx * currentImage.width - absW / 2;
        const absY = cy * currentImage.height - absH / 2;
        const oldClass = targetBbox.class;
        const oldArr = bboxes[currentImage.name][oldClass];
        const idx = oldArr.indexOf(targetBbox);
        if (idx !== -1) oldArr.splice(idx, 1);
        const newClass = result.prediction;
        if (!bboxes[currentImage.name][newClass]) {
            bboxes[currentImage.name][newClass] = [];
        }
        targetBbox.class = newClass;
        bboxes[currentImage.name][newClass].push(targetBbox);
        targetBbox.x = absX;
        targetBbox.y = absY;
        targetBbox.width = absW;
        targetBbox.height = absH;
        updateBboxAfterTransform();
        delete pendingApiBboxes[returnedUUID];
    }

    // A few standard parameters
    const saveInterval = 60;
    const fontBaseSize = 6;
    const fontColor = "#001f3f";
    const borderColor = "#001f3f";
    const backgroundColor = "rgba(0, 116, 217, 0.2)";
    const markedFontColor = "#FF4136";
    const markedBorderColor = "#FF4136";
    const markedBackgroundColor = "rgba(255, 133, 27, 0.2)";
    const minBBoxWidth = 5;
    const minBBoxHeight = 5;
    const scrollSpeed = 1.02;
    const minZoom = 0.1;
    const maxZoom = 5;
    const edgeSize = 5;
    const resetCanvasOnChange = true;
    const defaultScale = 0.5;
    const drawCenterX = true;
    const drawGuidelines = true;
    const fittedZoom = true;
    let canvas = null;
    let images = {};
    let classes = {};
    let bboxes = {};
    const extensions = ["jpg", "jpeg", "png", "JPG", "JPEG", "PNG"];
    let currentImage = null;
    let currentClass = null;
    let currentBbox = null;
    let imageListIndex = 0;
    let classListIndex = 0;
    let scale = defaultScale;
    let canvasX = 0;
    let canvasY = 0;
    let screenX = 0;
    let screenY = 0;
    const mouse = {
        x: 0,
        y: 0,
        realX: 0,
        realY: 0,
        buttonL: false,
        buttonR: false,
        startRealX: 0,
        startRealY: 0
    };

    document.addEventListener("contextmenu", function (e) {
        e.preventDefault();
    }, false);

    const isSupported = () => {
        try {
            const key = "__test_ls_key__";
            localStorage.setItem(key, key);
            localStorage.removeItem(key);
            return true;
        } catch (e) {
            return false;
        }
    };

    if (isSupported() === true) {
        setInterval(() => {
            if (Object.keys(bboxes).length > 0) {
                localStorage.setItem("bboxes", JSON.stringify(bboxes));
            }
        }, saveInterval * 1000);
    } else {
        alert("Restore function is not supported in this browser.");
    }

    document.onreadystatechange = () => {
        if (document.readyState === "complete") {
            listenCanvas();
            listenCanvasMouse();
            listenImageLoad();
            listenImageSelect();
            listenClassLoad();
            listenClassSelect();
            listenBboxLoad();
            listenBboxSave();
            listenBboxVocSave();
            listenBboxCocoSave();
            listenBboxRestore();
            listenKeyboard();
            listenImageSearch();
            listenImageCrop();
        }
    };

    const listenCanvas = () => {
        canvas = new Canvas("canvas", document.getElementById("right").clientWidth, window.innerHeight - 20);
        canvas.on("draw", (context) => {
            if (currentImage !== null) {
                drawImage(context);
                drawNewBbox(context);
                drawExistingBboxes(context);
                drawCross(context);
            } else {
                drawIntro(context);
            }
        }).start();
    };

    const drawImage = (context) => {
        context.drawImage(
            currentImage.object,
            zoomX(0),
            zoomY(0),
            zoom(currentImage.width),
            zoom(currentImage.height)
        );
    };

    const drawIntro = (context) => {
        setFontStyles(context, false);
        context.fillText("USAGE:", zoomX(20), zoomY(50));
        context.fillText("1. Load your images (jpg, png).", zoomX(20), zoomY(100));
        context.fillText("2. Load your classes (yolo *.names).", zoomX(20), zoomY(150));
        context.fillText("3. Create bboxes or restore from zipped yolo/voc/coco.", zoomX(20), zoomY(200));
        context.fillText("NOTES:", zoomX(20), zoomY(300));
        context.fillText("1: Reloading images resets bboxes.", zoomX(20), zoomY(350));
        context.fillText("2: Check out README.md for more info.", zoomX(20), zoomY(400));
    };

    const drawNewBbox = (context) => {
        if (mouse.buttonL === true && currentClass !== null && currentBbox === null) {
            const width = (mouse.realX - mouse.startRealX);
            const height = (mouse.realY - mouse.startRealY);
            const strokeColor = getColorFromClass(currentClass);
            const fillColor = withAlpha(strokeColor, 0.2);
            context.setLineDash([]);
            context.strokeStyle = strokeColor;
            context.fillStyle = fillColor;
            context.strokeRect(
                zoomX(mouse.startRealX),
                zoomY(mouse.startRealY),
                zoom(width),
                zoom(height)
            );
            context.fillRect(
                zoomX(mouse.startRealX),
                zoomY(mouse.startRealY),
                zoom(width),
                zoom(height)
            );
            drawX(context, mouse.startRealX, mouse.startRealY, width, height);
            setBboxCoordinates(mouse.startRealX, mouse.startRealY, width, height);
        }
    };

    const drawExistingBboxes = (context) => {
        const currentBboxes = bboxes[currentImage.name];
        for (let className in currentBboxes) {
            currentBboxes[className].forEach(bbox => {
                const strokeColor = getColorFromClass(className);
                const fillColor = withAlpha(strokeColor, 0.2);
                context.font = context.font.replace(/\d+px/, `${zoom(fontBaseSize)}px`);
                context.fillStyle = strokeColor;
                context.fillText(className, zoomX(bbox.x), zoomY(bbox.y - 2));
                context.setLineDash([]);
                context.strokeStyle = strokeColor;
                context.fillStyle = fillColor;
                context.strokeRect(
                    zoomX(bbox.x),
                    zoomY(bbox.y),
                    zoom(bbox.width),
                    zoom(bbox.height)
                );
                context.fillRect(
                    zoomX(bbox.x),
                    zoomY(bbox.y),
                    zoom(bbox.width),
                    zoom(bbox.height)
                );
                drawX(context, bbox.x, bbox.y, bbox.width, bbox.height);
                if (bbox.marked === true) {
                    setBboxCoordinates(bbox.x, bbox.y, bbox.width, bbox.height);
                }
            });
        }
    };

    const drawX = (context, x, y, width, height) => {
        if (drawCenterX === true) {
            const centerX = x + width / 2;
            const centerY = y + height / 2;
            context.beginPath();
            context.moveTo(zoomX(centerX), zoomY(centerY - 10));
            context.lineTo(zoomX(centerX), zoomY(centerY + 10));
            context.stroke();
            context.beginPath();
            context.moveTo(zoomX(centerX - 10), zoomY(centerY));
            context.lineTo(zoomX(centerX + 10), zoomY(centerY));
            context.stroke();
        }
    };

    const drawCross = (context) => {
        if (drawGuidelines === true) {
            context.setLineDash([5]);
            context.beginPath();
            context.moveTo(zoomX(mouse.realX), zoomY(0));
            context.lineTo(zoomX(mouse.realX), zoomY(currentImage.height));
            context.stroke();
            context.beginPath();
            context.moveTo(zoomX(0), zoomY(mouse.realY));
            context.lineTo(zoomX(currentImage.width), zoomY(mouse.realY));
            context.stroke();
        }
    };

    const setBBoxStyles = (context, marked) => {
        context.setLineDash([]);
        if (marked === false) {
            context.strokeStyle = borderColor;
            context.fillStyle = backgroundColor;
        } else {
            context.strokeStyle = markedBorderColor;
            context.fillStyle = markedBackgroundColor;
        }
    };

    const setBboxCoordinates = (x, y, width, height) => {
        const x2 = x + width;
        const y2 = y + height;
        document.getElementById("bboxInformation").innerHTML =
            `${width}x${height} (${x}, ${y}) (${x2}, ${y2})`;
    };

    const setFontStyles = (context, marked) => {
        if (marked === false) {
            context.fillStyle = fontColor;
        } else {
            context.fillStyle = markedFontColor;
        }
        context.font = context.font.replace(/\d+px/, `${zoom(fontBaseSize)}px`);
    };

    const listenCanvasMouse = () => {
        canvas.element.addEventListener("wheel", trackWheel, { passive: false });
        canvas.element.addEventListener("mousemove", trackPointer);
        canvas.element.addEventListener("mousedown", trackPointer);
        canvas.element.addEventListener("mouseup", trackPointer);
        canvas.element.addEventListener("mouseout", trackPointer);
    };

    const trackWheel = (event) => {
        if (event.shiftKey) {
            const panSpeed = -1;
            canvasX -= event.deltaX * panSpeed;
            canvasY -= event.deltaY * panSpeed;
            mouse.realX = zoomXInv(mouse.x);
            mouse.realY = zoomYInv(mouse.y);
            event.preventDefault();
            return;
        }
        if (event.deltaY < 0) {
            scale = Math.min(maxZoom, scale * scrollSpeed);
        } else {
            scale = Math.max(minZoom, scale * (1 / scrollSpeed));
        }
        canvasX = mouse.realX;
        canvasY = mouse.realY;
        screenX = mouse.x;
        screenY = mouse.y;
        mouse.realX = zoomXInv(mouse.x);
        mouse.realY = zoomYInv(mouse.y);
        event.preventDefault();
    };

    async function trackPointer(event) {
        // 1) Standard pointer tracking
        mouse.bounds = canvas.element.getBoundingClientRect();
        mouse.x = event.clientX - mouse.bounds.left;
        mouse.y = event.clientY - mouse.bounds.top;
        const oldRealX = mouse.realX;
        const oldRealY = mouse.realY;
        mouse.realX = zoomXInv(mouse.x);
        mouse.realY = zoomYInv(mouse.y);
    
        // 2) Check for mousedown/mouseup
        if (event.type === "mousedown") {
            mouse.startRealX = mouse.realX;
            mouse.startRealY = mouse.realY;
            if (event.which === 3) {
                mouse.buttonR = true;
            } else if (event.which === 1) {
                mouse.buttonL = true;
            }
        }
        else if (event.type === "mouseup" || event.type === "mouseout") {
            // if left button was down, we have an image + class
            if (mouse.buttonL && currentImage !== null && currentClass !== null) {
    
                // POINT MODE => forcibly create a 10×10 box
                // Point-mode logic with optional “samPointAutoMode”
                if (pointMode) {
                    // Clear any existing bbox
                    currentBbox = null;
                    const dotSize = 10;
                    const half = dotSize / 2;

                    // Shift the start coords so storeNewBbox yields a 10×10 box
                    mouse.startRealX = mouse.realX - half;
                    mouse.startRealY = mouse.realY - half;
                    storeNewBbox(dotSize, dotSize);

                    // Release the mouse so we don't keep dragging
                    mouse.buttonL = false;
                    mouse.buttonR = false;

                    if (samPointAutoMode) {
                        await sam2PointAutoPrompt(mouse.realX, mouse.realY);
                    } else {
                        await sam2PointPrompt(mouse.realX, mouse.realY);
                    }
                    setBboxMarkedState();
                    if (currentBbox) {
                        updateBboxAfterTransform();
                    }
                }
                else {
                    // Normal bounding-box mode
                    const movedWidth  = Math.abs(mouse.realX - mouse.startRealX);
                    const movedHeight = Math.abs(mouse.realY - mouse.startRealY);
    
                    if (movedWidth > minBBoxWidth && movedHeight > minBBoxHeight) {
                        if (currentBbox === null) {
                            storeNewBbox(movedWidth, movedHeight);
                            mouse.buttonL = false;
                            mouse.buttonR = false;
                            if (samMode && autoMode) {
                                await sam2BboxAutoPrompt(currentBbox.bbox);
                            }
                            else if (autoMode) {
                                await autoPredictNewCrop(currentBbox.bbox);
                            }
                            else if (samMode) {
                                await sam2BboxPrompt(currentBbox.bbox);
                            }
                            else {
                                setBboxMarkedState();
                                if (currentBbox) {
                                    updateBboxAfterTransform();
                                }
                            }
                            setBboxMarkedState();
                            if (currentBbox) {
                                updateBboxAfterTransform();
                            }
                        }
                        else {
                            updateBboxAfterTransform();
                        }
                    }
                    else {
                        // small => single click logic
                        if (currentBbox === null) {
                            setBboxMarkedState();
                            if (currentBbox !== null) {
                                updateBboxAfterTransform();
                            }
                        }
                        else {
                            setBboxMarkedState();
                            if (currentBbox !== null) {
                                updateBboxAfterTransform();
                            }
                        }
                    }
                }
            }
    
            mouse.buttonR = false;
            mouse.buttonL = false;
        }
    
        moveBbox();
        resizeBbox();
        changeCursorByLocation();
        panImage(oldRealX, oldRealY);
    }

    const storeNewBbox = (movedWidth, movedHeight) => {
        const bbox = {
            x: Math.min(mouse.startRealX, mouse.realX),
            y: Math.min(mouse.startRealY, mouse.realY),
            width: movedWidth,
            height: movedHeight,
            marked: true,
            class: currentClass,
            uuid: generateUUID()
        };
        if (!bboxes[currentImage.name]) {
            bboxes[currentImage.name] = {};
        }
        if (!bboxes[currentImage.name][currentClass]) {
            bboxes[currentImage.name][currentClass] = [];
        }
        bboxes[currentImage.name][currentClass].push(bbox);
        currentBbox = {
            bbox: bbox,
            index: bboxes[currentImage.name][currentClass].length - 1,
            originalX: bbox.x,
            originalY: bbox.y,
            originalWidth: bbox.width,
            originalHeight: bbox.height,
            moving: false,
            resizing: null
        };
        pendingApiBboxes[bbox.uuid] = bbox;
    };

    const updateBboxAfterTransform = () => {
        if (currentBbox && currentBbox.resizing !== null) {
            if (currentBbox.bbox.width < 0) {
                currentBbox.bbox.width = Math.abs(currentBbox.bbox.width);
                currentBbox.bbox.x -= currentBbox.bbox.width;
            }
            if (currentBbox.bbox.height < 0) {
                currentBbox.bbox.height = Math.abs(currentBbox.bbox.height);
                currentBbox.bbox.y -= currentBbox.bbox.height;
            }
            currentBbox.resizing = null;
        }
        if (currentBbox) {
            currentBbox.bbox.marked = true;
            currentBbox.originalX = currentBbox.bbox.x;
            currentBbox.originalY = currentBbox.bbox.y;
            currentBbox.originalWidth = currentBbox.bbox.width;
            currentBbox.originalHeight = currentBbox.bbox.height;
            currentBbox.moving = false;
        }
    };

    const setBboxMarkedState = () => {
        if (!currentBbox || (!currentBbox.moving && !currentBbox.resizing)) {
            const currentBxs = bboxes[currentImage.name];
            let wasInside = false;
            let smallestBbox = Number.MAX_SAFE_INTEGER;
            for (let className in currentBxs) {
                currentBxs[className].forEach((bx, i) => {
                    bx.marked = false;
                    const endX = bx.x + bx.width;
                    const endY = bx.y + bx.height;
                    const size = bx.width * bx.height;
                    if (
                        mouse.startRealX >= bx.x && mouse.startRealX <= endX &&
                        mouse.startRealY >= bx.y && mouse.startRealY <= endY
                    ) {
                        wasInside = true;
                        if (size < smallestBbox) {
                            smallestBbox = size;
                            currentBbox = {
                                bbox: bx,
                                index: i,
                                originalX: bx.x,
                                originalY: bx.y,
                                originalWidth: bx.width,
                                originalHeight: bx.height,
                                moving: false,
                                resizing: null
                            };
                        }
                    }
                });
            }
            if (!wasInside) {
                currentBbox = null;
            }
        }
    };

    const moveBbox = () => {
        if (mouse.buttonL && currentBbox) {
            const bx = currentBbox.bbox;
            const endX = bx.x + bx.width;
            const endY = bx.y + bx.height;
            if (
                mouse.startRealX >= bx.x + edgeSize && mouse.startRealX <= endX - edgeSize &&
                mouse.startRealY >= bx.y + edgeSize && mouse.startRealY <= endY - edgeSize
            ) {
                currentBbox.moving = true;
            }
            if (currentBbox.moving) {
                bx.x = currentBbox.originalX + (mouse.realX - mouse.startRealX);
                bx.y = currentBbox.originalY + (mouse.realY - mouse.startRealY);
            }
        }
    };

    const resizeBbox = () => {
        if (mouse.buttonL && currentBbox) {
            const bx = currentBbox.bbox;
            const tlx = bx.x;
            const tly = bx.y;
            const brx = bx.x + bx.width;
            const bry = bx.y + bx.height;

            if (nearCorner(mouse.startRealX, mouse.startRealY, tlx, tly)) {
                currentBbox.resizing = "topLeft";
            } else if (nearCorner(mouse.startRealX, mouse.startRealY, tlx, bry)) {
                currentBbox.resizing = "bottomLeft";
            } else if (nearCorner(mouse.startRealX, mouse.startRealY, brx, tly)) {
                currentBbox.resizing = "topRight";
            } else if (nearCorner(mouse.startRealX, mouse.startRealY, brx, bry)) {
                currentBbox.resizing = "bottomRight";
            }

            if (currentBbox.resizing === "topLeft") {
                bx.x = mouse.realX;
                bx.y = mouse.realY;
                bx.width = currentBbox.originalX + currentBbox.originalWidth - mouse.realX;
                bx.height = currentBbox.originalY + currentBbox.originalHeight - mouse.realY;
            }
            else if (currentBbox.resizing === "bottomLeft") {
                bx.x = mouse.realX;
                bx.y = mouse.realY - (mouse.realY - currentBbox.originalY);
                bx.width = currentBbox.originalX + currentBbox.originalWidth - mouse.realX;
                bx.height = mouse.realY - currentBbox.originalY;
            }
            else if (currentBbox.resizing === "topRight") {
                bx.x = mouse.realX - (mouse.realX - currentBbox.originalX);
                bx.y = mouse.realY;
                bx.width = mouse.realX - currentBbox.originalX;
                bx.height = currentBbox.originalY + currentBbox.originalHeight - mouse.realY;
            }
            else if (currentBbox.resizing === "bottomRight") {
                bx.x = mouse.realX - (mouse.realX - currentBbox.originalX);
                bx.y = mouse.realY - (mouse.realY - currentBbox.originalY);
                bx.width = mouse.realX - currentBbox.originalX;
                bx.height = mouse.realY - currentBbox.originalY;
            }
        }
    };

    function nearCorner(px, py, cx, cy) {
        return (px >= (cx - edgeSize) && px <= (cx + edgeSize) &&
                py >= (cy - edgeSize) && py <= (cy + edgeSize));
    }

    const changeCursorByLocation = () => {
        if (!currentImage) return;
        document.body.style.cursor = "default";
        const currentBxs = bboxes[currentImage.name];
        for (let className in currentBxs) {
            for (let bx of currentBxs[className]) {
                const endX = bx.x + bx.width;
                const endY = bx.y + bx.height;
                if (mouse.realX >= (bx.x + edgeSize) && mouse.realX <= (endX - edgeSize) &&
                    mouse.realY >= (bx.y + edgeSize) && mouse.realY <= (endY - edgeSize)) {
                    document.body.style.cursor = "pointer";
                    break;
                }
            }
        }
        if (currentBbox) {
            const bx = currentBbox.bbox;
            const brx = bx.x + bx.width;
            const bry = bx.y + bx.height;
            if (mouse.realX >= bx.x + edgeSize && mouse.realX <= brx - edgeSize &&
                mouse.realY >= bx.y + edgeSize && mouse.realY <= bry - edgeSize) {
                document.body.style.cursor = "move";
            }
        }
    };

    const panImage = (xx, yy) => {
        if (mouse.buttonR) {
            canvasX -= mouse.realX - xx;
            canvasY -= mouse.realY - yy;
            mouse.realX = zoomXInv(mouse.x);
            mouse.realY = zoomYInv(mouse.y);
        }
    };

    function zoom(n) { return Math.floor(n * scale); }
    function zoomX(n) { return Math.floor((n - canvasX) * scale + screenX); }
    function zoomY(n) { return Math.floor((n - canvasY) * scale + screenY); }
    function zoomXInv(n) { return Math.floor((n - screenX) / scale + canvasX); }
    function zoomYInv(n) { return Math.floor((n - screenY) / scale + canvasY); }

    /******************************************************
     * listenImageLoad
     * We still do one pass for each file to get .width, .height
     * But we do NOT store big .object for each file (to save memory).
     ******************************************************/
    const listenImageLoad = () => {
        document.getElementById("images").addEventListener("change", (event) => {
            const imageList = document.getElementById("imageList");
            const files = event.target.files;
            if (files.length > 0) {
                resetImageList();
                document.body.style.cursor = "wait";
                let fileCount = 0;

                // We'll do a small function that reads dimension only:
                function readDimensions(file) {
                    return new Promise((resolve) => {
                        const reader = new FileReader();
                        reader.onload = () => {
                            const tempImg = new Image();
                            tempImg.onload = () => {
                                const width = tempImg.width;
                                const height = tempImg.height;
                                resolve({ width, height });
                            };
                            tempImg.src = reader.result;
                        };
                        reader.readAsDataURL(file);
                    });
                }

                // For each file => store { meta, index }, plus width/height
                // but do NOT keep an .object
                const promises = [];
                for (let i = 0; i < files.length; i++) {
                    const file = files[i];
                    const nameParts = file.name.split(".");
                    const ext = nameParts[nameParts.length - 1];
                    if (extensions.indexOf(ext) !== -1) {
                        images[file.name] = {
                            meta: file,
                            index: fileCount,
                            width: 0,
                            height: 0,
                            object: undefined // we won't store an object now
                        };
                        fileCount++;
                        const option = document.createElement("option");
                        option.value = file.name;
                        option.innerHTML = file.name;
                        if (fileCount === 1) {
                            option.selected = true;
                        }
                        imageList.appendChild(option);

                        // read dimensions
                        promises.push(
                            readDimensions(file).then((dim) => {
                                images[file.name].width = dim.width;
                                images[file.name].height = dim.height;
                            })
                        );
                    }
                }

                Promise.all(promises).then(() => {
                    // all dimension reads done
                    document.body.style.cursor = "default";

                    // Auto-select the first if any
                    if (fileCount > 0) {
                        const firstName = imageList.options[0].innerHTML;
                        setCurrentImage(images[firstName]);
                    }

                    // If classes already loaded => enable bboxes
                    if (Object.keys(classes).length > 0) {
                        document.getElementById("bboxes").disabled = false;
                        document.getElementById("restoreBboxes").disabled = false;
                    }
                });
            }
        });
    };

    const resetImageList = () => {
        document.getElementById("imageList").innerHTML = "";
        images = {};
        bboxes = {};
        currentImage = null;
    };

    /**
     * setCurrentImage(image)
     * Now we check if image.object is defined:
     *   - if not, we do a FileReader + Image() => store in image.object
     *   - then we set currentImage = { ... } referencing that object
     */
    function setCurrentImage(image) {
        if (!image) return;

        // Reset canvas if needed
        if (resetCanvasOnChange) {
            resetCanvasPlacement();
        }

        // If the image is not loaded into memory yet, load it
        if (!image.object) {
            const reader = new FileReader();
            document.body.style.cursor = "wait";
            reader.onload = () => {
                const dataUrl = reader.result;
                const imageObject = new Image();
                imageObject.onload = () => {
                    // Now we store the actual HTMLImageElement
                    image.object = imageObject; 
                    // We already have image.width + image.height from dimension pass
                    currentImage = {
                        name: image.meta.name,
                        object: imageObject,
                        width: image.width,
                        height: image.height
                    };
                    document.body.style.cursor = "default";
                    if (fittedZoom) {
                        fitZoom(currentImage);
                    }
                    document.getElementById("imageInformation").innerHTML =
                        `${image.width}x${image.height}, ${formatBytes(image.meta.size)}`;
                };
                imageObject.src = dataUrl;
            };
            reader.readAsDataURL(image.meta);
        }
        else {
            // Already loaded => just set currentImage
            currentImage = {
                name: image.meta.name,
                object: image.object,
                width: image.width,
                height: image.height
            };
            if (fittedZoom) {
                fitZoom(currentImage);
            }
            document.getElementById("imageInformation").innerHTML =
                `${image.width}x${image.height}, ${formatBytes(image.meta.size)}`;
        }

        if (currentBbox !== null) {
            currentBbox.bbox.marked = false;
            currentBbox = null;
        }
    }

    const fitZoom = (image) => {
        if (image.width > image.height) {
            scale = canvas.width / image.width;
        } else {
            scale = canvas.height / image.height;
        }
    };

    function formatBytes(bytes, decimals) {
        if (bytes === 0) {
            return "0 Bytes";
        }
        const k = 1024;
        const dm = decimals === undefined ? 2 : decimals;
        const sizes = ["Bytes", "KB", "MB"];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + " " + sizes[i];
    }

    const listenImageSelect = () => {
        const imageList = document.getElementById("imageList");
        imageList.addEventListener("change", () => {
            imageListIndex = imageList.selectedIndex;
            const name = imageList.options[imageListIndex].innerHTML;
            setCurrentImage(images[name]);
        });
    };

    /*******************************************************
     * Classes
     *******************************************************/
    const listenClassLoad = () => {
        const classesElement = document.getElementById("classes");
        classesElement.addEventListener("click", () => {
            classesElement.value = null;
        });
        classesElement.addEventListener("change", (event) => {
            const files = event.target.files;
            if (files.length > 0) {
                resetClassList();
                const nameParts = files[0].name.split(".");
                if (nameParts[nameParts.length - 1] === "txt") {
                    const reader = new FileReader();
                    reader.addEventListener("load", () => {
                        const lines = reader.result;
                        const rows = lines.split(/[\r\n]+/);
                        if (rows.length > 0) {
                            const classList = document.getElementById("classList");
                            for (let i = 0; i < rows.length; i++) {
                                rows[i] = rows[i].trim();
                                if (rows[i] !== "") {
                                    classes[rows[i]] = i;
                                    const option = document.createElement("option");
                                    option.value = i;
                                    option.innerHTML = rows[i];
                                    if (i === 0) {
                                        option.selected = true;
                                        currentClass = rows[i];
                                    }
                                    classList.appendChild(option);
                                }
                            }
                            setCurrentClass();
                            if (Object.keys(images).length > 0) {
                                document.getElementById("bboxes").disabled = false;
                                document.getElementById("restoreBboxes").disabled = false;
                            }
                        }
                    });
                    reader.readAsText(files[0]);
                }
            }
        });
    };

    const resetClassList = () => {
        document.getElementById("classList").innerHTML = "";
        classes = {};
        currentClass = null;
    };

    const setCurrentClass = () => {
        const classList = document.getElementById("classList");
        currentClass = classList.options[classList.selectedIndex].text;
        if (currentBbox !== null) {
            currentBbox.bbox.marked = false;
            currentBbox = null;
        }
    };

    const listenClassSelect = () => {
        const classList = document.getElementById("classList");
        classList.addEventListener("change", () => {
            classListIndex = classList.selectedIndex;
            setCurrentClass();
        });
    };

    /*******************************************************
     * BBox load/save
     *******************************************************/
    const listenBboxLoad = () => {
        const bboxesElement = document.getElementById("bboxes");
        bboxesElement.addEventListener("click", () => {
            bboxesElement.value = null;
        });
        bboxesElement.addEventListener("change", (event) => {
            const files = event.target.files;
            if (files.length > 0) {
                resetBboxes();
                for (let i = 0; i < files.length; i++) {
                    const reader = new FileReader();
                    const extension = files[i].name.split(".").pop();
                    reader.addEventListener("load", () => {
                        if (extension === "txt" || extension === "xml" || extension === "json") {
                            storeBbox(files[i].name, reader.result);
                        } else {
                            const zip = new JSZip();
                            zip.loadAsync(reader.result)
                                .then((result) => {
                                    for (let filename in result.files) {
                                        result.file(filename).async("string")
                                            .then((text) => {
                                                storeBbox(filename, text);
                                            });
                                    }
                                });
                        }
                    });
                    if (extension === "txt" || extension === "xml" || extension === "json") {
                        reader.readAsText(files[i]);
                    } else {
                        reader.readAsArrayBuffer(event.target.files[i]);
                    }
                }
            }
        });
    };

    const resetBboxes = () => {
        bboxes = {};
    };

    const storeBbox = (filename, text) => {
        // unmodified logic from your code
        // ...
        let image = null;
        let bbox = null;
        const extension = filename.split(".").pop();
        if (extension === "txt" || extension === "xml") {
            for (let i = 0; i < extensions.length; i++) {
                const imageName = filename.replace(`.${extension}`, `.${extensions[i]}`);
                if (typeof images[imageName] !== "undefined") {
                    image = images[imageName];
                    if (typeof bboxes[imageName] === "undefined") {
                        bboxes[imageName] = {};
                    }
                    bbox = bboxes[imageName];
                    if (extension === "txt") {
                        const rows = text.split(/[\r\n]+/);
                        for (let i = 0; i < rows.length; i++) {
                            const cols = rows[i].split(" ");
                            cols[0] = parseInt(cols[0]);
                            for (let className in classes) {
                                if (classes[className] === cols[0]) {
                                    if (typeof bbox[className] === "undefined") {
                                        bbox[className] = [];
                                    }
                                    const width = cols[3] * image.width;
                                    const x = cols[1] * image.width - width * 0.5;
                                    const height = cols[4] * image.height;
                                    const y = cols[2] * image.height - height * 0.5;
                                    bbox[className].push({
                                        x: Math.floor(x),
                                        y: Math.floor(y),
                                        width: Math.floor(width),
                                        height: Math.floor(height),
                                        marked: false,
                                        class: className
                                    });
                                    break;
                                }
                            }
                        }
                    } else if (extension === "xml") {
                        const parser = new DOMParser();
                        const xmlDoc = parser.parseFromString(text, "text/xml");
                        const objects = xmlDoc.getElementsByTagName("object");
                        for (let i = 0; i < objects.length; i++) {
                            const objectName = objects[i].getElementsByTagName("name")[0].childNodes[0].nodeValue;
                            for (let className in classes) {
                                if (className === objectName) {
                                    if (typeof bbox[className] === "undefined") {
                                        bbox[className] = [];
                                    }
                                    const bndBox = objects[i].getElementsByTagName("bndbox")[0];
                                    const bndBoxX = bndBox.getElementsByTagName("xmin")[0].childNodes[0].nodeValue;
                                    const bndBoxY = bndBox.getElementsByTagName("ymin")[0].childNodes[0].nodeValue;
                                    const bndBoxMaxX = bndBox.getElementsByTagName("xmax")[0].childNodes[0].nodeValue;
                                    const bndBoxMaxY = bndBox.getElementsByTagName("ymax")[0].childNodes[0].nodeValue;
                                    bbox[className].push({
                                        x: parseInt(bndBoxX),
                                        y: parseInt(bndBoxY),
                                        width: parseInt(bndBoxMaxX) - parseInt(bndBoxX),
                                        height: parseInt(bndBoxMaxY) - parseInt(bndBoxY),
                                        marked: false,
                                        class: className
                                    });
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        } else {
            const json = JSON.parse(text);
            for (let i = 0; i < json.annotations.length; i++) {
                let imageName = null;
                let categoryName = null;
                for (let j = 0; j < json.images.length; j++) {
                    if (json.annotations[i].image_id === json.images[j].id) {
                        imageName = json.images[j].file_name;
                        if (typeof images[imageName] !== "undefined") {
                            image = images[imageName];
                            if (typeof bboxes[imageName] === "undefined") {
                                bboxes[imageName] = {};
                            }
                            bbox = bboxes[imageName];
                            break;
                        }
                    }
                }
                for (let j = 0; j < json.categories.length; j++) {
                    if (json.annotations[i].category_id === json.categories[j].id) {
                        categoryName = json.categories[j].name;
                        break;
                    }
                }
                for (let className in classes) {
                    if (className === categoryName) {
                        if (typeof bbox[className] === "undefined") {
                            bbox[className] = [];
                        }
                        const bboxX = json.annotations[i].bbox[0];
                        const bboxY = json.annotations[i].bbox[1];
                        const bboxWidth = json.annotations[i].bbox[2];
                        const bboxHeight = json.annotations[i].bbox[3];
                        bbox[className].push({
                            x: bboxX,
                            y: bboxY,
                            width: bboxWidth,
                            height: bboxHeight,
                            marked: false,
                            class: className
                        });
                        break;
                    }
                }
            }
        }
    };

    const listenBboxSave = () => {
        document.getElementById("saveBboxes").addEventListener("click", () => {
            const zip = new JSZip();
            for (let imageName in bboxes) {
                const image = images[imageName];
                if (!image) continue;
                const name = imageName.split(".");
                name[name.length - 1] = "txt";
                const result = [];
                for (let className in bboxes[imageName]) {
                    for (let i = 0; i < bboxes[imageName][className].length; i++) {
                        const bbox = bboxes[imageName][className][i];
                        const x = (bbox.x + bbox.width / 2) / image.width;
                        const y = (bbox.y + bbox.height / 2) / image.height;
                        const w = bbox.width / image.width;
                        const h = bbox.height / image.height;
                        result.push(`${classes[className]} ${x} ${y} ${w} ${h}`);
                    }
                }
                zip.file(name.join("."), result.join("\n"));
            }
            zip.generateAsync({ type: "blob" })
                .then((blob) => {
                    saveAs(blob, "bboxes_yolo.zip");
                });
        });
    };

    const listenBboxVocSave = () => {
        document.getElementById("saveVocBboxes").addEventListener("click", () => {
            const folderPath = document.getElementById("vocFolder").value;
            const zip = new JSZip();
            for (let imageName in bboxes) {
                const image = images[imageName];
                if (!image) continue;
                const name = imageName.split(".");
                name[name.length - 1] = "xml";
                const result = [
                    "<?xml version=\"1.0\"?>",
                    "<annotation>",
                    `<folder>${folderPath}</folder>`,
                    `<filename>${imageName}</filename>`,
                    "<path/>",
                    "<source>",
                    "<database>Unknown</database>",
                    "</source>",
                    "<size>",
                    `<width>${image.width}</width>`,
                    `<height>${image.height}</height>`,
                    "<depth>3</depth>",
                    "</size>",
                    "<segmented>0</segmented>"
                ];
                for (let className in bboxes[imageName]) {
                    for (let i = 0; i < bboxes[imageName][className].length; i++) {
                        const bbox = bboxes[imageName][className][i];
                        result.push("<object>");
                        result.push(`<name>${className}</name>`);
                        result.push("<pose>Unspecified</pose>");
                        result.push("<truncated>0</truncated>");
                        result.push("<occluded>0</occluded>");
                        result.push("<difficult>0</difficult>");
                        result.push("<bndbox>");
                        result.push(`<xmin>${bbox.x}</xmin>`);
                        result.push(`<ymin>${bbox.y}</ymin>`);
                        result.push(`<xmax>${bbox.x + bbox.width}</xmax>`);
                        result.push(`<ymax>${bbox.y + bbox.height}</ymax>`);
                        result.push("</bndbox>");
                        result.push("</object>");
                    }
                }
                result.push("</annotation>");
                if (result.length > 15) {
                    zip.file(name.join("."), result.join("\n"));
                }
            }
            zip.generateAsync({ type: "blob" })
                .then((blob) => {
                    saveAs(blob, "bboxes_voc.zip");
                });
        });
    };

    const listenBboxCocoSave = () => {
        document.getElementById("saveCocoBboxes").addEventListener("click", () => {
            const zip = new JSZip();
            const result = {
                images: [],
                type: "instances",
                annotations: [],
                categories: []
            };
            for (let className in classes) {
                result.categories.push({
                    supercategory: "none",
                    id: classes[className] + 1,
                    name: className
                });
            }
            for (let imageName in images) {
                const im = images[imageName];
                result.images.push({
                    id: im.index + 1,
                    file_name: imageName,
                    width: im.width,
                    height: im.height
                });
            }
            let id = 0;
            for (let imageName in bboxes) {
                const image = images[imageName];
                if (!image) continue;
                for (let className in bboxes[imageName]) {
                    for (let i = 0; i < bboxes[imageName][className].length; i++) {
                        const bbox = bboxes[imageName][className][i];
                        const segmentation = [
                            bbox.x, bbox.y,
                            bbox.x, bbox.y + bbox.height,
                            bbox.x + bbox.width, bbox.y + bbox.height,
                            bbox.x + bbox.width, bbox.y
                        ];
                        result.annotations.push({
                            segmentation: segmentation,
                            area: bbox.width * bbox.height,
                            iscrowd: 0,
                            ignore: 0,
                            image_id: image.index + 1,
                            bbox: [bbox.x, bbox.y, bbox.width, bbox.height],
                            category_id: classes[className] + 1,
                            id: ++id
                        });
                    }
                }
            }
            zip.file("coco.json", JSON.stringify(result));
            zip.generateAsync({ type: "blob" })
                .then((blob) => {
                    saveAs(blob, "bboxes_coco.zip");
                });
        });
    };

    const listenBboxRestore = () => {
        document.getElementById("restoreBboxes").addEventListener("click", () => {
            const item = localStorage.getItem("bboxes");
            if (item) {
                bboxes = JSON.parse(item);
            }
        });
    };

    const listenKeyboard = () => {
        const imageList = document.getElementById("imageList");
        const classList = document.getElementById("classList");
        document.addEventListener("keydown", (event) => {
            const key = event.keyCode || event.charCode;
            if (key === 8 || (key === 46 && event.metaKey === true)) {
                if (currentBbox !== null) {
                    bboxes[currentImage.name][currentBbox.bbox.class].splice(currentBbox.index, 1);
                    currentBbox = null;
                    document.body.style.cursor = "default";
                }
                event.preventDefault();
            }
            // 'a' => toggle autoMode
            if (key === 65) {
                const autoModeCheckbox = document.getElementById("autoMode");
                if (autoModeCheckbox) {
                    autoModeCheckbox.checked = !autoModeCheckbox.checked;
                    autoMode = autoModeCheckbox.checked;
                    samAutoMode = samMode && autoMode;
                    samPointAutoMode = pointMode && autoMode;
                    console.log("Auto mode toggled via 'A':", autoMode);
                }
                event.preventDefault();
            }
            // 's' => toggle SAM
            if (key === 83) {
                const samModeCheckbox = document.getElementById("samMode");
                if (samModeCheckbox) {
                    samModeCheckbox.checked = !samModeCheckbox.checked;
                    samMode = samModeCheckbox.checked;
                    samAutoMode = samMode && autoMode;
                    console.log("SAM mode toggled via 'S':", samMode);
                }
                event.preventDefault();
            }
            if (key === 37) {
                if (imageList.length > 1) {
                    imageList.options[imageListIndex].selected = false;
                    if (imageListIndex === 0) {
                        imageListIndex = imageList.length - 1;
                    } else {
                        imageListIndex--;
                    }
                    imageList.options[imageListIndex].selected = true;
                    imageList.selectedIndex = imageListIndex;
                    setCurrentImage(images[imageList.options[imageListIndex].innerHTML]);
                    document.body.style.cursor = "default";
                }
                event.preventDefault();
            }
            if (key === 39) {
                if (imageList.length > 1) {
                    imageList.options[imageListIndex].selected = false;
                    if (imageListIndex === imageList.length - 1) {
                        imageListIndex = 0;
                    } else {
                        imageListIndex++;
                    }
                    imageList.options[imageListIndex].selected = true;
                    imageList.selectedIndex = imageListIndex;
                    setCurrentImage(images[imageList.options[imageListIndex].innerHTML]);
                    document.body.style.cursor = "default";
                }
                event.preventDefault();
            }
            if (key === 38) {
                if (classList.length > 1) {
                    classList.options[classListIndex].selected = false;
                    if (classListIndex === 0) {
                        classListIndex = classList.length - 1;
                    } else {
                        classListIndex--;
                    }
                    classList.options[classListIndex].selected = true;
                    classList.selectedIndex = classListIndex;
                    setCurrentClass();
                }
                event.preventDefault();
            }
            if (key === 40) {
                if (classList.length > 1) {
                    classList.options[classListIndex].selected = false;
                    if (classListIndex === classList.length - 1) {
                        classListIndex = 0;
                    } else {
                        classListIndex++;
                    }
                    classList.options[classListIndex].selected = true;
                    classList.selectedIndex = classListIndex;
                    setCurrentClass();
                }
                event.preventDefault();
            }
        });
    };

    const resetCanvasPlacement = () => {
        scale = defaultScale;
        canvasX = 0;
        canvasY = 0;
        screenX = 0;
        screenY = 0;
        mouse.x = 0;
        mouse.y = 0;
        mouse.realX = 0;
        mouse.realY = 0;
        mouse.buttonL = 0;
        mouse.buttonR = 0;
        mouse.startRealX = 0;
        mouse.startRealY = 0;
    };

    const listenImageSearch = () => {
        document.getElementById("imageSearch").addEventListener("input", (event) => {
            const value = event.target.value;
            for (let imageName in images) {
                if (imageName.indexOf(value) !== -1) {
                    document.getElementById("imageList").selectedIndex = images[imageName].index;
                    setCurrentImage(images[imageName]);
                    break;
                }
            }
        });
    };

    function yieldToDom(delayMs = 50) {
        return new Promise((resolve) => {
            requestAnimationFrame(() => {
                setTimeout(resolve, delayMs);
            });
        });
    }

    async function extractBase64ForImage(imgObj) {
        const offCanvas = document.createElement("canvas");
        offCanvas.width = imgObj.width;
        offCanvas.height = imgObj.height;
        const ctx = offCanvas.getContext("2d");
        ctx.drawImage(imgObj.object, 0, 0, imgObj.width, imgObj.height);
        const dataUrl = offCanvas.toDataURL("image/jpeg");
        return dataUrl.split(",")[1];
    }

    async function listenImageCrop() {
        const btn = document.getElementById("cropImages");
        btn.addEventListener("click", async () => {
            const payloadImages = [];
            for (const imgName in bboxes) {
                const imgData = images[imgName];
                if (!imgData) continue;

                // Make sure it's loaded (not strictly necessary if you're only sending base64?)
                if (!imgData.object) {
                    // load it
                    await new Promise((resolve) => {
                        const r = new FileReader();
                        r.onload = () => {
                            const im = new Image();
                            im.onload = () => {
                                imgData.object = im;
                                resolve();
                            };
                            im.src = r.result;
                        };
                        r.readAsDataURL(imgData.meta);
                    });
                }

                const base64Img = await extractBase64ForImage(imgData);
                const bbs = [];
                for (const className in bboxes[imgName]) {
                    bboxes[imgName][className].forEach(bb => {
                        bbs.push({
                            className,
                            x: bb.x,
                            y: bb.y,
                            width: bb.width,
                            height: bb.height
                        });
                    });
                }
                if (bbs.length === 0) continue;
                payloadImages.push({
                    image_base64: base64Img,
                    originalName: imgName,
                    bboxes: bbs
                });
            }
            if (payloadImages.length === 0) {
                alert("No bounding boxes to crop.");
                return;
            }
            const progressModal = showProgressModal("Preparing chunked job...");
            document.body.style.cursor = "wait";
            try {
                let resp = await fetch("http://localhost:8000/crop_zip_init", {
                    method: "POST"
                });
                if (!resp.ok) {
                    throw new Error("crop_zip_init failed: " + resp.status);
                }
                const { jobId } = await resp.json();
                console.log("Got jobId:", jobId);
                const chunkSize = 5;
                const chunks = chunkArray(payloadImages, chunkSize);
                let count = 0;
                for (const batch of chunks) {
                    count++;
                    console.log(`Sending batch ${count}/${chunks.length} to /crop_zip_chunk...`);
                    resp = await fetch("http://localhost:8000/crop_zip_chunk?jobId=" + jobId, {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ images: batch })
                    });
                    if (!resp.ok) {
                        throw new Error("crop_zip_chunk failed: " + resp.status);
                    }
                }
                console.log("All chunks sent. Now finalizing /crop_zip_finalize...");
                resp = await fetch("http://localhost:8000/crop_zip_finalize?jobId=" + jobId);
                if (!resp.ok) {
                    throw new Error("crop_zip_finalize failed: " + resp.status);
                }
                const blob = await resp.blob();
                saveAs(blob, "crops.zip");
                alert("Done! Single crops.zip downloaded.");
            } catch (err) {
                console.error(err);
                alert("Crop & Save failed: " + err);
            } finally {
                progressModal.close();
                document.body.style.cursor = "default";
            }
        });
    }

    function chunkArray(array, size) {
        const result = [];
        for (let i = 0; i < array.length; i += size) {
            result.push(array.slice(i, i + size));
        }
        return result;
    }

    function showProgressModal(initialText = "") {
        const overlay = document.createElement("div");
        overlay.style.position = "fixed";
        overlay.style.top = "0";
        overlay.style.left = "0";
        overlay.style.width = "100%";
        overlay.style.height = "100%";
        overlay.style.backgroundColor = "rgba(0, 0, 0, 0.75)";
        overlay.style.zIndex = "9999";
        overlay.style.display = "flex";
        overlay.style.alignItems = "center";
        overlay.style.justifyContent = "center";
        overlay.style.flexDirection = "column";
        const textDiv = document.createElement("div");
        textDiv.style.color = "#fff";
        textDiv.style.fontSize = "20px";
        textDiv.style.marginBottom = "10px";
        textDiv.textContent = initialText;
        overlay.appendChild(textDiv);
        document.body.appendChild(overlay);
        return {
            close() {
                document.body.removeChild(overlay);
            }
        };
    }

})();
