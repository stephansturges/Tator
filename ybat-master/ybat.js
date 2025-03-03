(() => {
    "use strict";

    // 1) Define a palette of 100 colors spread around the hue wheel
    //    (First ~20 are roughly 0°, 18°, 36°, ..., 342°, then continuing).
    //    We'll dynamically reduce alpha for fills. Also random offset.
    const colorPalette = [];
    for (let i = 0; i < 100; i++) {
        const baseHue = i * 20; // increment by 20° each time
        const randomOffset = Math.random() * 20 * 0.3;  // up to ~6°
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

        // Add a global flag for "auto mode", default off
    let autoMode = false;

    // Once the DOM is ready, we can listen to a new checkbox #autoMode
    document.addEventListener("DOMContentLoaded", () => {
        const autoModeCheckbox = document.getElementById("autoMode");
        if (autoModeCheckbox) {
            autoModeCheckbox.addEventListener("change", () => {
                autoMode = autoModeCheckbox.checked;
            });
        }
    });

    // ----------------------------------------------------------------------------
    // 4) We'll add a helper function to create a local crop (base64) from the canvas
    //    for the newly drawn bbox, then call the Python API for a predicted class
    // ----------------------------------------------------------------------------
    async function autoPredictNewCrop(bbox) {
        // We assume you have a global "canvas" and "currentImage"
        // We'll create an offscreen canvas to draw just that sub-region
        const offCanvas = document.createElement("canvas");
        offCanvas.width = bbox.width;
        offCanvas.height = bbox.height;
        const ctx = offCanvas.getContext("2d");

        // We must read from the displayed image. Because we have transforms (zoom, pan),
        // we need the *actual* image data. So let's directly draw from currentImage.object.
        // We'll just draw the region from the original image (x, y, w, h).
        // Bbox coords are in real image coordinates, so we can directly do:
        ctx.drawImage(
            currentImage.object,
            bbox.x, bbox.y, bbox.width, bbox.height,
            0, 0, bbox.width, bbox.height
        );

        // Now get base64 from offCanvas
        const base64String = offCanvas.toDataURL("image/jpeg");
        // remove the "data:image/jpeg;base64," prefix:
        const base64Data = base64String.split(',')[1];

        try {
            // call your inference endpoint, e.g. http://localhost:8000/predict_base64
            const resp = await fetch("http://localhost:8000/predict_base64", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ image_base64: base64Data }),
            });
            const data = await resp.json();
            const predictedClass = data.prediction;
            // Show a popup with the predicted class
            //alert("Predicted class: " + predictedClass);
            // 1) Immediately call autoPredict for the newly created bbox
            // 2) Remove this bbox from its old class array
             const oldClass = bbox.class;
             const oldArr = bboxes[currentImage.name][oldClass];
             const idx = oldArr.indexOf(bbox);
             if (idx !== -1) {
                 oldArr.splice(idx, 1);
             }
     
             // 3) Add bbox to the new predicted class array
             if (!bboxes[currentImage.name][predictedClass]) {
                 bboxes[currentImage.name][predictedClass] = [];
             }
             bbox.class = predictedClass;
             bboxes[currentImage.name][predictedClass].push(bbox);
     
             console.log("Auto-assigned class:", predictedClass,
                         "→ numeric index:", classes[predictedClass]);
            // TODO: If you want to auto-assign to the newly created bbox, you can do so here
        } catch (e) {
            console.error("Error calling API: ", e);
            alert("Error calling prediction API");
        }
    }

    // ----------------------------------------------------------------------------------
    // All existing code below, now with a small change: after we create a new bbox,
    // we call autoPredictNewCrop(bbox).
    // ----------------------------------------------------------------------------------

    // Parameters
    const saveInterval = 60; // Bbox recovery save in seconds
    const fontBaseSize = 6; // Text size in pixels
    const fontColor = "#001f3f"; // Base font color (unused in box text now)
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

    // Draw new bbox with color
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

    const trackPointer = (event) => {
        mouse.bounds = canvas.element.getBoundingClientRect();
        mouse.x = event.clientX - mouse.bounds.left;
        mouse.y = event.clientY - mouse.bounds.top;
        const xx = mouse.realX;
        const yy = mouse.realY;
        mouse.realX = zoomXInv(mouse.x);
        mouse.realY = zoomYInv(mouse.y);

        if (event.type === "mousedown") {
            mouse.startRealX = mouse.realX;
            mouse.startRealY = mouse.realY;
            if (event.which === 3) {
                mouse.buttonR = true;
            } else if (event.which === 1) {
                mouse.buttonL = true;
            }
        } else if (event.type === "mouseup" || event.type === "mouseout") {
            if (mouse.buttonL === true && currentImage !== null && currentClass !== null) {
                const movedWidth = Math.max((mouse.startRealX - mouse.realX), (mouse.realX - mouse.startRealX));
                const movedHeight = Math.max((mouse.startRealY - mouse.realY), (mouse.realY - mouse.startRealY));
                if (movedWidth > minBBoxWidth && movedHeight > minBBoxHeight) {
                    if (currentBbox === null) {
                        storeNewBbox(movedWidth, movedHeight);

                        // <--- NEW CODE: we detect a newly created box, so call autoPredict
                        //const lastBbox = currentBbox.bbox;
                        //autoPredictNewCrop(lastBbox);
                        // 1) Only call autoPredict if autoMode is true
                        if (autoMode) {
                                autoPredictNewCrop(currentBbox.bbox);
                            } else {
                                // Normal mode: Bbox is already assigned the user-selected class
                                console.log("Created a new bbox in normal mode:", currentBbox.bbox.class);
                            }
                    } else {
                        updateBboxAfterTransform();
                    }
                } else {
                    setBboxMarkedState();
                    if (currentBbox !== null) {
                        updateBboxAfterTransform();
                    }
                }
            }
            mouse.buttonR = false;
            mouse.buttonL = false;
        }

        moveBbox();
        resizeBbox();
        changeCursorByLocation();
        panImage(xx, yy);
    };

    const storeNewBbox = (movedWidth, movedHeight) => {
        const bbox = {
            x: Math.min(mouse.startRealX, mouse.realX),
            y: Math.min(mouse.startRealY, mouse.realY),
            width: movedWidth,
            height: movedHeight,
            marked: true,
            class: currentClass
        };
        if (typeof bboxes[currentImage.name] === "undefined") {
            bboxes[currentImage.name] = {};
        }
        if (typeof bboxes[currentImage.name][currentClass] === "undefined") {
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
    };

    const updateBboxAfterTransform = () => {
        if (currentBbox.resizing !== null) {
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
        currentBbox.bbox.marked = true;
        currentBbox.originalX = currentBbox.bbox.x;
        currentBbox.originalY = currentBbox.bbox.y;
        currentBbox.originalWidth = currentBbox.bbox.width;
        currentBbox.originalHeight = currentBbox.bbox.height;
        currentBbox.moving = false;
    };

    const setBboxMarkedState = () => {
        if (currentBbox === null || (currentBbox.moving === false && currentBbox.resizing === null)) {
            const currentBboxes = bboxes[currentImage.name];
            let wasInside = false;
            let smallestBbox = Number.MAX_SAFE_INTEGER;
            for (let className in currentBboxes) {
                for (let i = 0; i < currentBboxes[className].length; i++) {
                    const bbox = currentBboxes[className][i];
                    bbox.marked = false;
                    const endX = bbox.x + bbox.width;
                    const endY = bbox.y + bbox.height;
                    const size = bbox.width * bbox.height;
                    if (mouse.startRealX >= bbox.x && mouse.startRealX <= endX
                        && mouse.startRealY >= bbox.y && mouse.startRealY <= endY) {
                        wasInside = true;
                        if (size < smallestBbox) {
                            smallestBbox = size;
                            currentBbox = {
                                bbox: bbox,
                                index: i,
                                originalX: bbox.x,
                                originalY: bbox.y,
                                originalWidth: bbox.width,
                                originalHeight: bbox.height,
                                moving: false,
                                resizing: null
                            };
                        }
                    }
                }
            }
            if (wasInside === false) {
                currentBbox = null;
            }
        }
    };

    const moveBbox = () => {
        if (mouse.buttonL === true && currentBbox !== null) {
            const endX = currentBbox.bbox.x + currentBbox.bbox.width;
            const endY = currentBbox.bbox.y + currentBbox.bbox.height;
            if (mouse.startRealX >= (currentBbox.bbox.x + edgeSize) &&
                mouse.startRealX <= (endX - edgeSize) &&
                mouse.startRealY >= (currentBbox.bbox.y + edgeSize) &&
                mouse.startRealY <= (endY - edgeSize)) {
                currentBbox.moving = true;
            }
            if (currentBbox.moving === true) {
                currentBbox.bbox.x = currentBbox.originalX + (mouse.realX - mouse.startRealX);
                currentBbox.bbox.y = currentBbox.originalY + (mouse.realY - mouse.startRealY);
            }
        }
    };

    const resizeBbox = () => {
        if (mouse.buttonL === true && currentBbox !== null) {
            const topLeftX = currentBbox.bbox.x;
            const topLeftY = currentBbox.bbox.y;
            const bottomLeftX = currentBbox.bbox.x;
            const bottomLeftY = currentBbox.bbox.y + currentBbox.bbox.height;
            const topRightX = currentBbox.bbox.x + currentBbox.bbox.width;
            const topRightY = currentBbox.bbox.y;
            const bottomRightX = currentBbox.bbox.x + currentBbox.bbox.width;
            const bottomRightY = currentBbox.bbox.y + currentBbox.bbox.height;

            // Get correct corner
            if (mouse.startRealX >= (topLeftX - edgeSize) && mouse.startRealX <= (topLeftX + edgeSize) &&
                mouse.startRealY >= (topLeftY - edgeSize) && mouse.startRealY <= (topLeftY + edgeSize)) {
                currentBbox.resizing = "topLeft";
            } else if (mouse.startRealX >= (bottomLeftX - edgeSize) && mouse.startRealX <= (bottomLeftX + edgeSize) &&
                mouse.startRealY >= (bottomLeftY - edgeSize) && mouse.startRealY <= (bottomLeftY + edgeSize)) {
                currentBbox.resizing = "bottomLeft";
            } else if (mouse.startRealX >= (topRightX - edgeSize) && mouse.startRealX <= (topRightX + edgeSize) &&
                mouse.startRealY >= (topRightY - edgeSize) && mouse.startRealY <= (topRightY + edgeSize)) {
                currentBbox.resizing = "topRight";
            } else if (mouse.startRealX >= (bottomRightX - edgeSize) && mouse.startRealX <= (bottomRightX + edgeSize) &&
                mouse.startRealY >= (bottomRightY - edgeSize) && mouse.startRealY <= (bottomRightY + edgeSize)) {
                currentBbox.resizing = "bottomRight";
            }

            if (currentBbox.resizing === "topLeft") {
                currentBbox.bbox.x = mouse.realX;
                currentBbox.bbox.y = mouse.realY;
                currentBbox.bbox.width = currentBbox.originalX + currentBbox.originalWidth - mouse.realX;
                currentBbox.bbox.height = currentBbox.originalY + currentBbox.originalHeight - mouse.realY;
            } else if (currentBbox.resizing === "bottomLeft") {
                currentBbox.bbox.x = mouse.realX;
                currentBbox.bbox.y = mouse.realY - (mouse.realY - currentBbox.originalY);
                currentBbox.bbox.width = currentBbox.originalX + currentBbox.originalWidth - mouse.realX;
                currentBbox.bbox.height = mouse.realY - currentBbox.originalY;
            } else if (currentBbox.resizing === "topRight") {
                currentBbox.bbox.x = mouse.realX - (mouse.realX - currentBbox.originalX);
                currentBbox.bbox.y = mouse.realY;
                currentBbox.bbox.width = mouse.realX - currentBbox.originalX;
                currentBbox.bbox.height = currentBbox.originalY + currentBbox.originalHeight - mouse.realY;
            } else if (currentBbox.resizing === "bottomRight") {
                currentBbox.bbox.x = mouse.realX - (mouse.realX - currentBbox.originalX);
                currentBbox.bbox.y = mouse.realY - (mouse.realY - currentBbox.originalY);
                currentBbox.bbox.width = mouse.realX - currentBbox.originalX;
                currentBbox.bbox.height = mouse.realY - currentBbox.originalY;
            }
        }
    };

    const changeCursorByLocation = () => {
        if (currentImage !== null) {
            const currentBboxes = bboxes[currentImage.name];
            for (let className in currentBboxes) {
                for (let i = 0; i < currentBboxes[className].length; i++) {
                    const bbox = currentBboxes[className][i];
                    const endX = bbox.x + bbox.width;
                    const endY = bbox.y + bbox.height;
                    if (mouse.realX >= (bbox.x + edgeSize) && mouse.realX <= (endX - edgeSize) &&
                        mouse.realY >= (bbox.y + edgeSize) && mouse.realY <= (endY - edgeSize)) {
                        document.body.style.cursor = "pointer";
                        break;
                    } else {
                        document.body.style.cursor = "default";
                    }
                }
            }
            if (currentBbox !== null) {
                const topLeftX = currentBbox.bbox.x;
                const topLeftY = currentBbox.bbox.y;
                const bottomLeftX = currentBbox.bbox.x;
                const bottomLeftY = currentBbox.bbox.y + currentBbox.bbox.height;
                const topRightX = currentBbox.bbox.x + currentBbox.bbox.width;
                const topRightY = currentBbox.bbox.y;
                const bottomRightX = currentBbox.bbox.x + currentBbox.bbox.width;
                const bottomRightY = currentBbox.bbox.y + currentBbox.bbox.height;
                if (mouse.realX >= (topLeftX + edgeSize) && mouse.realX <= (bottomRightX - edgeSize) &&
                    mouse.realY >= (topLeftY + edgeSize) && mouse.realY <= (bottomRightY - edgeSize)) {
                    document.body.style.cursor = "move";
                } else if (mouse.realX >= (topLeftX - edgeSize) && mouse.realX <= (topLeftX + edgeSize) &&
                    mouse.realY >= (topLeftY - edgeSize) && mouse.realY <= (topLeftY + edgeSize)) {
                    document.body.style.cursor = "nwse-resize";
                } else if (mouse.realX >= (bottomLeftX - edgeSize) && mouse.realX <= (bottomLeftX + edgeSize) &&
                    mouse.realY >= (bottomLeftY - edgeSize) && mouse.realY <= (bottomLeftY + edgeSize)) {
                    document.body.style.cursor = "nesw-resize";
                } else if (mouse.realX >= (topRightX - edgeSize) && mouse.realX <= (topRightX + edgeSize) &&
                    mouse.realY >= (topRightY - edgeSize) && mouse.realY <= (topRightY + edgeSize)) {
                    document.body.style.cursor = "nesw-resize";
                } else if (mouse.realX >= (bottomRightX - edgeSize) && mouse.realX <= (bottomRightX + edgeSize) &&
                    mouse.realY >= (bottomRightY - edgeSize) && mouse.realY <= (bottomRightY + edgeSize)) {
                    document.body.style.cursor = "nwse-resize";
                }
            }
        }
    };

    const panImage = (xx, yy) => {
        if (mouse.buttonR === true) {
            canvasX -= mouse.realX - xx;
            canvasY -= mouse.realY - yy;
            mouse.realX = zoomXInv(mouse.x);
            mouse.realY = zoomYInv(mouse.y);
        }
    };

    const zoom = (number) => {
        return Math.floor(number * scale);
    };
    const zoomX = (number) => {
        return Math.floor((number - canvasX) * scale + screenX);
    };
    const zoomY = (number) => {
        return Math.floor((number - canvasY) * scale + screenY);
    };
    const zoomXInv = (number) => {
        return Math.floor((number - screenX) * (1 / scale) + canvasX);
    };
    const zoomYInv = (number) => {
        return Math.floor((number - screenY) * (1 / scale) + canvasY);
    };

    const listenImageLoad = () => {
        document.getElementById("images").addEventListener("change", (event) => {
            const imageList = document.getElementById("imageList");
            const files = event.target.files;
            if (files.length > 0) {
                resetImageList();
                document.body.style.cursor = "wait";
                for (let i = 0; i < files.length; i++) {
                    const nameParts = files[i].name.split(".");
                    if (extensions.indexOf(nameParts[nameParts.length - 1]) !== -1) {
                        images[files[i].name] = {
                            meta: files[i],
                            index: i
                        };
                        const option = document.createElement("option");
                        option.value = files[i].name;
                        option.innerHTML = files[i].name;
                        if (i === 0) {
                            option.selected = true;
                        }
                        imageList.appendChild(option);
                    }
                }
                const imageArray = Object.keys(images);
                let async = imageArray.length;
                for (let image in images) {
                    const reader = new FileReader();
                    reader.addEventListener("load", () => {
                        const imageObject = new Image();
                        imageObject.addEventListener("load", (event) => {
                            images[image].width = event.target.width;
                            images[image].height = event.target.height;
                            if (--async === 0) {
                                document.body.style.cursor = "default";
                                setCurrentImage(images[imageArray[0]]);
                                if (Object.keys(classes).length > 0) {
                                    document.getElementById("bboxes").disabled = false;
                                    document.getElementById("restoreBboxes").disabled = false;
                                }
                            }
                        });
                        imageObject.src = reader.result;
                    });
                    reader.readAsDataURL(images[image].meta);
                }
            }
        });
    };

    const resetImageList = () => {
        const imageList = document.getElementById("imageList");
        imageList.innerHTML = "";
        images = {};
        bboxes = {};
        currentImage = null;
    };

    const setCurrentImage = (image) => {
        if (resetCanvasOnChange === true) {
            resetCanvasPlacement();
        }
        if (fittedZoom === true) {
            fitZoom(image);
        }
        const reader = new FileReader();
        reader.addEventListener("load", () => {
            const dataUrl = reader.result;
            const imageObject = new Image();
            imageObject.addEventListener("load", () => {
                currentImage = {
                    name: image.meta.name,
                    object: imageObject,
                    width: image.width,
                    height: image.height
                };
            });
            imageObject.src = dataUrl;
            document.getElementById("imageInformation")
                .innerHTML = `${image.width}x${image.height}, ${formatBytes(image.meta.size)}`;
        });
        reader.readAsDataURL(image.meta);
        if (currentBbox !== null) {
            currentBbox.bbox.marked = false;
            currentBbox = null;
        }
    };

    const fitZoom = (image) => {
        if (image.width > image.height) {
            scale = canvas.width / image.width;
        } else {
            scale = canvas.height / image.height;
        }
    };

    const formatBytes = (bytes, decimals) => {
        if (bytes === 0) {
            return "0 Bytes";
        }
        const k = 1024;
        const dm = decimals || 2;
        const sizes = ["Bytes", "KB", "MB"];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + " " + sizes[i];
    };

    const listenImageSelect = () => {
        const imageList = document.getElementById("imageList");
        imageList.addEventListener("change", () => {
            imageListIndex = imageList.selectedIndex;
            setCurrentImage(images[imageList.options[imageListIndex].innerHTML]);
        });
    };

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
                                    // Reverse engineer actual position and dimensions from yolo format
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
                const name = imageName.split(".");
                name[name.length - 1] = "txt";
                const result = [];
                for (let className in bboxes[imageName]) {
                    for (let i = 0; i < bboxes[imageName][className].length; i++) {
                        const bbox = bboxes[imageName][className][i];
                        // Prepare data for yolo format
                        const x = (bbox.x + bbox.width / 2) / image.width;
                        const y = (bbox.y + bbox.height / 2) / image.height;
                        const width = bbox.width / image.width;
                        const height = bbox.height / image.height;
                        result.push(`${classes[className]} ${x} ${y} ${width} ${height}`);
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
                result.images.push({
                    id: images[imageName].index + 1,
                    file_name: imageName,
                    width: images[imageName].width,
                    height: images[imageName].height
                });
            }
            let id = 0;
            for (let imageName in bboxes) {
                const image = images[imageName];
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
                zip.file("coco.json", JSON.stringify(result));
            }
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
            if (key === 8 || (key === 46 && event.metaKey === true)) { // Backspace or Delete + Meta (Command)
                if (currentBbox !== null) {
                    bboxes[currentImage.name][currentBbox.bbox.class].splice(currentBbox.index, 1);
                    currentBbox = null;
                    document.body.style.cursor = "default";
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

    const listenImageCrop = () => {
        document.getElementById("cropImages").addEventListener("click", () => {
            const zip = new JSZip();
            let x = 0;
            for (let imageName in bboxes) {
                const image = images[imageName];
                for (let className in bboxes[imageName]) {
                    for (let i = 0; i < bboxes[imageName][className].length; i++) {
                        x++;
                        if (x === 1) {
                            document.body.style.cursor = "wait"; // Mark as busy
                        }
                        const bbox = bboxes[imageName][className][i];
                        const reader = new FileReader();
                        reader.addEventListener("load", () => {
                            const dataUrl = reader.result;
                            const imageObject = new Image();
                            imageObject.addEventListener("load", () => {
                                const temporaryCanvas = document.createElement("canvas");
                                temporaryCanvas.style.display = "none";
                                temporaryCanvas.width = bbox.width;
                                temporaryCanvas.height = bbox.height;
                                temporaryCanvas.getContext("2d").drawImage(
                                    imageObject,
                                    bbox.x,
                                    bbox.y,
                                    bbox.width,
                                    bbox.height,
                                    0,
                                    0,
                                    bbox.width,
                                    bbox.height
                                );
                                temporaryCanvas.toBlob((blob) => {
                                    const imageNameParts = imageName.split(".");
                                    imageNameParts[imageNameParts.length - 2] += `-${className}-${i}`;
                                    zip.file(imageNameParts.join("."), blob);
                                    if (--x === 0) {
                                        document.body.style.cursor = "default";
                                        zip.generateAsync({ type: "blob" })
                                            .then((blob) => {
                                                saveAs(blob, "crops.zip");
                                            });
                                    }
                                }, image.meta.type);
                            });
                            imageObject.src = dataUrl;
                        });
                        reader.readAsDataURL(image.meta);
                    }
                }
            }
        });
    };
})();