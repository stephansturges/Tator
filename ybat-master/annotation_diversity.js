(() => {
    "use strict";

    const root = typeof globalThis !== "undefined" ? globalThis : window;

    function finiteNumber(value, fallback = 0) {
        const parsed = Number(value);
        return Number.isFinite(parsed) ? parsed : fallback;
    }

    function normalizeClassName(name) {
        return String(name || "").trim();
    }

    function isValidBox(record) {
        if (!record || typeof record !== "object") {
            return false;
        }
        if (Array.isArray(record.points) && record.points.length >= 3) {
            return record.points.filter((pt) => Number.isFinite(Number(pt?.x)) && Number.isFinite(Number(pt?.y))).length >= 3;
        }
        return Math.abs(finiteNumber(record.width)) > 0 && Math.abs(finiteNumber(record.height)) > 0;
    }

    function incrementCount(counts, className, amount = 1) {
        const normalized = normalizeClassName(className);
        if (!normalized) {
            return counts;
        }
        counts[normalized] = (counts[normalized] || 0) + amount;
        return counts;
    }

    function countBoxesByClassFromBuckets(classBuckets) {
        const counts = {};
        if (!classBuckets || typeof classBuckets !== "object") {
            return counts;
        }
        Object.entries(classBuckets).forEach(([className, records]) => {
            if (!Array.isArray(records)) {
                return;
            }
            records.forEach((record) => {
                if (isValidBox(record)) {
                    incrementCount(counts, record.class || className);
                }
            });
        });
        return counts;
    }

    function countBoxesByClassFromYoloLines(labelLines, labelmap = []) {
        const counts = {};
        if (!Array.isArray(labelLines)) {
            return counts;
        }
        labelLines.forEach((rawLine) => {
            const cols = String(rawLine || "").trim().split(/\s+/).filter(Boolean);
            if (cols.length < 5) {
                return;
            }
            const classIdx = parseInt(cols[0], 10);
            if (!Number.isFinite(classIdx)) {
                return;
            }
            const className = normalizeClassName(labelmap[classIdx] || String(classIdx));
            if (!className) {
                return;
            }
            const coords = cols.slice(1).map(Number);
            if (coords.some((value) => !Number.isFinite(value))) {
                return;
            }
            if (cols.length >= 7) {
                const pointCount = Math.floor(coords.length / 2);
                if (pointCount >= 3) {
                    incrementCount(counts, className);
                }
                return;
            }
            const width = coords[2];
            const height = coords[3];
            if (width > 0 && height > 0) {
                incrementCount(counts, className);
            }
        });
        return counts;
    }

    function mergeCounts(target, source, multiplier = 1) {
        const dest = target && typeof target === "object" ? target : {};
        if (!source || typeof source !== "object") {
            return dest;
        }
        Object.entries(source).forEach(([className, count]) => {
            const amount = finiteNumber(count) * multiplier;
            if (amount > 0) {
                incrementCount(dest, className, amount);
            }
        });
        return dest;
    }

    function sumCounts(counts) {
        if (!counts || typeof counts !== "object") {
            return 0;
        }
        return Object.values(counts).reduce((total, count) => total + Math.max(0, finiteNumber(count)), 0);
    }

    function medianPositive(values) {
        const positives = (Array.isArray(values) ? values : [])
            .map((value) => finiteNumber(value))
            .filter((value) => value > 0)
            .sort((a, b) => a - b);
        if (!positives.length) {
            return 0;
        }
        const middle = Math.floor(positives.length / 2);
        if (positives.length % 2) {
            return positives[middle];
        }
        return (positives[middle - 1] + positives[middle]) / 2;
    }

    function computeImageDiversityMetric({
        currentCounts = {},
        datasetCounts = {},
        classNames = [],
    } = {}) {
        const normalizedCurrent = {};
        mergeCounts(normalizedCurrent, currentCounts);
        const normalizedDataset = {};
        mergeCounts(normalizedDataset, datasetCounts);

        const totalCurrentBoxes = sumCounts(normalizedCurrent);
        const currentClassNames = Object.keys(normalizedCurrent).filter((name) => normalizedCurrent[name] > 0);
        const distinctCurrentClasses = currentClassNames.length;
        const totalDatasetBoxes = sumCounts(normalizedDataset);
        const knownClassCount = Math.max(
            1,
            Array.isArray(classNames) && classNames.length ? classNames.length : distinctCurrentClasses,
        );

        if (!totalCurrentBoxes || !distinctCurrentClasses) {
            return {
                score: 0,
                boxes: 0,
                classes: 0,
                knownClasses: knownClassCount,
                totalDatasetBoxes,
                currentCounts: normalizedCurrent,
                datasetCounts: normalizedDataset,
                classDetails: [],
                newClasses: [],
                rareClasses: [],
                status: "empty",
            };
        }

        const allClassNames = Array.from(new Set([
            ...(Array.isArray(classNames) ? classNames.map(normalizeClassName).filter(Boolean) : []),
            ...Object.keys(normalizedDataset),
            ...currentClassNames,
        ]));
        const existingCounts = {};
        allClassNames.forEach((className) => {
            existingCounts[className] = Math.max(
                0,
                finiteNumber(normalizedDataset[className]) - finiteNumber(normalizedCurrent[className]),
            );
        });
        const medianExisting = medianPositive(Object.values(existingCounts));

        let contribution = 0;
        let maxContribution = 0;
        const classDetails = currentClassNames.map((className) => {
            const currentCount = finiteNumber(normalizedCurrent[className]);
            const existingCount = finiteNumber(existingCounts[className]);
            const datasetCount = finiteNumber(normalizedDataset[className]);
            const repeatWeight = Math.log1p(currentCount);
            const rarityWeight = 1 / Math.sqrt(existingCount + 1);
            const classContribution = repeatWeight * rarityWeight;
            contribution += classContribution;
            maxContribution += repeatWeight;
            return {
                className,
                currentCount,
                existingCount,
                datasetCount,
                contribution: classContribution,
                rarityWeight,
            };
        }).sort((a, b) => b.contribution - a.contribution);

        const rarityScore = maxContribution > 0 ? 100 * (contribution / maxContribution) : 0;
        const richnessDenominator = Math.max(1, Math.min(knownClassCount, totalCurrentBoxes));
        const richnessScore = 100 * Math.min(1, distinctCurrentClasses / richnessDenominator);
        const rawScore = (0.75 * rarityScore) + (0.25 * richnessScore);
        const score = Math.max(0, Math.min(100, Math.round(rawScore)));
        const newClasses = classDetails
            .filter((entry) => entry.existingCount <= 0)
            .map((entry) => entry.className);
        const rareClasses = classDetails
            .filter((entry) => entry.existingCount <= 0 || (medianExisting > 0 && entry.existingCount <= medianExisting))
            .map((entry) => entry.className);

        return {
            score,
            rarityScore,
            richnessScore,
            boxes: totalCurrentBoxes,
            classes: distinctCurrentClasses,
            knownClasses: knownClassCount,
            totalDatasetBoxes,
            currentCounts: normalizedCurrent,
            datasetCounts: normalizedDataset,
            classDetails,
            newClasses,
            rareClasses,
            status: "ready",
        };
    }

    function pluralize(count, singular, plural = `${singular}s`) {
        return `${count} ${count === 1 ? singular : plural}`;
    }

    function formatImageDiversityMetric(metric) {
        if (!metric || metric.status === "empty") {
            return "Image value 0/100 - no bboxes on this image.";
        }
        const strongest = (metric.rareClasses && metric.rareClasses.length ? metric.rareClasses : [])
            .slice(0, 3)
            .join(", ");
        const rareText = strongest ? ` - coverage: ${strongest}` : "";
        return `Image value ${metric.score}/100 - ${pluralize(metric.boxes, "box", "boxes")} across ${pluralize(metric.classes, "class", "classes")}${rareText}.`;
    }

    const api = Object.freeze({
        countBoxesByClassFromBuckets,
        countBoxesByClassFromYoloLines,
        computeImageDiversityMetric,
        formatImageDiversityMetric,
        mergeCounts,
        sumCounts,
    });

    root.TatorAnnotationDiversity = api;
    if (typeof module !== "undefined" && module.exports) {
        module.exports = api;
    }
})();
