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

    function clampUnit(value) {
        const parsed = finiteNumber(value, 0);
        if (parsed <= 0) return 0;
        if (parsed >= 1) return 1;
        return parsed;
    }

    function median(values) {
        const sorted = (Array.isArray(values) ? values : [])
            .map((value) => finiteNumber(value, NaN))
            .filter((value) => Number.isFinite(value))
            .sort((a, b) => a - b);
        if (!sorted.length) {
            return 0;
        }
        const middle = Math.floor(sorted.length / 2);
        if (sorted.length % 2) {
            return sorted[middle];
        }
        return (sorted[middle - 1] + sorted[middle]) / 2;
    }

    function normalizeRange(value, minValue, maxValue) {
        const min = finiteNumber(minValue, 0);
        const max = finiteNumber(maxValue, min);
        if (!(max > min)) {
            return 0;
        }
        return clampUnit((finiteNumber(value, min) - min) / (max - min));
    }

    function datasetPointProjection(point) {
        if (Array.isArray(point?.projection) && point.projection.length >= 2) {
            return [finiteNumber(point.projection[0], NaN), finiteNumber(point.projection[1], NaN)];
        }
        return [
            finiteNumber(point?.embedding_x ?? point?.x, NaN),
            finiteNumber(point?.embedding_y ?? point?.y, NaN),
        ];
    }

    function datasetPointImageKey(point) {
        const frontendKey = String(point?.frontend_image_key || point?.image_key || "").trim();
        if (frontendKey) {
            return frontendKey;
        }
        const split = String(point?.split || "train").trim() || "train";
        const rel = String(point?.image_relpath || point?.image_name || point?.filename || "").trim();
        return rel ? `${split}/${rel}` : split;
    }

    function computeDatasetImageValueAnalysis(points = []) {
        const cleanPoints = (Array.isArray(points) ? points : [])
            .filter((point) => point && typeof point === "object" && String(point.point_id || "").trim());
        const groups = new Map();
        const classCounts = {};
        const clusterCounts = {};
        const logAreas = [];
        const projections = [];

        cleanPoints.forEach((point) => {
            const className = normalizeClassName(point.class_name || point.label || "");
            incrementCount(classCounts, className);
            const clusterId = String(point.cluster_id ?? "");
            if (clusterId) {
                clusterCounts[clusterId] = (clusterCounts[clusterId] || 0) + 1;
            }
            const width = Math.abs(finiteNumber(point.width, 0));
            const height = Math.abs(finiteNumber(point.height, 0));
            logAreas.push(Math.log1p(Math.max(0, width * height)));
            const projection = datasetPointProjection(point);
            if (Number.isFinite(projection[0]) && Number.isFinite(projection[1])) {
                projections.push(projection);
            }
        });

        const classWeights = cleanPoints.map((point) => {
            const className = normalizeClassName(point.class_name || point.label || "");
            const count = Math.max(1, finiteNumber(classCounts[className], 1));
            return 1 / Math.sqrt(count);
        });
        const clusterWeights = cleanPoints.map((point) => {
            const clusterId = String(point.cluster_id ?? "");
            const count = Math.max(1, finiteNumber(clusterCounts[clusterId], 1));
            return 1 / Math.sqrt(count);
        });
        const minClassWeight = classWeights.length ? Math.min(...classWeights) : 0;
        const maxClassWeight = classWeights.length ? Math.max(...classWeights) : 1;
        const minClusterWeight = clusterWeights.length ? Math.min(...clusterWeights) : 0;
        const maxClusterWeight = clusterWeights.length ? Math.max(...clusterWeights) : 1;
        const areaMedian = median(logAreas);
        const areaSpread = Math.max(0.25, median(logAreas.map((value) => Math.abs(value - areaMedian))) * 2.5);
        const globalX = projections.length ? projections.reduce((sum, projection) => sum + projection[0], 0) / projections.length : 0;
        const globalY = projections.length ? projections.reduce((sum, projection) => sum + projection[1], 0) / projections.length : 0;

        cleanPoints.forEach((point, idx) => {
            const imageKey = datasetPointImageKey(point);
            if (!groups.has(imageKey)) {
                groups.set(imageKey, {
                    image_key: imageKey,
                    split: String(point.split || "train"),
                    image_relpath: String(point.image_relpath || point.image_name || point.filename || imageKey),
                    image_name: String(point.image_name || point.filename || point.image_relpath || imageKey).split(/[\\/]/).pop(),
                    object_count: 0,
                    class_counts: {},
                    point_ids: [],
                    points: [],
                    projection_sum_x: 0,
                    projection_sum_y: 0,
                    projection_count: 0,
                    bbox_rarity_sum: 0,
                    feature_rarity_sum: 0,
                    preview_point_id: "",
                    preview_point_score: -1,
                });
            }
            const group = groups.get(imageKey);
            const className = normalizeClassName(point.class_name || point.label || "");
            const classWeight = normalizeRange(classWeights[idx], minClassWeight, maxClassWeight);
            const clusterWeight = normalizeRange(clusterWeights[idx], minClusterWeight, maxClusterWeight);
            const logArea = logAreas[idx] || 0;
            const areaRarity = clampUnit(Math.abs(logArea - areaMedian) / areaSpread);
            const outlier = clampUnit(point.outlier_score);
            const bboxRarity = clampUnit(0.65 * classWeight + 0.35 * areaRarity);
            const featureRarity = clampUnit(0.65 * outlier + 0.35 * clusterWeight);
            const pointScore = clampUnit(0.55 * bboxRarity + 0.45 * featureRarity);
            const projection = datasetPointProjection(point);

            group.object_count += 1;
            incrementCount(group.class_counts, className);
            group.point_ids.push(String(point.point_id));
            group.points.push(point);
            group.bbox_rarity_sum += bboxRarity;
            group.feature_rarity_sum += featureRarity;
            if (Number.isFinite(projection[0]) && Number.isFinite(projection[1])) {
                group.projection_sum_x += projection[0];
                group.projection_sum_y += projection[1];
                group.projection_count += 1;
            }
            if (pointScore > group.preview_point_score) {
                group.preview_point_score = pointScore;
                group.preview_point_id = String(point.point_id);
            }
        });

        const imageDistances = Array.from(groups.values()).map((group) => {
            if (!group.projection_count) {
                return 0;
            }
            const x = group.projection_sum_x / group.projection_count;
            const y = group.projection_sum_y / group.projection_count;
            return Math.hypot(x - globalX, y - globalY);
        });
        const maxImageDistance = imageDistances.length ? Math.max(...imageDistances) : 0;
        const items = Array.from(groups.values()).map((group, idx) => {
            const bboxRarity = group.object_count ? group.bbox_rarity_sum / group.object_count : 0;
            const featureRarity = group.object_count ? group.feature_rarity_sum / group.object_count : 0;
            const projectionRarity = maxImageDistance > 0 ? imageDistances[idx] / maxImageDistance : 0;
            const classes = Object.keys(group.class_counts)
                .filter(Boolean)
                .sort((a, b) => (group.class_counts[b] - group.class_counts[a]) || a.localeCompare(b));
            const richness = clampUnit((classes.length / Math.max(1, Math.min(5, Object.keys(classCounts).length || classes.length || 1))) * 0.7 + Math.log1p(group.object_count) / 8);
            const imageValue = clampUnit(
                0.35 * bboxRarity
                + 0.30 * featureRarity
                + 0.25 * projectionRarity
                + 0.10 * richness
            );
            return {
                image_key: group.image_key,
                split: group.split,
                image_relpath: group.image_relpath,
                image_name: group.image_name,
                object_count: group.object_count,
                class_count: classes.length,
                classes,
                top_class: classes[0] || "",
                bbox_rarity: bboxRarity,
                feature_rarity: featureRarity,
                projection_rarity: clampUnit(projectionRarity),
                richness,
                image_value: Math.round(imageValue * 100),
                image_value_score: imageValue,
                coverage_score: clampUnit(1 - imageValue),
                preview_point_id: group.preview_point_id,
                point_ids: group.point_ids,
            };
        }).sort((a, b) => (b.image_value_score - a.image_value_score) || a.image_relpath.localeCompare(b.image_relpath));

        const meanValue = items.length
            ? items.reduce((sum, item) => sum + item.image_value_score, 0) / items.length
            : 0;
        return {
            items,
            summary: {
                image_count: items.length,
                object_count: cleanPoints.length,
                class_count: Object.keys(classCounts).filter(Boolean).length,
                mean_image_value: Math.round(meanValue * 100),
                high_value_count: items.filter((item) => item.image_value_score >= 0.66).length,
                central_count: items.filter((item) => item.image_value_score <= 0.33).length,
            },
        };
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
        computeDatasetImageValueAnalysis,
        formatImageDiversityMetric,
        mergeCounts,
        sumCounts,
    });

    root.TatorAnnotationDiversity = api;
    if (typeof module !== "undefined" && module.exports) {
        module.exports = api;
    }
})();
