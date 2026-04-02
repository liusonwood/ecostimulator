/**
 * EcoSim - Simulation Worker
 * 负责并行执行 CPU 演替逻辑
 */

self.onmessage = function(e) {
    const { 
        startY, endY, width, height, numSpecies, 
        biomass, seedBank, soilDepth, climateMults, globalBiomass,
        CONFIG, randSeed 
    } = e.data;

    const chunkHeight = endY - startY;
    const nextB = new Float32Array(chunkHeight * width * numSpecies);
    const nextS = new Float32Array(chunkHeight * width * numSpecies);

    for (let y = startY; y < endY; y++) {
        for (let x = 0; x < width; x++) {
            const outIdxBase = ((y - startY) * width + x) * numSpecies;
            const inIdxBase = (y * width + x) * numSpecies;
            
            let totalB = 0;
            for (let k = 0; k < numSpecies; k++) totalB += biomass[inIdxBase + k];

            for (let k = 0; k < numSpecies; k++) {
                // 1. 种子扩散
                const lambda_k = CONFIG.lambda[k];
                const r_limit = Math.ceil(lambda_k * 2);
                let dispersed = 0;

                if (r_limit >= Math.min(width, height) / 2) {
                    dispersed = (globalBiomass[k] * CONFIG.rho[k] * 0.01) / (width * height);
                } else {
                    // 扩散预计算逻辑
                    for (let dy = -r_limit; dy <= r_limit; dy++) {
                        const ny = (y + dy + height) % height;
                        for (let dx = -r_limit; dx <= r_limit; dx++) {
                            const nx = (x + dx + width) % width;
                            const sourceB = biomass[(ny * width + nx) * numSpecies + k];
                            if (sourceB < 0.001) continue;
                            
                            const distSq = dx * dx + dy * dy;
                            const weight = Math.exp(-distSq / (2 * lambda_k * lambda_k));
                            dispersed += sourceB * CONFIG.rho[k] * weight * 0.01;
                        }
                    }
                }

                // 2. 种子库更新
                const randVal = Math.abs(Math.sin(x * 12.9898 + y * 78.233 + k * 37.1 + randSeed) * 43758.5453);
                const randRain = 0.5 + (randVal - Math.floor(randVal));
                let sk = seedBank[inIdxBase + k] + CONFIG.bgSeed[k] * randRain + dispersed;

                // 3. 生物量增长
                const depth = soilDepth[y * width + x];
                const k_local = CONFIG.K_base[k] * CONFIG.soilMult[k][depth] * climateMults[k];
                let compSum = 0;
                for (let l = 0; l < numSpecies; l++) compSum += CONFIG.alpha[k][l] * biomass[inIdxBase + l];
                
                let b_growth = biomass[inIdxBase + k] + CONFIG.r[k] * climateMults[k] * biomass[inIdxBase + k] * (1 - compSum / Math.max(0.01, k_local));
                
                // 4. 萌发
                let g_eff = CONFIG.g[k] * Math.max(0, 1 - totalB);
                if (totalB < CONFIG.thresholds[k]) g_eff = 0;

                const b_germ = Math.min(sk * g_eff * 0.1 * climateMults[k], Math.max(0, 1 - totalB));
                
                nextB[outIdxBase + k] = Math.max(0, b_growth + b_germ);
                nextS[outIdxBase + k] = Math.max(0, (sk - b_germ) * CONFIG.s[k]);
            }

            // 5. 归一化
            let finalTotalB = 0;
            for (let k = 0; k < numSpecies; k++) finalTotalB += nextB[outIdxBase + k];
            if (finalTotalB > 1.0) {
                for (let k = 0; k < numSpecies; k++) nextB[outIdxBase + k] /= finalTotalB;
            }
        }
    }

    // 返回计算结果，使用 Transferable 避免内存复制
    self.postMessage({
        startY, endY,
        nextB, nextS
    }, [nextB.buffer, nextS.buffer]);
};
