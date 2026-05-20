# EcoSim - 空间显式生态演替模拟器 (Ecological Succession Simulator) v3.3.1

[![Version](https://img.shields.io/badge/version-3.3.1-brightgreen.svg)](https://github.com/liusonwood/ecosimulator)
[![License](https://img.shields.io/badge/license-AGPL--3.0-blue.svg)](https://github.com/liusonwood/ecosimulator)
[![Platform](https://img.shields.io/badge/platform-Web%20%7C%20Electron-orange.svg)](https://github.com/liusonwood/ecosimulator)

`EcoSim` 是一个空间显式（Spatially Explicit）的植被动力学模拟系统。它基于 **Lotka-Volterra 竞争模型** 与 **高斯核种子扩散** 理论，在二维网格上逼真模拟了从地衣（Lichen）、苔藓（Moss）、草本（Herb）、灌木（Shrub）到乔木（Tree）的生态演替全过程，并融合了人为种植的农作物（Crop）及多种极端环境扰动。

---

## 🌍 项目概述与核心亮点

1. **极致视觉美学**：采用 16:9 Glassmorphism（玻璃拟态）响应式数据看板设计，融入 HSL 调和色卡与微动交互。网格采用生态物种盖度多色插值混合与优势种 Emoji 尺寸缩放技术，视觉张力拉满。
2. **三级降级计算引擎**：首创自适应异构计算，检测硬件支持并自动回退：**GPU.js (GLSL 并行渲染)** > **Multi-threaded CPU (Blob Inlined Workers)** > **Serial CPU (串行兜底)**。
3. **零依赖开箱即用**：逻辑与资源极致封装，全面兼容 `file://` 协议，无需架设本地服务器，双击 `index.html` 即可在任一现代浏览器中流畅运行 60 FPS 模拟。
4. **科研级参数可调性**：所有的生物学特性、竞争矩阵、土壤因子及演替阈值均集中在 `main.js` 的 `CONFIG` 中，专为生态模拟实验量身打造。

---

## 📂 项目结构说明

```bash
├── index.html           # 16:9 玻璃拟态 UI 布局与 Vue 3 全局挂载模版
├── style.css            # 统一的高端磨砂玻璃特效、动画过渡及自定义滚动条样式
├── main.js             # 核心计算引擎 (EcoSimulator Class)、多引擎逻辑与 Vue 应用控制
├── preload.js          # Electron 桌面客户端的预加载逻辑
├── electron-main.js    # Electron 应用的主进程启动配置
├── package.json        # 桌面客户端依赖与 Electron-Builder 构建配置
├── design/              # 数学模型、设计规范以及参考参数配置文件
│   ├── mathmodel.md    # 详细的数理公式推导与算法步骤
│   ├── uidesign.md     # 玻璃拟态看板 UI 的列行网格规范
│   └── parameters-2.0.csv # 官方生物学仿真参数基准对照表
└── lib/                # 兼容本地双击运行的离线 JS 依赖库 (Vue, GPU.js, Chart.js, Lucide)
```

---

## 🧬 仿真科学原理与数学模型

模拟器在每个网格（Grid Cell）上独立维护一个由 6 类功能型植物构成的动态微群落，每年演替迭代经历以下四个核心物理过程：

### 1. 种子产生与空间高斯扩散 (Seed Production & Dispersal)
每个网格上的物种根据其当前生物量盖度 $B_k(\mathbf{x})$ 按繁殖力 $\rho_k$ 产生种子，并以高斯分布核（Gaussian Kernel）向外空间卷积扩散：

$$P_k(\mathbf{x}) = B_k(\mathbf{x}) \cdot \rho_k$$

$$G(\mathbf{d}, \lambda_k) = \frac{1}{2\pi \lambda_k^2} \exp\left(-\frac{\|\mathbf{d}\|^2}{2\lambda_k^2}\right)$$

其中，$\mathbf{d}$ 为扩散距离向量，$\lambda_k$ 为物种 $k$ 的种子空间扩散标准差。同时，系统为各物种注入极微量的背景种子雨 $\beta_k$，模拟随风力或动物从远距离带入的随机种子，确保群落在遭受灭顶之灾后仍具自我复苏能力。

### 2. 土壤发育与演替阶段阈值 (Soil Development & Succession Gates)
为了忠实呈现“地衣/苔藓改良土壤 $\rightarrow$ 高等植被萌发”这一经典生态学演替路径：
- 高等物种（如乔木）的萌发率 $g_{eff}$ 受当前网格**累积总生物量** $\sum_{l=1}^{N} B_l$ 的严格阈值约束。
- 只有当先锋物种（地衣、苔藓）枯死腐殖并累积起足够的土壤有机质（即总盖度突破物种特有的门槛 `thresholds[k]`），高等物种的种子才能开始萌发。
- 萌发量进一步受当前网格内残存可用物理空间限制，以确保物理实体体积不发生膨胀。

### 3. Lotka-Volterra 空间非对称竞争 (Asymmetric Space Competition)
局部网格内的生长与空间排挤基于多物种 Lotka-Volterra 竞争微分方程的离散差分形式进行计算：

$$B_{new, k} = B_k + r_k \cdot \eta_{\text{climate}} \cdot B_k \left(1 - \frac{\sum_{l=1}^{N} \alpha_{kl} B_l}{K_{\text{base}, k} \cdot \mu_{\text{soil}}}\right)$$

- **$\alpha_{kl}$ (非对称竞争抑制矩阵)**：描述物种 $l$ 对物种 $k$ 在空间争夺上的压制强度。层级高的乔木对低层植物的抑制系数极高（例如 $\alpha_{lichen, tree} = 3.3$），而低层植物对高层的反馈抑制则趋近于 $0$。
- **$K_{\text{base}, k}$ (基础承载力)** 与 **$\mu_{\text{soil}}$ (土壤乘子)**：共同决定了物种的局部盖度上限。
- **$\eta_{\text{climate}}$ (气候波动因子)**：每年模拟时会产生 $\pm 15\%$ 的不确定波动，模拟现实气候的无常。

### 4. 土壤异质性与斑块生成 (Smoothed Potential Field Quantile Mapping)
为模拟真实的宏观地貌，系统采用**平滑势场分位数映射**算法：
1. 随机生成网格白噪声。
2. 进行 3 轮空间均值模糊（Mean Blur）滤波，融汇形成连续流畅、带有地理连续性的地貌势场。
3. 应用分位数进行区间切割，**严格确保**网格中精准包含 30% 浅土（支持地衣）、40% 中性土和 30% 深土（支持乔木），并呈现自然柔和的地理斑块分布。

---

## 🛠 开发者调参配置指南 (`CONFIG`)

在 `main.js` 的开头，开发或科研人员可以通过修改 `CONFIG` 对象中的关键物理量，自由定制并测试全新的生态场景：

```javascript
const CONFIG = {
    // 基础定义
    gridWidth: 20,              // 模拟网格宽度
    gridHeight: 13,             // 模拟网格高度
    numSpecies: 6,              // 物种总数：0-地衣, 1-苔藓, 2-草本, 3-灌木, 4-乔木, 5-农作物

    // 生物学核心参数 (数组索引 0 - 5 对应上述 6 个物种)
    rho: [30.0, 100.0, 50.0, 200.0, 80.0, 40.0],       // 繁殖力 (ρ)：单位盖度每年产生的种子数
    lambda: [0.5, 0.8, 1.5, 4.0, 6.0, 1.5],           // 扩散标准差 (λ)：控制高斯核的空间扩散半径
    r: [0.3, 0.4, 0.3, 0.6, 0.2, 0.3],                // 增长率 (r)：物种的最大年内禀增长速度
    g: [0.15, 0.4, 0.6, 0.4, 0.4, 0.6],               // 基础萌发率 (g)
    s: [0.6, 0.6, 0.2, 0.7, 0.6, 0.2],                // 种子库年存活率 (s)
    nu: [0.045, 0.06, 0.1, 0.12, 0.06, 0.1],         // 库萌发率 (ν)：种子库中种子的再萌发比率
    K_base: [0.3, 0.46, 0.7, 0.8, 0.95, 0.7],          // 基础环境承载力 (K)

    // 演替进化门槛：高等植物必须在该网格累积总生物量突破此阈值后才能萌发，模拟土壤有机质的改良
    thresholds: [0.0, 0.1, 0.4, 0.73, 0.85, 0.4], 

    // 非对称竞争强度矩阵 alpha[受体][施加体]
    alpha: [
        [1.0, 1.2, 2.0, 2.5, 3.3, 2.0], // 地衣受其他物种的竞争系数
        [0.4, 1.0, 1.2, 1.8, 2.3, 1.0], // 苔藓受其他物种的竞争系数
        [0.2, 0.3, 1.0, 1.6, 3.4, 1.0], // 草本受其他物种的竞争系数
        [0.1, 0.1, 0.3, 1.0, 1.3, 0.3], // 灌木受其他物种的竞争系数
        [0.01, 0.01, 0.05, 0.1, 1.0, 0.05], // 乔木受其他物种的竞争系数
        [0.2, 0.3, 1.3, 1.5, 3.4, 1.0]  // 农作物受其他物种的竞争系数
    ],

    // 气候类型对应的土壤厚度分布比 [浅土, 中土, 深土]
    soilDepthProportions: {
        '热带雨林气候': [0.1, 0.3, 0.6],
        '温带草原气候': [0.3, 0.5, 0.2],
        '寒带苔原气候': [0.7, 0.25, 0.05],
        '荒漠气候': [0.9, 0.1, 0.0],
        '农场': [0.05, 0.35, 0.6],
        'default': [0.3, 0.4, 0.3]
    }
};
```

---

## ⚡ 算力降级与多引擎架构

`EcoSim` 内部集成高度抽象的并发调度引擎，根据客户端软硬件特性在启动时完成检测与绑定：

1. **GPU 并行加速模式 (`GPU`)**
   - 依赖宿主环境的 WebGL。通过 `GPU.js` 库，在运行时动态将演替竞争算法、空间二维高斯卷积内核编译为高速运行的 GLSL 着色器（Shader）。
   - 实现每个网格并发由 GPU 流处理器计算，完美应对上万网格的大规模实时生态模拟。
2. **多线程 CPU 模式 (`Worker`)**
   - 若 WebGL 不可用，自动降级为 Web Workers 架构。
   - 内部利用 Blob 技术将 Worker 的逻辑代码以字符串形式内联，动态生成 Blob URL 并注入后台线程池，规避了浏览器的同源策略（Same-Origin Policy）限制，可在无服务器环境下流畅运行。
3. **串行 CPU 模式 (`Serial`)**
   - 作为最强兼容性的基石，当浏览器拦截了 Worker 或不支持多核时自动接入，用单线程循环稳健计算，提供一致的数据产出。

### 🔧 运行控制与调试

在运行模拟器时，你可以按下键盘 `F12` 打开浏览器控制台，通过手动调用全局函数切换和对比计算引擎：

```javascript
// 手动指定计算引擎并重置模拟系统
setMode('GPU');     // 强制切换至显卡并行计算
setMode('Worker');  // 强制切换至多线程 CPU 计算
setMode('Serial');  // 强制切换至串行单核 CPU 计算
setMode('Auto');    // 切换至自动检测自适应模式
```

---

## 📦 客户端安装与快速启动

### 方式一：零安装极速体验（推荐）
1. 将本项目整个克隆或下载到本地。
2. 直接双击目录下的 `index.html` 即可在本地浏览器中畅玩。

### 方式二：桌面客户端开发与打包 (Electron)
如果你需要将其作为本地独立的桌面客户端运行或分发，项目已集成了基于 Electron 的打包工具链：

1. **安装环境依赖**（确保本地已配置 [Node.js](https://nodejs.org/) 环境）：
   ```bash
   npm install
   ```

2. **本地开发调试运行**：
   ```bash
   npm start
   ```

3. **构建 Windows `.exe` 单文件桌面客户端**：
   ```bash
   npm run build:win
   ```

4. **构建 macOS 桌面客户端 (`.dmg` / `.app` / `.zip`)**：
   ```bash
   npm run build:mac
   ```
   构建产物将在项目根目录下的 `dist/` 文件夹中生成。

---

## 🔑 开源协议与原作说明

- **原作声明**：本项目原作名称为 **EcoSimulator (EcoSim)**，最初由 [liusonwood](https://github.com/liusonwood) 与 [yanyi-lin](https://github.com/yanyi-lin) 倾力设计并制作。
- **开源协议**：本项目秉承学术共享与开源极客精神，完整源代码遵循 **[GNU AGPL-3.0](https://www.gnu.org/licenses/agpl-3.0.html)** 开源协议。
- **派生要求**：如果您使用本项目进行二次开发、研究或构建衍生作品，请务必保证：
  1. 完整保留原作者署名信息与原项目 GitHub 链接。
  2. 衍生的应用、网页或服务同样必须在 AGPL-3.0 协议下公开全部源代码。
