这是该仓库的文件夹架构。

该仓库主要包含了机器学习相关课程的作业、模型代码以及课程期末竞赛（ChemE Car 化工车）的核心工程文件和硬件驱动。

````bash
ZJU-Medical-AI/
├── Get ready/                         # 课程准备材料
│   ├── 0. Get ready.md/.pdf           # 环境配置与前置知识说明
│   └── requirements.txt               # Python 依赖包列表
├── 作业1/                             # 第一次作业目录 (Class 1-2)
│   ├── Class1-2_HW/                   # 核心代码与作业文档
│   │   ├── code/                      # 包含 P1, P2, P3 的 Jupyter Notebook 源文件
│   │   ├── AI重构code/                # AI 辅助重构或优化的代码及模型权重文件 (.pkl)
│   │   └── Class1-2_HW.docx           # 作业报告
│   ├── ML Class 1-2 Homework/         # 导出的网页版笔记与静态网站资源（内含大量用于渲染排版的前端 js/css/fonts/图标）
│   └── README.md                      # 作业一说明文档
├── 作业2/                             # 第二次作业目录 (Class 3-5)
│   ├── Class3-5_HW/                   # 核心代码与作业文档
│   │   ├── code/                      # 包含源文件及 TLC_dataset 数据集
│   │   └── Class3-5_HW.docx           # 作业报告
│   └── ML Class 3-5 Homework/         # 导出的网页版笔记与前端资源
├── 作业3/                             # 第三次作业目录 (Class 6)
│   ├── Class6_HW/                     # 核心代码与测试文件
│   │   ├── code/                      # 包含 TLC_pred.py (预测脚本)、yaml 提示词配置文件、模型权重等
│   │   └── Class6_HW.docx             # 作业报告
│   └── ...                            # 相应的静态网页笔记资源
├── Class contest/                     # 课程期末竞赛核心工程 (ChemE Car)
│   ├── ChemE car contest.ipynb        # 竞赛主控 Jupyter Notebook
│   ├── camera_control.py              # 计算机视觉与摄像头控制逻辑
│   ├── MKSMotor_USB.py                # 电机驱动与串口通信控制代码
│   ├── nn.py                          # 神经网络分类逻辑
│   ├── mnist_model_test.py            # 基于 MNIST 的数字识别测试脚本
│   ├── iodine_clock_data_collect.ipynb# 碘钟反应数据采集脚本
│   └── 说明书与驱动/                  # MKS闭环步进电机说明书，以及 Windows/macOS/Linux 下的 CH340 串口驱动
├── CEAC-比赛报告.docx                 # 竞赛总结报告文档
├── 比赛报告模板(1) 2.docx             # 报告排版模板
├── 课程比赛说明(1).pdf                # 竞赛任务与规则说明
└── 实验数据.xlsx - 4.2.csv            # 相关的实验原始数据记录
````

-----

### 关于 `mnist.csv` 的网络获取链接

如果需要获取 CSV 格式的 MNIST 手写数字数据集（包含像素值展开和标签），通常可以通过以下两个可靠的渠道获取：

**1. PJ Reddie 提供的静态下载链接**
这是最直接的下载地址，非常适合使用代码（如 `wget` 或 Python 的 `requests`）直接拉取，无需登录：

  * **训练集** (mnist\_train.csv，约 104 MB): [https://pjreddie.com/media/files/mnist\_train.csv](https://www.google.com/search?q=https://pjreddie.com/media/files/mnist_train.csv)
  * **测试集** (mnist\_test.csv，约 17.4 MB): [https://pjreddie.com/media/files/mnist\_test.csv](https://www.google.com/search?q=https://pjreddie.com/media/files/mnist_test.csv)

**2. Kaggle 平台数据集**
如果你需要查看数据集的可视化分布或者将其导入 Kaggle Notebook，可以使用这个链接（需要注册/登录 Kaggle 账号才能下载）：

  * **链接**: [https://www.kaggle.com/datasets/oddrationale/mnist-in-csv](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)

*(注：在使用 PyTorch 开发多分类任务时，通常会直接使用 `torchvision.datasets.MNIST` API 来自动下载和解析，其底层处理会比读取庞大的 CSV 文件更加高效。)*