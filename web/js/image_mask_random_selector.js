import { app } from "../../../scripts/app.js";

app.registerExtension({
	name: "ImageMaskRandomSelector.Extension",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (nodeData.name === "ImageMaskRandomSelector") {
			// 保存原始的 onNodeCreated
			const onNodeCreated = nodeType.prototype.onNodeCreated;
			
			nodeType.prototype.onNodeCreated = function () {
				const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

				// 找到控制数量的 widget
				const countWidget = this.widgets.find((w) => w.name === "input_count");
				
				// 定义更新输入槽的函数
				const updateInputs = () => {
					const targetCount = countWidget.value;
					
					// 获取当前已有的动态输入槽（以 image_ 或 mask_ 开头）
					// 注意：this.inputs 可能包含其他固定输入（如果有的话）
					const dynamicInputs = this.inputs ? this.inputs.filter(i => i.name.startsWith("image_") || i.name.startsWith("mask_")) : [];
                    
                    // 计算当前的组数（每组包含 image 和 mask）
                    let currentMaxIndex = 0;
                    if (this.inputs) {
                        this.inputs.forEach(input => {
                            const match = input.name.match(/image_(\d+)/);
                            if (match) {
                                const idx = parseInt(match[1]);
                                if (idx > currentMaxIndex) currentMaxIndex = idx;
                            }
                        });
                    }

                    if (targetCount > currentMaxIndex) {
                        // 需要增加输入槽
                        for (let i = currentMaxIndex + 1; i <= targetCount; i++) {
                            this.addInput(`image_${i}`, "IMAGE");
                            this.addInput(`mask_${i}`, "MASK");
                        }
                    } else if (targetCount < currentMaxIndex) {
                        // 需要减少输入槽
                        for (let i = currentMaxIndex; i > targetCount; i--) {
                            const imgIdx = this.findInputSlot(`image_${i}`);
                            if (imgIdx !== -1) this.removeInput(imgIdx);
                            
                            const maskIdx = this.findInputSlot(`mask_${i}`);
                            if (maskIdx !== -1) this.removeInput(maskIdx);
                        }
                    }
                    
                    // 重新调整节点大小以适应新的输入槽
                    this.setSize(this.computeSize());
				};

				// 监听 widget 变化
				const self = this;
				const origCallback = countWidget.callback;
				countWidget.callback = function () {
					const result = origCallback ? origCallback.apply(this, arguments) : undefined;
					updateInputs();
					return result;
				};

				// 初始调用一次以匹配默认值
				setTimeout(() => {
                    updateInputs();
                }, 100);

				return r;
			};
		}
	},
});
